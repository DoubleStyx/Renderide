//! Blendshape and skinning compute dispatches before the main forward pass.
//!
//! Work items are collected per render space in parallel ([`rayon`]); compute is still recorded
//! sequentially on one [`wgpu::CommandEncoder`].

mod encode;
mod snapshot;

use std::{fmt, ops::Range};

use hashbrown::HashSet;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::mesh_deform::{
    SkinCacheKey, SkinCacheRendererKind, SkinningPaletteParams, write_skinning_palette_bytes_serial,
};
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{ComputePass, PassBuilder, PassPhase};
use crate::scene::{RenderSpaceId, SceneCoordinator};

use self::encode::{
    MeshDeformDispatchBatch, MeshDeformEncodeGpu, MeshDeformRecordInputs, MeshDeformRecordStats,
    flush_mesh_deform_batch, record_mesh_deform,
};
use self::snapshot::{
    MeshDeformSnapshot, deform_needs_skin_mesh, deform_needs_skin_snapshot,
    entry_need_for_snapshot, gpu_mesh_needs_deform_dispatch,
};

/// Encodes mesh deformation compute for all active render spaces.
///
/// Per-frame collection reuses scratch buffers (held inside a [`parking_lot::Mutex`]) to avoid
/// `Vec` allocations on the hot path. [`parking_lot::Mutex`] is chosen over [`std::sync::Mutex`]
/// so the `lock` API is infallible and the per-frame record path does not need poison-handling.
/// The mutex is never contended in practice because [`MeshDeformPass::phase`] returns
/// [`PassPhase::FrameGlobal`], so `record` always runs on the main thread before per-view
/// parallel encoding begins.
pub struct MeshDeformPass {
    scratch: Mutex<MeshDeformScratch>,
}

/// Reusable per-frame scratch buffers for mesh deform work collection.
struct MeshDeformScratch {
    /// Reused ordering of [`SceneCoordinator::render_space_ids`] for parallel per-space collection.
    space_ids: Vec<RenderSpaceId>,
    /// One bucket per render space; inner [`Vec`] capacities are reused across frames.
    chunks: Vec<Vec<DeformWorkItem>>,
    /// Flattened work list passed to encode (cleared after each successful dispatch).
    work: Vec<DeformWorkItem>,
    /// Work indices whose cache entries were allocated before batched encode begins.
    ready_work_indices: Vec<usize>,
    /// Reused compute dispatch batch for the encode phase.
    dispatch_batch: MeshDeformDispatchBatch,
}

impl MeshDeformScratch {
    fn new() -> Self {
        Self {
            space_ids: Vec::new(),
            chunks: Vec::new(),
            work: Vec::new(),
            ready_work_indices: Vec::new(),
            dispatch_batch: MeshDeformDispatchBatch::new(),
        }
    }
}

impl fmt::Debug for MeshDeformPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeshDeformPass").finish_non_exhaustive()
    }
}

impl Default for MeshDeformPass {
    fn default() -> Self {
        Self {
            scratch: Mutex::new(MeshDeformScratch::new()),
        }
    }
}

struct DeformWorkItem {
    space_id: RenderSpaceId,
    /// Stable renderer identity for GPU skin cache ownership.
    skin_cache_key: SkinCacheKey,
    mesh: MeshDeformSnapshot,
    /// Mesh pool generation observed when this snapshot was collected.
    mesh_pool_generation: u64,
    skinned: Option<Vec<i32>>,
    /// [`crate::scene::StaticMeshRenderer::node_id`] (SMR) for skinning fallbacks when a bone is unmapped.
    smr_node_id: i32,
    blend_weights: Vec<f32>,
}

/// Upload cursor state shared across all deform dispatches recorded for a frame.
#[derive(Default)]
struct MeshDeformRecordCursors {
    /// Current byte offset in the bone palette upload buffer.
    bone: u64,
    /// Current byte offset in the packed blendshape parameter upload buffer.
    blend_param: u64,
    /// Current index in the skin dispatch staging buffer.
    skin_dispatch: u64,
}

struct MeshDeformDispatchResult {
    work_item_count: u64,
    dispatch_stats: MeshDeformRecordStats,
    skipped_allocations: u64,
}

#[derive(Clone, Copy)]
struct MeshDeformDispatchCtx<'a> {
    scene: &'a SceneCoordinator,
    render_context: crate::shared::RenderingContext,
    head_output_transform: glam::Mat4,
}

/// Renderer count assigned to one deform collection worker chunk.
const DEFORM_COLLECT_RENDERER_CHUNK_SIZE: usize = 64;
/// Renderer count above which deform work collection fans out across two chunks.
const DEFORM_COLLECT_PARALLEL_MIN_RENDERERS: usize = DEFORM_COLLECT_RENDERER_CHUNK_SIZE * 2;
/// Renderer chunks assigned to one deform collection worker task.
const DEFORM_COLLECT_PARALLEL_CHUNK_TASKS: usize = 1;
/// Renderer chunk count required before deform collection fans out.
const DEFORM_COLLECT_PARALLEL_MIN_CHUNKS: usize = DEFORM_COLLECT_PARALLEL_CHUNK_TASKS * 2;
/// Render spaces assigned to one deform collection worker.
const DEFORM_SPACE_PARALLEL_CHUNK_SPACES: usize = 1;
/// Render-space count required before deform collection fans out across spaces.
const DEFORM_SPACE_PARALLEL_MIN_SPACES: usize = DEFORM_SPACE_PARALLEL_CHUNK_SPACES * 2;
/// Skinned deform work items assigned to one palette-preplan worker.
const DEFORM_PREPLAN_PARALLEL_CHUNK_ITEMS: usize = 16;
/// Skinned deform work item count required before palette preplanning uses Rayon.
const DEFORM_PREPLAN_PARALLEL_MIN_ITEMS: usize = DEFORM_PREPLAN_PARALLEL_CHUNK_ITEMS * 2;

#[derive(Clone, Copy)]
enum DeformCollectChunkKind {
    Static,
    Skinned,
}

#[derive(Clone)]
struct DeformCollectChunkSpec {
    kind: DeformCollectChunkKind,
    range: Range<usize>,
}

/// Collects deform work items for one render space (read-only scene + mesh pool).
fn collect_deform_work_for_space(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    visible_filter: Option<&HashSet<SkinCacheKey>>,
    space_id: RenderSpaceId,
    work: &mut Vec<DeformWorkItem>,
) {
    work.clear();
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    for r in space.static_mesh_renderers() {
        push_static_deform_work(scene, mesh_pool, visible_filter, space_id, r, work);
    }
    for skinned in space.skinned_mesh_renderers() {
        push_skinned_deform_work(scene, mesh_pool, visible_filter, space_id, skinned, work);
    }
}

fn push_static_deform_work(
    _scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    visible_filter: Option<&HashSet<SkinCacheKey>>,
    space_id: RenderSpaceId,
    r: &crate::scene::StaticMeshRenderer,
    work: &mut Vec<DeformWorkItem>,
) {
    if r.mesh_asset_id < 0 {
        return;
    }
    let skin_cache_key = SkinCacheKey::new(space_id, SkinCacheRendererKind::Static, r.instance_id);
    if visible_filter.is_some_and(|keys| !keys.contains(&skin_cache_key)) {
        return;
    }
    let Some(m) = mesh_pool.get(r.mesh_asset_id) else {
        return;
    };
    if !gpu_mesh_needs_deform_dispatch(m, None, &r.blend_shape_weights) {
        return;
    }
    work.push(DeformWorkItem {
        space_id,
        skin_cache_key,
        mesh: MeshDeformSnapshot::from_mesh(m, false),
        mesh_pool_generation: mesh_pool.mutation_generation(),
        skinned: None,
        smr_node_id: -1,
        blend_weights: r.blend_shape_weights.clone(),
    });
}

fn push_skinned_deform_work(
    _scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    visible_filter: Option<&HashSet<SkinCacheKey>>,
    space_id: RenderSpaceId,
    skinned: &crate::scene::SkinnedMeshRenderer,
    work: &mut Vec<DeformWorkItem>,
) {
    let r = &skinned.base;
    if r.mesh_asset_id < 0 {
        return;
    }
    let skin_cache_key = SkinCacheKey::new(space_id, SkinCacheRendererKind::Skinned, r.instance_id);
    if visible_filter.is_some_and(|keys| !keys.contains(&skin_cache_key)) {
        return;
    }
    let Some(m) = mesh_pool.get(r.mesh_asset_id) else {
        return;
    };
    let bone_ix = skinned.bone_transform_indices.as_slice();
    if !gpu_mesh_needs_deform_dispatch(m, Some(bone_ix), &r.blend_shape_weights) {
        return;
    }
    let clone_bind = deform_needs_skin_mesh(m, Some(bone_ix));
    work.push(DeformWorkItem {
        space_id,
        skin_cache_key,
        mesh: MeshDeformSnapshot::from_mesh(m, clone_bind),
        mesh_pool_generation: mesh_pool.mutation_generation(),
        skinned: Some(skinned.bone_transform_indices.clone()),
        smr_node_id: r.node_id,
        blend_weights: r.blend_shape_weights.clone(),
    });
}

/// Upper bound on deform work items (static + skinned) across active spaces for scratch reservation.
fn deform_work_upper_bound(scene: &SceneCoordinator) -> usize {
    let mut est = 0usize;
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if space.is_active() {
            est = est
                .saturating_add(space.static_mesh_renderers().len())
                .saturating_add(space.skinned_mesh_renderers().len());
        }
    }
    est
}

fn renderer_count_for_deform_space(scene: &SceneCoordinator, space_id: RenderSpaceId) -> usize {
    scene.space(space_id).map_or(0, |space| {
        if space.is_active() {
            space
                .static_mesh_renderers()
                .len()
                .saturating_add(space.skinned_mesh_renderers().len())
        } else {
            0
        }
    })
}

fn push_deform_collect_chunks(
    specs: &mut Vec<DeformCollectChunkSpec>,
    kind: DeformCollectChunkKind,
    len: usize,
) {
    let mut start = 0usize;
    while start < len {
        let end = len.min(start + DEFORM_COLLECT_RENDERER_CHUNK_SIZE);
        specs.push(DeformCollectChunkSpec {
            kind,
            range: start..end,
        });
        start = end;
    }
}

fn collect_deform_work_for_chunk(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    visible_filter: Option<&HashSet<SkinCacheKey>>,
    space_id: RenderSpaceId,
    spec: &DeformCollectChunkSpec,
    work: &mut Vec<DeformWorkItem>,
) {
    work.clear();
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    match spec.kind {
        DeformCollectChunkKind::Static => {
            for renderable_index in spec.range.clone() {
                let r = &space.static_mesh_renderers()[renderable_index];
                push_static_deform_work(scene, mesh_pool, visible_filter, space_id, r, work);
            }
        }
        DeformCollectChunkKind::Skinned => {
            for renderable_index in spec.range.clone() {
                let skinned = &space.skinned_mesh_renderers()[renderable_index];
                push_skinned_deform_work(scene, mesh_pool, visible_filter, space_id, skinned, work);
            }
        }
    }
}

fn collect_deform_work_for_space_aggressive(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    visible_filter: Option<&HashSet<SkinCacheKey>>,
    space_id: RenderSpaceId,
    chunks: &mut Vec<Vec<DeformWorkItem>>,
    work: &mut Vec<DeformWorkItem>,
) {
    work.clear();
    if renderer_count_for_deform_space(scene, space_id) < DEFORM_COLLECT_PARALLEL_MIN_RENDERERS {
        collect_deform_work_for_space(scene, mesh_pool, visible_filter, space_id, work);
        return;
    }
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }

    let mut specs = Vec::new();
    push_deform_collect_chunks(
        &mut specs,
        DeformCollectChunkKind::Static,
        space.static_mesh_renderers().len(),
    );
    push_deform_collect_chunks(
        &mut specs,
        DeformCollectChunkKind::Skinned,
        space.skinned_mesh_renderers().len(),
    );
    if specs.len() < DEFORM_COLLECT_PARALLEL_MIN_CHUNKS {
        collect_deform_work_for_space(scene, mesh_pool, visible_filter, space_id, work);
        return;
    }
    if chunks.len() < specs.len() {
        chunks.resize_with(specs.len(), Vec::new);
    }
    chunks
        .par_iter_mut()
        .take(specs.len())
        .with_min_len(DEFORM_COLLECT_PARALLEL_CHUNK_TASKS)
        .zip(
            specs
                .par_iter()
                .with_min_len(DEFORM_COLLECT_PARALLEL_CHUNK_TASKS),
        )
        .for_each(|(chunk, spec)| {
            profiling::scope!("mesh_deform::collect_work::renderer_chunk_worker");
            collect_deform_work_for_chunk(scene, mesh_pool, visible_filter, space_id, spec, chunk);
        });
    let est = chunks.iter().take(specs.len()).map(Vec::len).sum::<usize>();
    work.reserve(est);
    for chunk in chunks.iter_mut().take(specs.len()) {
        work.append(chunk);
    }
}

/// Fills `scratch` with deform work collected in parallel across all render spaces.
fn collect_deform_work_into_scratch(
    scratch: &mut MeshDeformScratch,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    visible_filter: Option<&HashSet<SkinCacheKey>>,
) {
    profiling::scope!("mesh_deform::collect_work");
    if visible_filter.is_some_and(|keys| keys.is_empty()) {
        scratch.space_ids.clear();
        scratch.work.clear();
        for chunk in &mut scratch.chunks {
            chunk.clear();
        }
        return;
    }
    let est = deform_work_upper_bound(scene);
    scratch.space_ids.clear();
    scratch.space_ids.extend(scene.render_space_ids());
    let space_count = scratch.space_ids.len();
    if scratch.chunks.len() < space_count {
        scratch.chunks.resize_with(space_count, Vec::new);
    } else {
        scratch.chunks.truncate(space_count);
    }

    match space_count {
        0 => {}
        1 => {
            let space_id = scratch.space_ids[0];
            collect_deform_work_for_space_aggressive(
                scene,
                mesh_pool,
                visible_filter,
                space_id,
                &mut scratch.chunks,
                &mut scratch.work,
            );
            return;
        }
        _ if space_count >= DEFORM_SPACE_PARALLEL_MIN_SPACES => {
            let space_ids = &scratch.space_ids;
            let chunks = &mut scratch.chunks;
            space_ids
                .par_iter()
                .with_min_len(DEFORM_SPACE_PARALLEL_CHUNK_SPACES)
                .copied()
                .zip(
                    chunks
                        .par_iter_mut()
                        .with_min_len(DEFORM_SPACE_PARALLEL_CHUNK_SPACES),
                )
                .for_each(|(space_id, chunk)| {
                    profiling::scope!("mesh_deform::collect_work::space_worker");
                    collect_deform_work_for_space(
                        scene,
                        mesh_pool,
                        visible_filter,
                        space_id,
                        chunk,
                    );
                });
        }
        _ => {}
    }

    scratch.work.clear();
    scratch.work.reserve(est);
    for chunk in &mut scratch.chunks {
        scratch.work.append(chunk);
    }
}

/// Emits mesh-deform Tracy plots and user-visible warnings for cache allocation pressure.
fn report_mesh_deform_stats(
    work_item_count: u64,
    dispatch_stats: MeshDeformRecordStats,
    scratch_buffer_grows: u64,
    skipped_allocations: u64,
    skin_cache: &crate::mesh_deform::GpuSkinCache,
) {
    let cache_stats = skin_cache.frame_stats();
    crate::profiling::plot_mesh_deform(crate::profiling::MeshDeformProfileSample {
        work_items: work_item_count,
        compute_passes: dispatch_stats.compute_passes,
        bind_groups_created: dispatch_stats.bind_groups_created,
        bind_group_cache_reuses: dispatch_stats.bind_group_cache_reuses,
        copy_ops: dispatch_stats.copy_ops,
        blend_dispatches: dispatch_stats.blend_dispatches,
        skin_dispatches: dispatch_stats.skin_dispatches,
        stable_skips: dispatch_stats.stable_skips,
        scratch_buffer_grows,
        skipped_allocations,
        cache_reuses: cache_stats.reuses,
        cache_allocations: cache_stats.allocations,
        cache_grows: cache_stats.grows,
        cache_evictions: cache_stats.evictions,
        cache_current_frame_eviction_refusals: cache_stats.current_frame_eviction_refusals,
    });
    if skipped_allocations > 0 {
        logger::warn!(
            "mesh deform: skipped {} work items because the skin cache could not allocate without evicting current-frame entries",
            skipped_allocations
        );
    }
}

/// Prepares CPU skinning palette bytes before the serial encode loop when enough skinned work exists.
fn preplan_skinning_palettes(
    work: &[DeformWorkItem],
    ready_work_indices: &[usize],
    dispatch_ctx: MeshDeformDispatchCtx<'_>,
) -> Vec<Option<Vec<u8>>> {
    let skinned_count = ready_work_indices
        .iter()
        .filter(|&&work_index| {
            work.get(work_index)
                .is_some_and(|item| deform_needs_skin_snapshot(&item.mesh, item.skinned.as_deref()))
        })
        .count();
    if skinned_count < DEFORM_PREPLAN_PARALLEL_MIN_ITEMS || rayon::current_num_threads() <= 1 {
        return Vec::new();
    }
    profiling::scope!("mesh_deform::preplan_skinning_palettes_parallel");
    ready_work_indices
        .par_iter()
        .with_min_len(DEFORM_PREPLAN_PARALLEL_CHUNK_ITEMS)
        .map(|&work_index| {
            work.get(work_index)
                .and_then(|item| preplan_skinning_palette_for_item(item, dispatch_ctx))
        })
        .collect()
}

/// Builds one work item's skinning palette using worker-owned scratch.
fn preplan_skinning_palette_for_item(
    item: &DeformWorkItem,
    dispatch_ctx: MeshDeformDispatchCtx<'_>,
) -> Option<Vec<u8>> {
    if !deform_needs_skin_snapshot(&item.mesh, item.skinned.as_deref()) {
        return None;
    }
    let mut bytes = Vec::new();
    write_skinning_palette_bytes_serial(
        SkinningPaletteParams {
            scene: dispatch_ctx.scene,
            space_id: item.space_id,
            skinning_bind_matrices: &item.mesh.skinning_bind_matrices,
            has_skeleton: item.mesh.has_skeleton,
            bone_transform_indices: item.skinned.as_deref()?,
            smr_node_id: item.smr_node_id,
            render_context: dispatch_ctx.render_context,
            head_output_transform: dispatch_ctx.head_output_transform,
        },
        &mut bytes,
    )?;
    Some(bytes)
}

fn dispatch_mesh_deform_work(
    mut gpu: MeshDeformEncodeGpu<'_>,
    skin_cache: &mut crate::mesh_deform::GpuSkinCache,
    work: &[DeformWorkItem],
    ready_work_indices: &mut Vec<usize>,
    batch: &mut MeshDeformDispatchBatch,
    dispatch_ctx: MeshDeformDispatchCtx<'_>,
) -> MeshDeformDispatchResult {
    profiling::scope!("mesh_deform::dispatch");
    let mut cursors = MeshDeformRecordCursors::default();
    let mut result = MeshDeformDispatchResult {
        work_item_count: work.len() as u64,
        dispatch_stats: MeshDeformRecordStats::default(),
        skipped_allocations: 0,
    };

    ready_work_indices.clear();
    ready_work_indices.reserve(work.len());
    for (work_index, item) in work.iter().enumerate() {
        let need =
            entry_need_for_snapshot(&item.mesh, item.skinned.as_deref(), &item.blend_weights);
        if skin_cache.prepare_entry(
            gpu.device,
            &mut *gpu.encoder,
            gpu.profiler,
            item.skin_cache_key,
            need,
            item.mesh.vertex_count,
        ) {
            ready_work_indices.push(work_index);
        } else {
            result.skipped_allocations = result.skipped_allocations.saturating_add(1);
        }
    }
    let preplanned_palettes = preplan_skinning_palettes(work, ready_work_indices, dispatch_ctx);

    batch.clear();
    for (ready_index, &work_index) in ready_work_indices.iter().enumerate() {
        let Some(item) = work.get(work_index) else {
            continue;
        };
        let prepared_skinning_palette_bytes = preplanned_palettes
            .get(ready_index)
            .and_then(Option::as_deref);
        let previous_signature = skin_cache.entry_deform_signature(&item.skin_cache_key);
        let record = {
            let Some((cache_entry, positions_arena, normals_arena, tangents_arena, temp_arena)) =
                skin_cache.lookup_current_with_arenas(&item.skin_cache_key)
            else {
                continue;
            };

            record_mesh_deform(
                &mut gpu,
                MeshDeformRecordInputs {
                    scene: dispatch_ctx.scene,
                    space_id: item.space_id,
                    mesh: &item.mesh,
                    mesh_pool_generation: item.mesh_pool_generation,
                    bone_transform_indices: item.skinned.as_deref(),
                    smr_node_id: item.smr_node_id,
                    render_context: dispatch_ctx.render_context,
                    head_output_transform: dispatch_ctx.head_output_transform,
                    prepared_skinning_palette_bytes,
                    blend_weights: &item.blend_weights,
                    previous_signature,
                    bone_cursor: &mut cursors.bone,
                    blend_param_cursor: &mut cursors.blend_param,
                    skin_dispatch_cursor: &mut cursors.skin_dispatch,
                    skin_cache_entry: cache_entry,
                    positions_arena,
                    normals_arena,
                    tangents_arena,
                    temp_arena,
                },
                batch,
            )
        };
        result.dispatch_stats.add(record.stats);
        if let Some(signature) = record.signature_to_store {
            skin_cache.set_entry_deform_signature(item.skin_cache_key, signature);
        }
    }
    result
        .dispatch_stats
        .add(flush_mesh_deform_batch(&mut gpu, batch));
    ready_work_indices.clear();
    result
}

impl MeshDeformPass {
    /// Creates a mesh deform pass with empty scratch buffers (filled lazily on first execute).
    pub fn new() -> Self {
        Self::default()
    }
}

impl ComputePass for MeshDeformPass {
    fn name(&self) -> &str {
        "MeshDeform"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.cull_exempt();
        Ok(())
    }

    fn phase(&self) -> PassPhase {
        PassPhase::FrameGlobal
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("mesh_deform::pass_record");
        let frame = &mut *ctx.pass_frame;

        if frame
            .shared
            .frame_resources
            .mesh_deform_dispatched_this_submission()
        {
            return Ok(());
        }

        let mesh_pool = frame.shared.asset_resources.mesh_pool();
        let visible_filter = frame
            .shared
            .frame_resources
            .visible_mesh_deform_keys_snapshot();

        let mut scratch = self.scratch.lock();
        collect_deform_work_into_scratch(
            &mut scratch,
            frame.shared.scene,
            mesh_pool,
            visible_filter.as_ref(),
        );

        let Some(pre) = frame.shared.mesh_preprocess else {
            scratch.work.clear();
            return Ok(());
        };
        let Some(deform_scratch) = frame.shared.mesh_deform_scratch.as_mut() else {
            scratch.work.clear();
            return Ok(());
        };
        let Some(skin_cache) = frame.shared.mesh_deform_skin_cache.as_mut() else {
            scratch.work.clear();
            return Ok(());
        };

        let render_context = frame.shared.scene.active_main_render_context();
        let head_output_transform = frame.view.host_camera.head_output_transform;
        let mut ready_work_indices = std::mem::take(&mut scratch.ready_work_indices);
        let mut dispatch_batch = std::mem::take(&mut scratch.dispatch_batch);
        // Iterate `scratch.work` in place under the lock and clear afterwards: previous code
        // did `drain(..).collect()` which heap-allocated a fresh Vec per frame just to drop
        // the lock early. The mutex is uncontended in practice (the pass runs single-threaded
        // before per-view fan-out), and clearing keeps the buffer's capacity for next frame.
        let dispatch = dispatch_mesh_deform_work(
            MeshDeformEncodeGpu {
                device: ctx.device,
                gpu_limits: ctx.gpu_limits,
                encoder: ctx.encoder,
                pre,
                scratch: deform_scratch,
                uploads: ctx.uploads,
                profiler: ctx.profiler,
            },
            skin_cache,
            &scratch.work,
            &mut ready_work_indices,
            &mut dispatch_batch,
            MeshDeformDispatchCtx {
                scene: frame.shared.scene,
                render_context,
                head_output_transform,
            },
        );
        scratch.ready_work_indices = ready_work_indices;
        scratch.dispatch_batch = dispatch_batch;
        scratch.work.clear();
        drop(scratch);
        let scratch_buffer_grows = deform_scratch.take_frame_grow_count();

        let fc = skin_cache.frame_counter();
        report_mesh_deform_stats(
            dispatch.work_item_count,
            dispatch.dispatch_stats,
            scratch_buffer_grows,
            dispatch.skipped_allocations,
            skin_cache,
        );
        skin_cache.sweep_stale(fc.saturating_sub(2));

        frame
            .shared
            .frame_resources
            .set_mesh_deform_dispatched_this_submission();
        Ok(())
    }
}

#[cfg(test)]
mod palette_tests {
    use super::*;
    use crate::scene::{MeshRendererInstanceId, SceneCoordinator, StaticMeshRenderer};
    use glam::{Mat3, Mat4, Vec3};

    #[test]
    fn palette_is_world_times_bind() {
        let world = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let bind = Mat4::from_scale(Vec3::splat(2.0));
        let pal = world * bind;
        let expected = world * bind;
        assert!(pal.abs_diff_eq(expected, 1e-5));
    }

    /// Matches WGSL `transpose(inverse(mat3_linear(M)))` for rigid rotations: equals the linear part.
    #[test]
    fn normal_matrix_inverse_transpose_is_rotation_for_orthogonal() {
        let m3 = Mat3::from_axis_angle(Vec3::Z, 1.15);
        let inv_t = m3.inverse().transpose();
        assert!(inv_t.abs_diff_eq(m3, 1e-5));
    }

    fn assert_deform_chunk(
        spec: &DeformCollectChunkSpec,
        kind: DeformCollectChunkKind,
        range: Range<usize>,
    ) {
        match (spec.kind, kind) {
            (DeformCollectChunkKind::Static, DeformCollectChunkKind::Static)
            | (DeformCollectChunkKind::Skinned, DeformCollectChunkKind::Skinned) => {}
            _ => panic!("unexpected deform collection chunk kind"),
        }
        assert_eq!(spec.range, range);
    }

    #[test]
    fn deform_collect_chunks_preserve_static_then_skinned_order() {
        let mut specs = Vec::new();
        push_deform_collect_chunks(&mut specs, DeformCollectChunkKind::Static, 130);
        push_deform_collect_chunks(&mut specs, DeformCollectChunkKind::Skinned, 70);

        assert_eq!(specs.len(), 5);
        assert_deform_chunk(&specs[0], DeformCollectChunkKind::Static, 0..64);
        assert_deform_chunk(&specs[1], DeformCollectChunkKind::Static, 64..128);
        assert_deform_chunk(&specs[2], DeformCollectChunkKind::Static, 128..130);
        assert_deform_chunk(&specs[3], DeformCollectChunkKind::Skinned, 0..64);
        assert_deform_chunk(&specs[4], DeformCollectChunkKind::Skinned, 64..70);
    }

    #[test]
    fn aggressive_deform_collect_matches_serial_for_missing_meshes() {
        let mut scene = SceneCoordinator::new();
        let space_id = RenderSpaceId(1);
        let renderers = (0..DEFORM_COLLECT_PARALLEL_MIN_RENDERERS + 9)
            .map(|idx| StaticMeshRenderer {
                instance_id: MeshRendererInstanceId(idx as u64 + 1),
                mesh_asset_id: 7,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        scene.test_insert_static_mesh_renderers(space_id, renderers);
        scene.test_set_space_active(space_id, true);
        let mesh_pool = MeshPool::default_pool();
        let mut serial = Vec::new();
        let mut aggressive = Vec::new();
        let mut chunks = Vec::new();

        collect_deform_work_for_space(&scene, &mesh_pool, None, space_id, &mut serial);
        collect_deform_work_for_space_aggressive(
            &scene,
            &mesh_pool,
            None,
            space_id,
            &mut chunks,
            &mut aggressive,
        );

        assert!(serial.is_empty());
        assert_eq!(aggressive.len(), serial.len());
        assert!(chunks.len() >= 2);
    }
}
