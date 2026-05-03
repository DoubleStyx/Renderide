//! Blendshape and skinning compute dispatches before the main forward pass.
//!
//! Work items are collected per render space in parallel ([`rayon`]); compute is still recorded
//! sequentially on one [`wgpu::CommandEncoder`].

mod encode;
mod snapshot;

use std::fmt;

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::mesh_deform::{SkinCacheKey, SkinCacheRendererKind};
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{ComputePass, PassBuilder, PassPhase};
use crate::scene::{RenderSpaceId, SceneCoordinator};

use self::encode::{
    MeshDeformEncodeGpu, MeshDeformRecordInputs, MeshDeformRecordStats, record_mesh_deform,
};
use self::snapshot::{
    MeshDeformSnapshot, deform_needs_skin_mesh, entry_need_for_snapshot,
    gpu_mesh_needs_deform_dispatch,
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
}

impl MeshDeformScratch {
    fn new() -> Self {
        Self {
            space_ids: Vec::new(),
            chunks: Vec::new(),
            work: Vec::new(),
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
    /// Current byte offset in the blendshape weight upload buffer.
    blend_weight: u64,
    /// Current byte offset in the packed blendshape parameter upload buffer.
    blend_param: u64,
    /// Current index in the skin dispatch staging buffer.
    skin_dispatch: u64,
}

/// Collects deform work items for one render space (read-only scene + mesh pool).
fn collect_deform_work_for_space(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    space_id: RenderSpaceId,
    work: &mut Vec<DeformWorkItem>,
) {
    work.clear();
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active {
        return;
    }
    for r in &space.static_mesh_renderers {
        if r.mesh_asset_id < 0 {
            continue;
        }
        let Some(m) = mesh_pool.get(r.mesh_asset_id) else {
            continue;
        };
        if !gpu_mesh_needs_deform_dispatch(m, None, &r.blend_shape_weights) {
            continue;
        }
        work.push(DeformWorkItem {
            space_id,
            skin_cache_key: SkinCacheKey::new(
                space_id,
                SkinCacheRendererKind::Static,
                r.instance_id,
            ),
            mesh: MeshDeformSnapshot::from_mesh(m, false),
            skinned: None,
            smr_node_id: -1,
            blend_weights: r.blend_shape_weights.clone(),
        });
    }
    for skinned in &space.skinned_mesh_renderers {
        let r = &skinned.base;
        if r.mesh_asset_id < 0 {
            continue;
        }
        let Some(m) = mesh_pool.get(r.mesh_asset_id) else {
            continue;
        };
        let bone_ix = skinned.bone_transform_indices.as_slice();
        if !gpu_mesh_needs_deform_dispatch(m, Some(bone_ix), &r.blend_shape_weights) {
            continue;
        }
        let clone_bind = deform_needs_skin_mesh(m, Some(bone_ix));
        work.push(DeformWorkItem {
            space_id,
            skin_cache_key: SkinCacheKey::new(
                space_id,
                SkinCacheRendererKind::Skinned,
                r.instance_id,
            ),
            mesh: MeshDeformSnapshot::from_mesh(m, clone_bind),
            skinned: Some(skinned.bone_transform_indices.clone()),
            smr_node_id: r.node_id,
            blend_weights: r.blend_shape_weights.clone(),
        });
    }
}

/// Upper bound on deform work items (static + skinned) across active spaces for scratch reservation.
fn deform_work_upper_bound(scene: &SceneCoordinator) -> usize {
    let mut est = 0usize;
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if space.is_active {
            est = est
                .saturating_add(space.static_mesh_renderers.len())
                .saturating_add(space.skinned_mesh_renderers.len());
        }
    }
    est
}

/// Fills `scratch` with deform work collected in parallel across all render spaces.
fn collect_deform_work_into_scratch(
    scratch: &mut MeshDeformScratch,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
) {
    profiling::scope!("mesh_deform::collect_work");
    let est = deform_work_upper_bound(scene);
    scratch.space_ids.clear();
    scratch.space_ids.extend(scene.render_space_ids());
    let space_count = scratch.space_ids.len();
    if scratch.chunks.len() < space_count {
        scratch.chunks.resize_with(space_count, Vec::new);
    } else {
        scratch.chunks.truncate(space_count);
    }

    {
        let space_ids = &scratch.space_ids;
        let chunks = &mut scratch.chunks;
        space_ids
            .par_iter()
            .copied()
            .zip(chunks.par_iter_mut())
            .for_each(|(space_id, chunk)| {
                collect_deform_work_for_space(scene, mesh_pool, space_id, chunk);
            });
    };

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
    skipped_allocations: u64,
    skin_cache: &crate::mesh_deform::GpuSkinCache,
) {
    let cache_stats = skin_cache.frame_stats();
    crate::profiling::plot_mesh_deform(crate::profiling::MeshDeformProfileSample {
        work_items: work_item_count,
        blend_dispatches: dispatch_stats.blend_dispatches,
        skin_dispatches: dispatch_stats.skin_dispatches,
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
        let frame = &mut *ctx.pass_frame;

        if frame
            .shared
            .frame_resources
            .mesh_deform_dispatched_this_tick()
        {
            return Ok(());
        }

        let mesh_pool = frame.shared.asset_transfers.mesh_pool();

        let mut scratch = self.scratch.lock();
        collect_deform_work_into_scratch(&mut scratch, frame.shared.scene, mesh_pool);

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

        let mut cursors = MeshDeformRecordCursors::default();
        let render_context = frame.shared.scene.active_main_render_context();
        let head_output_transform = frame.view.host_camera.head_output_transform;

        profiling::scope!("mesh_deform::dispatch");
        // Iterate `scratch.work` in place under the lock and clear afterwards: previous code
        // did `drain(..).collect()` which heap-allocated a fresh Vec per frame just to drop
        // the lock early. The mutex is uncontended in practice (the pass runs single-threaded
        // before per-view fan-out), and clearing keeps the buffer's capacity for next frame.
        let work_item_count = scratch.work.len() as u64;
        let (mut dispatch_stats, mut skipped_allocations) =
            (MeshDeformRecordStats::default(), 0u64);
        for item in &scratch.work {
            let need =
                entry_need_for_snapshot(&item.mesh, item.skinned.as_deref(), &item.blend_weights);
            let Some((cache_entry, positions_arena, normals_arena, tangents_arena, temp_arena)) =
                skin_cache.get_or_alloc_with_arenas(
                    ctx.device,
                    ctx.encoder,
                    item.skin_cache_key,
                    need,
                    item.mesh.vertex_count,
                )
            else {
                skipped_allocations = skipped_allocations.saturating_add(1);
                continue;
            };

            let stats = record_mesh_deform(
                MeshDeformEncodeGpu {
                    device: ctx.device,
                    gpu_limits: ctx.gpu_limits,
                    encoder: ctx.encoder,
                    pre,
                    scratch: deform_scratch,
                    upload_batch: ctx.upload_batch,
                    profiler: ctx.profiler,
                },
                MeshDeformRecordInputs {
                    scene: frame.shared.scene,
                    space_id: item.space_id,
                    mesh: &item.mesh,
                    bone_transform_indices: item.skinned.as_deref(),
                    smr_node_id: item.smr_node_id,
                    render_context,
                    head_output_transform,
                    blend_weights: &item.blend_weights,
                    bone_cursor: &mut cursors.bone,
                    blend_weight_cursor: &mut cursors.blend_weight,
                    blend_param_cursor: &mut cursors.blend_param,
                    skin_dispatch_cursor: &mut cursors.skin_dispatch,
                    skin_cache_entry: cache_entry,
                    positions_arena,
                    normals_arena,
                    tangents_arena,
                    temp_arena,
                },
            );
            dispatch_stats.blend_dispatches = dispatch_stats
                .blend_dispatches
                .saturating_add(stats.blend_dispatches);
            dispatch_stats.skin_dispatches = dispatch_stats
                .skin_dispatches
                .saturating_add(stats.skin_dispatches);
        }
        scratch.work.clear();
        drop(scratch);

        let fc = skin_cache.frame_counter();
        report_mesh_deform_stats(
            work_item_count,
            dispatch_stats,
            skipped_allocations,
            skin_cache,
        );
        skin_cache.sweep_stale(fc.saturating_sub(2));

        frame
            .shared
            .frame_resources
            .set_mesh_deform_dispatched_this_tick();
        Ok(())
    }
}

#[cfg(test)]
mod palette_tests {
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
}
