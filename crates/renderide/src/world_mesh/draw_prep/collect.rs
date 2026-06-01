//! Scene walk that pairs material slots with submesh ranges and applies optional CPU culling.
//!
//! [`queue_draws_with_parallelism`] walks each render space in 128-renderable parallel chunks
//! ([`rayon`]), merges in [`SceneCoordinator::render_space_ids`] order, assigns
//! [`WorldMeshDrawItem::collect_order`]. The caller then runs the explicit sort phase.
//!
//! Material-derived batch key fields are computed once per `(material_asset_id, property_block_id)`
//! per call via [`FrameMaterialBatchCache`] before the parallel phase begins. This eliminates
//! repeated dictionary and router lookups for the common case where hundreds of draws share a
//! few dozen materials.

use hashbrown::HashMap;

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::cpu_parallelism::{
    ParallelAdmission, admit_render_command_items, current_reference_worker_count,
    has_visibility_parallel_work, record_parallel_admission,
};
use crate::gpu_pools::MeshPool;
use crate::materials::ShaderPermutation;
use crate::materials::host_data::MaterialDictionary;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::WorldMeshCullInput;
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::arrange::arrange_draw_chunks_by_phase_bins;
use super::filter::CameraTransformDrawFilter;
use super::item::{WorldMeshDrawCollection, WorldMeshDrawItem, WorldMeshVisibilityStats};
use super::prepared_renderables::FramePreparedRenderables;

mod candidate;
mod filter_masks;
mod lod;
pub(super) mod prepared;
mod scene_walk;
mod world_matrix;

use filter_masks::build_per_space_filter_masks;
use lod::{LodVisibility, build_lod_visibility};
use prepared::collect_prepared_chunk;
use scene_walk::{build_chunk_specs, collect_chunk, estimate_active_renderable_count};

const SPATIAL_QUERY_RUN_CHUNK_TARGET: usize = 32;

#[cfg(test)]
use super::prepared_renderables::FramePreparedDraw;
#[cfg(test)]
use prepared::prepared_draws_share_renderer;
#[cfg(test)]
use scene_walk::transform_chain_has_degenerate_scale;

/// Prepared renderer-run chunks assigned to one draw-collection worker.
const PREPARED_COLLECT_PARALLEL_CHUNK_TASKS: usize = 1;
/// Scene-walk chunk specs assigned to one draw-collection worker.
const SCENE_COLLECT_PARALLEL_CHUNK_TASKS: usize = 1;
/// Scene-walk chunk count required before draw collection fans out.
const SCENE_COLLECT_PARALLEL_MIN_CHUNKS: usize = SCENE_COLLECT_PARALLEL_CHUNK_TASKS * 2;

/// Returns the admission decision for prepared renderer-run chunk collection.
fn prepared_collect_admission(
    task_count: usize,
    draw_count: usize,
    worker_count: usize,
) -> ParallelAdmission {
    let draw_admission = admit_render_command_items(draw_count, worker_count);
    if task_count >= PREPARED_COLLECT_PARALLEL_CHUNK_TASKS * 2 && draw_admission.is_parallel() {
        draw_admission
    } else {
        ParallelAdmission::Serial
    }
}

/// Returns the scene-walk fallback admission decision for a known worker count.
fn scene_collect_admission(
    chunk_count: usize,
    work_units: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if chunk_count >= SCENE_COLLECT_PARALLEL_MIN_CHUNKS
        && has_visibility_parallel_work(work_units, worker_count)
    {
        ParallelAdmission::Parallel {
            chunk_size: SCENE_COLLECT_PARALLEL_CHUNK_TASKS,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Per-chunk CPU cull counters: pre-cull, frustum-culled, and Hi-Z-culled draws.
type WorldMeshChunkCullStats = (usize, usize, usize);

/// Draw items and cull counters produced by one collection worker.
type WorldMeshDrawChunk = (Vec<WorldMeshDrawItem>, WorldMeshChunkCullStats);

/// Draw chunks plus broadphase visibility counters produced by collection.
struct WorldMeshCollectedChunks {
    /// Worker-local draw chunks in deterministic merge order.
    chunks: Vec<WorldMeshDrawChunk>,
    /// Visibility broadphase counters gathered during candidate selection.
    visibility: WorldMeshVisibilityStats,
}

impl WorldMeshCollectedChunks {
    /// Builds collected chunks with the supplied visibility counters.
    fn new(chunks: Vec<WorldMeshDrawChunk>, visibility: WorldMeshVisibilityStats) -> Self {
        Self { chunks, visibility }
    }
}

/// Read-only scene, material, and cull state shared across all spaces during draw collection.
pub struct DrawCollectionContext<'a> {
    /// Scene graph for mesh renderables.
    pub scene: &'a SceneCoordinator,
    /// Resident meshes (submeshes, deform buffers).
    pub mesh_pool: &'a MeshPool,
    /// Material property dictionary for batch keys.
    pub material_dict: &'a MaterialDictionary<'a>,
    /// Shader stem / pipeline routing.
    pub material_router: &'a MaterialRouter,
    /// Interned material property ids that affect pipeline state.
    pub pipeline_property_ids: &'a MaterialPipelinePropertyIds,
    /// Default vs multiview permutation for embedded materials.
    pub shader_perm: ShaderPermutation,
    /// Mono vs stereo / overlay render context.
    pub render_context: RenderingContext,
    /// Head / rig transform for world matrix resolution.
    pub head_output_transform: Mat4,
    /// Camera world position for back-to-front distance sorting of transparent draws.
    ///
    /// Populate from `HostCameraFrame::view_origin_world()`.
    pub view_origin_world: Vec3,
    /// Optional CPU frustum + Hi-Z cull inputs.
    pub culling: Option<&'a WorldMeshCullInput<'a>>,
    /// Unity-style mesh LOD bias multiplier for relative screen-height selection.
    pub mesh_lod_bias: f32,
    /// Optional per-camera node filter.
    pub transform_filter: Option<&'a CameraTransformDrawFilter>,
    /// Optional render-space scope for offscreen cameras/tasks.
    pub render_space_filter: Option<RenderSpaceId>,
    /// Optional pre-built material batch cache shared across multiple views in the same frame.
    ///
    /// When `Some`, collection reuses the shared cache instead of rebuilding one per call. Callers
    /// that render multiple views in one frame (secondary render-texture cameras + main
    /// swapchain) should build the cache once via [`FrameMaterialBatchCache::build_for_frame`] and
    /// hand the same borrow to every per-view context. When `None`, a fresh cache is built
    /// internally for this call (backwards-compatible single-view path).
    pub material_cache: Option<&'a FrameMaterialBatchCache>,
    /// Optional frame reflection-probe selector used to choose the set of specular IBL probes to use per draw.
    pub reflection_probes: Option<&'a ReflectionProbeFrameSelection>,
    /// Optional pre-expanded dense draw list shared across multiple views in the same frame.
    ///
    /// When `Some`, collection iterates the flat list instead of walking every active render
    /// space and looking up mesh pool entries per view. The prepared list must have been built
    /// for the **same** [`Self::render_context`] used here; otherwise material-override
    /// resolution may disagree. Single-view callers can leave this `None` and fall back to the
    /// scene-walk path.
    pub prepared: Option<&'a FramePreparedRenderables>,
}

/// How [`queue_draws_with_parallelism`] parallelizes per-chunk collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawCollectParallelism {
    /// Per-chunk collection uses rayon.
    Full,
    /// Serial per-chunk collection; use when an outer `par_iter` already fans out (e.g. multiple secondary RTs).
    SerialInnerForNestedBatch,
}

/// How final per-view draw arrangement may use Rayon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawArrangeParallelism {
    /// Draw chunk partitioning and strict-order sorting may use rayon when the work is large enough.
    Full,
    /// Draw arrangement stays on the caller thread.
    Serial,
}

/// Draw candidates queued for one view before final phase sorting and arrangement.
pub struct QueuedWorldMeshDraws {
    /// Candidate draw chunks in deterministic scene collection order.
    chunks: Vec<Vec<WorldMeshDrawItem>>,
    /// Number of queued draw candidates across all chunks.
    len: usize,
    /// Number of candidate draws before CPU culling.
    draws_pre_cull: usize,
    /// Number of candidate draws rejected by CPU frustum culling.
    draws_culled: usize,
    /// Number of candidate draws rejected by temporal Hi-Z culling.
    draws_hi_z_culled: usize,
    /// Visibility broadphase counters gathered before per-renderer prepared draw expansion.
    visibility: WorldMeshVisibilityStats,
}

/// Prepared draw collection state derived once per view before chunk dispatch.
struct PreparedCollectionState<'a> {
    /// Pre-expanded frame draw snapshot for this view's render context.
    prepared: &'a FramePreparedRenderables,
    /// Material batch keys refreshed for this view's render context and shader permutation.
    cache: &'a FrameMaterialBatchCache,
    /// Active prepared render spaces relevant to this view.
    space_ids: Vec<RenderSpaceId>,
    /// Per-space camera transform-filter masks.
    filter_masks: HashMap<RenderSpaceId, Vec<bool>>,
    /// Per-view LOD visibility decisions.
    lod_visibility: LodVisibility,
    /// Expected retained draw count for output capacity.
    cap_hint: usize,
}

/// One flat prepared-collection task for a specific view and run chunk.
#[derive(Clone, Copy)]
struct PreparedViewChunkTask {
    /// View index in the caller-provided context slice.
    view_index: usize,
    /// Prepared renderer-run chunk index within the view's prepared snapshot.
    chunk_index: usize,
}

impl QueuedWorldMeshDraws {
    /// Number of queued draw candidates before arrangement.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Sorts and arranges queued draws into render-phase submission order.
    pub fn sort_and_arrange(
        self,
        parallelism: WorldMeshDrawArrangeParallelism,
    ) -> WorldMeshDrawCollection {
        let (items, arrangement) = {
            profiling::scope!("mesh::arrange");
            arrange_draw_chunks_by_phase_bins(
                self.chunks,
                parallelism == WorldMeshDrawArrangeParallelism::Full,
            )
        };
        WorldMeshDrawCollection {
            items,
            draws_pre_cull: self.draws_pre_cull,
            draws_culled: self.draws_culled,
            draws_hi_z_culled: self.draws_hi_z_culled,
            visibility: self.visibility,
            arrangement,
        }
    }
}

/// Queues draws from active spaces with control over inner rayon use.
pub fn queue_draws_with_parallelism(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> QueuedWorldMeshDraws {
    profiling::scope!("mesh::queue_draws");
    let owned_space_ids;
    let space_ids: &[RenderSpaceId] = {
        profiling::scope!("mesh::queue_draws::resolve_space_ids");
        if let Some(prepared) = ctx.prepared {
            if let Some(space_id) = ctx.render_space_filter {
                owned_space_ids = prepared
                    .active_space_ids()
                    .iter()
                    .copied()
                    .filter(|id| *id == space_id)
                    .collect::<Vec<_>>();
                &owned_space_ids
            } else {
                prepared.active_space_ids()
            }
        } else {
            owned_space_ids = match ctx.render_space_filter {
                Some(space_id) => ctx
                    .scene
                    .space(space_id)
                    .filter(|space| space.is_active())
                    .map_or_else(Vec::new, |_| vec![space_id]),
                None => ctx.scene.render_space_ids().collect::<Vec<_>>(),
            };
            &owned_space_ids
        }
    };
    let cap_hint = {
        profiling::scope!("mesh::queue_draws::estimate_capacity");
        if let Some(prepared) = ctx.prepared {
            match ctx.render_space_filter {
                Some(space_id) => prepared
                    .draws()
                    .iter()
                    .filter(|draw| draw.space_id == space_id)
                    .count(),
                None => prepared.len(),
            }
        } else {
            estimate_active_renderable_count(space_ids, ctx)
        }
    };

    let owned_cache;
    let cache: &FrameMaterialBatchCache = {
        profiling::scope!("mesh::queue_draws::resolve_material_cache");
        if let Some(shared) = ctx.material_cache {
            shared
        } else {
            let mut local = FrameMaterialBatchCache::new();
            local.refresh_for_frame(
                ctx.scene,
                ctx.material_dict,
                ctx.material_router,
                ctx.pipeline_property_ids,
                ctx.shader_perm,
            );
            owned_cache = local;
            &owned_cache
        }
    };
    let filter_masks = {
        profiling::scope!("mesh::queue_draws::build_filter_masks");
        build_per_space_filter_masks(space_ids, ctx)
    };
    let lod_visibility = {
        profiling::scope!("mesh::queue_draws::build_lod_visibility");
        build_lod_visibility(ctx, space_ids)
    };

    let collected = {
        profiling::scope!("mesh::queue_draws::collect_chunks");
        collect_world_mesh_chunks(
            ctx,
            parallelism,
            cache,
            &filter_masks,
            &lod_visibility,
            space_ids,
        )
    };

    merge_collected_chunks(collected, cap_hint)
}

/// Queues prepared draws for multiple view contexts through one flat `(view, chunk)` workload.
///
/// Returns `None` when any context lacks a prepared snapshot or material cache, letting callers
/// fall back to the general per-view queue path. When the combined prepared work is large enough,
/// this avoids spawning one Rayon job per view that then serially walks every prepared chunk.
pub(crate) fn queue_prepared_draws_for_views_with_parallelism(
    contexts: &[DrawCollectionContext<'_>],
    parallelism: WorldMeshDrawCollectParallelism,
) -> Option<Vec<QueuedWorldMeshDraws>> {
    profiling::scope!("mesh::queue_prepared_draws_for_views");
    if contexts.is_empty() {
        return Some(Vec::new());
    }
    let states = build_prepared_collection_states(contexts)?;
    let task_count = states
        .iter()
        .map(|state| state.prepared.run_chunks().len())
        .sum::<usize>();
    let draw_count = states.iter().map(|state| state.cap_hint).sum::<usize>();
    let admission =
        prepared_collect_admission(task_count, draw_count, current_reference_worker_count());
    record_parallel_admission("prepared_draw_collect", draw_count, draw_count, admission);
    if contexts.len() > 1
        && contexts.iter().all(|ctx| ctx.culling.is_none())
        && admission.is_parallel()
    {
        Some(collect_prepared_views_flat(contexts, &states, task_count))
    } else {
        Some(
            states
                .iter()
                .enumerate()
                .map(|(view_index, state)| {
                    let allow_parallel_chunks =
                        contexts.len() == 1 && parallelism == WorldMeshDrawCollectParallelism::Full;
                    let collected = collect_prepared_chunks_for_state(
                        &contexts[view_index],
                        state,
                        allow_parallel_chunks,
                    );
                    merge_collected_chunks(collected, state.cap_hint)
                })
                .collect(),
        )
    }
}

/// Packages per-chunk collection output for later arrangement.
fn merge_collected_chunks(
    collected: WorldMeshCollectedChunks,
    _cap_hint: usize,
) -> QueuedWorldMeshDraws {
    let mut chunks = Vec::with_capacity(collected.chunks.len());
    let mut len = 0usize;
    let mut cull_stats = (0usize, 0usize, 0usize);
    profiling::scope!("mesh::collect::package_chunks");
    for (items, cs) in collected.chunks {
        cull_stats.0 += cs.0;
        cull_stats.1 += cs.1;
        cull_stats.2 += cs.2;
        len += items.len();
        if !items.is_empty() {
            chunks.push(items);
        }
    }
    QueuedWorldMeshDraws {
        chunks,
        len,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
        visibility: collected.visibility,
    }
}

/// Builds per-view prepared collection state for all contexts.
fn build_prepared_collection_states<'a>(
    contexts: &[DrawCollectionContext<'a>],
) -> Option<Vec<PreparedCollectionState<'a>>> {
    let mut states = Vec::with_capacity(contexts.len());
    for ctx in contexts {
        let prepared = ctx.prepared?;
        let cache = ctx.material_cache?;
        let space_ids = prepared_space_ids_for_context(ctx, prepared);
        let cap_hint = prepared_capacity_hint_for_context(ctx, prepared);
        let filter_masks = {
            profiling::scope!("mesh::queue_prepared_draws_for_views::build_filter_masks");
            build_per_space_filter_masks(&space_ids, ctx)
        };
        let lod_visibility = {
            profiling::scope!("mesh::queue_prepared_draws_for_views::build_lod_visibility");
            build_lod_visibility(ctx, &space_ids)
        };
        states.push(PreparedCollectionState {
            prepared,
            cache,
            space_ids,
            filter_masks,
            lod_visibility,
            cap_hint,
        });
    }
    Some(states)
}

/// Resolves the active prepared render spaces relevant to one view context.
fn prepared_space_ids_for_context(
    ctx: &DrawCollectionContext<'_>,
    prepared: &FramePreparedRenderables,
) -> Vec<RenderSpaceId> {
    match ctx.render_space_filter {
        Some(space_id) => prepared
            .active_space_ids()
            .iter()
            .copied()
            .filter(|id| *id == space_id)
            .collect(),
        None => prepared.active_space_ids().to_vec(),
    }
}

/// Estimates output capacity for one prepared view context.
fn prepared_capacity_hint_for_context(
    ctx: &DrawCollectionContext<'_>,
    prepared: &FramePreparedRenderables,
) -> usize {
    match ctx.render_space_filter {
        Some(space_id) => prepared
            .draws()
            .iter()
            .filter(|draw| draw.space_id == space_id)
            .count(),
        None => prepared.len(),
    }
}

/// Collects all prepared view chunks through one flat Rayon workload.
fn collect_prepared_views_flat(
    contexts: &[DrawCollectionContext<'_>],
    states: &[PreparedCollectionState<'_>],
    task_count: usize,
) -> Vec<QueuedWorldMeshDraws> {
    profiling::scope!("mesh::collect_prepared::flat_views");
    let tasks = build_prepared_view_chunk_tasks(states, task_count);
    let outputs = {
        profiling::scope!("mesh::collect_prepared::flat_views::parallel_chunks");
        tasks
            .par_iter()
            .with_min_len(PREPARED_COLLECT_PARALLEL_CHUNK_TASKS)
            .map(|task| {
                profiling::scope!("mesh::collect_prepared::flat_views::chunk_worker");
                let state = &states[task.view_index];
                let chunk = state.prepared.run_chunks()[task.chunk_index];
                let runs = state.prepared.runs_for_chunk(chunk);
                let collected = collect_prepared_chunk(
                    state.prepared.draws(),
                    runs,
                    &contexts[task.view_index],
                    state.cache,
                    &state.filter_masks,
                    &state.lod_visibility,
                );
                (task.view_index, collected)
            })
            .collect::<Vec<_>>()
    };
    let mut per_view = states
        .iter()
        .map(|state| Vec::with_capacity(state.prepared.run_chunks().len()))
        .collect::<Vec<_>>();
    {
        profiling::scope!("mesh::collect_prepared::flat_views::demux");
        for (view_index, collected) in outputs {
            if let Some(view_chunks) = per_view.get_mut(view_index) {
                view_chunks.push(collected);
            }
        }
    }
    per_view
        .into_iter()
        .zip(states)
        .map(|(per_chunk, state)| {
            let collected =
                WorldMeshCollectedChunks::new(per_chunk, WorldMeshVisibilityStats::default());
            merge_collected_chunks(collected, state.cap_hint)
        })
        .collect()
}

/// Builds flat prepared view-chunk tasks in deterministic view/chunk order.
fn build_prepared_view_chunk_tasks(
    states: &[PreparedCollectionState<'_>],
    task_count: usize,
) -> Vec<PreparedViewChunkTask> {
    let mut tasks = Vec::with_capacity(task_count);
    for (view_index, state) in states.iter().enumerate() {
        tasks.extend((0..state.prepared.run_chunks().len()).map(|chunk_index| {
            PreparedViewChunkTask {
                view_index,
                chunk_index,
            }
        }));
    }
    tasks
}

/// Collects prepared chunks for one view state.
fn collect_prepared_chunks_for_state(
    ctx: &DrawCollectionContext<'_>,
    state: &PreparedCollectionState<'_>,
    allow_parallel_chunks: bool,
) -> WorldMeshCollectedChunks {
    if ctx.culling.is_some() {
        let parallelism = if allow_parallel_chunks {
            WorldMeshDrawCollectParallelism::Full
        } else {
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        };
        return collect_prepared_spatial_chunks(
            state.prepared,
            ctx,
            parallelism,
            state.cache,
            &state.filter_masks,
            &state.lod_visibility,
            &state.space_ids,
        );
    }
    WorldMeshCollectedChunks::new(
        collect_prepared_chunks(
            ctx,
            state.prepared,
            state.cache,
            &state.filter_masks,
            &state.lod_visibility,
            state.cap_hint,
            allow_parallel_chunks,
        ),
        WorldMeshVisibilityStats::default(),
    )
}

/// Collects prepared chunks for one view from borrowed per-view state.
fn collect_prepared_chunks(
    ctx: &DrawCollectionContext<'_>,
    prepared: &FramePreparedRenderables,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    lod_visibility: &LodVisibility,
    cap_hint: usize,
    allow_parallel_chunks: bool,
) -> Vec<(Vec<WorldMeshDrawItem>, (usize, usize, usize))> {
    let run_chunks = prepared.run_chunks();
    let draws = prepared.draws();
    let admission =
        prepared_collect_admission(run_chunks.len(), cap_hint, current_reference_worker_count());
    record_parallel_admission("prepared_draw_collect", cap_hint, cap_hint, admission);
    if allow_parallel_chunks && admission.is_parallel() {
        profiling::scope!("mesh::collect_prepared::parallel_chunks");
        run_chunks
            .par_iter()
            .with_min_len(PREPARED_COLLECT_PARALLEL_CHUNK_TASKS)
            .map(|&chunk| {
                profiling::scope!("mesh::collect_prepared::chunk_worker");
                let runs = prepared.runs_for_chunk(chunk);
                collect_prepared_chunk(draws, runs, ctx, cache, filter_masks, lod_visibility)
            })
            .collect()
    } else {
        profiling::scope!("mesh::collect_prepared::serial_chunks");
        run_chunks
            .iter()
            .map(|&chunk| {
                let runs = prepared.runs_for_chunk(chunk);
                collect_prepared_chunk(draws, runs, ctx, cache, filter_masks, lod_visibility)
            })
            .collect()
    }
}

/// Dispatches chunk collection to the prepared-draw path or the scene-walk fallback.
///
/// `Full` parallelism maps chunks via rayon; `SerialInnerForNestedBatch` keeps iteration serial
/// so nested multi-view batches don't hammer rayon with contention.
fn collect_world_mesh_chunks(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    lod_visibility: &LodVisibility,
    space_ids: &[RenderSpaceId],
) -> WorldMeshCollectedChunks {
    if let Some(prepared) = ctx.prepared {
        return collect_prepared_world_mesh_chunks(
            prepared,
            ctx,
            parallelism,
            cache,
            filter_masks,
            lod_visibility,
            space_ids,
        );
    }
    collect_scene_world_mesh_chunks(
        ctx,
        parallelism,
        cache,
        filter_masks,
        lod_visibility,
        space_ids,
    )
}

/// Collects chunks from the prepared draw snapshot.
fn collect_prepared_world_mesh_chunks(
    prepared: &FramePreparedRenderables,
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    lod_visibility: &LodVisibility,
    space_ids: &[RenderSpaceId],
) -> WorldMeshCollectedChunks {
    debug_assert!(
        prepared.is_compatible_with_render_context(ctx.render_context),
        "prepared renderables were built for a different render context than the per-view draw collection -- material overrides would disagree"
    );
    profiling::scope!("mesh::collect_prepared");
    if ctx.culling.is_some() {
        return collect_prepared_spatial_chunks(
            prepared,
            ctx,
            parallelism,
            cache,
            filter_masks,
            lod_visibility,
            space_ids,
        );
    }
    WorldMeshCollectedChunks::new(
        collect_prepared_chunks(
            ctx,
            prepared,
            cache,
            filter_masks,
            lod_visibility,
            prepared_capacity_hint_for_context(ctx, prepared),
            parallelism == WorldMeshDrawCollectParallelism::Full,
        ),
        WorldMeshVisibilityStats::default(),
    )
}

/// Collects prepared chunks from spatial broadphase candidates.
fn collect_prepared_spatial_chunks(
    prepared: &FramePreparedRenderables,
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    lod_visibility: &LodVisibility,
    space_ids: &[RenderSpaceId],
) -> WorldMeshCollectedChunks {
    profiling::scope!("mesh::collect_prepared::spatial_candidates");
    let draws = prepared.draws();
    let candidates = prepared.spatial_run_candidates(space_ids, ctx.scene, ctx.culling);
    let candidate_draw_count = candidates
        .runs
        .iter()
        .map(|run| (run.end - run.start) as usize)
        .sum::<usize>();
    let admission = prepared_collect_admission(
        candidates.runs.len(),
        candidate_draw_count,
        current_reference_worker_count(),
    );
    record_parallel_admission(
        "prepared_draw_collect",
        candidate_draw_count,
        candidate_draw_count,
        admission,
    );
    let mut chunks = if candidates.runs.is_empty() {
        Vec::new()
    } else if parallelism == WorldMeshDrawCollectParallelism::Full && admission.is_parallel() {
        profiling::scope!("mesh::collect_prepared::spatial_parallel_chunks");
        candidates
            .runs
            .par_chunks(SPATIAL_QUERY_RUN_CHUNK_TARGET)
            .map(|runs| {
                profiling::scope!("mesh::collect_prepared::spatial_chunk_worker");
                collect_prepared_chunk(draws, runs, ctx, cache, filter_masks, lod_visibility)
            })
            .collect()
    } else {
        profiling::scope!("mesh::collect_prepared::spatial_serial_chunks");
        vec![collect_prepared_chunk(
            draws,
            &candidates.runs,
            ctx,
            cache,
            filter_masks,
            lod_visibility,
        )]
    };
    merge_spatial_candidate_cull_stats(&mut chunks, candidates.cull_stats);
    WorldMeshCollectedChunks::new(chunks, candidates.visibility)
}

/// Merges broadphase cull counters into collected draw chunks.
fn merge_spatial_candidate_cull_stats(
    chunks: &mut Vec<WorldMeshDrawChunk>,
    cull_stats: WorldMeshChunkCullStats,
) {
    if cull_stats == (0, 0, 0) {
        return;
    }
    if let Some((_, stats)) = chunks.first_mut() {
        stats.0 += cull_stats.0;
        stats.1 += cull_stats.1;
        stats.2 += cull_stats.2;
    } else {
        chunks.push((Vec::new(), cull_stats));
    }
}

/// Collects chunks by walking scene render spaces when no prepared draw snapshot exists.
fn collect_scene_world_mesh_chunks(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    lod_visibility: &LodVisibility,
    space_ids: &[RenderSpaceId],
) -> WorldMeshCollectedChunks {
    let chunks = {
        profiling::scope!("mesh::collect::build_chunk_specs");
        build_chunk_specs(space_ids, ctx)
    };
    let work_units = {
        profiling::scope!("mesh::collect::estimate_parallel_work");
        estimate_active_renderable_count(space_ids, ctx)
    };
    let admission =
        scene_collect_admission(chunks.len(), work_units, current_reference_worker_count());
    record_parallel_admission("scene_draw_collect", work_units, chunks.len(), admission);
    profiling::scope!("mesh::collect");
    let collected =
        if parallelism == WorldMeshDrawCollectParallelism::Full && admission.is_parallel() {
            profiling::scope!("mesh::collect::parallel_chunks");
            chunks
                .par_iter()
                .with_min_len(SCENE_COLLECT_PARALLEL_CHUNK_TASKS)
                .map(|spec| {
                    profiling::scope!("mesh::collect::chunk_worker");
                    collect_chunk(spec, ctx, cache, filter_masks, lod_visibility)
                })
                .collect()
        } else {
            profiling::scope!("mesh::collect::serial_chunks");
            chunks
                .iter()
                .map(|spec| collect_chunk(spec, ctx, cache, filter_masks, lod_visibility))
                .collect()
        };
    WorldMeshCollectedChunks::new(collected, WorldMeshVisibilityStats::default())
}

#[cfg(test)]
mod tests;
