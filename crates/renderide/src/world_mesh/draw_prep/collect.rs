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
use crate::shared::{LayerType, RenderingContext};
use crate::world_mesh::culling::WorldMeshCullInput;
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::arrange::arrange_draw_chunks_by_phase_bins;
use super::command_cache::WorldMeshCommandCache;
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

/// Scene and resident asset tables shared across draw collection helpers.
#[derive(Clone, Copy)]
pub struct DrawCollectionSceneAssets<'a> {
    /// Scene graph for mesh renderables.
    pub scene: &'a SceneCoordinator,
    /// Resident meshes (submeshes, deform buffers).
    pub mesh_pool: &'a MeshPool,
}

/// Material routing inputs used to resolve draw batch keys.
#[derive(Clone, Copy)]
pub struct DrawCollectionMaterialInputs<'a> {
    /// Material property dictionary for batch keys.
    pub dict: &'a MaterialDictionary<'a>,
    /// Shader stem / pipeline routing.
    pub router: &'a MaterialRouter,
    /// Interned material property ids that affect pipeline state.
    pub pipeline_property_ids: &'a MaterialPipelinePropertyIds,
    /// Default vs multiview permutation for embedded materials.
    pub shader_perm: ShaderPermutation,
}

/// Per-view draw selection, culling, and sorting inputs.
#[derive(Clone, Copy)]
pub struct DrawCollectionViewInputs<'a> {
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
    /// Per-view Unity layer visibility policy for camera-specific culling behavior.
    pub layer_policy: ViewLayerPolicy,
    /// Optional frame reflection-probe selector used to choose the set of specular IBL probes to use per draw.
    pub reflection_probes: Option<&'a ReflectionProbeFrameSelection>,
}

/// Optional frame-level caches shared by all draw collection workers for one view.
#[derive(Clone, Copy)]
pub struct DrawCollectionFrameCaches<'a> {
    /// Optional pre-built material batch cache shared across multiple views in the same frame.
    ///
    /// When `Some`, collection reuses the shared cache instead of rebuilding one per call. Callers
    /// that render multiple views in one frame (secondary render-texture cameras + main
    /// swapchain) should build the cache once via [`FrameMaterialBatchCache::build_for_frame`] and
    /// hand the same borrow to every per-view context. When `None`, a fresh cache is built
    /// internally for this call (backwards-compatible single-view path).
    pub material_cache: Option<&'a FrameMaterialBatchCache>,
    /// Optional pre-expanded dense draw list shared across multiple views in the same frame.
    ///
    /// When `Some`, collection iterates the flat list instead of walking every active render
    /// space and looking up mesh pool entries per view. The prepared list must have been built for
    /// the same [`DrawCollectionViewInputs::render_context`] used here; otherwise material-override
    /// resolution may disagree. Single-view callers can leave this `None` and fall back to the
    /// scene-walk path.
    pub prepared: Option<&'a FramePreparedRenderables>,
}

/// Unity layer visibility behavior applied while collecting draws for one view.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ViewLayerPolicy {
    /// Regular world rendering. Hidden and overlay roots are excluded unless a selective camera
    /// transform filter explicitly exposes them.
    #[default]
    MainView,
    /// Host camera rendering with camera culling masks and private-UI opt-in.
    Camera {
        /// Whether private render spaces are visible to a non-selective camera.
        render_private_ui: bool,
    },
    /// Desktop overlay camera rendering. Only overlay roots are included.
    DesktopOverlay,
}

impl ViewLayerPolicy {
    /// Builds a camera layer policy from the host camera's `renderPrivateUI` flag.
    pub const fn camera(render_private_ui: bool) -> Self {
        Self::Camera { render_private_ui }
    }

    fn shows_private_render_space(self, has_selective_roots: bool) -> bool {
        match self {
            Self::MainView => true,
            Self::Camera { render_private_ui } => has_selective_roots || render_private_ui,
            Self::DesktopOverlay => true,
        }
    }

    fn shows_special_layer(
        self,
        special_layer: Option<LayerType>,
        has_selective_roots: bool,
    ) -> bool {
        match self {
            Self::MainView => match special_layer {
                Some(LayerType::Hidden | LayerType::Overlay) => has_selective_roots,
                _ => true,
            },
            Self::Camera { .. } => match special_layer {
                Some(LayerType::Hidden | LayerType::Overlay) => has_selective_roots,
                _ => true,
            },
            Self::DesktopOverlay => matches!(special_layer, Some(LayerType::Overlay)),
        }
    }

    fn effective_overlay(self, is_overlay: bool) -> bool {
        is_overlay && !matches!(self, Self::Camera { .. })
    }
}

/// Read-only scene, material, and cull state shared across all spaces during draw collection.
#[derive(Clone, Copy)]
pub struct DrawCollectionInputs<'a> {
    /// Scene graph and resident mesh tables.
    pub scene_assets: DrawCollectionSceneAssets<'a>,
    /// Material routing inputs used by batch-key resolution.
    pub materials: DrawCollectionMaterialInputs<'a>,
    /// Per-view transform, culling, filtering, and sorting inputs.
    pub view: DrawCollectionViewInputs<'a>,
    /// Optional frame-level caches reused across this view.
    pub caches: DrawCollectionFrameCaches<'a>,
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

/// Per-view derived lookup state shared by every chunk-collection worker for one queue call.
///
/// Built inside [`queue_draws_with_parallelism`] (or borrowed from a
/// [`PreparedCollectionState`]) after space, material-cache, filter-mask, and LOD resolution, and
/// threaded through chunk dispatch as one bundle instead of four loose parameters. All fields are
/// shared borrows, so the bundle stays [`Copy`] and hands out to rayon workers exactly like the
/// individual references did.
#[derive(Clone, Copy)]
struct CollectState<'a> {
    /// Material batch keys refreshed for this view's render context and shader permutation.
    cache: &'a FrameMaterialBatchCache,
    /// Per-space camera transform-filter masks.
    filter_masks: &'a HashMap<RenderSpaceId, Vec<bool>>,
    /// Per-view LOD visibility decisions.
    lod_visibility: &'a LodVisibility,
    /// Active render spaces relevant to this view.
    space_ids: &'a [RenderSpaceId],
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

impl PreparedCollectionState<'_> {
    /// Borrows the shared per-view lookup state for chunk-collection workers.
    fn as_collect_state(&self) -> CollectState<'_> {
        CollectState {
            cache: self.cache,
            filter_masks: &self.filter_masks,
            lod_visibility: &self.lod_visibility,
            space_ids: &self.space_ids,
        }
    }
}

/// One flat prepared-collection task for a specific view and run chunk.
#[derive(Clone, Copy)]
struct PreparedViewChunkTask {
    /// View index in the caller-provided context slice.
    view_index: usize,
    /// Prepared renderer-run chunk index within the view's prepared snapshot.
    chunk_index: usize,
}

/// Returns whether this view has a non-empty selective camera transform list.
fn transform_filter_has_selective_roots(ctx: &DrawCollectionInputs<'_>) -> bool {
    ctx.view
        .transform_filter
        .is_some_and(CameraTransformDrawFilter::has_selective_roots)
}

/// Returns whether `space_id` is visible under this view's render-space and private-UI policy.
fn render_space_visible_in_view(ctx: &DrawCollectionInputs<'_>, space_id: RenderSpaceId) -> bool {
    let Some(space) = ctx.scene_assets.scene.space(space_id) else {
        return false;
    };
    if !space.is_active() {
        return false;
    }
    !space.is_private()
        || ctx
            .view
            .layer_policy
            .shows_private_render_space(transform_filter_has_selective_roots(ctx))
}

/// Returns whether a renderer with `special_layer` is visible under this view's layer policy.
fn special_layer_visible_in_view(
    ctx: &DrawCollectionInputs<'_>,
    special_layer: Option<LayerType>,
) -> bool {
    ctx.view
        .layer_policy
        .shows_special_layer(special_layer, transform_filter_has_selective_roots(ctx))
}

/// Returns the overlay flag that should be emitted for a visible renderer in this view.
fn effective_overlay_in_view(ctx: &DrawCollectionInputs<'_>, is_overlay: bool) -> bool {
    ctx.view.layer_policy.effective_overlay(is_overlay)
}

impl QueuedWorldMeshDraws {
    /// Number of queued draw candidates before arrangement.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Sorts and arranges queued draws, reusing a retained command-list cache when provided.
    pub(crate) fn sort_and_arrange_with_cache(
        self,
        parallelism: WorldMeshDrawArrangeParallelism,
        command_cache: Option<&WorldMeshCommandCache>,
    ) -> WorldMeshDrawCollection {
        let allow_parallel_sort = parallelism == WorldMeshDrawArrangeParallelism::Full;
        let (items, arrangement) = {
            profiling::scope!("mesh::arrange");
            if let Some(command_cache) = command_cache {
                command_cache.arrange_draw_chunks(self.chunks, allow_parallel_sort)
            } else {
                arrange_draw_chunks_by_phase_bins(self.chunks, allow_parallel_sort)
            }
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
    ctx: &DrawCollectionInputs<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> QueuedWorldMeshDraws {
    profiling::scope!("mesh::queue_draws");
    let owned_space_ids;
    let space_ids: &[RenderSpaceId] = {
        profiling::scope!("mesh::queue_draws::resolve_space_ids");
        if let Some(prepared) = ctx.caches.prepared {
            if let Some(space_id) = ctx.view.render_space_filter {
                owned_space_ids = prepared
                    .active_space_ids()
                    .iter()
                    .copied()
                    .filter(|id| *id == space_id)
                    .filter(|id| render_space_visible_in_view(ctx, *id))
                    .collect::<Vec<_>>();
                &owned_space_ids
            } else if matches!(ctx.view.layer_policy, ViewLayerPolicy::MainView) {
                prepared.active_space_ids()
            } else {
                owned_space_ids = prepared
                    .active_space_ids()
                    .iter()
                    .copied()
                    .filter(|id| render_space_visible_in_view(ctx, *id))
                    .collect::<Vec<_>>();
                &owned_space_ids
            }
        } else {
            owned_space_ids = match ctx.view.render_space_filter {
                Some(space_id) => ctx
                    .scene_assets
                    .scene
                    .space(space_id)
                    .filter(|_| render_space_visible_in_view(ctx, space_id))
                    .map_or_else(Vec::new, |_| vec![space_id]),
                None => ctx
                    .scene_assets
                    .scene
                    .render_space_ids()
                    .filter(|id| render_space_visible_in_view(ctx, *id))
                    .collect::<Vec<_>>(),
            };
            &owned_space_ids
        }
    };
    let cap_hint = queued_capacity_hint(ctx, space_ids);

    let owned_cache;
    let cache: &FrameMaterialBatchCache = {
        profiling::scope!("mesh::queue_draws::resolve_material_cache");
        if let Some(shared) = ctx.caches.material_cache {
            shared
        } else {
            let mut local = FrameMaterialBatchCache::new();
            local.refresh_for_frame(
                ctx.scene_assets.scene,
                ctx.materials.dict,
                ctx.materials.router,
                ctx.materials.pipeline_property_ids,
                ctx.materials.shader_perm,
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
            CollectState {
                cache,
                filter_masks: &filter_masks,
                lod_visibility: &lod_visibility,
                space_ids,
            },
        )
    };

    merge_collected_chunks(collected, cap_hint)
}

/// Estimates the retained draw capacity for one queue call over the resolved spaces.
fn queued_capacity_hint(ctx: &DrawCollectionInputs<'_>, space_ids: &[RenderSpaceId]) -> usize {
    profiling::scope!("mesh::queue_draws::estimate_capacity");
    if let Some(prepared) = ctx.caches.prepared {
        if ctx.view.render_space_filter.is_none()
            && matches!(ctx.view.layer_policy, ViewLayerPolicy::MainView)
        {
            prepared.len()
        } else {
            prepared
                .draws()
                .iter()
                .filter(|draw| space_ids.contains(&draw.space_id))
                .count()
        }
    } else {
        estimate_active_renderable_count(space_ids, ctx)
    }
}

/// Queues prepared draws for multiple view contexts through one flat `(view, chunk)` workload.
///
/// Returns `None` when any context lacks a prepared snapshot or material cache, letting callers
/// fall back to the general per-view queue path. When the combined prepared work is large enough,
/// this avoids spawning one Rayon job per view that then serially walks every prepared chunk.
pub(crate) fn queue_prepared_draws_for_views_with_parallelism(
    contexts: &[DrawCollectionInputs<'_>],
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
        && contexts.iter().all(|ctx| ctx.view.culling.is_none())
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
    contexts: &[DrawCollectionInputs<'a>],
) -> Option<Vec<PreparedCollectionState<'a>>> {
    let mut states = Vec::with_capacity(contexts.len());
    for ctx in contexts {
        let prepared = ctx.caches.prepared?;
        let cache = ctx.caches.material_cache?;
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
    ctx: &DrawCollectionInputs<'_>,
    prepared: &FramePreparedRenderables,
) -> Vec<RenderSpaceId> {
    match ctx.view.render_space_filter {
        Some(space_id) => prepared
            .active_space_ids()
            .iter()
            .copied()
            .filter(|id| *id == space_id)
            .filter(|id| render_space_visible_in_view(ctx, *id))
            .collect(),
        None if matches!(ctx.view.layer_policy, ViewLayerPolicy::MainView) => {
            prepared.active_space_ids().to_vec()
        }
        None => prepared
            .active_space_ids()
            .iter()
            .copied()
            .filter(|id| render_space_visible_in_view(ctx, *id))
            .collect(),
    }
}

/// Estimates output capacity for one prepared view context.
fn prepared_capacity_hint_for_context(
    ctx: &DrawCollectionInputs<'_>,
    prepared: &FramePreparedRenderables,
) -> usize {
    if ctx.view.render_space_filter.is_none()
        && matches!(ctx.view.layer_policy, ViewLayerPolicy::MainView)
    {
        return prepared.len();
    }
    prepared
        .draws()
        .iter()
        .filter(|draw| render_space_visible_in_view(ctx, draw.space_id))
        .filter(|draw| {
            ctx.view
                .render_space_filter
                .is_none_or(|space_id| draw.space_id == space_id)
        })
        .count()
}

/// Collects all prepared view chunks through one flat Rayon workload.
fn collect_prepared_views_flat(
    contexts: &[DrawCollectionInputs<'_>],
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
                    state.as_collect_state(),
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
    ctx: &DrawCollectionInputs<'_>,
    state: &PreparedCollectionState<'_>,
    allow_parallel_chunks: bool,
) -> WorldMeshCollectedChunks {
    if ctx.view.culling.is_some() {
        let parallelism = if allow_parallel_chunks {
            WorldMeshDrawCollectParallelism::Full
        } else {
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        };
        return collect_prepared_spatial_chunks(
            state.prepared,
            ctx,
            parallelism,
            state.as_collect_state(),
        );
    }
    WorldMeshCollectedChunks::new(
        collect_prepared_chunks(
            ctx,
            state.prepared,
            state.as_collect_state(),
            state.cap_hint,
            allow_parallel_chunks,
        ),
        WorldMeshVisibilityStats::default(),
    )
}

/// Collects prepared chunks for one view from borrowed per-view state.
fn collect_prepared_chunks(
    ctx: &DrawCollectionInputs<'_>,
    prepared: &FramePreparedRenderables,
    state: CollectState<'_>,
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
                collect_prepared_chunk(draws, runs, ctx, state)
            })
            .collect()
    } else {
        profiling::scope!("mesh::collect_prepared::serial_chunks");
        run_chunks
            .iter()
            .map(|&chunk| {
                let runs = prepared.runs_for_chunk(chunk);
                collect_prepared_chunk(draws, runs, ctx, state)
            })
            .collect()
    }
}

/// Dispatches chunk collection to the prepared-draw path or the scene-walk fallback.
///
/// `Full` parallelism maps chunks via rayon; `SerialInnerForNestedBatch` keeps iteration serial
/// so nested multi-view batches don't hammer rayon with contention.
fn collect_world_mesh_chunks(
    ctx: &DrawCollectionInputs<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    state: CollectState<'_>,
) -> WorldMeshCollectedChunks {
    if let Some(prepared) = ctx.caches.prepared {
        return collect_prepared_world_mesh_chunks(prepared, ctx, parallelism, state);
    }
    collect_scene_world_mesh_chunks(ctx, parallelism, state)
}

/// Collects chunks from the prepared draw snapshot.
fn collect_prepared_world_mesh_chunks(
    prepared: &FramePreparedRenderables,
    ctx: &DrawCollectionInputs<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    state: CollectState<'_>,
) -> WorldMeshCollectedChunks {
    debug_assert!(
        prepared.is_compatible_with_render_context(ctx.view.render_context),
        "prepared renderables were built for a different render context than the per-view draw collection -- material overrides would disagree"
    );
    profiling::scope!("mesh::collect_prepared");
    if ctx.view.culling.is_some() {
        return collect_prepared_spatial_chunks(prepared, ctx, parallelism, state);
    }
    WorldMeshCollectedChunks::new(
        collect_prepared_chunks(
            ctx,
            prepared,
            state,
            prepared_capacity_hint_for_context(ctx, prepared),
            parallelism == WorldMeshDrawCollectParallelism::Full,
        ),
        WorldMeshVisibilityStats::default(),
    )
}

/// Collects prepared chunks from spatial broadphase candidates.
fn collect_prepared_spatial_chunks(
    prepared: &FramePreparedRenderables,
    ctx: &DrawCollectionInputs<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    state: CollectState<'_>,
) -> WorldMeshCollectedChunks {
    profiling::scope!("mesh::collect_prepared::spatial_candidates");
    let draws = prepared.draws();
    let candidates =
        prepared.spatial_run_candidates(state.space_ids, ctx.scene_assets.scene, ctx.view.culling);
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
                collect_prepared_chunk(draws, runs, ctx, state)
            })
            .collect()
    } else {
        profiling::scope!("mesh::collect_prepared::spatial_serial_chunks");
        vec![collect_prepared_chunk(draws, &candidates.runs, ctx, state)]
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
    ctx: &DrawCollectionInputs<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    state: CollectState<'_>,
) -> WorldMeshCollectedChunks {
    let chunks = {
        profiling::scope!("mesh::collect::build_chunk_specs");
        build_chunk_specs(state.space_ids, ctx)
    };
    let work_units = {
        profiling::scope!("mesh::collect::estimate_parallel_work");
        estimate_active_renderable_count(state.space_ids, ctx)
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
                    collect_chunk(spec, ctx, state)
                })
                .collect()
        } else {
            profiling::scope!("mesh::collect::serial_chunks");
            chunks
                .iter()
                .map(|spec| collect_chunk(spec, ctx, state))
                .collect()
        };
    WorldMeshCollectedChunks::new(collected, WorldMeshVisibilityStats::default())
}

#[cfg(test)]
mod tests;
