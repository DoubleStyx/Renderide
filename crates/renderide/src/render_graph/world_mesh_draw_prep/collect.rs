//! Scene walk that pairs material slots with submesh ranges and applies optional CPU culling.
//!
//! [`collect_and_sort_world_mesh_draws`] walks each render space in 128-renderable parallel chunks
//! ([`rayon`]), merges in [`SceneCoordinator::render_space_ids`] order, assigns
//! [`WorldMeshDrawItem::collect_order`], then sorts.
//!
//! Material-derived batch key fields are computed once per `(material_asset_id, property_block_id)`
//! per call via [`FrameMaterialBatchCache`] before the parallel phase begins. This eliminates
//! repeated dictionary and router lookups for the common case where hundreds of draws share a
//! few dozen materials.

use hashbrown::HashMap;

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterFrontFace};
use crate::pipelines::ShaderPermutation;
use crate::resources::MeshPool;
use crate::scene::{
    MeshMaterialSlot, RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::shared::RenderingContext;

use super::material_batch_cache::FrameMaterialBatchCache;
use super::material_batch_cache::MaterialResolveCtx;
use super::prepared::{FramePreparedDraw, FramePreparedRenderables};
use super::sort::{batch_key_for_slot_cached, sort_world_mesh_draws, sort_world_mesh_draws_serial};
use super::types::{
    resolved_material_slot_count, stacked_material_submesh_range, WorldMeshDrawCollection,
    WorldMeshDrawItem,
};

use super::super::world_mesh_cull_eval::{
    mesh_draw_passes_cpu_cull, CpuCullFailure, MeshCullTarget,
};

#[path = "collect_candidate.rs"]
mod candidate;

use candidate::{evaluate_draw_candidate, DrawCandidate};

/// Renders per chunk (static or skinned slice of one render space).
const WORLD_MESH_COLLECT_CHUNK_SIZE: usize = 128;

/// Rayon chunk width when iterating a pre-expanded [`FramePreparedRenderables`] list.
///
/// Matches [`WORLD_MESH_COLLECT_CHUNK_SIZE`] so per-view CPU cost stays bounded by the same
/// per-task overhead as the scene-walk path.
const PREPARED_CHUNK_SIZE: usize = 128;

/// Submesh index range for one material slot pairing during draw collection.
pub(crate) struct SubmeshSlotIndices {
    /// Slot index in [`StaticMeshRenderer`] material slots.
    pub slot_index: usize,
    /// First index in the mesh index buffer for this submesh.
    pub first_index: u32,
    /// Index count for this submesh draw.
    pub index_count: u32,
}

/// Layer and skin deform flags that affect CPU cull and [`WorldMeshDrawItem`] fields.
pub(crate) struct OverlayDeformCullFlags {
    /// Overlay layer uses alternate cull behavior.
    pub is_overlay: bool,
    /// Skinned mesh with world-space deform from the skin cache.
    pub world_space_deformed: bool,
}

/// Resolves the draw's world matrix when the selected vertex stream is still local-space.
#[inline]
fn world_matrix_for_local_vertex_stream(
    ctx: &DrawCollectionContext<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
) -> Option<Mat4> {
    if node_id < 0 {
        return None;
    }
    ctx.scene.world_matrix_for_render_context(
        space_id,
        node_id as usize,
        ctx.render_context,
        ctx.head_output_transform,
    )
}

/// Returns `true` when a renderer node's effective transform chain collapses object scale.
#[inline]
fn transform_chain_has_degenerate_scale(
    ctx: &DrawCollectionContext<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
) -> bool {
    node_id >= 0
        && ctx.scene.transform_has_degenerate_scale_for_context(
            space_id,
            node_id as usize,
            ctx.render_context,
        )
}

/// Resolves the raster front face for the model matrix used by the forward vertex shader.
#[inline]
fn front_face_for_world_matrix(world_matrix: Option<Mat4>) -> RasterFrontFace {
    world_matrix
        .map(RasterFrontFace::from_model_matrix)
        .unwrap_or_default()
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
    /// Populate from `HostCameraFrame::explicit_camera_world_position.unwrap_or_else(|| head_output_transform.col(3).truncate())`.
    pub view_origin_world: Vec3,
    /// Optional CPU frustum + Hi-Z cull inputs.
    pub culling: Option<&'a super::super::world_mesh_cull::WorldMeshCullInput<'a>>,
    /// Optional per-camera node filter.
    pub transform_filter: Option<&'a super::types::CameraTransformDrawFilter>,
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
    /// space and looking up mesh pool entries per view. The prepared list must have been built
    /// for the **same** [`Self::render_context`] used here; otherwise material-override
    /// resolution may disagree. Single-view callers can leave this `None` and fall back to the
    /// scene-walk path.
    pub prepared: Option<&'a FramePreparedRenderables>,
}

/// One static or skinned mesh renderer with its resolved [`crate::assets::mesh::GpuMesh`] and submesh index ranges.
struct StaticMeshDrawSource<'a> {
    space_id: RenderSpaceId,
    renderer: &'a StaticMeshRenderer,
    renderable_index: usize,
    skinned: bool,
    skinned_renderer: Option<&'a SkinnedMeshRenderer>,
    mesh: &'a crate::assets::mesh::GpuMesh,
    submeshes: &'a [(u32, u32)],
}

/// Mutable expansion state while expanding one chunk into draw items.
struct DrawCollectionAccumulator<'a> {
    out: &'a mut Vec<WorldMeshDrawItem>,
    cull_stats: &'a mut (usize, usize, usize),
    /// Precomputed filter result per node index. When `Some`, used in place of
    /// [`super::types::CameraTransformDrawFilter::passes_scene_node`] to avoid per-draw ancestor walks.
    filter_pass_mask: Option<&'a [bool]>,
}

/// How [`collect_and_sort_world_mesh_draws_with_parallelism`] parallelizes per-chunk collection and sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawCollectParallelism {
    /// Per-chunk collection and draw sort both use rayon.
    Full,
    /// Serial per-chunk merge and serial sort; use when an outer `par_iter` already fans out (e.g. multiple secondary RTs).
    SerialInnerForNestedBatch,
}

/// Whether a chunk covers the static or skinned renderer list of a render space.
#[derive(Clone, Copy)]
enum ChunkKind {
    Static,
    Skinned,
}

/// One 128-renderable slice of a render space's static or skinned renderer array.
struct WorldMeshChunkSpec {
    space_id: RenderSpaceId,
    kind: ChunkKind,
    range: std::ops::Range<usize>,
}

/// Expands one static mesh renderer into draw items (material slots mapped to submesh ranges).
///
/// `collect_order` is filled with a placeholder; [`collect_and_sort_world_mesh_draws`] assigns the
/// final stable index after per-chunk results are merged.
fn push_draws_for_renderer(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: StaticMeshDrawSource<'_>,
    cache: &FrameMaterialBatchCache,
) {
    if let Some(f) = ctx.transform_filter {
        let passes = match acc.filter_pass_mask {
            Some(mask) => {
                let nid = draw.renderer.node_id;
                nid >= 0 && (nid as usize) < mask.len() && mask[nid as usize]
            }
            None => f.passes_scene_node(ctx.scene, draw.space_id, draw.renderer.node_id),
        };
        if !passes {
            return;
        }
    }
    if transform_chain_has_degenerate_scale(ctx, draw.space_id, draw.renderer.node_id) {
        return;
    }

    // Resolve material slots inline to avoid the Cow::Owned(vec![..]) allocation for the
    // primary-material fallback path.
    let fallback_slot;
    let slots: &[MeshMaterialSlot] = if !draw.renderer.material_slots.is_empty() {
        &draw.renderer.material_slots
    } else if let Some(mat_id) = draw.renderer.primary_material_asset_id {
        fallback_slot = MeshMaterialSlot {
            material_asset_id: mat_id,
            property_block_id: draw.renderer.primary_property_block_id,
        };
        std::slice::from_ref(&fallback_slot)
    } else {
        return;
    };

    if slots.is_empty() {
        return;
    }
    let n_sub = draw.submeshes.len();
    let n_slot = slots.len();
    if n_slot > n_sub {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} exceeds submesh count {}; stacking extra slots onto the last submesh",
            draw.renderer.mesh_asset_id,
            n_slot,
            n_sub,
        );
    } else if n_slot < n_sub {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} is below submesh count {}; only material-backed submeshes draw",
            draw.renderer.mesh_asset_id,
            n_slot,
            n_sub,
        );
    }
    if n_sub == 0 {
        return;
    }

    let is_overlay = draw.renderer.layer == crate::shared::LayerType::Overlay;
    let world_space_deformed = draw.skinned
        && draw.mesh.supports_world_space_skin_deform(
            draw.skinned_renderer
                .map(|skinned| skinned.bone_transform_indices.as_slice()),
        );

    for (slot_index, slot) in slots.iter().enumerate() {
        let Some((first_index, index_count)) =
            stacked_material_submesh_range(slot_index, draw.submeshes)
        else {
            continue;
        };
        push_one_slot_draw(
            ctx,
            acc,
            &draw,
            slot,
            SubmeshSlotIndices {
                slot_index,
                first_index,
                index_count,
            },
            OverlayDeformCullFlags {
                is_overlay,
                world_space_deformed,
            },
            cache,
        );
    }
}

/// One material slot mapped to a submesh range: optional CPU cull, batch key, and [`WorldMeshDrawItem`] push.
fn push_one_slot_draw(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: &StaticMeshDrawSource<'_>,
    slot: &MeshMaterialSlot,
    indices: SubmeshSlotIndices,
    flags: OverlayDeformCullFlags,
    cache: &FrameMaterialBatchCache,
) {
    let SubmeshSlotIndices {
        slot_index,
        first_index,
        index_count,
    } = indices;
    let OverlayDeformCullFlags {
        is_overlay,
        world_space_deformed,
    } = flags;
    let material_asset_id = ctx
        .scene
        .overridden_material_asset_id(
            draw.space_id,
            ctx.render_context,
            draw.skinned,
            draw.renderable_index,
            slot_index,
        )
        .unwrap_or(slot.material_asset_id);
    if index_count == 0 || material_asset_id < 0 {
        return;
    }
    let mut rigid_world_matrix = None;
    if !draw.skinned {
        if let Some(c) = ctx.culling {
            acc.cull_stats.0 += 1;
            let target = MeshCullTarget {
                scene: ctx.scene,
                space_id: draw.space_id,
                mesh: draw.mesh,
                skinned: draw.skinned,
                skinned_renderer: draw.skinned_renderer,
                node_id: draw.renderer.node_id,
            };
            match mesh_draw_passes_cpu_cull(&target, is_overlay, c, ctx.render_context) {
                Err(CpuCullFailure::Frustum) => {
                    acc.cull_stats.1 += 1;
                    return;
                }
                Err(CpuCullFailure::HiZ) => {
                    acc.cull_stats.2 += 1;
                    return;
                }
                Ok(m) => {
                    rigid_world_matrix = m;
                }
            }
        }
    }
    if !world_space_deformed && rigid_world_matrix.is_none() {
        rigid_world_matrix =
            world_matrix_for_local_vertex_stream(ctx, draw.space_id, draw.renderer.node_id);
    }
    let front_face = front_face_for_world_matrix(rigid_world_matrix);
    let alpha_distance_sq = rigid_world_matrix
        .map(|m| (m.col(3).truncate() - ctx.view_origin_world).length_squared())
        .unwrap_or(0.0);
    let candidate = DrawCandidate {
        space_id: draw.space_id,
        node_id: draw.renderer.node_id,
        mesh_asset_id: draw.renderer.mesh_asset_id,
        slot_index,
        first_index,
        index_count,
        is_overlay,
        sorting_order: draw.renderer.sorting_order,
        skinned: draw.skinned,
        world_space_deformed,
        material_asset_id,
        property_block_id: slot.property_block_id,
    };
    if let Some(item) = evaluate_draw_candidate(
        ctx,
        cache,
        candidate,
        front_face,
        rigid_world_matrix,
        alpha_distance_sq,
    ) {
        acc.out.push(item);
    }
}

/// Builds the chunk list: one entry per 128-renderer slice of static or skinned renderers per space.
fn build_chunk_specs(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> Vec<WorldMeshChunkSpec> {
    profiling::scope!("mesh::build_chunk_specs");
    let mut chunks = Vec::new();
    for &space_id in space_ids {
        let Some(space) = ctx.scene.space(space_id) else {
            continue;
        };
        if !space.is_active {
            continue;
        }
        let n_static = space.static_mesh_renderers.len();
        let mut start = 0;
        while start < n_static {
            let end = n_static.min(start + WORLD_MESH_COLLECT_CHUNK_SIZE);
            chunks.push(WorldMeshChunkSpec {
                space_id,
                kind: ChunkKind::Static,
                range: start..end,
            });
            start = end;
        }
        let n_skinned = space.skinned_mesh_renderers.len();
        start = 0;
        while start < n_skinned {
            let end = n_skinned.min(start + WORLD_MESH_COLLECT_CHUNK_SIZE);
            chunks.push(WorldMeshChunkSpec {
                space_id,
                kind: ChunkKind::Skinned,
                range: start..end,
            });
            start = end;
        }
    }
    chunks
}

/// Collects draw items for one chunk (one 128-renderer slice of static or skinned renderers).
fn collect_chunk(
    spec: &WorldMeshChunkSpec,
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    let mut out = Vec::new();
    let mut cull_stats = (0usize, 0usize, 0usize);

    let Some(space) = ctx.scene.space(spec.space_id) else {
        return (out, cull_stats);
    };
    if !space.is_active {
        return (out, cull_stats);
    }

    let filter_pass_mask = filter_masks.get(&spec.space_id).map(|m| m.as_slice());
    let mut acc = DrawCollectionAccumulator {
        out: &mut out,
        cull_stats: &mut cull_stats,
        filter_pass_mask,
    };

    match spec.kind {
        ChunkKind::Static => {
            for renderable_index in spec.range.clone() {
                let r = &space.static_mesh_renderers[renderable_index];
                if r.mesh_asset_id < 0 || r.node_id < 0 {
                    continue;
                }
                let Some(mesh) = ctx.mesh_pool.get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                if mesh.submeshes.is_empty() {
                    continue;
                }
                push_draws_for_renderer(
                    ctx,
                    &mut acc,
                    StaticMeshDrawSource {
                        space_id: spec.space_id,
                        renderer: r,
                        renderable_index,
                        skinned: false,
                        skinned_renderer: None,
                        mesh,
                        submeshes: &mesh.submeshes,
                    },
                    cache,
                );
            }
        }
        ChunkKind::Skinned => {
            for renderable_index in spec.range.clone() {
                let skinned = &space.skinned_mesh_renderers[renderable_index];
                let r = &skinned.base;
                if r.mesh_asset_id < 0 || r.node_id < 0 {
                    continue;
                }
                let Some(mesh) = ctx.mesh_pool.get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                if mesh.submeshes.is_empty() {
                    continue;
                }
                push_draws_for_renderer(
                    ctx,
                    &mut acc,
                    StaticMeshDrawSource {
                        space_id: spec.space_id,
                        renderer: r,
                        renderable_index,
                        skinned: true,
                        skinned_renderer: Some(skinned),
                        mesh,
                        submeshes: &mesh.submeshes,
                    },
                    cache,
                );
            }
        }
    }
    (out, cull_stats)
}

/// Returns true when two prepared slot entries came from the same source renderer.
#[inline]
fn prepared_draws_share_renderer(a: &FramePreparedDraw, b: &FramePreparedDraw) -> bool {
    a.space_id == b.space_id
        && a.renderable_index == b.renderable_index
        && a.node_id == b.node_id
        && a.mesh_asset_id == b.mesh_asset_id
        && a.is_overlay == b.is_overlay
        && a.sorting_order == b.sorting_order
        && a.skinned == b.skinned
        && a.world_space_deformed == b.world_space_deformed
}

/// Per-renderer view-local state shared by every material slot in a prepared run.
#[derive(Clone, Copy)]
struct PreparedRunViewState {
    /// Rigid model matrix reused by all emitted slot draws.
    rigid_world_matrix: Option<Mat4>,
    /// Raster front-face winding selected from [`Self::rigid_world_matrix`].
    front_face: RasterFrontFace,
    /// Camera distance reused by alpha-blended slot draws.
    alpha_distance_sq: f32,
}

/// Skinned renderer lookup result for a prepared renderer run.
enum PreparedRunSkinning<'a> {
    /// The renderer uses the rigid static-mesh path.
    Rigid,
    /// The renderer uses the skinned path and still has a valid scene entry.
    Skinned(&'a SkinnedMeshRenderer),
    /// The prepared index no longer points at a valid skinned renderer this frame.
    Stale,
}

impl<'a> PreparedRunSkinning<'a> {
    /// Returns the culling target's optional skinned renderer borrow.
    fn as_renderer(&self) -> Option<&'a SkinnedMeshRenderer> {
        match self {
            Self::Rigid | Self::Stale => None,
            Self::Skinned(renderer) => Some(renderer),
        }
    }
}

/// Returns whether the renderer run passes the view's optional transform filter.
fn prepared_run_passes_filter(
    first: &FramePreparedDraw,
    ctx: &DrawCollectionContext<'_>,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> bool {
    let Some(filter) = ctx.transform_filter else {
        return true;
    };
    match filter_masks.get(&first.space_id) {
        Some(mask) => {
            first.node_id >= 0
                && (first.node_id as usize) < mask.len()
                && mask[first.node_id as usize]
        }
        None => filter.passes_scene_node(ctx.scene, first.space_id, first.node_id),
    }
}

/// Returns the skinned renderer backing a prepared run, or `None` when stale scene indices should skip it.
fn prepared_run_skinned_renderer<'a>(
    first: &FramePreparedDraw,
    ctx: &'a DrawCollectionContext<'_>,
) -> PreparedRunSkinning<'a> {
    if !first.skinned {
        return PreparedRunSkinning::Rigid;
    }
    let Some(space) = ctx.scene.space(first.space_id) else {
        return PreparedRunSkinning::Stale;
    };
    space
        .skinned_mesh_renderers
        .get(first.renderable_index)
        .map_or(PreparedRunSkinning::Stale, PreparedRunSkinning::Skinned)
}

/// Builds shared view-local state for one prepared renderer run and reports draw-slot cull stats.
fn prepared_run_view_state(
    run: &[FramePreparedDraw],
    first: &FramePreparedDraw,
    mesh: &crate::assets::mesh::GpuMesh,
    skinning: &PreparedRunSkinning<'_>,
    ctx: &DrawCollectionContext<'_>,
) -> (Option<PreparedRunViewState>, (usize, usize, usize)) {
    let mut cull_stats = (0usize, 0usize, 0usize);
    let mut rigid_world_matrix = None;
    if let Some(c) = ctx.culling {
        cull_stats.0 += run.len();
        let target = MeshCullTarget {
            scene: ctx.scene,
            space_id: first.space_id,
            mesh,
            skinned: first.skinned,
            skinned_renderer: skinning.as_renderer(),
            node_id: first.node_id,
        };
        match mesh_draw_passes_cpu_cull(&target, first.is_overlay, c, ctx.render_context) {
            Err(CpuCullFailure::Frustum) => {
                cull_stats.1 += run.len();
                return (None, cull_stats);
            }
            Err(CpuCullFailure::HiZ) => {
                cull_stats.2 += run.len();
                return (None, cull_stats);
            }
            Ok(m) => {
                rigid_world_matrix = m;
            }
        }
    }
    if !first.world_space_deformed && rigid_world_matrix.is_none() {
        rigid_world_matrix =
            world_matrix_for_local_vertex_stream(ctx, first.space_id, first.node_id);
    }
    let front_face = front_face_for_world_matrix(rigid_world_matrix);
    let alpha_distance_sq = rigid_world_matrix
        .map(|m| (m.col(3).truncate() - ctx.view_origin_world).length_squared())
        .unwrap_or(0.0);
    (
        Some(PreparedRunViewState {
            rigid_world_matrix,
            front_face,
            alpha_distance_sq,
        }),
        cull_stats,
    )
}

/// Emits one [`WorldMeshDrawItem`] per material slot in a surviving prepared renderer run.
fn append_prepared_run_draws(
    run: &[FramePreparedDraw],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    state: PreparedRunViewState,
    out: &mut Vec<WorldMeshDrawItem>,
) {
    for d in run {
        let candidate = DrawCandidate {
            space_id: d.space_id,
            node_id: d.node_id,
            mesh_asset_id: d.mesh_asset_id,
            slot_index: d.slot_index,
            first_index: d.first_index,
            index_count: d.index_count,
            is_overlay: d.is_overlay,
            sorting_order: d.sorting_order,
            skinned: d.skinned,
            world_space_deformed: d.world_space_deformed,
            material_asset_id: d.material_asset_id,
            property_block_id: d.property_block_id,
        };
        if let Some(item) = evaluate_draw_candidate(
            ctx,
            cache,
            candidate,
            state.front_face,
            state.rigid_world_matrix,
            state.alpha_distance_sq,
        ) {
            out.push(item);
        }
    }
}

/// Collects one prepared renderer run after frame-global slot expansion.
fn collect_prepared_renderer_run(
    run: &[FramePreparedDraw],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    out: &mut Vec<WorldMeshDrawItem>,
) -> (usize, usize, usize) {
    let Some(first) = run.first() else {
        return (0, 0, 0);
    };
    if !prepared_run_passes_filter(first, ctx, filter_masks) {
        return (0, 0, 0);
    }
    let Some(mesh) = ctx.mesh_pool.get_mesh(first.mesh_asset_id) else {
        return (0, 0, 0);
    };
    let skinning = prepared_run_skinned_renderer(first, ctx);
    if matches!(skinning, PreparedRunSkinning::Stale) {
        return (0, 0, 0);
    }
    let (state, cull_stats) = prepared_run_view_state(run, first, mesh, &skinning, ctx);
    if let Some(state) = state {
        append_prepared_run_draws(run, ctx, cache, state, out);
    }
    cull_stats
}

/// Collects draw items for one chunk of a pre-expanded [`FramePreparedRenderables`] list.
///
/// Unlike [`collect_chunk`], there is no scene walk: the prepared draws already captured every
/// valid `(renderer × material slot)` tuple plus its frame-global resolution (material override,
/// submesh index range, overlay flag, skin deform flag). This per-view pass only applies filters
/// and per-view CPU culling per renderer, then builds [`WorldMeshDrawItem`]s for each material slot.
fn collect_prepared_chunk(
    draws: &[FramePreparedDraw],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    let mut out: Vec<WorldMeshDrawItem> = Vec::with_capacity(draws.len());
    let mut cull_stats = (0usize, 0usize, 0usize);

    let mut run_start = 0usize;
    while run_start < draws.len() {
        let first = &draws[run_start];
        let mut run_end = run_start + 1;
        while run_end < draws.len() && prepared_draws_share_renderer(first, &draws[run_end]) {
            run_end += 1;
        }
        let run = &draws[run_start..run_end];
        run_start = run_end;
        let run_stats = collect_prepared_renderer_run(run, ctx, cache, filter_masks, &mut out);
        cull_stats.0 += run_stats.0;
        cull_stats.1 += run_stats.1;
        cull_stats.2 += run_stats.2;
    }

    (out, cull_stats)
}

/// Collects draws from active spaces, then sorts for batching (material / pipeline boundaries).
///
/// When `culling` is [`Some`], instances outside the frustum (and optional Hi-Z) are dropped (see
/// [`mesh_draw_passes_cpu_cull`](super::super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull)).
///
/// Collection runs over 128-renderer chunks in parallel via [`rayon`] by default; results are
/// merged in the same order as [`SceneCoordinator::render_space_ids`], then
/// [`WorldMeshDrawItem::collect_order`] is assigned for transparent sort stability.
pub fn collect_and_sort_world_mesh_draws(
    ctx: &DrawCollectionContext<'_>,
) -> WorldMeshDrawCollection {
    collect_and_sort_world_mesh_draws_with_parallelism(ctx, WorldMeshDrawCollectParallelism::Full)
}

/// Like [`collect_and_sort_world_mesh_draws`], with control over inner rayon use (see [`WorldMeshDrawCollectParallelism`]).
pub fn collect_and_sort_world_mesh_draws_with_parallelism(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> WorldMeshDrawCollection {
    profiling::scope!("mesh::collect_and_sort");
    let space_ids: Vec<RenderSpaceId> = ctx.scene.render_space_ids().collect();
    let cap_hint = estimate_active_renderable_count(&space_ids, ctx);

    // Build per-material cache and per-space filter masks before the parallel phase. When the
    // caller already shared a frame-scope cache (multi-view paths), reuse it instead of rebuilding
    // — that is the whole point of `material_cache` on [`DrawCollectionContext`]. When no cache
    // is shared (single-view fallback), refresh a local one in place; this path loses
    // cross-frame reuse but keeps the within-call deduplication.
    let owned_cache;
    let cache: &FrameMaterialBatchCache = match ctx.material_cache {
        Some(shared) => shared,
        None => {
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
    let filter_masks = build_per_space_filter_masks(&space_ids, ctx);

    let per_chunk = collect_world_mesh_chunks(ctx, parallelism, cache, &filter_masks, &space_ids);

    let mut out = Vec::with_capacity(cap_hint);
    let mut cull_stats = (0usize, 0usize, 0usize);
    for (items, cs) in per_chunk {
        cull_stats.0 += cs.0;
        cull_stats.1 += cs.1;
        cull_stats.2 += cs.2;
        out.extend(items);
    }

    for (i, item) in out.iter_mut().enumerate() {
        item.collect_order = i;
    }

    {
        profiling::scope!("mesh::sort");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => sort_world_mesh_draws(&mut out),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => {
                sort_world_mesh_draws_serial(&mut out);
            }
        }
    }
    WorldMeshDrawCollection {
        items: out,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
    }
}

/// Upper bound on expanded draw slots across active render spaces (capacity hint for the output vec).
fn estimate_active_renderable_count(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> usize {
    let mut cap_hint = 0usize;
    for space_id in space_ids {
        let Some(space) = ctx.scene.space(*space_id) else {
            continue;
        };
        if !space.is_active {
            continue;
        }
        for renderer in &space.static_mesh_renderers {
            if renderer.mesh_asset_id < 0 || renderer.node_id < 0 {
                continue;
            }
            if ctx
                .mesh_pool
                .get_mesh(renderer.mesh_asset_id)
                .is_some_and(|mesh| !mesh.submeshes.is_empty())
            {
                cap_hint = cap_hint.saturating_add(resolved_material_slot_count(renderer));
            }
        }
        for skinned in &space.skinned_mesh_renderers {
            let renderer = &skinned.base;
            if renderer.mesh_asset_id < 0 || renderer.node_id < 0 {
                continue;
            }
            if ctx
                .mesh_pool
                .get_mesh(renderer.mesh_asset_id)
                .is_some_and(|mesh| !mesh.submeshes.is_empty())
            {
                cap_hint = cap_hint.saturating_add(resolved_material_slot_count(renderer));
            }
        }
    }
    cap_hint
}

/// Builds per-space `Vec<bool>` masks from [`DrawCollectionContext::transform_filter`].
///
/// Returns an empty map when no transform filter was provided.
fn build_per_space_filter_masks(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> HashMap<RenderSpaceId, Vec<bool>> {
    if ctx.transform_filter.is_some() {
        space_ids
            .iter()
            .copied()
            .filter_map(|sid| {
                let mask = ctx.transform_filter?.build_pass_mask(ctx.scene, sid)?;
                Some((sid, mask))
            })
            .collect()
    } else {
        HashMap::new()
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
    space_ids: &[RenderSpaceId],
) -> Vec<(Vec<WorldMeshDrawItem>, (usize, usize, usize))> {
    // Prefer the pre-expanded dense draw list when the caller shared one (multi-view paths build
    // it once per frame). The scene-walk path below stays as the single-view fallback.
    if let Some(prepared) = ctx.prepared {
        debug_assert_eq!(
            prepared.render_context(),
            ctx.render_context,
            "prepared renderables were built for a different render context than the per-view draw collection — material overrides would disagree"
        );
        profiling::scope!("mesh::collect_prepared");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => prepared
                .draws
                .par_chunks(PREPARED_CHUNK_SIZE)
                .map(|chunk| collect_prepared_chunk(chunk, ctx, cache, filter_masks))
                .collect(),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => prepared
                .draws
                .chunks(PREPARED_CHUNK_SIZE)
                .map(|chunk| collect_prepared_chunk(chunk, ctx, cache, filter_masks))
                .collect(),
        }
    } else {
        let chunks = build_chunk_specs(space_ids, ctx);
        profiling::scope!("mesh::collect");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => chunks
                .par_iter()
                .map(|spec| collect_chunk(spec, ctx, cache, filter_masks))
                .collect(),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => chunks
                .iter()
                .map(|spec| collect_chunk(spec, ctx, cache, filter_masks))
                .collect(),
        }
    }
}

#[cfg(test)]
#[path = "collect_tests.rs"]
mod tests;
