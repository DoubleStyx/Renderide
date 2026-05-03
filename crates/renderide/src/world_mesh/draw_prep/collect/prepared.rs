//! Prepared-renderable draw collection path for world-mesh renderables.

use hashbrown::HashMap;

use glam::Mat4;

use crate::materials::RasterFrontFace;
use crate::scene::{RenderSpaceId, SkinnedMeshRenderer};

use crate::world_mesh::culling::{
    CpuCullFailure, MeshCullTarget, mesh_cpu_cull_with_geometry,
    mesh_world_geometry_for_cull_with_head,
};
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::super::item::{WorldMeshDrawItem, stacked_material_submesh_topology};
use super::super::prepared_renderables::FramePreparedDraw;
use super::candidate::{DrawCandidate, evaluate_draw_candidate};
use super::{
    DrawCollectionContext, front_face_for_world_matrix, world_matrix_for_local_vertex_stream,
};

/// Rayon chunk width when iterating a pre-expanded [`super::FramePreparedRenderables`] list.
///
/// Matches the scene-walk chunk width so per-view CPU cost stays bounded by the same per-task
/// overhead as the scene-walk path.
pub(super) const PREPARED_CHUNK_SIZE: usize = 256;

/// Returns true when two prepared slot entries came from the same source renderer.
#[inline]
pub(in crate::world_mesh::draw_prep) fn prepared_draws_share_renderer(
    a: &FramePreparedDraw,
    b: &FramePreparedDraw,
) -> bool {
    a.space_id == b.space_id
        && a.renderable_index == b.renderable_index
        && a.instance_id == b.instance_id
        && a.node_id == b.node_id
        && a.mesh_asset_id == b.mesh_asset_id
        && a.is_overlay == b.is_overlay
        && a.sorting_order == b.sorting_order
        && a.skinned == b.skinned
        && a.world_space_deformed == b.world_space_deformed
        && a.blendshape_deformed == b.blendshape_deformed
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
        // Reuse the per-renderer geometry that `FramePreparedRenderables::build_for_frame` already
        // computed for non-overlay spaces. Overlay spaces (geometry depends on the per-view
        // `head_output_transform`) keep recomputing per-view via the fallback path below.
        let geom = match first.cull_geometry {
            Some(g) => g,
            None => {
                let target = MeshCullTarget {
                    scene: ctx.scene,
                    space_id: first.space_id,
                    mesh,
                    skinned: first.skinned,
                    skinned_renderer: skinning.as_renderer(),
                    node_id: first.node_id,
                };
                mesh_world_geometry_for_cull_with_head(
                    &target,
                    c.host_camera.head_output_transform,
                    ctx.render_context,
                )
            }
        };
        match mesh_cpu_cull_with_geometry(geom, ctx.scene, first.space_id, first.is_overlay, c) {
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
    let alpha_distance_sq = rigid_world_matrix.map_or(0.0, |m| {
        (m.col(3).truncate() - ctx.view_origin_world).length_squared()
    });
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
    mesh: &crate::assets::mesh::GpuMesh,
    state: PreparedRunViewState,
    out: &mut Vec<WorldMeshDrawItem>,
) {
    for d in run {
        let primitive_topology =
            stacked_material_submesh_topology(d.slot_index, &mesh.submesh_topologies);
        let candidate = DrawCandidate {
            space_id: d.space_id,
            node_id: d.node_id,
            renderable_index: d.renderable_index,
            instance_id: d.instance_id,
            mesh_asset_id: d.mesh_asset_id,
            slot_index: d.slot_index,
            first_index: d.first_index,
            index_count: d.index_count,
            is_overlay: d.is_overlay,
            sorting_order: d.sorting_order,
            skinned: d.skinned,
            world_space_deformed: d.world_space_deformed,
            blendshape_deformed: d.blendshape_deformed,
            material_asset_id: d.material_asset_id,
            property_block_id: d.property_block_id,
        };
        if let Some(item) = evaluate_draw_candidate(
            ctx,
            cache,
            candidate,
            state.front_face,
            primitive_topology,
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
    let Some(mesh) = ctx.mesh_pool.get(first.mesh_asset_id) else {
        return (0, 0, 0);
    };
    let skinning = prepared_run_skinned_renderer(first, ctx);
    if matches!(skinning, PreparedRunSkinning::Stale) {
        return (0, 0, 0);
    }
    let (state, cull_stats) = prepared_run_view_state(run, first, mesh, &skinning, ctx);
    if let Some(state) = state {
        append_prepared_run_draws(run, ctx, cache, mesh, state, out);
    }
    cull_stats
}

/// Collects draw items for one chunk of a pre-expanded [`super::FramePreparedRenderables`] list.
///
/// Unlike the scene-walk chunk collector, there is no scene walk: the prepared draws already
/// captured every valid `(renderer x material slot)` tuple plus its frame-global resolution
/// (material override, submesh index range, overlay flag, skin deform flag). This per-view pass
/// only applies filters and per-view CPU culling per renderer, then builds [`WorldMeshDrawItem`]s
/// for each material slot.
pub(super) fn collect_prepared_chunk(
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
