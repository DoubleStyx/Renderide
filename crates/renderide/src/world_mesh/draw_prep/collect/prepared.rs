//! Prepared-renderable draw collection path for world-mesh renderables.

use hashbrown::HashMap;

use glam::{Mat4, Vec3};

use crate::materials::RasterFrontFace;
use crate::scene::{RenderSpaceId, SkinnedMeshRenderer};
use crate::shared::LayerType;

use crate::world_mesh::WorldMeshPhase;
use crate::world_mesh::culling::{
    CpuCullFailure, MeshCullGeometry, MeshCullTarget, mesh_cpu_cull_with_geometry,
    mesh_world_geometry_for_cull_with_head,
};
use crate::world_mesh::materials::FrameMaterialBatchCache;
use crate::world_mesh::phase_classification::classify_world_mesh_batch;

use super::super::item::{WorldMeshDrawItem, stacked_material_submesh_topology};
use super::super::prepared_renderables::{FramePreparedDraw, FramePreparedRun};
use super::candidate::{DrawCandidate, evaluate_draw_candidate};
use super::transform_chain_has_degenerate_scale;
use super::world_matrix::{front_face_for_draw_matrices, world_matrix_for_local_vertex_stream};
use super::{CollectState, DrawCollectionInputs};
use super::{effective_overlay_in_view, special_layer_visible_in_view, transform_filter_for_space};

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
        && a.is_hidden == b.is_hidden
        && a.sorting_order == b.sorting_order
        && a.shadow_cast_mode == b.shadow_cast_mode
        && a.skinned == b.skinned
        && a.world_space_deformed == b.world_space_deformed
        && a.blendshape_deformed == b.blendshape_deformed
        && a.tangent_blendshape_deform_active == b.tangent_blendshape_deform_active
        && a.rigid_world_matrix_override == b.rigid_world_matrix_override
}

/// Per-renderer view-local state shared by every material slot in a prepared run.
#[derive(Clone, Copy)]
struct PreparedRunViewState {
    /// Rigid model matrix reused by all emitted slot draws.
    rigid_world_matrix: Option<Mat4>,
    /// World-space object AABB reused by all emitted slot draws for transparent sorting and probes.
    world_aabb: Option<(Vec3, Vec3)>,
    /// Raster front-face winding selected from [`Self::rigid_world_matrix`].
    front_face: RasterFrontFace,
    /// Camera distance reused by alpha-blended slot draws.
    alpha_distance_sq: f32,
    /// Geometry retained so unsupported material slots in an otherwise GPU-static renderer run
    /// can still use the CPU visibility fallback after material classification.
    cull_geometry: Option<MeshCullGeometry>,
}

/// Skinned renderer lookup result for a prepared renderer run.
enum PreparedRunSkinning<'a> {
    /// The renderer uses the rigid static-mesh path.
    Rigid,
    /// The renderer uses the skinned path and still has a valid scene entry.
    Skinned(&'a SkinnedMeshRenderer),
    /// The prepared index does not resolve to a valid skinned renderer.
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
    ctx: &DrawCollectionInputs<'_>,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> bool {
    let Some(filter) = transform_filter_for_space(ctx, first.space_id) else {
        return true;
    };
    match filter_masks.get(&first.space_id) {
        Some(mask) => {
            first.node_id >= 0
                && (first.node_id as usize) < mask.len()
                && mask[first.node_id as usize]
        }
        None => filter.passes_scene_node(ctx.scene_assets.scene, first.space_id, first.node_id),
    }
}

/// Returns the skinned renderer backing a prepared run, or `None` when stale scene indices should skip it.
fn prepared_run_skinned_renderer<'a>(
    first: &FramePreparedDraw,
    ctx: &'a DrawCollectionInputs<'_>,
) -> PreparedRunSkinning<'a> {
    if !first.skinned {
        return PreparedRunSkinning::Rigid;
    }
    let Some(space) = ctx.scene_assets.scene.space(first.space_id) else {
        return PreparedRunSkinning::Stale;
    };
    space
        .skinned_mesh_renderers()
        .get(first.renderable_index)
        .map_or(PreparedRunSkinning::Stale, PreparedRunSkinning::Skinned)
}

/// Builds shared view-local state for one prepared renderer run and reports draw-slot cull stats.
fn prepared_run_view_state(
    run: &[FramePreparedDraw],
    first: &FramePreparedDraw,
    is_overlay: bool,
    mesh: &crate::assets::mesh::GpuMesh,
    skinning: &PreparedRunSkinning<'_>,
    ctx: &DrawCollectionInputs<'_>,
    defer_visibility_to_gpu: bool,
) -> (Option<PreparedRunViewState>, (usize, usize, usize)) {
    let mut cull_stats = (0usize, 0usize, 0usize);
    let mut rigid_world_matrix = None;
    let mut world_aabb = None;
    let mut deformed_front_face_world_matrix = None;
    if let Some(override_matrix) = first.rigid_world_matrix_override {
        rigid_world_matrix = Some(override_matrix);
    }
    let needs_geometry = ctx.view.reflection_probes.is_some()
        || ctx.view.culling.is_some()
        || first.world_space_deformed
        || defer_visibility_to_gpu;
    let geometry = (needs_geometry && first.rigid_world_matrix_override.is_none()).then(|| {
        // Reuse the per-renderer geometry that `FramePreparedRenderables::build_for_frame` already
        // computed for non-overlay spaces. Overlay spaces (geometry depends on the per-view
        // `head_output_transform`) keep recomputing per-view via the fallback path below.
        first.cull_geometry.unwrap_or_else(|| {
            let target = MeshCullTarget {
                scene: ctx.scene_assets.scene,
                space_id: first.space_id,
                mesh,
                skinned: first.skinned,
                skinned_renderer: skinning.as_renderer(),
                node_id: first.node_id,
            };
            mesh_world_geometry_for_cull_with_head(
                &target,
                ctx.view.head_output_transform,
                ctx.view.render_context,
            )
        })
    });
    if let Some(geom) = geometry {
        world_aabb = geom.world_aabb;
        deformed_front_face_world_matrix = geom.front_face_world_matrix;
        if let Some(c) = ctx.view.culling
            && !defer_visibility_to_gpu
        {
            cull_stats.0 += run.len();
            match mesh_cpu_cull_with_geometry(
                geom,
                ctx.scene_assets.scene,
                first.space_id,
                is_overlay,
                c,
                None,
            ) {
                Err(CpuCullFailure::Frustum | CpuCullFailure::UiRectMask) => {
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
        } else if rigid_world_matrix.is_none() {
            rigid_world_matrix = geom.rigid_world_matrix;
        }
    }
    if is_overlay && !first.world_space_deformed {
        rigid_world_matrix =
            world_matrix_for_local_vertex_stream(ctx, first.space_id, first.node_id, true);
    } else if !first.world_space_deformed && rigid_world_matrix.is_none() {
        rigid_world_matrix =
            world_matrix_for_local_vertex_stream(ctx, first.space_id, first.node_id, false);
    }
    let front_face = front_face_for_draw_matrices(
        first.world_space_deformed,
        rigid_world_matrix,
        deformed_front_face_world_matrix,
    );
    let alpha_distance_sq = rigid_world_matrix.map_or(0.0, |m| {
        (m.col(3).truncate() - ctx.view.view_origin_world).length_squared()
    });
    (
        Some(PreparedRunViewState {
            rigid_world_matrix,
            world_aabb,
            front_face,
            alpha_distance_sq,
            cull_geometry: geometry,
        }),
        cull_stats,
    )
}

/// Returns whether a prepared renderer can source every draw stream from the shared static arena.
///
/// Tangent-only blendshape activation is material-dependent, so a renderer carrying any active
/// tangent blendshape is kept on the conservative CPU/deform path even if a particular material
/// might not consume tangents.
#[inline]
fn renderer_can_defer_visibility_to_gpu(
    first: &FramePreparedDraw,
    mesh: &crate::assets::mesh::GpuMesh,
    is_overlay: bool,
    ctx: &DrawCollectionInputs<'_>,
) -> bool {
    ctx.view.retain_gpu_static_candidates
        && ctx.view.culling.is_some()
        && !is_overlay
        && !first.skinned
        && !first.world_space_deformed
        && !first.blendshape_deformed
        && !first.tangent_blendshape_deform_active
        && !mesh.dynamic_geometry
        && mesh.has_raster_core_residency()
}

/// Returns whether a resolved material slot belongs to the opaque/alpha-test phases supported by
/// the GPU visibility handoff. Intersection, transparent, and grab-pass slots retain CPU culling.
#[inline]
fn material_can_defer_visibility_to_gpu(item: &WorldMeshDrawItem) -> bool {
    item.ui_rect_clip_local.is_none()
        && matches!(
            classify_world_mesh_batch(&item.batch_key).phase,
            WorldMeshPhase::ForwardOpaque | WorldMeshPhase::ForwardAlphaTest
        )
}

/// Applies CPU culling only to unsupported slots in a renderer whose opaque/alpha-test slots were
/// retained for GPU visibility. This keeps mixed-material renderers correct without paying a CPU
/// frustum/Hi-Z test when every emitted slot can move to compute.
fn cull_gpu_static_run_fallback_slots(
    ctx: &DrawCollectionInputs<'_>,
    first: &FramePreparedDraw,
    is_overlay: bool,
    state: &PreparedRunViewState,
    run_output_start: usize,
    out: &mut Vec<WorldMeshDrawItem>,
) -> (usize, usize, usize) {
    let fallback_slot_count = out[run_output_start..]
        .iter()
        .filter(|item| !material_can_defer_visibility_to_gpu(item))
        .count();
    if fallback_slot_count == 0 {
        return (0, 0, 0);
    }
    let (Some(culling), Some(geometry)) = (ctx.view.culling, state.cull_geometry) else {
        return (0, 0, 0);
    };

    let mut stats = (fallback_slot_count, 0usize, 0usize);
    let failure = mesh_cpu_cull_with_geometry(
        geometry,
        ctx.scene_assets.scene,
        first.space_id,
        is_overlay,
        culling,
        None,
    )
    .err();
    match failure {
        None => return stats,
        Some(CpuCullFailure::Frustum | CpuCullFailure::UiRectMask) => {
            stats.1 = fallback_slot_count;
        }
        Some(CpuCullFailure::HiZ) => {
            stats.2 = fallback_slot_count;
        }
    }

    let mut run_output = out.split_off(run_output_start);
    run_output.retain(material_can_defer_visibility_to_gpu);
    out.append(&mut run_output);
    stats
}

/// Emits one [`WorldMeshDrawItem`] per material slot in a surviving prepared renderer run.
fn append_prepared_run_draws(
    run: &[FramePreparedDraw],
    ctx: &DrawCollectionInputs<'_>,
    cache: &FrameMaterialBatchCache,
    mesh: &crate::assets::mesh::GpuMesh,
    is_overlay: bool,
    state: &PreparedRunViewState,
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
            material_stack_order: d.material_stack_order,
            first_index: d.first_index,
            index_count: d.index_count,
            is_overlay,
            sorting_order: d.sorting_order,
            shadow_cast_mode: d.shadow_cast_mode,
            skinned: d.skinned,
            world_space_deformed: d.world_space_deformed,
            blendshape_deformed: d.blendshape_deformed,
            tangent_blendshape_deform_active: d.tangent_blendshape_deform_active,
            material_asset_id: d.material_asset_id,
            property_block_id: d.property_block_id,
            world_aabb: state.world_aabb,
            particle_draw: d.particle_draw,
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
    ctx: &DrawCollectionInputs<'_>,
    state: CollectState<'_>,
    out: &mut Vec<WorldMeshDrawItem>,
) -> (usize, usize, usize) {
    let Some(first) = run.first() else {
        return (0, 0, 0);
    };
    if !ctx.view.render_space_scope.includes(first.space_id) {
        return (0, 0, 0);
    }
    if !prepared_run_passes_filter(first, ctx, state.filter_masks) {
        return (0, 0, 0);
    }
    let source_is_overlay = first.is_overlay;
    let source_special_layer = if first.is_hidden {
        Some(LayerType::Hidden)
    } else if source_is_overlay {
        Some(LayerType::Overlay)
    } else {
        None
    };
    if !special_layer_visible_in_view(ctx, source_special_layer) {
        return (0, 0, 0);
    }
    if !state
        .lod_visibility
        .renderer_visible(first.space_id, first.renderer_ordinal)
    {
        return (0, 0, 0);
    }
    if transform_chain_has_degenerate_scale(ctx, first.space_id, first.node_id) {
        return (0, 0, 0);
    }
    let is_overlay = effective_overlay_in_view(ctx, source_is_overlay);
    let Some(mesh) = ctx.scene_assets.mesh_pool.get(first.mesh_asset_id) else {
        return (0, 0, 0);
    };
    let skinning = prepared_run_skinned_renderer(first, ctx);
    if matches!(skinning, PreparedRunSkinning::Stale) {
        return (0, 0, 0);
    }
    let defer_visibility_to_gpu =
        renderer_can_defer_visibility_to_gpu(first, mesh, is_overlay, ctx);
    let (view_state, mut cull_stats) = prepared_run_view_state(
        run,
        first,
        is_overlay,
        mesh,
        &skinning,
        ctx,
        defer_visibility_to_gpu,
    );
    if let Some(view_state) = view_state {
        let run_output_start = out.len();
        append_prepared_run_draws(run, ctx, state.cache, mesh, is_overlay, &view_state, out);
        if defer_visibility_to_gpu {
            cull_stats = cull_gpu_static_run_fallback_slots(
                ctx,
                first,
                is_overlay,
                &view_state,
                run_output_start,
                out,
            );
        }
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
    runs: &[FramePreparedRun],
    ctx: &DrawCollectionInputs<'_>,
    state: CollectState<'_>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    profiling::scope!("mesh::collect_prepared::chunk");
    let chunk_draws = {
        profiling::scope!("mesh::collect_prepared::chunk_capacity");
        runs.first()
            .and_then(|first| runs.last().map(|last| last.end - first.start))
            .unwrap_or(0) as usize
    };
    let mut out: Vec<WorldMeshDrawItem> = Vec::with_capacity(chunk_draws);
    let mut cull_stats = (0usize, 0usize, 0usize);

    {
        profiling::scope!("mesh::collect_prepared::renderer_runs");
        for prepared_run in runs {
            let start = prepared_run.start as usize;
            let end = prepared_run.end as usize;
            let run = &draws[start..end];
            let run_stats = collect_prepared_renderer_run(run, ctx, state, &mut out);
            cull_stats.0 += run_stats.0;
            cull_stats.1 += run_stats.1;
            cull_stats.2 += run_stats.2;
        }
    }

    (out, cull_stats)
}

#[cfg(test)]
mod tests {
    use crate::materials::UNITY_RENDER_QUEUE_ALPHA_TEST;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use super::material_can_defer_visibility_to_gpu;

    fn draw(alpha_blended: bool) -> crate::world_mesh::WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended,
        })
    }

    #[test]
    fn gpu_visibility_handoff_accepts_opaque_and_alpha_test_only() {
        let opaque = draw(false);
        assert!(material_can_defer_visibility_to_gpu(&opaque));

        let mut alpha_test = draw(false);
        alpha_test.batch_key.render_queue = UNITY_RENDER_QUEUE_ALPHA_TEST;
        assert!(material_can_defer_visibility_to_gpu(&alpha_test));

        let transparent = draw(true);
        assert!(!material_can_defer_visibility_to_gpu(&transparent));

        let mut intersection = draw(false);
        intersection.batch_key.embedded_requires_intersection_pass = true;
        assert!(!material_can_defer_visibility_to_gpu(&intersection));

        let mut grab = draw(false);
        grab.batch_key.embedded_uses_scene_color_snapshot = true;
        assert!(!material_can_defer_visibility_to_gpu(&grab));

        let mut scissored = draw(false);
        scissored.ui_rect_clip_local = Some(glam::Vec4::ZERO);
        assert!(!material_can_defer_visibility_to_gpu(&scissored));
    }
}
