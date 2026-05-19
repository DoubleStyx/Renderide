//! Per material-slot draw emission for the scene-walk collection path.

use glam::Mat4;

use crate::scene::MeshMaterialSlot;
use crate::world_mesh::culling::CpuCullFailure;
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::super::super::item::stacked_material_submesh_topology;
use super::super::DrawCollectionContext;
use super::super::candidate::{DrawCandidate, evaluate_draw_candidate};
use super::super::world_matrix::{
    front_face_for_draw_matrices, skinned_front_face_world_matrix,
    world_matrix_for_local_vertex_stream,
};
use super::cull_cache::{cull_result_for_slot, world_aabb_for_reflection_probe_selection};
use super::{
    DrawCollectionAccumulator, OverlayDeformCullFlags, StaticMeshDrawSource, SubmeshSlotIndices,
};

/// Per-slot cull-resolution outcome consumed by [`push_one_slot_draw`].
enum SlotCullDecision {
    /// Draw is visible; carries the resolved rigid world matrix if the cache provided one.
    Visible(Option<Mat4>),
    /// Draw is rejected by frustum culling (or the equivalent UI rect mask check).
    RejectedFrustum,
    /// Draw is rejected by Hi-Z culling.
    RejectedHiZ,
    /// No cull pass was active for this draw (skinned, or no culling config); skip cull bookkeeping.
    NotApplicable,
}

/// One material slot mapped to a submesh range: optional CPU cull, batch key, and [`WorldMeshDrawItem`] push.
pub(super) fn push_one_slot_draw(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: &StaticMeshDrawSource<'_>,
    slot: &MeshMaterialSlot,
    indices: SubmeshSlotIndices,
    flags: OverlayDeformCullFlags<'_>,
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
        blendshape_deformed,
        tangent_blendshape_deform_active,
        cached_cull,
    } = flags;

    let Some(material_asset_id) = resolve_material_asset_id(ctx, draw, slot, slot_index) else {
        return;
    };
    if index_count == 0 {
        return;
    }

    let rigid_world_matrix = match resolve_cull_decision(ctx, acc, draw, is_overlay, cached_cull) {
        SlotCullDecision::Visible(m) => {
            fill_rigid_world_matrix(ctx, draw, is_overlay, world_space_deformed, m)
        }
        SlotCullDecision::NotApplicable => {
            fill_rigid_world_matrix(ctx, draw, is_overlay, world_space_deformed, None)
        }
        SlotCullDecision::RejectedFrustum | SlotCullDecision::RejectedHiZ => return,
    };

    let deformed_front_face_world_matrix = if world_space_deformed {
        skinned_front_face_world_matrix(
            ctx,
            draw.space_id,
            draw.renderer.node_id,
            draw.skinned_renderer,
        )
    } else {
        None
    };
    let front_face = front_face_for_draw_matrices(
        world_space_deformed,
        rigid_world_matrix,
        deformed_front_face_world_matrix,
    );
    let primitive_topology =
        stacked_material_submesh_topology(slot_index, &draw.mesh.submesh_topologies);
    let alpha_distance_sq = rigid_world_matrix.map_or(0.0, |m| {
        (m.col(3).truncate() - ctx.view_origin_world).length_squared()
    });
    let world_aabb = world_aabb_for_reflection_probe_selection(ctx, draw);
    let candidate = DrawCandidate {
        space_id: draw.space_id,
        node_id: draw.renderer.node_id,
        renderable_index: draw.renderable_index,
        instance_id: draw.instance_id,
        mesh_asset_id: draw.renderer.mesh_asset_id,
        slot_index,
        first_index,
        index_count,
        is_overlay,
        sorting_order: draw.renderer.sorting_order,
        skinned: draw.skinned,
        world_space_deformed,
        blendshape_deformed,
        tangent_blendshape_deform_active,
        shadow_cast_mode: draw.renderer.shadow_cast_mode,
        material_asset_id,
        property_block_id: slot.property_block_id,
        world_aabb,
    };
    if let Some(item) = evaluate_draw_candidate(
        ctx,
        cache,
        candidate,
        front_face,
        primitive_topology,
        rigid_world_matrix,
        alpha_distance_sq,
    ) {
        acc.out.push(item);
    }
}

/// Resolves the effective material id for this slot, honoring scene-level overrides.
fn resolve_material_asset_id(
    ctx: &DrawCollectionContext<'_>,
    draw: &StaticMeshDrawSource<'_>,
    slot: &MeshMaterialSlot,
    slot_index: usize,
) -> Option<i32> {
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
    (material_asset_id >= 0).then_some(material_asset_id)
}

/// Reads the cached per-renderer cull outcome (or runs an inline cull for single-slot renderers)
/// and folds it into a [`SlotCullDecision`], updating the running pre-cull / culled counters.
fn resolve_cull_decision(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: &StaticMeshDrawSource<'_>,
    is_overlay: bool,
    cached_cull: Option<&super::cull_cache::CachedCull>,
) -> SlotCullDecision {
    let Some(outcome) = cull_result_for_slot(ctx, draw, is_overlay, cached_cull) else {
        return SlotCullDecision::NotApplicable;
    };
    acc.cull_stats.0 += 1;
    match outcome {
        Err(CpuCullFailure::Frustum | CpuCullFailure::UiRectMask) => {
            acc.cull_stats.1 += 1;
            SlotCullDecision::RejectedFrustum
        }
        Err(CpuCullFailure::HiZ) => {
            acc.cull_stats.2 += 1;
            SlotCullDecision::RejectedHiZ
        }
        Ok(m) => SlotCullDecision::Visible(m),
    }
}

/// Picks the rigid world matrix for the local-vertex-stream path: prefer the value the cull cache
/// already computed; otherwise fall back to the overlay-aware scene lookup.
fn fill_rigid_world_matrix(
    ctx: &DrawCollectionContext<'_>,
    draw: &StaticMeshDrawSource<'_>,
    is_overlay: bool,
    world_space_deformed: bool,
    cached: Option<Mat4>,
) -> Option<Mat4> {
    if is_overlay && !world_space_deformed {
        return world_matrix_for_local_vertex_stream(
            ctx,
            draw.space_id,
            draw.renderer.node_id,
            true,
        );
    }
    if !world_space_deformed && cached.is_none() {
        return world_matrix_for_local_vertex_stream(
            ctx,
            draw.space_id,
            draw.renderer.node_id,
            false,
        );
    }
    cached
}
