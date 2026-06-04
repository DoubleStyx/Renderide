//! Per-renderer fan-out for the scene-walk collection path.

use crate::scene::MeshMaterialSlot;
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::super::super::item::stacked_material_submesh_range;
use super::super::{
    DrawCollectionInputs, effective_overlay_in_view, special_layer_visible_in_view,
};
use super::cull_cache::compute_cached_cull;
use super::per_slot::push_one_slot_draw;
use super::{
    DrawCollectionAccumulator, OverlayDeformCullFlags, StaticMeshDrawSource, SubmeshSlotIndices,
    transform_chain_has_degenerate_scale,
};

/// Expands one static mesh renderer into draw items (material slots mapped to submesh ranges).
///
/// `collect_order` is filled with a placeholder; the merge step in
/// [`super::super::queue_draws_with_parallelism`] assigns the final stable index after
/// per-chunk results are merged.
pub(super) fn push_draws_for_renderer(
    ctx: &DrawCollectionInputs<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: StaticMeshDrawSource<'_>,
    cache: &FrameMaterialBatchCache,
) {
    if !renderer_passes_view_filters(ctx, acc, &draw) {
        return;
    }
    let special_layer = if draw.renderer.node_id >= 0 {
        ctx.scene_assets
            .scene
            .transform_special_layer(draw.space_id, draw.renderer.node_id as usize)
    } else {
        None
    };
    if !special_layer_visible_in_view(ctx, special_layer) {
        return;
    }

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

    let n_sub = draw.submeshes.len();
    let n_slot = slots.len();
    if n_sub == 0 || n_slot == 0 {
        return;
    }

    let is_overlay = effective_overlay_in_view(
        ctx,
        matches!(special_layer, Some(crate::shared::LayerType::Overlay)),
    );
    let world_space_deformed = draw.skinned
        && draw.mesh.supports_world_space_skin_deform(
            draw.skinned_renderer
                .map(|skinned| skinned.bone_transform_indices.as_slice()),
        );
    let blendshape_deformed = draw
        .mesh
        .supports_active_blendshape_deform(&draw.renderer.blend_shape_weights);
    let tangent_blendshape_deform_active = draw
        .mesh
        .supports_active_tangent_blendshape_deform(&draw.renderer.blend_shape_weights);

    // Cull hoist: when a renderer expands to multiple material slots, every slot's cull would
    // otherwise re-test the same world AABB / Hi-Z. Compute it once and feed every slot the
    // cached outcome. For single-slot renderers (the common case for static scenery) the hoist
    // has no work to amortize, so let `push_one_slot_draw` cull inline via its `None` branch.
    let cached_cull = if n_slot > 1 {
        compute_cached_cull(ctx, &draw, is_overlay)
    } else {
        None
    };

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
                blendshape_deformed,
                tangent_blendshape_deform_active,
                cached_cull: cached_cull.as_ref(),
            },
            cache,
        );
    }
}

/// Applies the per-view transform filter and the renderer's transform-scale gate.
fn renderer_passes_view_filters(
    ctx: &DrawCollectionInputs<'_>,
    acc: &DrawCollectionAccumulator<'_>,
    draw: &StaticMeshDrawSource<'_>,
) -> bool {
    if let Some(f) = ctx.view.transform_filter {
        let passes = match acc.filter_pass_mask {
            Some(mask) => {
                let nid = draw.renderer.node_id;
                nid >= 0 && (nid as usize) < mask.len() && mask[nid as usize]
            }
            None => {
                f.passes_scene_node(ctx.scene_assets.scene, draw.space_id, draw.renderer.node_id)
            }
        };
        if !passes {
            return false;
        }
    }
    if transform_chain_has_degenerate_scale(ctx, draw.space_id, draw.renderer.node_id) {
        return false;
    }
    true
}
