//! Builds [`DrawEntry`] rows and [`SpaceDrawBatch`] from filtered drawables.

use crate::render::batch::{DrawEntry, SpaceDrawBatch};
use crate::scene::Scene;

use super::pipeline::FilteredDrawable;

/// Builds draw entries from filtered drawables.
///
/// Converts [`FilteredDrawable`] tuples into [`DrawEntry`] for batch construction.
pub(in crate::session) fn build_draw_entries(filtered: Vec<FilteredDrawable>) -> Vec<DrawEntry> {
    filtered
        .into_iter()
        .map(|f| {
            let material_id = f.drawable.material_handle.unwrap_or(-1);
            DrawEntry {
                model_matrix: f.world_matrix,
                node_id: f.drawable.node_id,
                mesh_asset_id: f.drawable.mesh_handle,
                is_skinned: f.drawable.is_skinned,
                material_id,
                sort_key: f.drawable.sort_key,
                bone_transform_ids: if f.drawable.is_skinned {
                    f.drawable.bone_transform_ids.clone()
                } else {
                    None
                },
                root_bone_transform_id: if f.drawable.is_skinned {
                    f.drawable.root_bone_transform_id
                } else {
                    None
                },
                blendshape_weights: if f.drawable.is_skinned {
                    f.drawable.blend_shape_weights.clone()
                } else {
                    None
                },
                pipeline_variant: f.pipeline_variant,
                shader_key: f.shader_key,
                stencil_state: f.drawable.stencil_state,
                shadow_cast_mode: f.drawable.shadow_cast_mode,
                mesh_renderer_property_block_slot0_id: f
                    .drawable
                    .mesh_renderer_property_block_slot0_id,
                submesh_index_range: f.submesh_index_range,
            }
        })
        .collect()
}

/// Creates a space batch if draws is non-empty.
///
/// Returns `None` when draws is empty; otherwise builds [`SpaceDrawBatch`] from scene metadata.
/// For overlay spaces, when `view_override` is `Some`, uses it as the batch view transform
/// (primary/head view) instead of `scene.view_transform` (root).
pub(in crate::session) fn create_space_batch(
    space_id: i32,
    scene: &Scene,
    draws: Vec<DrawEntry>,
    view_override: Option<crate::shared::RenderTransform>,
) -> Option<SpaceDrawBatch> {
    if draws.is_empty() {
        return None;
    }
    let view_transform = if scene.is_overlay {
        view_override.unwrap_or(scene.view_transform)
    } else {
        scene.view_transform
    };
    Some(SpaceDrawBatch {
        space_id,
        is_overlay: scene.is_overlay,
        view_transform,
        draws,
    })
}
