//! Draw batch collection: filters drawables, builds draw entries, and creates space batches.
//!
//! Used by [`Session::collect_draw_batches`] and [`Session::collect_draw_batches_for_task`].

use std::collections::HashSet;

use glam::Mat4;

use crate::assets::{self, AssetRegistry};
use crate::gpu::PipelineVariant;
use crate::render::batch::{DrawEntry, SpaceDrawBatch};
use crate::scene::{Drawable, Scene, SceneGraph, render_transform_to_matrix};
use crate::shared::{LayerType, VertexAttributeType};
use crate::stencil::{StencilOperation, StencilState};

/// Filtered drawable with world matrix and pipeline variant.
///
/// Output of [`filter_and_collect_drawables`]; input to [`build_draw_entries`].
pub(super) struct FilteredDrawable {
    pub(super) drawable: Drawable,
    pub(super) world_matrix: Mat4,
    pub(super) pipeline_variant: PipelineVariant,
}

/// Filters drawables by layer, render lists, and skinned validity; collects world matrices.
///
/// Skips Hidden layer, applies only/exclude lists, validates bone_transform_ids and bind_poses
/// for skinned draws. Returns (Drawable, world_matrix, pipeline_variant) for each valid draw.
#[allow(clippy::too_many_arguments)]
pub(super) fn filter_and_collect_drawables(
    scene: &Scene,
    only_render_list: &[i32],
    exclude_render_list: &[i32],
    scene_graph: &SceneGraph,
    space_id: i32,
    asset_registry: &AssetRegistry,
    use_debug_uv: bool,
    use_pbr: bool,
) -> Vec<FilteredDrawable> {
    let only_set: HashSet<i32> = only_render_list.iter().copied().collect();
    let exclude_set: HashSet<i32> = exclude_render_list.iter().copied().collect();
    let use_only = !only_set.is_empty();
    let use_exclude = !exclude_set.is_empty();

    let mut out = Vec::new();
    let combined = scene
        .drawables
        .iter()
        .map(|d| (d, false))
        .chain(scene.skinned_drawables.iter().map(|d| (d, true)));

    for (entry, is_skinned) in combined {
        if entry.node_id < 0 {
            continue;
        }
        if entry.layer == LayerType::hidden {
            continue;
        }
        if use_only && !only_set.contains(&entry.node_id) {
            continue;
        }
        if use_exclude && exclude_set.contains(&entry.node_id) {
            continue;
        }
        if is_skinned {
            if entry
                .bone_transform_ids
                .as_ref()
                .is_none_or(|b| b.is_empty())
            {
                logger::trace!(
                    "Skinned draw skipped: bone_transform_ids missing or empty (node_id={})",
                    entry.node_id
                );
                continue;
            }
            if let Some(mesh) = asset_registry.get_mesh(entry.mesh_handle)
                && mesh.bind_poses.as_ref().is_none_or(|b| b.is_empty())
            {
                logger::trace!(
                    "Skinned draw skipped: mesh missing bind_poses (mesh={}, node_id={})",
                    entry.mesh_handle,
                    entry.node_id
                );
                continue;
            }
        }
        let idx = entry.node_id as usize;
        let world_matrix = match scene_graph.get_world_matrix(space_id, idx) {
            Some(m) => m,
            None => {
                if idx >= scene.nodes.len() {
                    continue;
                }
                render_transform_to_matrix(&scene.nodes[idx])
            }
        };

        let stencil_state = resolve_overlay_stencil_state(scene.is_overlay, entry, asset_registry);
        let mut drawable = entry.clone();
        drawable.stencil_state = stencil_state;

        let pipeline_variant = compute_pipeline_variant_for_drawable(
            scene.is_overlay,
            is_skinned,
            &drawable,
            entry.mesh_handle,
            use_debug_uv,
            use_pbr,
            asset_registry,
        );
        out.push(FilteredDrawable {
            drawable,
            world_matrix,
            pipeline_variant,
        });
    }

    out
}

/// Builds draw entries from filtered drawables.
///
/// Converts [`FilteredDrawable`] tuples into [`DrawEntry`] for batch construction.
pub(super) fn build_draw_entries(filtered: Vec<FilteredDrawable>) -> Vec<DrawEntry> {
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
                stencil_state: f.drawable.stencil_state,
                shadow_cast_mode: f.drawable.shadow_cast_mode,
            }
        })
        .collect()
}

/// Creates a space batch if draws is non-empty.
///
/// Returns `None` when draws is empty; otherwise builds [`SpaceDrawBatch`] from scene metadata.
/// For overlay spaces, when `view_override` is `Some`, uses it as the batch view transform
/// (primary/head view) instead of `scene.view_transform` (root).
pub(super) fn create_space_batch(
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

/// Resolves overlay stencil state from material property store when scene is overlay.
pub(super) fn resolve_overlay_stencil_state(
    is_overlay: bool,
    entry: &Drawable,
    asset_registry: &AssetRegistry,
) -> Option<StencilState> {
    if !is_overlay {
        return None;
    }
    if let Some(block_id) = entry.material_override_block_id {
        StencilState::from_property_store(&asset_registry.material_property_store, block_id)
            .or(entry.stencil_state)
    } else {
        entry.stencil_state
    }
}

/// Computes pipeline variant for a drawable based on overlay, skinned, stencil, and mesh.
pub(super) fn compute_pipeline_variant_for_drawable(
    is_overlay: bool,
    is_skinned: bool,
    drawable: &Drawable,
    mesh_asset_id: i32,
    use_debug_uv: bool,
    use_pbr: bool,
    asset_registry: &AssetRegistry,
) -> PipelineVariant {
    if is_overlay {
        if let Some(ref stencil) = drawable.stencil_state {
            if stencil.pass_op == StencilOperation::Replace && stencil.write_mask != 0 {
                if is_skinned {
                    PipelineVariant::OverlayStencilMaskWriteSkinned
                } else {
                    PipelineVariant::OverlayStencilMaskWrite
                }
            } else if stencil.pass_op == StencilOperation::Zero {
                if is_skinned {
                    PipelineVariant::OverlayStencilMaskClearSkinned
                } else {
                    PipelineVariant::OverlayStencilMaskClear
                }
            } else if is_skinned {
                PipelineVariant::OverlayStencilSkinned
            } else {
                PipelineVariant::OverlayStencilContent
            }
        } else if is_skinned {
            PipelineVariant::Skinned
        } else {
            compute_pipeline_variant(false, mesh_asset_id, use_debug_uv, false, asset_registry)
        }
    } else if is_skinned {
        if use_pbr {
            PipelineVariant::SkinnedPbr
        } else {
            PipelineVariant::Skinned
        }
    } else {
        compute_pipeline_variant(false, mesh_asset_id, use_debug_uv, use_pbr, asset_registry)
    }
}

/// Computes pipeline variant from is_skinned, mesh UVs, use_debug_uv, and use_pbr.
fn compute_pipeline_variant(
    is_skinned: bool,
    mesh_asset_id: i32,
    use_debug_uv: bool,
    use_pbr: bool,
    asset_registry: &AssetRegistry,
) -> PipelineVariant {
    if is_skinned {
        return PipelineVariant::Skinned;
    }
    let has_uvs = asset_registry
        .get_mesh(mesh_asset_id)
        .and_then(|m| {
            assets::attribute_offset_size_format(&m.vertex_attributes, VertexAttributeType::uv0)
        })
        .map(|(_, s, _)| s >= 4)
        .unwrap_or(false);
    if use_debug_uv && has_uvs {
        PipelineVariant::UvDebug
    } else if use_pbr {
        PipelineVariant::Pbr
    } else {
        PipelineVariant::NormalDebug
    }
}

#[cfg(test)]
mod tests {
    use super::{FilteredDrawable, build_draw_entries, create_space_batch};
    use crate::gpu::PipelineVariant;
    use crate::render::batch::DrawEntry;
    use crate::scene::{Drawable, Scene};
    use crate::shared::ShadowCastMode;
    use glam::Mat4;

    fn make_scene(space_id: i32, is_overlay: bool) -> Scene {
        Scene {
            id: space_id,
            is_overlay,
            ..Default::default()
        }
    }

    #[test]
    fn create_space_batch_returns_none_when_empty() {
        let scene = make_scene(0, false);
        let batch = create_space_batch(0, &scene, vec![], None);
        assert!(batch.is_none());
    }

    #[test]
    fn create_space_batch_returns_some_when_non_empty() {
        let mut scene = make_scene(5, false);
        scene.view_transform = crate::shared::RenderTransform::default();
        let draw = DrawEntry {
            model_matrix: Mat4::IDENTITY,
            node_id: 0,
            mesh_asset_id: 1,
            is_skinned: false,
            material_id: -1,
            sort_key: 0,
            bone_transform_ids: None,
            root_bone_transform_id: None,
            blendshape_weights: None,
            pipeline_variant: PipelineVariant::NormalDebug,
            stencil_state: None,
            shadow_cast_mode: crate::shared::ShadowCastMode::on,
        };
        let batch = create_space_batch(5, &scene, vec![draw], None);
        let batch = batch.expect("should have batch");
        assert_eq!(batch.space_id, 5);
        assert!(!batch.is_overlay);
        assert_eq!(batch.draws.len(), 1);
    }

    #[test]
    fn build_draw_entries_preserves_order() {
        let filtered = vec![FilteredDrawable {
            drawable: Drawable {
                node_id: 0,
                mesh_handle: 1,
                material_handle: Some(10),
                sort_key: 5,
                is_skinned: false,
                ..Default::default()
            },
            world_matrix: Mat4::IDENTITY,
            pipeline_variant: PipelineVariant::NormalDebug,
        }];
        let entries = build_draw_entries(filtered);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].material_id, 10);
        assert_eq!(entries[0].sort_key, 5);
    }

    #[test]
    fn build_draw_entries_propagates_shadow_cast_mode() {
        let filtered = vec![FilteredDrawable {
            drawable: Drawable {
                node_id: 0,
                mesh_handle: 1,
                shadow_cast_mode: ShadowCastMode::off,
                ..Default::default()
            },
            world_matrix: Mat4::IDENTITY,
            pipeline_variant: PipelineVariant::NormalDebug,
        }];
        let entries = build_draw_entries(filtered);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].shadow_cast_mode, ShadowCastMode::off);
    }
}
