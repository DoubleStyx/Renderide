//! Draw batch collection: filters drawables, builds draw entries, and creates space batches.
//!
//! Used by [`Session::collect_draw_batches`] and [`Session::collect_draw_batches_for_task`].

use std::collections::HashSet;

use glam::Mat4;

use crate::assets::{self, AssetRegistry, NativeUiShaderFamily, resolve_native_ui_shader_family};
use crate::config::{RenderConfig, ShaderDebugOverride};
use crate::gpu::{PipelineVariant, ShaderKey};
use crate::render::batch::{DrawEntry, SpaceDrawBatch};
use crate::scene::{Drawable, Scene, SceneGraph, render_transform_to_matrix};
use crate::shared::{LayerType, VertexAttributeType};
use crate::stencil::{StencilOperation, StencilState};

/// Returns true when the mesh has UV0 and vertex color (legacy strict UI canvas check).
///
/// Only compiled for unit tests; production routing uses [`mesh_has_native_ui_vertices`].
#[cfg(test)]
pub(crate) fn mesh_has_ui_canvas_vertices(
    asset_registry: &AssetRegistry,
    mesh_asset_id: i32,
) -> bool {
    let Some(mesh) = asset_registry.get_mesh(mesh_asset_id) else {
        return false;
    };
    let uv =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::uv0);
    let color =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::color);
    uv.map(|(_, s, _)| s >= 4).unwrap_or(false) && color.map(|(_, s, _)| s >= 4).unwrap_or(false)
}

/// Returns true when the mesh has UV0 so native UI routing may use [`crate::gpu::mesh::GpuMeshBuffers::ui_canvas_buffers`]
/// (vertex color defaults to white when absent).
pub(crate) fn mesh_has_native_ui_vertices(
    asset_registry: &AssetRegistry,
    mesh_asset_id: i32,
) -> bool {
    let Some(mesh) = asset_registry.get_mesh(mesh_asset_id) else {
        return false;
    };
    let uv =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::uv0);
    uv.map(|(_, s, _)| s >= 4).unwrap_or(false)
}

/// After [`ShaderKey::effective_variant`], maps overlay (and optionally world) draws to native UI pipelines when allowed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_native_ui_pipeline_variant(
    is_overlay: bool,
    is_skinned: bool,
    stencil_state: Option<&StencilState>,
    render_config: &RenderConfig,
    host_shader_asset_id: Option<i32>,
    material_block_id: i32,
    mesh_asset_id: i32,
    current: PipelineVariant,
    asset_registry: &AssetRegistry,
) -> PipelineVariant {
    if !render_config.use_native_ui_wgsl
        || matches!(
            render_config.shader_debug_override,
            ShaderDebugOverride::ForceLegacyGlobalShading
        )
        || is_skinned
        || material_block_id < 0
    {
        return current;
    }
    let allow_surface = is_overlay || render_config.native_ui_world_space;
    if !allow_surface {
        return current;
    }
    if !is_overlay && stencil_state.is_some() {
        return current;
    }
    let has_stencil = stencil_state.is_some();
    if has_stencil && !render_config.native_ui_overlay_stencil_pipelines {
        return current;
    }
    let Some(shader_id) = host_shader_asset_id else {
        if render_config.log_native_ui_routing && is_overlay {
            logger::trace!(
                "native_ui: skip (no set_shader) material_block={} mesh={}",
                material_block_id,
                mesh_asset_id
            );
        }
        return current;
    };
    if !mesh_has_native_ui_vertices(asset_registry, mesh_asset_id) {
        if render_config.log_native_ui_routing && is_overlay {
            logger::trace!(
                "native_ui: skip (mesh missing uv0) shader_id={} material_block={} mesh={}",
                shader_id,
                material_block_id,
                mesh_asset_id
            );
        }
        return current;
    }
    let Some(family) = resolve_native_ui_shader_family(
        shader_id,
        render_config.native_ui_unlit_shader_id,
        render_config.native_ui_text_unlit_shader_id,
        asset_registry,
    ) else {
        if render_config.log_native_ui_routing && is_overlay {
            logger::trace!(
                "native_ui: skip (shader not recognized as UI) shader_id={} material_block={}",
                shader_id,
                material_block_id
            );
        }
        return current;
    };
    match family {
        NativeUiShaderFamily::UiUnlit => {
            if has_stencil {
                PipelineVariant::NativeUiUnlitStencil {
                    material_id: material_block_id,
                }
            } else {
                PipelineVariant::NativeUiUnlit {
                    material_id: material_block_id,
                }
            }
        }
        NativeUiShaderFamily::UiTextUnlit => {
            if has_stencil {
                PipelineVariant::NativeUiTextUnlitStencil {
                    material_id: material_block_id,
                }
            } else {
                PipelineVariant::NativeUiTextUnlit {
                    material_id: material_block_id,
                }
            }
        }
    }
}

/// Filtered drawable with world matrix and pipeline variant.
///
/// Output of [`filter_and_collect_drawables`]; input to [`build_draw_entries`].
pub(super) struct FilteredDrawable {
    pub(super) drawable: Drawable,
    pub(super) world_matrix: Mat4,
    pub(super) pipeline_variant: PipelineVariant,
    pub(super) shader_key: ShaderKey,
}

/// Filters drawables by layer, render lists, and skinned validity; collects world matrices.
///
/// Skips Hidden layer, applies only/exclude lists, validates bone_transform_ids and bind_poses
/// for skinned draws. Returns [`FilteredDrawable`] for each valid draw (including [`ShaderKey`]
/// and resolved [`PipelineVariant`](crate::gpu::PipelineVariant)).
#[allow(clippy::too_many_arguments)]
pub(super) fn filter_and_collect_drawables(
    scene: &Scene,
    only_render_list: &[i32],
    exclude_render_list: &[i32],
    scene_graph: &SceneGraph,
    space_id: i32,
    asset_registry: &AssetRegistry,
    render_config: &RenderConfig,
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

        let fallback_variant = compute_pipeline_variant_for_drawable(
            scene.is_overlay,
            is_skinned,
            &drawable,
            entry.mesh_handle,
            use_debug_uv,
            use_pbr,
            asset_registry,
        );
        let material_block_id = drawable.material_handle.unwrap_or(-1);
        let host_shader_asset_id = asset_registry
            .material_property_store
            .shader_asset_for_block(material_block_id);
        let shader_key = ShaderKey {
            host_shader_asset_id,
            fallback_variant,
        };
        let force_legacy = matches!(
            render_config.shader_debug_override,
            ShaderDebugOverride::ForceLegacyGlobalShading
        );
        let pipeline_variant = shader_key.effective_variant(
            render_config.use_host_unlit_pilot,
            force_legacy,
            material_block_id,
            false,
            is_skinned,
            scene.is_overlay,
        );
        let pipeline_variant = apply_native_ui_pipeline_variant(
            scene.is_overlay,
            is_skinned,
            drawable.stencil_state.as_ref(),
            render_config,
            host_shader_asset_id,
            material_block_id,
            drawable.mesh_handle,
            pipeline_variant,
            asset_registry,
        );
        out.push(FilteredDrawable {
            drawable,
            world_matrix,
            pipeline_variant,
            shader_key,
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
                shader_key: f.shader_key,
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
    use super::{
        AssetRegistry, FilteredDrawable, apply_native_ui_pipeline_variant, build_draw_entries,
        create_space_batch, mesh_has_ui_canvas_vertices,
    };
    use crate::config::{RenderConfig, ShaderDebugOverride};
    use crate::gpu::{PipelineVariant, ShaderKey};
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
            shader_key: ShaderKey::builtin_only(PipelineVariant::NormalDebug),
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
            shader_key: ShaderKey::builtin_only(PipelineVariant::NormalDebug),
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
            shader_key: ShaderKey::builtin_only(PipelineVariant::NormalDebug),
        }];
        let entries = build_draw_entries(filtered);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].shadow_cast_mode, ShadowCastMode::off);
    }

    #[test]
    fn mesh_has_ui_canvas_vertices_false_without_mesh() {
        let reg = AssetRegistry::new();
        assert!(!mesh_has_ui_canvas_vertices(&reg, 1));
    }

    #[test]
    fn apply_native_ui_overlay_respects_overlay_only() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            native_ui_unlit_shader_id: 42,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            false,
            false,
            None,
            &rc,
            Some(42),
            7,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn apply_native_ui_overlay_disabled_when_config_off() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: false,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            true,
            false,
            None,
            &rc,
            Some(99),
            7,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn apply_native_ui_overlay_skips_legacy_shader_override() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_native_ui_wgsl: true,
            native_ui_unlit_shader_id: 42,
            shader_debug_override: ShaderDebugOverride::ForceLegacyGlobalShading,
            ..Default::default()
        };
        let v = apply_native_ui_pipeline_variant(
            true,
            false,
            None,
            &rc,
            Some(42),
            7,
            1,
            PipelineVariant::NormalDebug,
            &reg,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }
}
