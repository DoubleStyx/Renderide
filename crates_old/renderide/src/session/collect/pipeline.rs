//! Pipeline and material resolution: filtered drawable type, shader keys, stencil, and variant selection.

use glam::Mat4;

use crate::assets::{
    self, AssetRegistry, MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
    resolve_native_shader_route, texture2d_asset_id_from_packed,
};
use crate::config::{RenderConfig, ShaderDebugOverride};
use crate::gpu::{PipelineVariant, ShaderKey};
use crate::scene::{Drawable, MeshMaterialSlot, Scene};
use crate::shared::VertexAttributeType;
use crate::stencil::{StencilOperation, StencilState};

use super::native_ui::{
    apply_native_ui_pipeline_variant, apply_ui_mesh_pbr_fallback_for_non_native_shader,
};

/// Filtered drawable with world matrix and pipeline variant.
///
/// Output of [`super::filter_and_collect_drawables`]; input to [`super::build_draw_entries`].
pub(in crate::session) struct FilteredDrawable {
    pub(in crate::session) drawable: Drawable,
    pub(in crate::session) world_matrix: Mat4,
    pub(in crate::session) pipeline_variant: PipelineVariant,
    pub(in crate::session) shader_key: ShaderKey,
    /// When set, mesh recording draws only this `(index_start, index_count)` slice.
    pub(in crate::session) submesh_index_range: Option<(u32, u32)>,
}

/// Material slots from [`Drawable::material_slots`], or a single synthetic slot from legacy fields.
pub(in crate::session) fn resolved_material_slots(drawable: &Drawable) -> Vec<MeshMaterialSlot> {
    if !drawable.material_slots.is_empty() {
        return drawable.material_slots.clone();
    }
    match drawable.material_handle {
        Some(material_asset_id) => vec![MeshMaterialSlot {
            material_asset_id,
            property_block_id: drawable.mesh_renderer_property_block_slot0_id,
        }],
        None => Vec::new(),
    }
}

/// When forward PBR would apply and the host bound `_MainTex` with UV0 on the mesh, use [`PipelineVariant::PbrHostAlbedo`].
pub(in crate::session::collect) fn maybe_upgrade_pbr_host_albedo(
    pipeline_variant: PipelineVariant,
    render_config: &RenderConfig,
    store: &MaterialPropertyStore,
    drawable: &Drawable,
    material_asset_id: i32,
    mesh_asset_id: i32,
    asset_registry: &AssetRegistry,
) -> PipelineVariant {
    if pipeline_variant != PipelineVariant::Pbr {
        return pipeline_variant;
    }
    if !render_config.pbr_bind_host_main_texture || render_config.pbr_host_main_tex_property_id < 0
    {
        return pipeline_variant;
    }
    let mesh_has_uv = asset_registry
        .get_mesh(mesh_asset_id)
        .and_then(|m| {
            assets::attribute_offset_size_format(&m.vertex_attributes, VertexAttributeType::uv0)
        })
        .map(|(_, s, _)| s >= 4)
        .unwrap_or(false);
    if !mesh_has_uv {
        return pipeline_variant;
    }
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: drawable.mesh_renderer_property_block_slot0_id,
    };
    let has_tex = store
        .get_merged(lookup, render_config.pbr_host_main_tex_property_id)
        .and_then(|v| match v {
            MaterialPropertyValue::Texture(packed) => texture2d_asset_id_from_packed(*packed),
            _ => None,
        })
        .is_some();
    if !has_tex {
        return pipeline_variant;
    }
    PipelineVariant::PbrHostAlbedo
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_pipeline_for_material_draw_internal(
    scene: &Scene,
    render_config: &RenderConfig,
    drawable: &Drawable,
    use_pbr: bool,
    is_skinned: bool,
    asset_registry: &AssetRegistry,
    material_block_id: i32,
    fallback_variant: PipelineVariant,
) -> (PipelineVariant, ShaderKey) {
    let host_shader_asset_id = asset_registry
        .material_property_store
        .shader_asset_for_material(material_block_id);
    let shader_key = ShaderKey {
        host_shader_asset_id,
        fallback_variant,
    };
    let force_legacy = matches!(
        render_config.shader_debug_override,
        ShaderDebugOverride::ForceLegacyGlobalShading
    );
    let native_shader_route = resolve_native_shader_route(host_shader_asset_id, asset_registry);
    let pipeline_variant = shader_key.effective_variant(
        force_legacy,
        material_block_id,
        false,
        is_skinned,
        scene.is_overlay,
        native_shader_route,
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
    let pipeline_variant = apply_ui_mesh_pbr_fallback_for_non_native_shader(
        render_config,
        asset_registry,
        drawable,
        pipeline_variant,
        use_pbr,
        fallback_variant,
    );
    let pipeline_variant = maybe_upgrade_pbr_host_albedo(
        pipeline_variant,
        render_config,
        &asset_registry.material_property_store,
        drawable,
        material_block_id,
        drawable.mesh_handle,
        asset_registry,
    );
    (pipeline_variant, shader_key)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::session::collect) fn resolve_pipeline_for_material_draw(
    scene: &Scene,
    render_config: &RenderConfig,
    drawable: &Drawable,
    use_pbr: bool,
    is_skinned: bool,
    asset_registry: &AssetRegistry,
    material_block_id: i32,
    fallback_variant: PipelineVariant,
) -> (PipelineVariant, ShaderKey) {
    resolve_pipeline_for_material_draw_internal(
        scene,
        render_config,
        drawable,
        use_pbr,
        is_skinned,
        asset_registry,
        material_block_id,
        fallback_variant,
    )
}

/// Resolves overlay stencil state from material property store when scene is overlay.
pub(in crate::session) fn resolve_overlay_stencil_state(
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
pub(in crate::session) fn compute_pipeline_variant_for_drawable(
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
