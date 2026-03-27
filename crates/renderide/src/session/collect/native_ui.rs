//! Native UI WGSL routing: mesh vertex checks and pipeline variant selection for UI shaders.

use crate::assets::{self, AssetRegistry, NativeUiShaderFamily, resolve_native_ui_shader_family};
use crate::config::{RenderConfig, ShaderDebugOverride};
use crate::gpu::PipelineVariant;
use crate::scene::Drawable;
use crate::shared::VertexAttributeType;
use crate::stencil::StencilState;

use super::super::native_ui_routing_metrics::{
    NativeUiRoutedFamily, NativeUiSkipKind, record_native_ui_routed, record_native_ui_skip,
    record_pbr_uivert_fallback,
};

/// When true with [`RenderConfig::log_native_ui_routing`], emit trace lines from this module.
fn should_log_native_ui_routing(rc: &RenderConfig, is_overlay: bool) -> bool {
    rc.log_native_ui_routing && (is_overlay || rc.native_ui_world_space)
}

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

/// Strangler Fig routing: after [`ShaderKey::effective_variant`](crate::gpu::ShaderKey::effective_variant), maps draws to native WGSL `UI_Unlit` / `UI_TextUnlit` when allowed.
///
/// # Routing contract (all must pass for native UI)
///
/// 1. [`RenderConfig::use_native_ui_wgsl`] is true.
/// 2. [`RenderConfig::shader_debug_override`] is not [`ShaderDebugOverride::ForceLegacyGlobalShading`].
/// 3. Drawable is not skinned (`is_skinned == false`).
/// 4. `material_block_id >= 0` (material property block selected).
/// 5. Surface is allowed: `is_overlay` **or** [`RenderConfig::native_ui_world_space`].
/// 6. World-space rule: if not overlay, stencil state must be absent (stencil + world is not routed here).
/// 7. If stencil is present, [`RenderConfig::native_ui_overlay_stencil_pipelines`] must be true.
/// 8. Host material store exposes `set_shader` for this block (`host_shader_asset_id`).
/// 9. [`mesh_has_native_ui_vertices`]: mesh declares UV0 (4+ bytes).
/// 10. [`resolve_native_ui_shader_family`] yields [`NativeUiShaderFamily::UiUnlit`] or [`NativeUiShaderFamily::UiTextUnlit`]
///     (INI shader ids, [`crate::assets::ShaderAsset::unity_shader_name`], or upload label / path hint).
///
/// Counters: [`crate::session::native_ui_routing_metrics`] when [`RenderConfig::native_ui_routing_metrics`] is on.
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
    let m = render_config.native_ui_routing_metrics;
    if !render_config.use_native_ui_wgsl {
        record_native_ui_skip(m, NativeUiSkipKind::NativeUiWgslOff);
        return current;
    }
    if matches!(
        render_config.shader_debug_override,
        ShaderDebugOverride::ForceLegacyGlobalShading
    ) {
        record_native_ui_skip(m, NativeUiSkipKind::ShaderDebugForceLegacy);
        return current;
    }
    if is_skinned {
        record_native_ui_skip(m, NativeUiSkipKind::Skinned);
        return current;
    }
    if material_block_id < 0 {
        record_native_ui_skip(m, NativeUiSkipKind::BadMaterialBlock);
        return current;
    }
    let allow_surface = is_overlay || render_config.native_ui_world_space;
    if !allow_surface {
        record_native_ui_skip(m, NativeUiSkipKind::NoSurface);
        return current;
    }
    if !is_overlay && stencil_state.is_some() {
        record_native_ui_skip(m, NativeUiSkipKind::StencilOnWorldMesh);
        return current;
    }
    let has_stencil = stencil_state.is_some();
    if has_stencil && !render_config.native_ui_overlay_stencil_pipelines {
        record_native_ui_skip(m, NativeUiSkipKind::StencilPipelinesOff);
        return current;
    }
    let Some(shader_id) = host_shader_asset_id else {
        record_native_ui_skip(m, NativeUiSkipKind::NoHostShader);
        if should_log_native_ui_routing(render_config, is_overlay) {
            logger::trace!(
                "native_ui: skip (no set_shader) material_block={} mesh={}",
                material_block_id,
                mesh_asset_id
            );
        }
        return current;
    };
    if !mesh_has_native_ui_vertices(asset_registry, mesh_asset_id) {
        record_native_ui_skip(m, NativeUiSkipKind::MeshNoUv0);
        if should_log_native_ui_routing(render_config, is_overlay) {
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
        record_native_ui_skip(m, NativeUiSkipKind::UnrecognizedShader);
        if should_log_native_ui_routing(render_config, is_overlay) {
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
                record_native_ui_routed(m, NativeUiRoutedFamily::UiUnlitStencil);
                PipelineVariant::NativeUiUnlitStencil {
                    material_id: material_block_id,
                }
            } else {
                record_native_ui_routed(m, NativeUiRoutedFamily::UiUnlit);
                PipelineVariant::NativeUiUnlit {
                    material_id: material_block_id,
                }
            }
        }
        NativeUiShaderFamily::UiTextUnlit => {
            if has_stencil {
                record_native_ui_routed(m, NativeUiRoutedFamily::UiTextUnlitStencil);
                PipelineVariant::NativeUiTextUnlitStencil {
                    material_id: material_block_id,
                }
            } else {
                record_native_ui_routed(m, NativeUiRoutedFamily::UiTextUnlit);
                PipelineVariant::NativeUiTextUnlit {
                    material_id: material_block_id,
                }
            }
        }
    }
}

/// Coexistence branch: when native UI WGSL is on, the mesh has UI-capable vertices (UV0), but routing did not
/// select a native UI variant, optionally replace with [`PipelineVariant::Pbr`].
///
/// Requires [`RenderConfig::native_ui_uivert_pbr_fallback`], global PBR, non-skinned, and no stencil. Otherwise
/// keeps `fallback_variant` (e.g. overlay debug unlit) so canvases are not forced through untextured PBR.
///
/// Stock PBR may still read host `_Color` / `_Metallic` / `_Glossiness` into the uniform ring when
/// [`RenderConfig::pbr_bind_host_material_properties`] is on (no arbitrary albedo texture bind yet).
/// See [`crate::gpu::pipeline::pbr_host_material_plan::GpuPbrHostMaterialPlan`].
pub(crate) fn apply_ui_mesh_pbr_fallback_for_non_native_shader(
    render_config: &RenderConfig,
    asset_registry: &AssetRegistry,
    drawable: &Drawable,
    pipeline_variant: PipelineVariant,
    use_pbr: bool,
    fallback_variant: PipelineVariant,
) -> PipelineVariant {
    if !render_config.use_native_ui_wgsl || !use_pbr || drawable.is_skinned {
        return pipeline_variant;
    }
    if drawable.stencil_state.is_some() {
        return pipeline_variant;
    }
    if !mesh_has_native_ui_vertices(asset_registry, drawable.mesh_handle) {
        return pipeline_variant;
    }
    if matches!(
        pipeline_variant,
        PipelineVariant::NativeUiUnlit { .. }
            | PipelineVariant::NativeUiTextUnlit { .. }
            | PipelineVariant::NativeUiUnlitStencil { .. }
            | PipelineVariant::NativeUiTextUnlitStencil { .. }
    ) {
        return pipeline_variant;
    }
    if render_config.native_ui_uivert_pbr_fallback {
        record_pbr_uivert_fallback(render_config.native_ui_routing_metrics);
        PipelineVariant::Pbr
    } else {
        fallback_variant
    }
}
