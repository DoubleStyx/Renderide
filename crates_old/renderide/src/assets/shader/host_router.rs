//! Canonical routing from host shader identity to the small native Renderide shader set.

use crate::assets::util::compact_alnum_lower;

use super::{AssetRegistry, ShaderAsset, ShaderPipelineFamily};

/// Native Renderide shader route selected from the host-requested shader.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NativeShaderRoute {
    /// Host shader does not map to a known native route.
    Unsupported,
    /// Physically based metallic shading (`PBSMetallic` and related).
    PbsMetallic,
    /// World unlit / overlay / billboard / volume unlit families.
    WorldUnlit,
    /// UI canvas unlit (`UI/Unlit` and similar).
    UiUnlit,
    /// UI text unlit (`UI/Text/Unlit` and similar).
    UiTextUnlit,
}

/// Returns true when the first whitespace-delimited token of `name` matches the PBS metallic family.
pub fn pbs_metallic_family_from_unity_shader_name(name: &str) -> bool {
    let Some(token) = name.split_whitespace().next() else {
        return false;
    };
    compact_alnum_lower(token) == compact_alnum_lower("PBSMetallic")
}

/// Heuristic PBS metallic detection from a shader path or upload label string.
pub fn pbs_metallic_family_from_shader_path_hint(hint: &str) -> bool {
    let h = hint.to_ascii_lowercase();
    h.contains("pbsmetallic")
        || h.contains("pbs_specular")
        || h.contains("pbsspecular")
        || h.contains("pbs/specular")
}

/// Derives a route from [`ShaderAsset::pipeline_family`] and name/path fallbacks.
fn route_from_shader_asset(shader: &ShaderAsset) -> NativeShaderRoute {
    match shader.pipeline_family {
        ShaderPipelineFamily::Pbr => return NativeShaderRoute::PbsMetallic,
        ShaderPipelineFamily::WorldUnlit => return NativeShaderRoute::WorldUnlit,
        ShaderPipelineFamily::UiUnlit => return NativeShaderRoute::UiUnlit,
        ShaderPipelineFamily::UiTextUnlit => return NativeShaderRoute::UiTextUnlit,
        ShaderPipelineFamily::Unsupported => {}
    }

    if let Some(name) = shader.unity_shader_name.as_deref() {
        if let Some(f) = super::native_ui_family_from_unity_shader_name(name) {
            return match f {
                super::NativeUiShaderFamily::UiUnlit => NativeShaderRoute::UiUnlit,
                super::NativeUiShaderFamily::UiTextUnlit => NativeShaderRoute::UiTextUnlit,
            };
        }
        if super::world_unlit_family_from_unity_shader_name(name).is_some() {
            return NativeShaderRoute::WorldUnlit;
        }
        if pbs_metallic_family_from_unity_shader_name(name) {
            return NativeShaderRoute::PbsMetallic;
        }
    }

    if let Some(label) = shader.wgsl_source.as_deref() {
        if let Some(f) = super::native_ui_family_from_unity_shader_name(label) {
            return match f {
                super::NativeUiShaderFamily::UiUnlit => NativeShaderRoute::UiUnlit,
                super::NativeUiShaderFamily::UiTextUnlit => NativeShaderRoute::UiTextUnlit,
            };
        }
        if super::world_unlit_family_from_unity_shader_name(label).is_some() {
            return NativeShaderRoute::WorldUnlit;
        }
        if pbs_metallic_family_from_shader_path_hint(label) {
            return NativeShaderRoute::PbsMetallic;
        }
    }

    NativeShaderRoute::Unsupported
}

/// True when the shader asset at `shader_asset_id` resolves to [`NativeShaderRoute::PbsMetallic`].
pub fn resolve_pbs_metallic_shader_family(shader_asset_id: i32, registry: &AssetRegistry) -> bool {
    matches!(
        resolve_native_shader_route(Some(shader_asset_id), registry),
        NativeShaderRoute::PbsMetallic
    )
}

/// Maps a host material's bound shader asset (if any) to a [`NativeShaderRoute`].
pub fn resolve_native_shader_route(
    host_shader_asset_id: Option<i32>,
    registry: &AssetRegistry,
) -> NativeShaderRoute {
    let Some(shader_asset_id) = host_shader_asset_id else {
        return NativeShaderRoute::Unsupported;
    };
    let Some(shader) = registry.get_shader(shader_asset_id) else {
        return NativeShaderRoute::Unsupported;
    };
    route_from_shader_asset(shader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::AssetRegistry;
    use crate::shared::ShaderUpload;

    #[test]
    fn pbs_metallic_from_unity_name() {
        assert!(pbs_metallic_family_from_unity_shader_name("PBSMetallic"));
        assert!(!pbs_metallic_family_from_unity_shader_name("Unlit"));
    }

    #[test]
    fn resolves_route_from_registry_program() {
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(ShaderUpload {
            asset_id: 9,
            file: Some("PBSMetallic".to_string()),
        });
        assert_eq!(
            resolve_native_shader_route(Some(9), &reg),
            NativeShaderRoute::PbsMetallic
        );
    }

    #[test]
    fn resolves_route_for_world_unlit() {
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(ShaderUpload {
            asset_id: 2,
            file: Some("Unlit".to_string()),
        });
        assert_eq!(
            resolve_native_shader_route(Some(2), &reg),
            NativeShaderRoute::WorldUnlit
        );
    }
}
