//! Canonical essential WGSL shader programs resolved from uploaded Unity shader names.

/// Coarse pipeline family chosen after mapping a host shader name to a Renderide WGSL shader.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ShaderPipelineFamily {
    /// No usable native family mapping exists.
    Unsupported,
    /// Forward/deferred lit PBR-style family.
    Pbr,
    /// World-space unlit family.
    WorldUnlit,
    /// UI unlit family.
    UiUnlit,
    /// UI text family.
    UiTextUnlit,
}

/// Small supported shader set for the renderer's native WGSL implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EssentialShaderProgram {
    /// No native WGSL equivalent is registered for this shader name.
    Unsupported,
    /// Resonite `PBSMetallic` family.
    PbsMetallic,
    /// Resonite world `Shader "Unlit"`.
    WorldUnlit,
    /// Resonite `UI/Unlit`.
    UiUnlit,
    /// Resonite `UI/Text/Unlit`.
    UiTextUnlit,
}

/// Generic shader binding result: host shader name mapped to a Renderide WGSL asset and pipeline family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResolvedRenderideShader {
    pub rel_path: &'static str,
    pub family: ShaderPipelineFamily,
}

/// Relative path under `RENDERIDESHADERS/` for the native WGSL file that should represent
/// a given uploaded Unity shader name.
pub fn resolve_renderide_shader_rel_path(name: Option<&str>) -> Option<&'static str> {
    resolve_renderide_shader_binding(name).map(|binding| binding.rel_path)
}

/// Generic mapping from host shader name / label to a Renderide WGSL asset plus pipeline family.
pub fn resolve_renderide_shader_binding(name: Option<&str>) -> Option<ResolvedRenderideShader> {
    let name = name?;
    let token = name.split_whitespace().next()?;
    let key = compact_alnum_lower(token);

    let (rel_path, family) = match key.as_str() {
        "unlit" => ("world/unlit.wgsl", ShaderPipelineFamily::WorldUnlit),
        "overlayunlit" => ("world/overlay_unlit.wgsl", ShaderPipelineFamily::WorldUnlit),
        "volumeunlit" => ("world/volume_unlit.wgsl", ShaderPipelineFamily::WorldUnlit),
        "billboardunlit" => ("world/billboard_unlit.wgsl", ShaderPipelineFamily::WorldUnlit),
        "uiunlit" => ("ui/ui_unlit.wgsl", ShaderPipelineFamily::UiUnlit),
        "uicirclesegment" => ("ui/ui_circle_segment.wgsl", ShaderPipelineFamily::UiUnlit),
        "uitextunlit" => ("ui/ui_text_unlit.wgsl", ShaderPipelineFamily::UiTextUnlit),
        "textunlit" => ("ui/text_unlit.wgsl", ShaderPipelineFamily::UiTextUnlit),
        _ if is_pbs_family(&key) || is_toon_lit_family(&key) => {
            ("pbr/pbs_metallic.wgsl", ShaderPipelineFamily::Pbr)
        }
        _ if is_world_unlit_family(&key) => ("world/unlit.wgsl", ShaderPipelineFamily::WorldUnlit),
        _ => return None,
    };

    Some(ResolvedRenderideShader { rel_path, family })
}

fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn is_pbs_family(key: &str) -> bool {
    key.starts_with("pbs") || key.starts_with("paintpbs")
}

fn is_toon_lit_family(key: &str) -> bool {
    key.starts_with("xstoon") || key.starts_with("toon")
}

fn is_world_unlit_family(key: &str) -> bool {
    key == "unlit"
        || key.ends_with("unlit")
        || matches!(
            key,
            "overlayunlit"
                | "fresnel"
                | "fresnellerp"
                | "overlayfresnel"
                | "matcap"
                | "projection360"
                | "cubemapprojection"
                | "reflection"
                | "proceduralskybox"
                | "uvrect"
                | "billboardunlit"
                | "volumeunlit"
                | "wireframeunlittransition"
        )
}

/// Resolves the essential WGSL program from a Unity shader name or plain upload label.
pub fn resolve_essential_shader_program(name: Option<&str>) -> EssentialShaderProgram {
    match resolve_renderide_shader_binding(name).map(|binding| binding.family) {
        Some(ShaderPipelineFamily::Pbr) => EssentialShaderProgram::PbsMetallic,
        Some(ShaderPipelineFamily::WorldUnlit) => EssentialShaderProgram::WorldUnlit,
        Some(ShaderPipelineFamily::UiUnlit) => EssentialShaderProgram::UiUnlit,
        Some(ShaderPipelineFamily::UiTextUnlit) => EssentialShaderProgram::UiTextUnlit,
        Some(ShaderPipelineFamily::Unsupported) | None => EssentialShaderProgram::Unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EssentialShaderProgram, ShaderPipelineFamily, resolve_essential_shader_program,
        resolve_renderide_shader_binding, resolve_renderide_shader_rel_path,
    };

    #[test]
    fn resolves_essential_programs() {
        assert_eq!(
            resolve_essential_shader_program(Some("PBSMetallic")),
            EssentialShaderProgram::PbsMetallic
        );
        assert_eq!(
            resolve_essential_shader_program(Some("Unlit")),
            EssentialShaderProgram::WorldUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("UI_Unlit")),
            EssentialShaderProgram::UiUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("UI_TextUnlit")),
            EssentialShaderProgram::UiTextUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("PBSLerpSpecular")),
            EssentialShaderProgram::PbsMetallic
        );
        assert_eq!(
            resolve_essential_shader_program(Some("OverlayUnlit")),
            EssentialShaderProgram::WorldUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("TextUnlit")),
            EssentialShaderProgram::UiTextUnlit
        );
        assert_eq!(
            resolve_essential_shader_program(Some("XSToon2.0")),
            EssentialShaderProgram::PbsMetallic
        );
    }

    #[test]
    fn resolves_explicit_renderide_variant_paths() {
        assert_eq!(
            resolve_renderide_shader_rel_path(Some("UI/CircleSegment")),
            Some("ui/ui_circle_segment.wgsl")
        );
        assert_eq!(
            resolve_renderide_shader_rel_path(Some("Text/Unlit")),
            Some("ui/text_unlit.wgsl")
        );
        assert_eq!(
            resolve_renderide_shader_rel_path(Some("OverlayUnlit")),
            Some("world/overlay_unlit.wgsl")
        );
        assert_eq!(
            resolve_renderide_shader_rel_path(Some("Volume/Unlit")),
            Some("world/volume_unlit.wgsl")
        );
        assert_eq!(
            resolve_renderide_shader_rel_path(Some("Billboard/Unlit")),
            Some("world/billboard_unlit.wgsl")
        );
    }

    #[test]
    fn binding_carries_family_and_path() {
        let binding = resolve_renderide_shader_binding(Some("UI/CircleSegment")).unwrap();
        assert_eq!(binding.rel_path, "ui/ui_circle_segment.wgsl");
        assert_eq!(binding.family, ShaderPipelineFamily::UiUnlit);
    }
}
