//! Unity logical shader names → composed WGSL stems under `shaders/target/` (embedded at build time).
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and probes
//! `{normalized_key}_default`. ShaderLab **path** forms under `UI/…` are mapped to the same asset-style
//! keys as material sources under `shaders/source/materials/*.wgsl` (see crate `build.rs`).

use crate::assets::util::{compact_alnum_lower, normalize_unity_shader_lookup_key};
use crate::embedded_shaders;

/// Maps `UI/…` ShaderLab path strings to the compact `ui_*` keys used by on-disk material stems.
///
/// Example: `UI/Text/Unlit` → `ui_textunlit` (matches `ui_textunlit.wgsl`).
fn shader_lab_ui_path_to_asset_lookup_key(name: &str) -> Option<String> {
    let token = name.split_whitespace().next().unwrap_or(name).trim();
    if !token.contains('/') {
        return None;
    }
    let parts: Vec<&str> = token.split('/').collect();
    if parts.is_empty() {
        return None;
    }
    if parts[0].eq_ignore_ascii_case("ui") && parts.len() >= 2 {
        let rest: String = parts[1..]
            .iter()
            .flat_map(|s| s.chars())
            .flat_map(|c| c.to_lowercase())
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        return Some(format!("ui_{rest}"));
    }
    None
}

/// Maps Xiexe asset/container stems and source filenames to their ShaderLab path-backed material keys.
///
/// Some Unity AssetBundle paths expose stems such as `XSToon2.0 CutoutA2C Outlined` instead of the
/// quoted ShaderLab path `Xiexe/Toon2.0/XSToon2.0_CutoutA2C_Outlined`.
fn xiexe_toon2_asset_lookup_key(name: &str) -> Option<&'static str> {
    let raw = name
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_ascii_lowercase();
    match raw.as_str() {
        "xstoon2.0" => return Some("xiexe_xstoon2.0"),
        "xstoon2.0_outlined" => return Some("xiexe_xstoon2.0_outlined"),
        _ => {}
    }

    match compact_alnum_lower(name).as_str() {
        "xstoon20cutout" => Some("xiexe_toon2.0_xstoon2.0_cutout"),
        "xstoon20cutouta2c" => Some("xiexe_toon2.0_xstoon2.0_cutouta2c"),
        "xstoon20cutouta2coutlined" => Some("xiexe_toon2.0_xstoon2.0_cutouta2c_outlined"),
        "xstoon20cutouta2cmasked" => Some("xiexe_toon2.0_xstoon2.0_cutouta2c_masked"),
        "xstoon20dithered" => Some("xiexe_toon2.0_xstoon2.0_dithered"),
        "xstoon20ditheredoutlined" => Some("xiexe_toon2.0_xstoon2.0_dithered_outlined"),
        "xstoon20fade" => Some("xiexe_toon2.0_xstoon2.0_fade"),
        "xstoon20outlined" => Some("xiexe_toon2.0_xstoon2.0_outlined"),
        "xstoon20transparent" => Some("xiexe_toon2.0_xstoon2.0_transparent"),
        "xstoonstenciler" => Some("xiexe_toon2.0_xstoonstenciler"),
        _ => None,
    }
}

/// Returns `{normalized_key}_default` when that composed target exists in the embedded table.
pub fn embedded_default_stem_for_unity_name(name: &str) -> Option<String> {
    let key = normalize_unity_shader_lookup_key(name);
    let stem = format!("{key}_default");
    if embedded_shaders::embedded_target_wgsl(&stem).is_some() {
        return Some(stem);
    }
    if let Some(asset_key) = shader_lab_ui_path_to_asset_lookup_key(name) {
        let stem2 = format!("{asset_key}_default");
        if embedded_shaders::embedded_target_wgsl(&stem2).is_some() {
            return Some(stem2);
        }
    }
    if let Some(asset_key) = xiexe_toon2_asset_lookup_key(name) {
        let stem2 = format!("{asset_key}_default");
        if embedded_shaders::embedded_target_wgsl(&stem2).is_some() {
            return Some(stem2);
        }
    }
    None
}

/// Returns the composed WGSL stem for `name` when an embedded `{key}_default` target exists (routing).
pub fn embedded_stem_for_unity_name(name: &str) -> Option<String> {
    embedded_default_stem_for_unity_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_asset_style_stem_directly() {
        assert_eq!(
            embedded_default_stem_for_unity_name("ui_textunlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn resolves_pbs_metallic_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSMetallic").as_deref(),
            Some("pbsmetallic_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSDualSided").as_deref(),
            Some("pbsdualsided_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_specular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSDualSidedSpecular").as_deref(),
            Some("pbsdualsidedspecular_default")
        );
    }

    #[test]
    fn resolves_ui_textunlit_from_unity_asset_token() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UI_TextUnlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn shader_lab_ui_path_form_maps_to_embedded_stem() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UI/Text/Unlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn shader_lab_ui_unlit_path_maps() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UI/Unlit").as_deref(),
            Some("ui_unlit_default")
        );
    }

    #[test]
    fn shader_lab_ui_circle_segment_path_maps() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UI/CircleSegment").as_deref(),
            Some("ui_circlesegment_default")
        );
    }

    #[test]
    fn resolves_overlay_unlit_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("OverlayUnlit").as_deref(),
            Some("overlayunlit_default")
        );
    }

    #[test]
    fn resolves_overlay_fresnel_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("OverlayFresnel").as_deref(),
            Some("overlayfresnel_default")
        );
    }

    #[test]
    fn resolves_projection360_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Projection360").as_deref(),
            Some("projection360_default")
        );
    }

    #[test]
    fn resolves_fresnel_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Fresnel").as_deref(),
            Some("fresnel_default")
        );
    }

    #[test]
    fn resolves_fresnel_lerp_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("FresnelLerp").as_deref(),
            Some("fresnellerp_default")
        );
    }

    #[test]
    fn resolves_textunit_from_asset_style_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("TextUnit").as_deref(),
            Some("textunit_default")
        );
    }

    #[test]
    fn resolves_text_unlit_from_shader_lab_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Text/Unlit").as_deref(),
            Some("text_unlit_default")
        );
    }

    #[test]
    fn resolves_textunlit_from_plain_label() {
        assert_eq!(
            embedded_default_stem_for_unity_name("TextUnlit").as_deref(),
            Some("textunlit_default")
        );
    }

    #[test]
    fn resolves_uvrect_from_asset_style_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UVRect").as_deref(),
            Some("uvrect_default")
        );
    }

    #[test]
    fn resolves_uvrect_from_shader_lab_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Unlit/UVRect").as_deref(),
            Some("unlit_uvrect_default")
        );
    }

    #[test]
    fn resolves_pbsrim_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSRim").as_deref(),
            Some("pbsrim_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparent_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSRimTransparent").as_deref(),
            Some("pbsrimtransparent_default")
        );
    }

    #[test]
    fn resolves_pbslerp_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSLerp").as_deref(),
            Some("pbslerp_default")
        );
    }

    #[test]
    fn resolves_pbslerpspecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSLerpSpecular").as_deref(),
            Some("pbslerpspecular_default")
        );
    }

    #[test]
    fn resolves_pbsintersectspecular_from_plain_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSIntersectSpecular").as_deref(),
            Some("pbsintersectspecular_default")
        );
    }

    #[test]
    fn resolves_custom_pbsintersectspecular_from_shader_lab_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Custom/PBSIntersectSpecular").as_deref(),
            Some("custom_pbsintersectspecular_default")
        );
    }

    #[test]
    fn resolves_matcap_from_plain_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Matcap").as_deref(),
            Some("matcap_default")
        );
    }

    #[test]
    fn resolves_xiexe_toon2_cutout_from_shader_lab_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Xiexe/Toon2.0/XSToon2.0_Cutout").as_deref(),
            Some("xiexe_toon2.0_xstoon2.0_cutout_default")
        );
    }

    #[test]
    fn resolves_xiexe_toon2_cutout_a2c_outlined_from_asset_stem() {
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToon2.0 CutoutA2C Outlined").as_deref(),
            Some("xiexe_toon2.0_xstoon2.0_cutouta2c_outlined_default")
        );
    }

    #[test]
    fn resolves_xiexe_legacy_outlined_from_underscore_asset_stem() {
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToon2.0_Outlined").as_deref(),
            Some("xiexe_xstoon2.0_outlined_default")
        );
    }

    #[test]
    fn resolves_xiexe_stenciler_from_asset_stem() {
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToonStenciler").as_deref(),
            Some("xiexe_toon2.0_xstoonstenciler_default")
        );
    }
}
