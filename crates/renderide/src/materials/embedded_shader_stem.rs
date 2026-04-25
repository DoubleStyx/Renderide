//! Unity logical shader names → composed WGSL stems under `shaders/target/` (embedded at build time).
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and probes
//! `{normalized_key}_default`. ShaderLab **path** forms under `UI/…` are mapped to the same asset-style
//! keys as material sources under `shaders/source/materials/*.wgsl` (see crate `build.rs`).

use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::embedded_shaders;

/// Maps `UI/…` ShaderLab path strings to the compact `ui_*` keys used by on-disk material stems.
///
/// Example: `UI/Text/Unlit` → `ui_textunlit` (matches `ui_textunlit.wgsl`). This covers the
/// case where a shader upload ships inline ShaderLab source with a nested `UI/…/…` path; the
/// primary filename-based routing handles the AssetBundle case via
/// [`normalize_unity_shader_lookup_key`] directly.
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
    fn resolves_pbsrimtransparentzwrite_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSRimTransparentZWrite").as_deref(),
            Some("pbsrimtransparentzwrite_default")
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
    fn resolves_pbsintersect_from_plain_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSIntersect").as_deref(),
            Some("pbsintersect_default")
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
    fn resolves_custom_pbsintersect_from_shader_lab_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("Custom/PBSIntersect").as_deref(),
            Some("custom_pbsintersect_default")
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
    fn resolves_billboard_unlit_from_filename() {
        assert_eq!(
            embedded_default_stem_for_unity_name("BillboardUnlit").as_deref(),
            Some("billboardunlit_default")
        );
    }

    #[test]
    fn resolves_unlit_distance_lerp_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UnlitDistanceLerp").as_deref(),
            Some("unlitdistancelerp_default")
        );
    }

    #[test]
    fn resolves_xiexe_toon2_cutout_from_filename() {
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToon2.0 Cutout").as_deref(),
            Some("xstoon2.0-cutout_default")
        );
    }

    #[test]
    fn resolves_xiexe_toon2_cutout_a2c_outlined_from_filename() {
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToon2.0 CutoutA2C Outlined").as_deref(),
            Some("xstoon2.0-cutouta2c-outlined_default")
        );
    }

    #[test]
    fn resolves_xiexe_outlined_from_underscore_filename() {
        // The underscore-spelled `XSToon2.0_Outlined.shader` is a distinct Unity asset from
        // the space-spelled `XSToon2.0 Outlined.shader` — the normalizer preserves the
        // underscore/dash distinction so they resolve to different stems.
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToon2.0_Outlined").as_deref(),
            Some("xstoon2.0_outlined_default")
        );
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToon2.0 Outlined").as_deref(),
            Some("xstoon2.0-outlined_default")
        );
    }

    #[test]
    fn resolves_xiexe_stenciler_from_filename() {
        assert_eq!(
            embedded_default_stem_for_unity_name("XSToonStenciler").as_deref(),
            Some("xstoonstenciler_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_transparent_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSDualSidedTransparent").as_deref(),
            Some("pbsdualsidedtransparent_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_transparent_specular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSDualSidedTransparentSpecular").as_deref(),
            Some("pbsdualsidedtransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbs_color_mask_from_asset_filename() {
        // Unity's `Shader "ColorMask"` ships in `PBSColorMask.shader`; the host can route the
        // material by either label, so both forms must resolve to the same embedded stem.
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSColorMask").as_deref(),
            Some("pbscolormask_default")
        );
    }

    #[test]
    fn resolves_pbs_color_mask_specular_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSColorMaskSpecular").as_deref(),
            Some("pbscolormaskspecular_default")
        );
    }

    #[test]
    fn resolves_pbs_triplanar_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSTriplanar").as_deref(),
            Some("pbstriplanar_default")
        );
    }

    #[test]
    fn resolves_pbs_triplanar_specular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSTriplanarSpecular").as_deref(),
            Some("pbstriplanarspecular_default")
        );
    }

    #[test]
    fn resolves_pbs_multi_uv_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSMultiUV").as_deref(),
            Some("pbsmultiuv_default")
        );
    }

    #[test]
    fn resolves_pbs_multi_uv_specular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSMultiUVSpecular").as_deref(),
            Some("pbsmultiuvspecular_default")
        );
    }

    #[test]
    fn resolves_pbsrimspecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSRimSpecular").as_deref(),
            Some("pbsrimspecular_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparentspecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSRimTransparentSpecular").as_deref(),
            Some("pbsrimtransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparentzwritespecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSRimTransparentZWriteSpecular").as_deref(),
            Some("pbsrimtransparentzwritespecular_default")
        );
    }

    #[test]
    fn resolves_pbsslicespecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSSliceSpecular").as_deref(),
            Some("pbsslicespecular_default")
        );
    }

    #[test]
    fn resolves_pbsslicetransparent_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSSliceTransparent").as_deref(),
            Some("pbsslicetransparent_default")
        );
    }

    #[test]
    fn resolves_pbsslicetransparentspecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSSliceTransparentSpecular").as_deref(),
            Some("pbsslicetransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbstriplanartransparent_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSTriplanarTransparent").as_deref(),
            Some("pbstriplanartransparent_default")
        );
    }

    #[test]
    fn resolves_pbstriplanartransparentspecular_from_unity_name() {
        assert_eq!(
            embedded_default_stem_for_unity_name("PBSTriplanarTransparentSpecular").as_deref(),
            Some("pbstriplanartransparentspecular_default")
        );
    }
}
