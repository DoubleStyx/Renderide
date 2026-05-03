//! Unity shader asset names mapped to composed WGSL stems under `shaders/target/` (embedded at build time).
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and probes
//! `{normalized_key}_default`, matching material source stems under `shaders/materials/*.wgsl`
//! (see crate `build.rs`).

use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::embedded_shaders;

/// Returns `{normalized_key}_default` when that composed target exists in the embedded table.
pub fn embedded_default_stem_for_shader_asset_name(name: &str) -> Option<String> {
    let key = normalize_unity_shader_lookup_key(name);
    let stem = format!("{key}_default");
    if embedded_shaders::embedded_target_wgsl(&stem).is_some() {
        return Some(stem);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_asset_style_stem_directly() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("ui_textunlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn resolves_pbs_metallic_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSMetallic").as_deref(),
            Some("pbsmetallic_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDualSided").as_deref(),
            Some("pbsdualsided_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_specular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDualSidedSpecular").as_deref(),
            Some("pbsdualsidedspecular_default")
        );
    }

    #[test]
    fn resolves_ui_textunlit_from_unity_asset_token() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("UI_TextUnlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn resolves_overlay_unlit_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("OverlayUnlit").as_deref(),
            Some("overlayunlit_default")
        );
    }

    #[test]
    fn resolves_overlay_fresnel_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("OverlayFresnel").as_deref(),
            Some("overlayfresnel_default")
        );
    }

    #[test]
    fn resolves_projection360_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Projection360").as_deref(),
            Some("projection360_default")
        );
    }

    #[test]
    fn resolves_gradient_skybox_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("GradientSkybox").as_deref(),
            Some("gradientskybox_default")
        );
    }

    #[test]
    fn resolves_procedural_skybox_from_unity_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("ProceduralSkybox").as_deref(),
            Some("proceduralskybox_default")
        );
    }

    #[test]
    fn resolves_fresnel_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Fresnel").as_deref(),
            Some("fresnel_default")
        );
    }

    #[test]
    fn resolves_fresnel_lerp_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("FresnelLerp").as_deref(),
            Some("fresnellerp_default")
        );
    }

    #[test]
    fn resolves_textunit_from_asset_style_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("TextUnit").as_deref(),
            Some("textunit_default")
        );
    }

    #[test]
    fn resolves_textunlit_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("TextUnlit").as_deref(),
            Some("textunlit_default")
        );
    }

    #[test]
    fn resolves_uvrect_from_asset_style_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("UVRect").as_deref(),
            Some("uvrect_default")
        );
    }

    #[test]
    fn resolves_pbsrim_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSRim").as_deref(),
            Some("pbsrim_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSRimTransparent").as_deref(),
            Some("pbsrimtransparent_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparentzwrite_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSRimTransparentZWrite").as_deref(),
            Some("pbsrimtransparentzwrite_default")
        );
    }

    #[test]
    fn resolves_pbslerp_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSLerp").as_deref(),
            Some("pbslerp_default")
        );
    }

    #[test]
    fn resolves_pbslerpspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSLerpSpecular").as_deref(),
            Some("pbslerpspecular_default")
        );
    }

    #[test]
    fn resolves_pbsintersect_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSIntersect").as_deref(),
            Some("pbsintersect_default")
        );
    }

    #[test]
    fn resolves_pbsintersectspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSIntersectSpecular").as_deref(),
            Some("pbsintersectspecular_default")
        );
    }

    #[test]
    fn resolves_matcap_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Matcap").as_deref(),
            Some("matcap_default")
        );
    }

    #[test]
    fn resolves_filter_assets_from_asset_filenames() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Blur").as_deref(),
            Some("blur_default")
        );
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Blur_PerObject").as_deref(),
            Some("blur_perobject_default")
        );
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("HSV").as_deref(),
            Some("hsv_default")
        );
    }

    #[test]
    fn resolves_billboard_unlit_from_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("BillboardUnlit").as_deref(),
            Some("billboardunlit_default")
        );
    }

    #[test]
    fn resolves_unlit_distance_lerp_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("UnlitDistanceLerp").as_deref(),
            Some("unlitdistancelerp_default")
        );
    }

    #[test]
    fn resolves_xiexe_toon2_cutout_from_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("XSToon2.0 Cutout").as_deref(),
            Some("xstoon2.0-cutout_default")
        );
    }

    #[test]
    fn resolves_xiexe_toon2_cutout_a2c_outlined_from_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("XSToon2.0 CutoutA2C Outlined").as_deref(),
            Some("xstoon2.0-cutouta2c-outlined_default")
        );
    }

    #[test]
    fn resolves_xiexe_outlined_from_underscore_filename() {
        // The underscore-spelled `XSToon2.0_Outlined.shader` is a distinct Unity asset from
        // the space-spelled `XSToon2.0 Outlined.shader` -- the normalizer preserves the
        // underscore/dash distinction so they resolve to different stems.
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("XSToon2.0_Outlined").as_deref(),
            Some("xstoon2.0_outlined_default")
        );
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("XSToon2.0 Outlined").as_deref(),
            Some("xstoon2.0-outlined_default")
        );
    }

    #[test]
    fn resolves_xiexe_stenciler_from_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("XSToonStenciler").as_deref(),
            Some("xstoonstenciler_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_transparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDualSidedTransparent").as_deref(),
            Some("pbsdualsidedtransparent_default")
        );
    }

    #[test]
    fn resolves_pbs_dual_sided_transparent_specular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDualSidedTransparentSpecular")
                .as_deref(),
            Some("pbsdualsidedtransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbs_color_mask_from_asset_filename() {
        // The internal shader label differs from the asset filename; routing only uses the asset filename.
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSColorMask").as_deref(),
            Some("pbscolormask_default")
        );
    }

    #[test]
    fn resolves_pbs_color_mask_specular_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSColorMaskSpecular").as_deref(),
            Some("pbscolormaskspecular_default")
        );
    }

    #[test]
    fn resolves_pbs_triplanar_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSTriplanar").as_deref(),
            Some("pbstriplanar_default")
        );
    }

    #[test]
    fn resolves_pbs_triplanar_specular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSTriplanarSpecular").as_deref(),
            Some("pbstriplanarspecular_default")
        );
    }

    #[test]
    fn resolves_pbs_multi_uv_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSMultiUV").as_deref(),
            Some("pbsmultiuv_default")
        );
    }

    #[test]
    fn resolves_pbs_multi_uv_specular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSMultiUVSpecular").as_deref(),
            Some("pbsmultiuvspecular_default")
        );
    }

    #[test]
    fn resolves_pbsrimspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSRimSpecular").as_deref(),
            Some("pbsrimspecular_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparentspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSRimTransparentSpecular").as_deref(),
            Some("pbsrimtransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbsrimtransparentzwritespecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSRimTransparentZWriteSpecular")
                .as_deref(),
            Some("pbsrimtransparentzwritespecular_default")
        );
    }

    #[test]
    fn resolves_pbsslicespecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSSliceSpecular").as_deref(),
            Some("pbsslicespecular_default")
        );
    }

    #[test]
    fn resolves_pbsslicetransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSSliceTransparent").as_deref(),
            Some("pbsslicetransparent_default")
        );
    }

    #[test]
    fn resolves_pbsslicetransparentspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSSliceTransparentSpecular").as_deref(),
            Some("pbsslicetransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbstriplanartransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSTriplanarTransparent").as_deref(),
            Some("pbstriplanartransparent_default")
        );
    }

    #[test]
    fn resolves_pbstriplanartransparentspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSTriplanarTransparentSpecular")
                .as_deref(),
            Some("pbstriplanartransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbsvertexcolortransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSVertexColorTransparent").as_deref(),
            Some("pbsvertexcolortransparent_default")
        );
    }

    #[test]
    fn resolves_pbsvertexcolortransparentspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSVertexColorTransparentSpecular")
                .as_deref(),
            Some("pbsvertexcolortransparentspecular_default")
        );
    }

    #[test]
    fn resolves_pbscolorsplat_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSColorSplat").as_deref(),
            Some("pbscolorsplat_default")
        );
    }

    #[test]
    fn resolves_pbscolorsplatspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSColorSplatSpecular").as_deref(),
            Some("pbscolorsplatspecular_default")
        );
    }

    #[test]
    fn resolves_pbsdistancelerp_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDistanceLerp").as_deref(),
            Some("pbsdistancelerp_default")
        );
    }

    #[test]
    fn resolves_pbsdistancelerpspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDistanceLerpSpecular").as_deref(),
            Some("pbsdistancelerpspecular_default")
        );
    }

    #[test]
    fn resolves_pbsdistancelerptransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDistanceLerpTransparent").as_deref(),
            Some("pbsdistancelerptransparent_default")
        );
    }

    #[test]
    fn resolves_pbsdistancelerpspeculartransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDistanceLerpSpecularTransparent")
                .as_deref(),
            Some("pbsdistancelerpspeculartransparent_default")
        );
    }

    #[test]
    fn resolves_circle_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Circle").as_deref(),
            Some("circle_default")
        );
    }

    #[test]
    fn resolves_polargrid_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PolarGrid").as_deref(),
            Some("polargrid_default")
        );
    }

    #[test]
    fn resolves_invisible_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Invisible").as_deref(),
            Some("invisible_default")
        );
    }

    #[test]
    fn resolves_null_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Null").as_deref(),
            Some("null_default")
        );
    }

    #[test]
    fn resolves_testshader_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("TestShader").as_deref(),
            Some("testshader_default")
        );
    }

    #[test]
    fn resolves_newunlitshader_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("NewUnlitShader").as_deref(),
            Some("newunlitshader_default")
        );
    }

    #[test]
    fn resolves_overlay_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Overlay").as_deref(),
            Some("overlay_default")
        );
    }

    #[test]
    fn resolves_texturedebug_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("TextureDebug").as_deref(),
            Some("texturedebug_default")
        );
    }

    #[test]
    fn resolves_unlitpolarmapping_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("UnlitPolarMapping").as_deref(),
            Some("unlitpolarmapping_default")
        );
    }

    #[test]
    fn resolves_faceexplodeshader_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("FaceExplodeShader").as_deref(),
            Some("faceexplodeshader_default")
        );
    }

    #[test]
    fn resolves_testblend_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("TestBlend").as_deref(),
            Some("testblend_default")
        );
    }

    #[test]
    fn resolves_paintpbs_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PaintPBS").as_deref(),
            Some("paintpbs_default")
        );
    }

    #[test]
    fn resolves_pbsvoronoicrystal_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSVoronoiCrystal").as_deref(),
            Some("pbsvoronoicrystal_default")
        );
    }

    #[test]
    fn resolves_reflection_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Reflection").as_deref(),
            Some("reflection_default")
        );
    }

    #[test]
    fn resolves_nosamplers_from_asset_filename() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("Nosamplers").as_deref(),
            Some("nosamplers_default")
        );
    }

    #[test]
    fn resolves_pbsstencil_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSStencil").as_deref(),
            Some("pbsstencil_default")
        );
    }

    #[test]
    fn resolves_pbsstencilspecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSStencilSpecular").as_deref(),
            Some("pbsstencilspecular_default")
        );
    }

    #[test]
    fn resolves_cadshader_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("CADShader").as_deref(),
            Some("cadshader_default")
        );
    }

    #[test]
    fn resolves_pbsdisplace_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDisplace").as_deref(),
            Some("pbsdisplace_default")
        );
    }

    #[test]
    fn resolves_pbsdisplacespecular_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDisplaceSpecular").as_deref(),
            Some("pbsdisplacespecular_default")
        );
    }

    #[test]
    fn resolves_pbsdisplacetransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDisplaceTransparent").as_deref(),
            Some("pbsdisplacetransparent_default")
        );
    }

    #[test]
    fn resolves_pbsdisplacespeculartransparent_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDisplaceSpecularTransparent")
                .as_deref(),
            Some("pbsdisplacespeculartransparent_default")
        );
    }

    #[test]
    fn resolves_pbsdisplaceshadow_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("PBSDisplaceShadow").as_deref(),
            Some("pbsdisplaceshadow_default")
        );
    }

    #[test]
    fn resolves_toonstandard_from_asset_name() {
        assert_eq!(
            embedded_default_stem_for_shader_asset_name("ToonStandard").as_deref(),
            Some("toonstandard_default")
        );
    }
}
