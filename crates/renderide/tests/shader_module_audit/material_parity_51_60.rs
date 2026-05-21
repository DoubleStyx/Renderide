//! Source audits for the 51-60 alphabetical material parity slice.

use std::io;

use super::{material_source, module_source};

/// Asserts that a shader source contains every required snippet.
fn assert_contains_all(src: &str, label: &str, snippets: &[&str]) {
    for snippet in snippets {
        assert!(src.contains(snippet), "{label} must contain `{snippet}`");
    }
}

/// Verifies that PaintPBS declares the transparent alpha-fade pass state authored by Unity.
#[test]
fn paintpbs_uses_source_authored_alpha_fade_pass() -> io::Result<()> {
    let src = material_source("paintpbs.wgsl")?;
    assert!(
        src.contains(
            "//#pass type=forward name=forward_alpha_fade blend=alpha zwrite=off ztest=main cull=back color_mask=rgba offset=0,0"
        ),
        "PaintPBS must preserve Unity alpha:fade render-state parity"
    );
    Ok(())
}

/// Verifies that vertex color sRGB helpers preserve the authored LDR and HDR conversion profiles.
#[test]
fn vertex_color_srgb_helpers_match_common_cginc_profiles() -> io::Result<()> {
    let src = module_source("material/vertex_color.wgsl")?;
    assert_contains_all(
        &src,
        "vertex_color.wgsl",
        &[
            "fn srgb_ldr_channel_to_linear(value: f32) -> f32",
            "if (value < 1.0 && value > -1.0) {",
            "return value;",
            "fn srgb_hdr_channel_to_linear(value: f32) -> f32",
            "if (value >= 1.0) {",
            "return pow(value, 0.6666667);",
        ],
    );
    Ok(())
}

/// Verifies that regular material texture samples in this slice receive host LOD bias values.
#[test]
fn material_slice_51_60_threads_lod_bias_for_regular_samples() -> io::Result<()> {
    for (file, snippets) in [
        (
            "lut_perobject.wgsl",
            &[
                "_LUT_LodBias: f32",
                "_SecondaryLUT_LodBias: f32",
                "ts::sample_tex_3d(_LUT, _LUT_sampler, coords, mat._LUT_LodBias)",
                "mat._SecondaryLUT_LodBias",
            ][..],
        ),
        (
            "matcap.wgsl",
            &[
                "_MainTex_LodBias: f32",
                "_NormalMap_LodBias: f32",
                "ts::sample_tex_2d(_NormalMap, _NormalMap_sampler, uv_normal, mat._NormalMap_LodBias)",
                "ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv, mat._MainTex_LodBias)",
            ],
        ),
        (
            "nosamplers.wgsl",
            &[
                "_Albedo_LodBias: f32",
                "_MetallicMap_LodBias: f32",
                "_EmissionMap_LodBias: f32",
                "_EmissionMap1_LodBias: f32",
                "ts::sample_tex_2d(_MetallicMap, _Albedo_sampler, uv, mat._MetallicMap_LodBias)",
            ],
        ),
        (
            "overlay.wgsl",
            &[
                "_MainTexture_LodBias: f32",
                "ts::sample_tex_2d(_MainTexture, _MainTexture_sampler, uv, mat._MainTexture_LodBias)",
            ],
        ),
        (
            "overlayfresnel.wgsl",
            &[
                "_BehindFarTex_LodBias: f32",
                "_BehindNearTex_LodBias: f32",
                "_FrontFarTex_LodBias: f32",
                "_FrontNearTex_LodBias: f32",
                "_NormalMap_LodBias: f32",
                "textureSampleGrad(tex, samp, mapped.uv, mapped.ddx_uv, mapped.ddy_uv)",
                "ts::sample_tex_2d(tex, samp, uvu::apply_st(uv, st), lod_bias)",
            ],
        ),
        (
            "overlayunlit.wgsl",
            &[
                "_BehindTex_LodBias: f32",
                "_FrontTex_LodBias: f32",
                "textureSampleGrad(tex, samp, mapped.uv, mapped.ddx_uv, mapped.ddy_uv)",
                "ts::sample_tex_2d(tex, samp, uvu::apply_st(uv, st), lod_bias)",
            ],
        ),
        (
            "paintpbs.wgsl",
            &[
                "_MainTex_LodBias: f32",
                "_PaintTex_LodBias: f32",
                "ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias)",
                "mat._PaintTex_LodBias",
            ],
        ),
        (
            "pbscolormask.wgsl",
            &[
                "_MainTex_LodBias: f32",
                "_ColorMask_LodBias: f32",
                "_NormalMap_LodBias: f32",
                "_EmissionMap_LodBias: f32",
                "_OcclusionMap_LodBias: f32",
                "_MetallicMap_LodBias: f32",
                "ts::sample_tex_2d(_ColorMask, _ColorMask_sampler, uv_mask, mat._ColorMask_LodBias)",
            ],
        ),
    ] {
        let src = material_source(file)?;
        assert_contains_all(&src, file, snippets);
    }
    Ok(())
}
