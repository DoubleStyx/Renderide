//! Audit that every variant-bit-migrated shader still decodes its `#pragma multi_compile`
//! keywords from `_RenderideVariantBits` rather than from legacy f32 keyword uniforms.

use super::*;

fn assert_variant_bits_shader(
    file_name: &str,
    forbidden_f32_fields: &[&str],
    keyword_constants: &[(&str, u32)],
) -> io::Result<()> {
    let src = material_source(file_name)?;
    assert!(
        src.contains("_RenderideVariantBits: u32"),
        "{file_name}: must declare _RenderideVariantBits: u32"
    );
    for field_name in forbidden_f32_fields {
        assert!(
            !declares_f32_field(&src, field_name),
            "{file_name}: {field_name} must be decoded from _RenderideVariantBits, not packed as f32"
        );
    }
    for (constant_name, bit_index) in keyword_constants {
        assert!(
            src.contains(&format!("const {constant_name}: u32 = 1u << {bit_index}u;")),
            "{file_name}: {constant_name} must match the Froox sorted UniqueKeywords bit order"
        );
    }
    Ok(())
}

#[test]
fn billboardunlit_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "billboardunlit.wgsl",
        &[
            "_ALPHATEST",
            "_COLOR",
            "_MUL_ALPHA_INTENSITY",
            "_MUL_RGB_BY_ALPHA",
            "_OFFSET_TEXTURE",
            "_POINT_ROTATION",
            "_POINT_SIZE",
            "_POINT_UV",
            "_POLARUV",
            "_RIGHT_EYE_ST",
            "_TEXTURE",
            "_VERTEX_HDRSRGB_COLOR",
            "_VERTEX_HDRSRGBALPHA_COLOR",
            "_VERTEX_LINEAR_COLOR",
            "_VERTEX_SRGB_COLOR",
            "_VERTEXCOLORS",
        ],
        &[
            ("BILLBOARDUNLIT_KW_ALPHATEST", 0),
            ("BILLBOARDUNLIT_KW_COLOR", 1),
            ("BILLBOARDUNLIT_KW_MUL_ALPHA_INTENSITY", 2),
            ("BILLBOARDUNLIT_KW_MUL_RGB_BY_ALPHA", 3),
            ("BILLBOARDUNLIT_KW_OFFSET_TEXTURE", 4),
            ("BILLBOARDUNLIT_KW_POINT_ROTATION", 5),
            ("BILLBOARDUNLIT_KW_POINT_SIZE", 6),
            ("BILLBOARDUNLIT_KW_POINT_UV", 7),
            ("BILLBOARDUNLIT_KW_POLARUV", 8),
            ("BILLBOARDUNLIT_KW_RIGHT_EYE_ST", 9),
            ("BILLBOARDUNLIT_KW_TEXTURE", 10),
            ("BILLBOARDUNLIT_KW_VERTEX_HDRSRGB_COLOR", 11),
            ("BILLBOARDUNLIT_KW_VERTEX_HDRSRGBALPHA_COLOR", 12),
            ("BILLBOARDUNLIT_KW_VERTEX_LINEAR_COLOR", 13),
            ("BILLBOARDUNLIT_KW_VERTEX_SRGB_COLOR", 14),
            ("BILLBOARDUNLIT_KW_VERTEXCOLORS", 15),
        ],
    )
}

#[test]
fn blur_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "blur.wgsl",
        &[
            "POISSON_DISC",
            "RECTCLIP",
            "REFRACT",
            "REFRACT_NORMALMAP",
            "SPREAD_TEX",
        ],
        &[
            ("BLUR_KW_POISSON_DISC", 0),
            ("BLUR_KW_RECTCLIP", 1),
            ("BLUR_KW_REFRACT", 2),
            ("BLUR_KW_REFRACT_NORMALMAP", 3),
            ("BLUR_KW_SPREAD_TEX", 4),
        ],
    )
}

#[test]
fn channelmatrix_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "channelmatrix.wgsl",
        &["RECTCLIP"],
        &[("CHANNELMATRIX_KW_RECTCLIP", 0)],
    )
}

#[test]
fn hsv_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader("hsv.wgsl", &["RECTCLIP"], &[("HSV_KW_RECTCLIP", 0)])
}

#[test]
fn invert_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader("invert.wgsl", &["RECTCLIP"], &[("INVERT_KW_RECTCLIP", 0)])
}

#[test]
fn lut_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "lut.wgsl",
        &["LERP", "RECTCLIP", "SRGB"],
        &[
            ("LUT_KW_LERP", 0),
            ("LUT_KW_RECTCLIP", 1),
            ("LUT_KW_SRGB", 2),
        ],
    )
}

#[test]
fn lut_perobject_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "lut_perobject.wgsl",
        &["LERP", "RECTCLIP"],
        &[
            ("LUT_PEROBJECT_KW_LERP", 0),
            ("LUT_PEROBJECT_KW_RECTCLIP", 1),
        ],
    )
}

#[test]
fn matcap_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "matcap.wgsl",
        &["_NORMALMAP"],
        &[("MATCAP_KW_NORMALMAP", 0)],
    )
}

#[test]
fn cubemapprojection_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "cubemapprojection.wgsl",
        &["FLIP"],
        &[("CUBEMAPPROJECTION_KW_FLIP", 0)],
    )
}

const PROJECTION360_KEYWORD_ORDER: [(&str, &str, u32); 18] = [
    ("_CLAMP_INTENSITY", "P360_KW_CLAMP_INTENSITY", 0),
    ("_NORMAL", "P360_KW_NORMAL", 1),
    ("_OFFSET", "P360_KW_OFFSET", 2),
    ("_PERSPECTIVE", "P360_KW_PERSPECTIVE", 3),
    ("_RIGHT_EYE_ST", "P360_KW_RIGHT_EYE_ST", 4),
    ("_VIEW", "P360_KW_VIEW", 5),
    ("_WORLD_VIEW", "P360_KW_WORLD_VIEW", 6),
    ("CUBEMAP", "P360_KW_CUBEMAP", 7),
    ("CUBEMAP_LOD", "P360_KW_CUBEMAP_LOD", 8),
    ("EQUIRECTANGULAR", "P360_KW_EQUIRECTANGULAR", 9),
    ("OUTSIDE_CLAMP", "P360_KW_OUTSIDE_CLAMP", 10),
    ("OUTSIDE_CLIP", "P360_KW_OUTSIDE_CLIP", 11),
    ("OUTSIDE_COLOR", "P360_KW_OUTSIDE_COLOR", 12),
    ("RECTCLIP", "P360_KW_RECTCLIP", 13),
    ("SECOND_TEXTURE", "P360_KW_SECOND_TEXTURE", 14),
    ("TINT_TEX_DIRECT", "P360_KW_TINT_TEX_DIRECT", 15),
    ("TINT_TEX_LERP", "P360_KW_TINT_TEX_LERP", 16),
    ("TINT_TEX_NONE", "P360_KW_TINT_TEX_NONE", 17),
];

fn projection360_sources() -> io::Result<[(&'static str, String); 2]> {
    Ok([
        (
            "materials/projection360.wgsl",
            material_source("projection360.wgsl")?,
        ),
        (
            "passes/backend/skybox_projection360.wgsl",
            source_file(manifest_dir().join("shaders/passes/backend/skybox_projection360.wgsl"))?,
        ),
    ])
}

fn decode_projection360_keywords(bits: u32) -> Vec<&'static str> {
    PROJECTION360_KEYWORD_ORDER
        .into_iter()
        .filter_map(|(keyword, _, bit_index)| {
            ((bits & (1u32 << bit_index)) != 0).then_some(keyword)
        })
        .collect()
}

#[test]
fn projection360_uses_froox_sorted_variant_bits() -> io::Result<()> {
    for (path_label, src) in projection360_sources()? {
        for (_, constant_name, bit_index) in PROJECTION360_KEYWORD_ORDER {
            assert!(
                src.contains(&format!("const {constant_name}: u32 = 1u << {bit_index}u;")),
                "{path_label}: {constant_name} must match Froox's sorted UniqueKeywords bit order",
            );
        }
    }
    assert_eq!(
        decode_projection360_keywords(0x0002_0A02),
        [
            "_NORMAL",
            "EQUIRECTANGULAR",
            "OUTSIDE_CLIP",
            "TINT_TEX_NONE"
        ],
    );
    assert_eq!(
        decode_projection360_keywords(0x0001_4A24),
        [
            "_OFFSET",
            "_VIEW",
            "EQUIRECTANGULAR",
            "OUTSIDE_CLIP",
            "SECOND_TEXTURE",
            "TINT_TEX_LERP",
        ],
    );
    Ok(())
}

#[test]
fn projection360_reconstructs_implicit_first_keyword_defaults() -> io::Result<()> {
    for (path_label, src) in projection360_sources()? {
        for group_const in [
            "P360_GROUP_VIEW",
            "P360_GROUP_OUTSIDE",
            "P360_GROUP_TINT_TEX",
            "P360_GROUP_TEXTURE_MODE",
        ] {
            assert!(
                src.contains(group_const),
                "{path_label}: {group_const} must be declared so zero bits preserve Unity's implicit first keyword",
            );
        }
        for (helper, group, bit) in [
            ("kw_VIEW", "P360_GROUP_VIEW", "P360_KW_VIEW"),
            (
                "kw_OUTSIDE_CLIP",
                "P360_GROUP_OUTSIDE",
                "P360_KW_OUTSIDE_CLIP",
            ),
            (
                "kw_TINT_TEX_NONE",
                "P360_GROUP_TINT_TEX",
                "P360_KW_TINT_TEX_NONE",
            ),
            (
                "kw_EQUIRECTANGULAR",
                "P360_GROUP_TEXTURE_MODE",
                "P360_KW_EQUIRECTANGULAR",
            ),
        ] {
            let expected = format!("proj360_group_default({group}, {bit})");
            assert!(
                src.contains(&format!("fn {helper}()")) && src.contains(&expected),
                "{path_label}: {helper}() must route through {expected}",
            );
        }
    }
    Ok(())
}

#[test]
fn projection360_clamp_intensity_guards_zero_max() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/skybox/projection360.wgsl"))?;
    assert!(
        src.contains("if (clamp_intensity && max_intensity > 0.0) {"),
        "Projection360 clamp_intensity must not zero colors when _MaxIntensity is missing or zero",
    );
    Ok(())
}

#[test]
fn depthprojection_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "depthprojection.wgsl",
        &["DEPTH_HUE", "DEPTH_GRAYSCALE"],
        &[
            ("DEPTHPROJECTION_KW_DEPTH_GRAYSCALE", 0),
            ("DEPTHPROJECTION_KW_DEPTH_HUE", 1),
        ],
    )
}

#[test]
fn fresnel_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "fresnel.wgsl",
        &[
            "_ALPHATEST",
            "_MASK_TEXTURE_CLIP",
            "_MASK_TEXTURE_MUL",
            "_MUL_ALPHA_INTENSITY",
            "_NORMALMAP",
            "_POLARUV",
            "_TEXTURE",
            "_VERTEX_HDRSRGB_COLOR",
            "_VERTEX_LINEAR_COLOR",
            "_VERTEX_SRGB_COLOR",
            "_VERTEXCOLORS",
        ],
        &[
            ("FRESNEL_KW_ALPHATEST", 0),
            ("FRESNEL_KW_MASK_TEXTURE_CLIP", 1),
            ("FRESNEL_KW_MASK_TEXTURE_MUL", 2),
            ("FRESNEL_KW_MUL_ALPHA_INTENSITY", 3),
            ("FRESNEL_KW_NORMALMAP", 4),
            ("FRESNEL_KW_POLARUV", 5),
            ("FRESNEL_KW_TEXTURE", 6),
            ("FRESNEL_KW_VERTEX_HDRSRGB_COLOR", 7),
            ("FRESNEL_KW_VERTEX_LINEAR_COLOR", 8),
            ("FRESNEL_KW_VERTEX_SRGB_COLOR", 9),
            ("FRESNEL_KW_VERTEXCOLORS", 10),
        ],
    )
}

#[test]
fn fresnel_applies_unity_alpha_output_after_alpha_intensity() -> io::Result<()> {
    let src = material_source("fresnel.wgsl")?;
    let alpha_intensity = src
        .find("color.a = ma::alpha_intensity_squared(color.a, color.rgb);")
        .expect("fresnel.wgsl must apply _MUL_ALPHA_INTENSITY before final alpha output");
    let alpha_output = src
        .find("color.a = pow(color.a, mat._GammaCurve);")
        .expect("fresnel.wgsl must apply Unity EVR_APPLY_ALPHA_OUTPUT parity");
    let ret = src
        .find("return rg::retain_globals_additive(color);")
        .expect("fresnel.wgsl must return the final color");

    assert!(
        alpha_intensity < alpha_output && alpha_output < ret,
        "fresnel.wgsl must apply final alpha gamma after alpha intensity and before return",
    );
    Ok(())
}

#[test]
fn fresnellerp_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "fresnellerp.wgsl",
        &[
            "_LERPTEX",
            "_LERPTEX_POLARUV",
            "_MULTI_VALUES",
            "_NORMALMAP",
            "_TEXTURE",
        ],
        &[
            ("FRESNELLERP_KW_LERPTEX", 0),
            ("FRESNELLERP_KW_LERPTEX_POLARUV", 1),
            ("FRESNELLERP_KW_MULTI_VALUES", 2),
            ("FRESNELLERP_KW_NORMALMAP", 3),
            ("FRESNELLERP_KW_TEXTURE", 4),
        ],
    )
}

#[test]
fn fresnellerp_preserves_unity_lerp_extrapolation() -> io::Result<()> {
    let src = material_source("fresnellerp.wgsl")?;
    assert!(
        src.contains("fn compute_lerp(uv: vec2<f32>) -> f32")
            && src.contains("return l;")
            && !src.contains("return clamp(l"),
        "fresnellerp.wgsl must not clamp _Lerp; Unity lerp extrapolates outside [0, 1]",
    );
    Ok(())
}

#[test]
fn gamma_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader("gamma.wgsl", &["RECTCLIP"], &[("GAMMA_KW_RECTCLIP", 0)])
}

#[test]
fn gamma_preserves_unity_pow_operands() -> io::Result<()> {
    let src = material_source("gamma.wgsl")?;
    assert!(
        src.contains("pow(c.rgb, vec3<f32>(mat._Gamma))")
            && !src.contains("max(mat._Gamma")
            && !src.contains("max(c.rgb"),
        "gamma.wgsl must match Unity's direct pow(c.rgb, _Gamma)",
    );
    Ok(())
}

#[test]
fn getdepth_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "getdepth.wgsl",
        &["CLIP", "RECTCLIP"],
        &[("GETDEPTH_KW_CLIP", 0), ("GETDEPTH_KW_RECTCLIP", 1)],
    )
}

#[test]
fn getdepth_preserves_unity_clip_division() -> io::Result<()> {
    let src = material_source("getdepth.wgsl")?;
    assert!(
        src.contains("depth = depth - mat._ClipMin;")
            && src.contains("depth = depth / (mat._ClipMax - mat._ClipMin);")
            && !src.contains("max(mat._ClipMax - mat._ClipMin"),
        "getdepth.wgsl must match Unity's unguarded clip-range division",
    );
    Ok(())
}

#[test]
fn grayscale_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "grayscale.wgsl",
        &["GRADIENT", "RECTCLIP"],
        &[("GRAYSCALE_KW_GRADIENT", 0), ("GRAYSCALE_KW_RECTCLIP", 1)],
    )
}

#[test]
fn hsv_uses_unity_branch_color_conversion() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/post/filter_math.wgsl"))?;
    for required in [
        "var min_channel: f32;",
        "var max_channel: f32;",
        "if (delta != 0.0)",
        "let del_rgb = (hsv.zzz - rgb + vec3<f32>(3.0 * delta)) / (6.0 * delta);",
        "let var_i = floor(var_h);",
        "rgb = vec3<f32>(var_3, var_1, hsv.z);",
    ] {
        assert!(
            src.contains(required),
            "filter_math.wgsl must contain Unity HSV conversion fragment `{required}`",
        );
    }
    Ok(())
}

#[test]
fn pixelate_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "pixelate.wgsl",
        &["RECTCLIP", "RESOLUTION_TEX", "_RectClip"],
        &[
            ("PIXELATE_KW_RECTCLIP", 0),
            ("PIXELATE_KW_RESOLUTION_TEX", 1),
        ],
    )
}

#[test]
fn pixelate_resolution_tex_sample_is_keyword_gated() -> io::Result<()> {
    let src = material_source("pixelate.wgsl")?;
    let kw_position = src.find("kw_RESOLUTION_TEX()").expect(
        "pixelate.wgsl must reference kw_RESOLUTION_TEX() to gate the _ResolutionTex sample",
    );
    let sample_position = src
        .find("textureSample(_ResolutionTex,")
        .expect("pixelate.wgsl must sample _ResolutionTex");
    assert!(
        kw_position < sample_position,
        "pixelate.wgsl must guard the _ResolutionTex sample on kw_RESOLUTION_TEX()"
    );
    Ok(())
}

#[test]
fn material_shaders_21_30_preserve_source_parity_gaps() -> io::Result<()> {
    let invisible = material_source("invisible.wgsl")?;
    assert!(
        !invisible.contains("struct InvisibleMaterial")
            && !invisible.contains("@group(1)")
            && invisible.contains("rg::retain_globals_additive(vec4<f32>(0.0))"),
        "Invisible must not declare synthetic material properties absent from Unity"
    );

    let lut = material_source("lut.wgsl")?;
    assert!(
        !lut.contains("_LUT_LodBias")
            && !lut.contains("_SecondaryLUT_LodBias")
            && !lut.contains("clamp(normalized")
            && lut.contains("ts::sample_tex_3d_level(_LUT, _LUT_sampler, coords, 0.0)")
            && lut.contains(
                "ts::sample_tex_3d_level(_SecondaryLUT, _SecondaryLUT_sampler, coords, 0.0)"
            ),
        "LUT must mirror Unity tex3Dlod level-0 sampling without synthetic LOD-bias uniforms or coordinate clamping"
    );

    let lut_perobject = material_source("lut_perobject.wgsl")?;
    assert!(
        !lut_perobject.contains("_LUT_LodBias")
            && !lut_perobject.contains("_SecondaryLUT_LodBias")
            && !lut_perobject.contains("clamp(c.rgb")
            && lut_perobject.contains("ts::sample_tex_3d(_LUT, _LUT_sampler, coords, 0.0)")
            && lut_perobject
                .contains("ts::sample_tex_3d(_SecondaryLUT, _SecondaryLUT_sampler, coords, 0.0)"),
        "LUT_PerObject must mirror Unity tex3D sampling without synthetic LOD-bias uniforms or coordinate clamping"
    );

    let nosamplers = material_source("nosamplers.wgsl")?;
    assert!(
        nosamplers.contains("var _Albedo1: texture_2d<f32>")
            && nosamplers.contains("var _Albedo2: texture_2d<f32>")
            && nosamplers.contains("var _Albedo3: texture_2d<f32>")
            && !nosamplers.contains("textureSample(_Albedo1")
            && !nosamplers.contains("textureSample(_Albedo2")
            && !nosamplers.contains("textureSample(_Albedo3"),
        "Nosamplers must preserve declared texture property names without adding non-source texture reads"
    );

    assert!(
        material_source("newunlitshader.wgsl")?
            .contains("//#pass forward offset_factor=0 offset_units=1"),
        "NewUnlitShader must preserve Unity Offset 0, 1"
    );
    assert!(
        material_source("null.wgsl")?.contains("//#pass forward offset_factor=2 offset_units=2"),
        "Null must preserve Unity Offset 2, 2"
    );
    Ok(())
}

#[test]
fn unlitdistancelerp_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "unlitdistancelerp.wgsl",
        &[
            "_ALPHATEST",
            "_VERTEXCOLORS",
            "_LOCAL_SPACE",
            "_WORLD_SPACE",
        ],
        &[
            ("UNLITDISTANCELERP_KW_ALPHATEST", 0),
            ("UNLITDISTANCELERP_KW_VERTEXCOLORS", 1),
            ("UNLITDISTANCELERP_KW_LOCAL_SPACE", 2),
            ("UNLITDISTANCELERP_KW_WORLD_SPACE", 3),
        ],
    )
}

#[test]
fn uvrect_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader("uvrect.wgsl", &[], &[("UVRECT_KW_RECTCLIP", 0)])
}

fn is_meaningful_wrapper_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.starts_with("//")
}

fn assert_source_alias_wrapper(file_name: &str, parent_stem: &str) -> io::Result<()> {
    let src = material_source(file_name)?;
    let directive = format!("//#source_alias {parent_stem}");
    assert!(
        src.lines().any(|line| line.trim() == directive),
        "{file_name}: must contain `{directive}` directive"
    );
    let meaningful_lines: Vec<&str> = src
        .lines()
        .filter(|l| is_meaningful_wrapper_line(l))
        .collect();
    assert!(
        meaningful_lines.is_empty(),
        "{file_name}: source-alias wrapper must contain no code, found: {meaningful_lines:?}"
    );
    Ok(())
}

#[test]
fn blur_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("blur_perobject.wgsl", "blur")
}

#[test]
fn channelmatrix_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("channelmatrix_perobject.wgsl", "channelmatrix")
}

#[test]
fn hsv_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("hsv_perobject.wgsl", "hsv")
}

#[test]
fn invert_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("invert_perobject.wgsl", "invert")
}

#[test]
fn gamma_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("gamma_perobject.wgsl", "gamma")
}

#[test]
fn getdepth_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("getdepth_perobject.wgsl", "getdepth")
}

#[test]
fn grayscale_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("grayscale_perobject.wgsl", "grayscale")
}

#[test]
fn threshold_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "threshold.wgsl",
        &["RECTCLIP"],
        &[("THRESHOLD_KW_RECTCLIP", 0)],
    )
}

#[test]
fn threshold_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("threshold_perobject.wgsl", "threshold")
}

#[test]
fn textunlit_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "textunlit.wgsl",
        &["_TextMode"],
        &[
            ("TEXTUNLIT_KW_MSDF", 0),
            ("TEXTUNLIT_KW_OUTLINE", 1),
            ("TEXTUNLIT_KW_RASTER", 2),
            ("TEXTUNLIT_KW_SDF", 3),
        ],
    )
}

#[test]
fn textunit_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("textunit.wgsl", "textunlit")
}

#[test]
fn ui_textunlit_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "ui_textunlit.wgsl",
        &["_TextMode", "_RectClip", "_OVERLAY"],
        &[
            ("UITEXTUNLIT_KW_MSDF", 0),
            ("UITEXTUNLIT_KW_OUTLINE", 1),
            ("UITEXTUNLIT_KW_OVERLAY", 2),
            ("UITEXTUNLIT_KW_RASTER", 3),
            ("UITEXTUNLIT_KW_RECTCLIP", 4),
            ("UITEXTUNLIT_KW_SDF", 5),
        ],
    )
}

#[test]
fn ui_circlesegment_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "ui_circlesegment.wgsl",
        &["_RectClip", "_OVERLAY"],
        &[
            ("UICIRCLESEGMENT_KW_OVERLAY", 0),
            ("UICIRCLESEGMENT_KW_RECTCLIP", 1),
        ],
    )
}

#[test]
fn ui_unlit_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "ui_unlit.wgsl",
        &[
            "_ALPHACLIP",
            "_ALPHATEST_ON",
            "_ALPHABLEND_ON",
            "_TEXTURE_NORMALMAP",
            "_TEXTURE_LERPCOLOR",
            "_MASK_TEXTURE_MUL",
            "_MASK_TEXTURE_CLIP",
            "_RectClip",
            "_OVERLAY",
        ],
        &[
            ("UIUNLIT_KW_MASK_TEXTURE_CLIP", 0),
            ("UIUNLIT_KW_MASK_TEXTURE_MUL", 1),
            ("UIUNLIT_KW_ALPHACLIP", 2),
            ("UIUNLIT_KW_OVERLAY", 3),
            ("UIUNLIT_KW_RECTCLIP", 4),
            ("UIUNLIT_KW_TEXTURE_LERPCOLOR", 5),
            ("UIUNLIT_KW_TEXTURE_NORMALMAP", 6),
        ],
    )
}
