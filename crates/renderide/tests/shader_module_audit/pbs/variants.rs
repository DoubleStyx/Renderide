//! PBS shader variant and packed-channel audits.

use super::super::*;
use super::helpers::assert_keyword_bits;

#[test]
fn selected_pbs_materials_keep_sorted_shader_variant_bits() -> io::Result<()> {
    assert_keyword_bits(
        "pbsstencilspecular.wgsl",
        &[
            ("PBSSTENCILSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSSTENCILSPECULAR_KW_EMISSIONTEX", 1),
            ("PBSSTENCILSPECULAR_KW_NORMALMAP", 2),
            ("PBSSTENCILSPECULAR_KW_OCCLUSION", 3),
            ("PBSSTENCILSPECULAR_KW_SPECULARMAP", 4),
        ],
    )?;
    assert_keyword_bits(
        "pbstriplanar.wgsl",
        &[
            ("PBSTRIPLANAR_KW_ALBEDOTEX", 0),
            ("PBSTRIPLANAR_KW_EMISSIONTEX", 1),
            ("PBSTRIPLANAR_KW_METALLICMAP", 2),
            ("PBSTRIPLANAR_KW_NORMALMAP", 3),
            ("PBSTRIPLANAR_KW_OBJECTSPACE", 4),
            ("PBSTRIPLANAR_KW_OCCLUSION", 5),
            ("PBSTRIPLANAR_KW_WORLDSPACE", 6),
        ],
    )?;
    assert_keyword_bits(
        "pbstriplanarspecular.wgsl",
        &[
            ("PBSTRIPLANARSPEC_KW_ALBEDOTEX", 0),
            ("PBSTRIPLANARSPEC_KW_EMISSIONTEX", 1),
            ("PBSTRIPLANARSPEC_KW_NORMALMAP", 2),
            ("PBSTRIPLANARSPEC_KW_OBJECTSPACE", 3),
            ("PBSTRIPLANARSPEC_KW_OCCLUSION", 4),
            ("PBSTRIPLANARSPEC_KW_SPECULARMAP", 5),
            ("PBSTRIPLANARSPEC_KW_WORLDSPACE", 6),
        ],
    )?;
    assert_keyword_bits(
        "pbsslice.wgsl",
        &[
            ("PBSSLICE_KW_ALBEDOTEX", 0),
            ("PBSSLICE_KW_ALPHACLIP", 1),
            ("PBSSLICE_KW_DETAIL_ALBEDOTEX", 2),
            ("PBSSLICE_KW_DETAIL_NORMALMAP", 3),
            ("PBSSLICE_KW_EMISSIONTEX", 4),
            ("PBSSLICE_KW_METALLICMAP", 5),
            ("PBSSLICE_KW_NORMALMAP", 6),
            ("PBSSLICE_KW_OCCLUSION", 7),
            ("PBSSLICE_KW_OBJECT_SPACE", 8),
            ("PBSSLICE_KW_WORLD_SPACE", 9),
        ],
    )?;
    assert_keyword_bits(
        "pbsslicespecular.wgsl",
        &[
            ("PBSSLICESPECULAR_KW_ALBEDOTEX", 0),
            ("PBSSLICESPECULAR_KW_ALPHACLIP", 1),
            ("PBSSLICESPECULAR_KW_DETAIL_ALBEDOTEX", 2),
            ("PBSSLICESPECULAR_KW_DETAIL_NORMALMAP", 3),
            ("PBSSLICESPECULAR_KW_EMISSIONTEX", 4),
            ("PBSSLICESPECULAR_KW_METALLICMAP", 5),
            ("PBSSLICESPECULAR_KW_NORMALMAP", 6),
            ("PBSSLICESPECULAR_KW_OCCLUSION", 7),
            ("PBSSLICESPECULAR_KW_OBJECT_SPACE", 8),
            ("PBSSLICESPECULAR_KW_WORLD_SPACE", 9),
        ],
    )?;
    assert_keyword_bits(
        "pbsvertexcolortransparent.wgsl",
        &[
            ("PBSVCT_KW_ALBEDOTEX", 0),
            ("PBSVCT_KW_ALPHACLIP", 1),
            ("PBSVCT_KW_EMISSIONTEX", 2),
            ("PBSVCT_KW_METALLICMAP", 3),
            ("PBSVCT_KW_NORMALMAP", 4),
            ("PBSVCT_KW_OCCLUSION", 5),
            ("PBSVCT_KW_VCOLOR_ALBEDO", 6),
            ("PBSVCT_KW_VCOLOR_EMIT", 7),
            ("PBSVCT_KW_VCOLOR_METALLIC", 8),
        ],
    )?;
    assert_keyword_bits(
        "pbsvertexcolortransparentspecular.wgsl",
        &[
            ("PBSVCTS_KW_ALBEDOTEX", 0),
            ("PBSVCTS_KW_ALPHACLIP", 1),
            ("PBSVCTS_KW_EMISSIONTEX", 2),
            ("PBSVCTS_KW_NORMALMAP", 3),
            ("PBSVCTS_KW_OCCLUSION", 4),
            ("PBSVCTS_KW_SPECULARMAP", 5),
            ("PBSVCTS_KW_VCOLOR_ALBEDO", 6),
            ("PBSVCTS_KW_VCOLOR_EMIT", 7),
            ("PBSVCTS_KW_VCOLOR_SPECULAR", 8),
        ],
    )?;
    assert_keyword_bits(
        "pixelate.wgsl",
        &[
            ("PIXELATE_KW_RECTCLIP", 0),
            ("PIXELATE_KW_RESOLUTION_TEX", 1),
        ],
    )?;
    Ok(())
}

#[test]
fn pbs_lerp_preserves_variant_channels_and_raw_lerp() -> io::Result<()> {
    let metallic = material_source("pbslerp.wgsl")?;
    for required in [
        "return l;",
        "occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;",
        "occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;",
        "metallic0 = m0.r;",
        "metallic1 = m1.r;",
        "smoothness0 = m0.a;",
        "smoothness1 = m1.a;",
    ] {
        assert!(
            metallic.contains(required),
            "pbslerp.wgsl must contain `{required}`"
        );
    }
    assert!(
        !metallic.contains("return clamp(l, 0.0, 1.0);"),
        "pbslerp.wgsl must use Unity's raw lerp factor"
    );

    let specular = material_source("pbslerpspecular.wgsl")?;
    for required in [
        "return l;",
        "occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;",
        "occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;",
        "spec0 = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main0);",
        "spec1 = textureSample(_SpecularMap1, _SpecularMap1_sampler, uv_main1);",
    ] {
        assert!(
            specular.contains(required),
            "pbslerpspecular.wgsl must contain `{required}`"
        );
    }
    assert!(
        !specular.contains("return clamp(l, 0.0, 1.0);"),
        "pbslerpspecular.wgsl must use Unity's raw lerp factor"
    );

    Ok(())
}

#[test]
fn standard_pbs_optional_gloss_maps_sample_inside_keyword_branches() -> io::Result<()> {
    let metallic = material_source("pbsmetallic.wgsl")?;
    assert!(
        metallic.contains(
            "if (metallic_gloss_map_enabled()) {\n        let mg = ts::sample_tex_2d(_MetallicGlossMap"
        ),
        "pbsmetallic.wgsl must sample _MetallicGlossMap inside the keyword branch"
    );
    assert!(
        !metallic.contains(
            "let mg = ts::sample_tex_2d(_MetallicGlossMap, _MetallicGlossMap_sampler, uv_main, mat._MetallicGlossMap_LodBias);\n    var metallic"
        ),
        "pbsmetallic.wgsl must not pre-sample _MetallicGlossMap before testing the keyword"
    );

    let specular = material_source("pbsspecular.wgsl")?;
    assert!(
        specular.contains(
            "if (spec_gloss_map_enabled()) {\n        let spec_gloss = ts::sample_tex_2d(_SpecGlossMap"
        ),
        "pbsspecular.wgsl must sample _SpecGlossMap inside the keyword branch"
    );
    assert!(
        !specular.contains(
            "let spec_gloss = ts::sample_tex_2d(_SpecGlossMap, _SpecGlossMap_sampler, uv_main, mat._SpecGlossMap_LodBias);\n    var specular_color"
        ),
        "pbsspecular.wgsl must not pre-sample _SpecGlossMap before testing the keyword"
    );

    Ok(())
}

/// Verifies PBSMultiUVSpecular propagates specular RGB and smoothness alpha.
#[test]
fn pbs_multiuv_specular_uses_specular_channels() -> io::Result<()> {
    let src = material_source("pbsmultiuvspecular.wgsl")?;
    for required in [
        "let f0 = spec.rgb;",
        "let smoothness = clamp(spec.a, 0.0, 1.0);",
    ] {
        assert!(
            src.contains(required),
            "pbsmultiuvspecular.wgsl must contain `{required}`"
        );
    }
    for rejected in [
        "spec.rgb - spec.rgb",
        "spec.a - spec.a",
        "one_minus_reflectivity",
    ] {
        assert!(
            !src.contains(rejected),
            "pbsmultiuvspecular.wgsl must not contain `{rejected}`"
        );
    }
    Ok(())
}
