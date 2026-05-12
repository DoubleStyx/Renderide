//! Per-shader audits locking in the `_RenderideVariantBits` migration for the PBS material family.

use super::*;

/// Asserts that `material` declares `_RenderideVariantBits: u32`, omits the listed legacy
/// f32 keyword fields, defines each bit constant at the expected index, and contains the
/// `vb::enabled(mat._RenderideVariantBits` decode call.
fn assert_variant_bits_migration(
    file_name: &str,
    legacy_kw_fields: &[&str],
    bits: &[(&str, u32)],
) -> io::Result<()> {
    let src = material_source(file_name)?;
    assert!(
        src.contains("_RenderideVariantBits: u32"),
        "{file_name} must declare _RenderideVariantBits: u32"
    );
    assert!(
        src.contains("#import renderide::material::variant_bits as vb"),
        "{file_name} must import renderide::material::variant_bits"
    );
    assert!(
        src.contains("vb::enabled(mat._RenderideVariantBits"),
        "{file_name} must decode keywords through vb::enabled(mat._RenderideVariantBits, ...)"
    );
    for kw in legacy_kw_fields {
        assert!(
            !declares_f32_field(&src, kw),
            "{file_name} must not declare legacy f32 keyword field {kw}; \
             decode it from _RenderideVariantBits instead"
        );
    }
    for (constant_name, bit_index) in bits {
        let needle = format!("const {constant_name}: u32 = 1u << {bit_index}u;");
        assert!(
            src.contains(&needle),
            "{file_name} must define `{needle}` (Froox sorted UniqueKeywords bit order)"
        );
    }
    Ok(())
}

#[test]
fn pbsdualsided_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsdualsided.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
            "VCOLOR_ALBEDO",
            "VCOLOR_EMIT",
            "VCOLOR_METALLIC",
        ],
        &[
            ("PBSDUALSIDED_KW_ALBEDOTEX", 0),
            ("PBSDUALSIDED_KW_ALPHACLIP", 1),
            ("PBSDUALSIDED_KW_EMISSIONTEX", 2),
            ("PBSDUALSIDED_KW_METALLICMAP", 3),
            ("PBSDUALSIDED_KW_NORMALMAP", 4),
            ("PBSDUALSIDED_KW_OCCLUSION", 5),
            ("PBSDUALSIDED_KW_VCOLOR_ALBEDO", 6),
            ("PBSDUALSIDED_KW_VCOLOR_EMIT", 7),
            ("PBSDUALSIDED_KW_VCOLOR_METALLIC", 8),
        ],
    )
}

#[test]
fn pbsintersectspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsintersectspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSINTERSECTSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSINTERSECTSPECULAR_KW_EMISSIONTEX", 1),
            ("PBSINTERSECTSPECULAR_KW_NORMALMAP", 2),
            ("PBSINTERSECTSPECULAR_KW_OCCLUSION", 3),
            ("PBSINTERSECTSPECULAR_KW_SPECULARMAP", 4),
        ],
    )
}

#[test]
fn pbsintersect_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsintersect.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSINTERSECT_KW_ALBEDOTEX", 0),
            ("PBSINTERSECT_KW_EMISSIONTEX", 1),
            ("PBSINTERSECT_KW_METALLICMAP", 2),
            ("PBSINTERSECT_KW_NORMALMAP", 3),
            ("PBSINTERSECT_KW_OCCLUSION", 4),
        ],
    )
}

#[test]
fn pbslerpspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbslerpspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_DUALSIDED",
            "_EMISSIONTEX",
            "_LERPTEX",
            "_MULTI_VALUES",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSLERPSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSLERPSPECULAR_KW_ALPHACLIP", 1),
            ("PBSLERPSPECULAR_KW_DUALSIDED", 2),
            ("PBSLERPSPECULAR_KW_EMISSIONTEX", 3),
            ("PBSLERPSPECULAR_KW_LERPTEX", 4),
            ("PBSLERPSPECULAR_KW_MULTI_VALUES", 5),
            ("PBSLERPSPECULAR_KW_NORMALMAP", 6),
            ("PBSLERPSPECULAR_KW_OCCLUSION", 7),
            ("PBSLERPSPECULAR_KW_SPECULARMAP", 8),
        ],
    )
}

#[test]
fn pbslerp_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbslerp.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_DUALSIDED",
            "_EMISSIONTEX",
            "_LERPTEX",
            "_METALLICMAP",
            "_MULTI_VALUES",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSLERP_KW_ALBEDOTEX", 0),
            ("PBSLERP_KW_ALPHACLIP", 1),
            ("PBSLERP_KW_DUALSIDED", 2),
            ("PBSLERP_KW_EMISSIONTEX", 3),
            ("PBSLERP_KW_LERPTEX", 4),
            ("PBSLERP_KW_METALLICMAP", 5),
            ("PBSLERP_KW_MULTI_VALUES", 6),
            ("PBSLERP_KW_NORMALMAP", 7),
            ("PBSLERP_KW_OCCLUSION", 8),
        ],
    )
}

#[test]
fn pbsmetallic_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsmetallic.wgsl",
        &[
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
            "_ALPHATEST_ON",
            "_DETAIL_MULX2",
            "_EMISSION",
            "_GLOSSYREFLECTIONS_OFF",
            "_METALLICGLOSSMAP",
            "_MUL_RGB_BY_ALPHA",
            "_NORMALMAP",
            "_PARALLAXMAP",
            "_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A",
            "_SPECULARHIGHLIGHTS_OFF",
        ],
        &[
            ("PBSMETALLIC_KW_ALPHABLEND_ON", 0),
            ("PBSMETALLIC_KW_ALPHAPREMULTIPLY_ON", 1),
            ("PBSMETALLIC_KW_ALPHATEST_ON", 2),
            ("PBSMETALLIC_KW_DETAIL_MULX2", 3),
            ("PBSMETALLIC_KW_EMISSION", 4),
            ("PBSMETALLIC_KW_GLOSSYREFLECTIONS_OFF", 5),
            ("PBSMETALLIC_KW_METALLICGLOSSMAP", 6),
            ("PBSMETALLIC_KW_NORMALMAP", 7),
            ("PBSMETALLIC_KW_PARALLAXMAP", 8),
            ("PBSMETALLIC_KW_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A", 9),
            ("PBSMETALLIC_KW_SPECULARHIGHLIGHTS_OFF", 10),
            ("PBSMETALLIC_KW_EDITOR_VISUALIZATION", 11),
        ],
    )
}

#[test]
fn pbsmetallic_emission_gated_by_variant_bit_not_runtime_check() -> io::Result<()> {
    let src = material_source("pbsmetallic.wgsl")?;
    assert!(
        !src.contains("dot(emission_color, emission_color)"),
        "pbsmetallic.wgsl must not use the runtime `dot(emission_color, emission_color) > 1e-8` \
         guard; the _EMISSION variant bit controls the optional emission sample"
    );
    assert!(
        src.contains("pbs_kw(PBSMETALLIC_KW_EMISSION)"),
        "pbsmetallic.wgsl must gate emission sampling on PBSMETALLIC_KW_EMISSION"
    );
    Ok(())
}

#[test]
fn pbsmultiuvspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsmultiuvspecular.wgsl",
        &[
            "_ALPHACLIP",
            "_DUAL_ALBEDO",
            "_DUAL_EMISSIONTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSMULTIUVSPECULAR_KW_ALPHACLIP", 0),
            ("PBSMULTIUVSPECULAR_KW_DUAL_ALBEDO", 1),
            ("PBSMULTIUVSPECULAR_KW_DUAL_EMISSIONTEX", 2),
            ("PBSMULTIUVSPECULAR_KW_EMISSIONTEX", 3),
            ("PBSMULTIUVSPECULAR_KW_NORMALMAP", 4),
            ("PBSMULTIUVSPECULAR_KW_OCCLUSION", 5),
            ("PBSMULTIUVSPECULAR_KW_SPECULARMAP", 6),
        ],
    )
}

#[test]
fn pbsmultiuv_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsmultiuv.wgsl",
        &[
            "_ALPHACLIP",
            "_DUAL_ALBEDO",
            "_DUAL_EMISSIONTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSMULTIUV_KW_ALPHACLIP", 0),
            ("PBSMULTIUV_KW_DUAL_ALBEDO", 1),
            ("PBSMULTIUV_KW_DUAL_EMISSIONTEX", 2),
            ("PBSMULTIUV_KW_EMISSIONTEX", 3),
            ("PBSMULTIUV_KW_METALLICMAP", 4),
            ("PBSMULTIUV_KW_NORMALMAP", 5),
            ("PBSMULTIUV_KW_OCCLUSION", 6),
        ],
    )
}

#[test]
fn pbsrimspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsrimspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSRIMSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSRIMSPECULAR_KW_EMISSIONTEX", 1),
            ("PBSRIMSPECULAR_KW_NORMALMAP", 2),
            ("PBSRIMSPECULAR_KW_OCCLUSION", 3),
            ("PBSRIMSPECULAR_KW_SPECULARMAP", 4),
        ],
    )
}

#[test]
fn pbsrimtransparentspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsrimtransparentspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSRIMTRANSPARENTSPECULAR_KW_EMISSIONTEX", 0),
            ("PBSRIMTRANSPARENTSPECULAR_KW_NORMALMAP", 1),
            ("PBSRIMTRANSPARENTSPECULAR_KW_OCCLUSION", 2),
            ("PBSRIMTRANSPARENTSPECULAR_KW_SPECULARMAP", 3),
        ],
    )
}

#[test]
fn pbsrimtransparentspecular_drops_dead_albedo_path() -> io::Result<()> {
    let src = material_source("pbsrimtransparentspecular.wgsl")?;
    assert!(
        !src.contains("_MainTex:"),
        "pbsrimtransparentspecular.wgsl must not bind _MainTex; Unity never declares \
         #pragma multi_compile _ _ALBEDOTEX for this shader, so the #ifdef _ALBEDOTEX \
         branch is dead code and `_Color` is always the base color"
    );
    assert!(
        !src.contains("PBSRIMTRANSPARENTSPECULAR_KW_ALBEDOTEX"),
        "pbsrimtransparentspecular.wgsl must not declare a _ALBEDOTEX bit (dead in Unity)"
    );
    Ok(())
}

#[test]
fn pbsrim_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsrim.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSRIM_KW_ALBEDOTEX", 0),
            ("PBSRIM_KW_EMISSIONTEX", 1),
            ("PBSRIM_KW_METALLICMAP", 2),
            ("PBSRIM_KW_NORMALMAP", 3),
            ("PBSRIM_KW_OCCLUSION", 4),
        ],
    )
}

#[test]
fn pbsrimtransparent_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsrimtransparent.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSRIMTRANSPARENT_KW_ALBEDOTEX", 0),
            ("PBSRIMTRANSPARENT_KW_EMISSIONTEX", 1),
            ("PBSRIMTRANSPARENT_KW_METALLICMAP", 2),
            ("PBSRIMTRANSPARENT_KW_NORMALMAP", 3),
            ("PBSRIMTRANSPARENT_KW_OCCLUSION", 4),
        ],
    )
}

#[test]
fn pbsrimtransparentzwrite_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsrimtransparentzwrite.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSRIMTRANSPARENTZWRITE_KW_ALBEDOTEX", 0),
            ("PBSRIMTRANSPARENTZWRITE_KW_EMISSIONTEX", 1),
            ("PBSRIMTRANSPARENTZWRITE_KW_METALLICMAP", 2),
            ("PBSRIMTRANSPARENTZWRITE_KW_NORMALMAP", 3),
            ("PBSRIMTRANSPARENTZWRITE_KW_OCCLUSION", 4),
        ],
    )
}

#[test]
fn pbsrimtransparentzwritespecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsrimtransparentzwritespecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSRIMTRANSPARENTZWRITESPECULAR_KW_EMISSIONTEX", 0),
            ("PBSRIMTRANSPARENTZWRITESPECULAR_KW_NORMALMAP", 1),
            ("PBSRIMTRANSPARENTZWRITESPECULAR_KW_OCCLUSION", 2),
            ("PBSRIMTRANSPARENTZWRITESPECULAR_KW_SPECULARMAP", 3),
        ],
    )
}

#[test]
fn pbsrimtransparentzwritespecular_drops_dead_albedo_path() -> io::Result<()> {
    let src = material_source("pbsrimtransparentzwritespecular.wgsl")?;
    assert!(
        !src.contains("_MainTex:"),
        "pbsrimtransparentzwritespecular.wgsl must not bind _MainTex; Unity never declares \
         #pragma multi_compile _ _ALBEDOTEX for this shader, so the #ifdef _ALBEDOTEX \
         branch is dead code and `_Color` is always the base color"
    );
    assert!(
        !src.contains("PBSRIMTRANSPARENTZWRITESPECULAR_KW_ALBEDOTEX"),
        "pbsrimtransparentzwritespecular.wgsl must not declare a _ALBEDOTEX bit (dead in Unity)"
    );
    Ok(())
}

#[test]
fn pbsslice_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsslice.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_DETAIL_ALBEDOTEX",
            "_DETAIL_NORMALMAP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OBJECT_SPACE",
            "_OCCLUSION",
            "_WORLD_SPACE",
        ],
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
    )
}

#[test]
fn pbsslicespecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsslicespecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_DETAIL_ALBEDOTEX",
            "_DETAIL_NORMALMAP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OBJECT_SPACE",
            "_OCCLUSION",
            "_SPECULARMAP",
            "_WORLD_SPACE",
        ],
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
    )
}

#[test]
fn pbsslicetransparent_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsslicetransparent.wgsl",
        &[
            "_ALBEDOTEX",
            "_DETAIL_ALBEDOTEX",
            "_DETAIL_NORMALMAP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OBJECT_SPACE",
            "_OCCLUSION",
            "_WORLD_SPACE",
        ],
        &[
            ("PBSSLICETRANSPARENT_KW_ALBEDOTEX", 0),
            ("PBSSLICETRANSPARENT_KW_DETAIL_ALBEDOTEX", 1),
            ("PBSSLICETRANSPARENT_KW_DETAIL_NORMALMAP", 2),
            ("PBSSLICETRANSPARENT_KW_EMISSIONTEX", 3),
            ("PBSSLICETRANSPARENT_KW_METALLICMAP", 4),
            ("PBSSLICETRANSPARENT_KW_NORMALMAP", 5),
            ("PBSSLICETRANSPARENT_KW_OCCLUSION", 6),
            ("PBSSLICETRANSPARENT_KW_OBJECT_SPACE", 7),
            ("PBSSLICETRANSPARENT_KW_WORLD_SPACE", 8),
        ],
    )
}

#[test]
fn pbsslicetransparentspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsslicetransparentspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_DETAIL_ALBEDOTEX",
            "_DETAIL_NORMALMAP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OBJECT_SPACE",
            "_OCCLUSION",
            "_SPECULARMAP",
            "_WORLD_SPACE",
        ],
        &[
            ("PBSSLICETRANSPARENTSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSSLICETRANSPARENTSPECULAR_KW_DETAIL_ALBEDOTEX", 1),
            ("PBSSLICETRANSPARENTSPECULAR_KW_DETAIL_NORMALMAP", 2),
            ("PBSSLICETRANSPARENTSPECULAR_KW_EMISSIONTEX", 3),
            ("PBSSLICETRANSPARENTSPECULAR_KW_METALLICMAP", 4),
            ("PBSSLICETRANSPARENTSPECULAR_KW_NORMALMAP", 5),
            ("PBSSLICETRANSPARENTSPECULAR_KW_OCCLUSION", 6),
            ("PBSSLICETRANSPARENTSPECULAR_KW_OBJECT_SPACE", 7),
            ("PBSSLICETRANSPARENTSPECULAR_KW_WORLD_SPACE", 8),
        ],
    )
}

#[test]
fn pbsspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsspecular.wgsl",
        &[
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
            "_ALPHATEST_ON",
            "_DETAIL_MULX2",
            "_EMISSION",
            "_GLOSSYREFLECTIONS_OFF",
            "_MUL_RGB_BY_ALPHA",
            "_NORMALMAP",
            "_PARALLAXMAP",
            "_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A",
            "_SPECGLOSSMAP",
            "_SPECULARHIGHLIGHTS_OFF",
        ],
        &[
            ("PBSSPECULAR_KW_ALPHABLEND_ON", 0),
            ("PBSSPECULAR_KW_ALPHAPREMULTIPLY_ON", 1),
            ("PBSSPECULAR_KW_ALPHATEST_ON", 2),
            ("PBSSPECULAR_KW_DETAIL_MULX2", 3),
            ("PBSSPECULAR_KW_EMISSION", 4),
            ("PBSSPECULAR_KW_GLOSSYREFLECTIONS_OFF", 5),
            ("PBSSPECULAR_KW_NORMALMAP", 6),
            ("PBSSPECULAR_KW_PARALLAXMAP", 7),
            ("PBSSPECULAR_KW_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A", 8),
            ("PBSSPECULAR_KW_SPECGLOSSMAP", 9),
            ("PBSSPECULAR_KW_SPECULARHIGHLIGHTS_OFF", 10),
        ],
    )
}

#[test]
fn pbsspecular_emission_gated_by_variant_bit_not_runtime_check() -> io::Result<()> {
    let src = material_source("pbsspecular.wgsl")?;
    assert!(
        !src.contains("dot(emission_color, emission_color)"),
        "pbsspecular.wgsl must not use the runtime `dot(emission_color, emission_color) > 1e-8` \
         guard; the _EMISSION variant bit controls the optional emission sample"
    );
    assert!(
        src.contains("pbs_kw(PBSSPECULAR_KW_EMISSION)"),
        "pbsspecular.wgsl must gate emission sampling on PBSSPECULAR_KW_EMISSION"
    );
    Ok(())
}

#[test]
fn pbsstencilspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsstencilspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSSTENCILSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSSTENCILSPECULAR_KW_EMISSIONTEX", 1),
            ("PBSSTENCILSPECULAR_KW_NORMALMAP", 2),
            ("PBSSTENCILSPECULAR_KW_OCCLUSION", 3),
            ("PBSSTENCILSPECULAR_KW_SPECULARMAP", 4),
        ],
    )
}

#[test]
fn pbsstencil_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsstencil.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSSTENCIL_KW_ALBEDOTEX", 0),
            ("PBSSTENCIL_KW_EMISSIONTEX", 1),
            ("PBSSTENCIL_KW_METALLICMAP", 2),
            ("PBSSTENCIL_KW_NORMALMAP", 3),
            ("PBSSTENCIL_KW_OCCLUSION", 4),
        ],
    )
}

#[test]
fn pbsstencil_emission_includes_rim_term() -> io::Result<()> {
    let src = material_source("pbsstencil.wgsl")?;
    assert!(
        src.contains("mf::rim_factor("),
        "pbsstencil.wgsl must compute rim through renderide::material::fresnel::rim_factor"
    );
    assert!(
        src.contains("mat._RimColor.rgb"),
        "pbsstencil.wgsl must add the _RimColor contribution to emission"
    );
    Ok(())
}

#[test]
fn pbstriplanar_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbstriplanar.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OBJECTSPACE",
            "_OCCLUSION",
            "_WORLDSPACE",
        ],
        &[
            ("PBSTRIPLANAR_KW_ALBEDOTEX", 0),
            ("PBSTRIPLANAR_KW_EMISSIONTEX", 1),
            ("PBSTRIPLANAR_KW_METALLICMAP", 2),
            ("PBSTRIPLANAR_KW_NORMALMAP", 3),
            ("PBSTRIPLANAR_KW_OBJECTSPACE", 4),
            ("PBSTRIPLANAR_KW_OCCLUSION", 5),
            ("PBSTRIPLANAR_KW_WORLDSPACE", 6),
        ],
    )
}

#[test]
fn pbstriplanarspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbstriplanarspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OBJECTSPACE",
            "_OCCLUSION",
            "_SPECULARMAP",
            "_WORLDSPACE",
        ],
        &[
            ("PBSTRIPLANARSPEC_KW_ALBEDOTEX", 0),
            ("PBSTRIPLANARSPEC_KW_EMISSIONTEX", 1),
            ("PBSTRIPLANARSPEC_KW_NORMALMAP", 2),
            ("PBSTRIPLANARSPEC_KW_OBJECTSPACE", 3),
            ("PBSTRIPLANARSPEC_KW_OCCLUSION", 4),
            ("PBSTRIPLANARSPEC_KW_SPECULARMAP", 5),
            ("PBSTRIPLANARSPEC_KW_WORLDSPACE", 6),
        ],
    )
}

#[test]
fn pbstriplanar_shaders_flip_normal_for_back_face() -> io::Result<()> {
    for file in ["pbstriplanar.wgsl", "pbstriplanarspecular.wgsl"] {
        let src = material_source(file)?;
        assert!(
            src.contains("ptri::flip_normal_for_back_face("),
            "{file} must call ptri::flip_normal_for_back_face for dual-sided shading"
        );
        assert!(
            src.contains("@builtin(front_facing)"),
            "{file} must take @builtin(front_facing) so it can flip normals on back faces"
        );
    }
    Ok(())
}

#[test]
fn pbsvertexcolortransparent_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsvertexcolortransparent.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
            "VCOLOR_ALBEDO",
            "VCOLOR_EMIT",
            "VCOLOR_METALLIC",
        ],
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
    )
}

#[test]
fn pbsvertexcolortransparentspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsvertexcolortransparentspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
            "VCOLOR_ALBEDO",
            "VCOLOR_EMIT",
            "VCOLOR_SPECULAR",
        ],
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
    )
}
