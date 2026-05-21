//! Shader source audits for this behavior family.

use super::*;

#[test]
fn text_shaders_use_one_font_atlas_sample_for_coverage() -> io::Result<()> {
    for file_name in ["ui_textunlit.wgsl", "textunlit.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("#import renderide::core::texture_sampling as ts"),
            "{file_name} must import biased texture sampling for _FontAtlas"
        );
        assert!(
            declares_f32_field(&src, "_FontAtlas_LodBias"),
            "{file_name} must expose _FontAtlas_LodBias in the material uniform"
        );
        assert_eq!(
            count_font_atlas_lod_bias_samples(&src),
            1,
            "{file_name} must sample _FontAtlas exactly once through the LOD-bias helper"
        );
        assert!(
            !src.contains("textureSample(_FontAtlas")
                && !src.contains("textureSampleLevel(_FontAtlas"),
            "{file_name} must not directly sample _FontAtlas outside the shared helper"
        );
        assert!(
            !src.contains("texture_rgba_base_mip(_FontAtlas"),
            "{file_name} must not force base-mip atlas sampling for text coverage"
        );
        assert!(
            !src.contains("atlas_clip"),
            "{file_name} must route text coverage through the same atlas sample as color"
        );
    }

    let module_src = source_file(manifest_dir().join("shaders/modules/material/text_sdf.wgsl"))?;
    assert!(
        !module_src.contains("atlas_clip"),
        "text_sdf.wgsl must not expose a second atlas sample for coverage"
    );
    Ok(())
}

#[test]
fn text_shaders_route_font_extra_data_through_normal_stream() -> io::Result<()> {
    for file_name in ["ui_textunlit.wgsl", "textunlit.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("@location(1) extra_n: vec4<f32>"),
            "{file_name} must read glyph extra data from the normal stream"
        );
        assert!(
            src.contains("@location(2) uv: vec2<f32>"),
            "{file_name} must keep atlas UVs on vertex location 2"
        );
        assert!(
            src.contains("@location(3) color: vec4<f32>"),
            "{file_name} must keep vertex tint on vertex location 3"
        );
        assert!(
            src.contains("out.extra_data = extra_n;"),
            "{file_name} must pass glyph extra data through to the fragment shader"
        );
    }
    Ok(())
}

#[test]
fn text_outline_keyword_uses_per_glyph_width_even_when_material_size_is_zero() -> io::Result<()> {
    let module_src = source_file(manifest_dir().join("shaders/modules/material/text_sdf.wgsl"))?;
    assert!(
        module_src.contains("if (outline_enabled) {")
            && !module_src.contains("outline_enabled && style.outline_size > 1e-6"),
        "text_sdf.wgsl must not drop OUTLINE when glyph extra data supplies the outline width"
    );
    Ok(())
}
