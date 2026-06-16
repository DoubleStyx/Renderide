//! Shader source audits for light-cookie scalar behavior.

use super::*;

#[test]
fn light_cookie_atlas_sampling_uses_scalar_red_channel() -> io::Result<()> {
    let lighting = source_file(manifest_dir().join("shaders/modules/lighting/light_cookies.wgsl"))?;
    for required in [
        "textureSample(\n        rg::light_cookie_2d_atlas,",
        "atlas_cookie_uv(light.cookie_layer, wrap_cookie_uv(uv, light.cookie_reserved))",
        "wrap_cookie_uv(uv, light.cookie_reserved)",
        "atlas_cookie_uv(rect_index, face_uv.xy)",
    ] {
        assert!(
            lighting.contains(required),
            "lighting light-cookie atlas sampling must read scalar red channel: `{required}`"
        );
    }
    for forbidden in [
        "light_cookie_spot_atlas",
        "textureSample(rg::light_cookie_point_atlas, rg::light_cookie_sampler, face_uv.xy, i32(layer)).a",
        "texture_2d_array<f32>",
    ] {
        assert!(
            !lighting.contains(forbidden),
            "lighting light-cookie atlas sampling must not read alpha from scalar atlases: `{forbidden}`"
        );
    }

    let globals = source_file(manifest_dir().join("shaders/modules/frame/globals.wgsl"))?;
    assert!(
        globals.contains("textureSampleLevel(light_cookie_2d_atlas, light_cookie_sampler, vec2<f32>(0.5), 0.0).r")
            && globals.contains("textureSampleLevel(light_cookie_point_atlas, light_cookie_sampler, vec2<f32>(0.5), 0.0).r")
            && globals.contains("light_cookie_rects[0u].origin_scale.x"),
        "frame globals retain path must read scalar red from light-cookie atlases"
    );
    assert!(
        !globals.contains("textureSampleLevel(light_cookie_2d_atlas, light_cookie_sampler, vec2<f32>(0.5), 0.0).a")
            && !globals.contains("textureSampleLevel(light_cookie_point_atlas, light_cookie_sampler, vec2<f32>(0.5), 0.0).a"),
        "frame globals retain path must not read alpha from scalar light-cookie atlases"
    );

    Ok(())
}

#[test]
fn light_cookie_blit_shader_stays_mask_alpha_or_red() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/passes/backend/light_cookie_blit_2d.wgsl"))?;
    assert!(
        src.contains("fn source_alpha") && src.contains("return source_sample(in).a;"),
        "light-cookie blit shader must expose an alpha-channel mask path"
    );
    assert!(
        src.contains("fn source_red") && src.contains("return source_sample(in).r;"),
        "light-cookie blit shader must expose a red-channel mask path"
    );
    assert!(
        src.contains("fn fs_alpha_scalar") && src.contains("fn fs_red_scalar"),
        "light-cookie blit shader must expose scalar atlas fragment outputs"
    );
    assert!(
        src.contains("fn fs_alpha_rgba") && src.contains("fn fs_red_rgba"),
        "light-cookie blit shader must expose RGBA atlas fragment outputs"
    );
    for forbidden in [".rgb *", "fn fs_main"] {
        assert!(
            !src.contains(forbidden),
            "light-cookie blit shader must not use the old RGBA/color-cookie path: `{forbidden}`"
        );
    }
    Ok(())
}

#[test]
fn directional_lights_apply_cookie_multiplier() -> io::Result<()> {
    let lighting = source_file(manifest_dir().join("shaders/modules/lighting/light_cookies.wgsl"))?;
    assert!(
        lighting.contains("fn directional_cookie_multiplier")
            && lighting.contains("ft::LIGHT_COOKIE_KIND_DIRECTIONAL_2D"),
        "light-cookie helpers must include a directional 2D cookie projection path"
    );

    let pbs = source_file(manifest_dir().join("shaders/modules/pbs/brdf.wgsl"))?;
    assert!(
        pbs.contains("out.attenuation = bl::direct_light_scale();")
            && pbs.contains(
                "out.attenuation = out.attenuation * cookies::multiplier(light, world_pos);"
            ),
        "PBS directional lighting must apply cookie attenuation"
    );

    let toon = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;
    assert!(
        toon.contains("bl::direct_light_scale() * cookies::multiplier(light, world_pos)"),
        "Xiexe toon directional lighting must apply cookie attenuation"
    );

    for material in ["toonstandard.wgsl", "toonwater.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("attenuation = bl::direct_light_scale();")
                && src
                    .contains("attenuation = attenuation * cookies::multiplier(light, world_pos);"),
            "{material} directional lighting must apply cookie attenuation"
        );
    }
    Ok(())
}

#[test]
fn light_cookie_pipeline_stems_do_not_reference_removed_targets() -> io::Result<()> {
    let src = source_file(manifest_dir().join("src/backend/frame_gpu/light_cookies.rs"))?;
    for forbidden in ["light_cookie_blit_2d_default", "light_cookie_blit_cube"] {
        assert!(
            !src.contains(forbidden),
            "light-cookie backend must not reference removed composed target `{forbidden}`"
        );
    }
    Ok(())
}
