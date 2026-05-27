//! Shader source audits for this behavior family.

use super::*;

fn pass_directives(src: &str) -> Vec<&str> {
    src.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("//#pass ")?;
            let pass_type = rest
                .split_whitespace()
                .find_map(|token| token.strip_prefix("type="))?;
            Some(
                rest.split_whitespace()
                    .find_map(|token| token.strip_prefix("name="))
                    .unwrap_or(pass_type),
            )
        })
        .collect()
}

fn assert_keyword_bit(src: &str, file_name: &str, constant_name: &str, bit_index: u32) {
    let needle = format!("const {constant_name}: u32 = 1u << {bit_index}u;");
    assert!(src.contains(&needle), "{file_name} must define `{needle}`");
}

/// Asserts all expected shader variant bit constants for a material root.
fn assert_keyword_bits(file_name: &str, expected: &[(&str, u32)]) -> io::Result<()> {
    let src = material_source(file_name)?;
    for (constant_name, bit_index) in expected.iter().copied() {
        assert_keyword_bit(&src, file_name, constant_name, bit_index);
    }
    Ok(())
}

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
fn pbs_slice_keeps_unity_space_and_alpha_clip_precedence() -> io::Result<()> {
    let slice_family = module_source("pbs/families/slice.wgsl")?;
    assert!(
        slice_family.contains("if (object_space_enabled) {\n        return false;\n    }"),
        "PBSSlice must match Unity's OBJECT_SPACE branch taking precedence over WORLD_SPACE"
    );
    assert!(
        slice_family.contains("return world_space_enabled || (!object_space_enabled);"),
        "PBSSlice must keep WORLD_SPACE as the implicit fallback when neither space bit is set"
    );

    for material in ["pbsslice.wgsl", "pbsslicespecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("&& c.a < mat._AlphaClip"),
            "{material} must match Unity `clip(c.a - _AlphaClip)` equality behavior"
        );
        assert!(
            !src.contains("&& c.a <= mat._AlphaClip"),
            "{material} must not reject alpha exactly equal to `_AlphaClip`"
        );
    }

    Ok(())
}

#[test]
fn pbs_transparent_roots_keep_authored_pass_directives() -> io::Result<()> {
    for material in [
        "pbsdisplacetransparent.wgsl",
        "pbsdisplacespeculartransparent.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
        "pbsintersect.wgsl",
        "pbsintersectspecular.wgsl",
        "pbsslicetransparent.wgsl",
        "pbsslicetransparentspecular.wgsl",
        "pbstriplanartransparent.wgsl",
        "pbstriplanartransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(pass_directives(&src), ["forward_transparent"], "{material}");
    }

    for material in ["pbsrimtransparent.wgsl", "pbsrimtransparentspecular.wgsl"] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            ["forward_transparent_cull_back"],
            "{material}"
        );
    }

    for material in [
        "pbsrimtransparentzwrite.wgsl",
        "pbsrimtransparentzwritespecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            ["depth_prepass", "forward_transparent_cull_back"],
            "{material}"
        );
    }

    for material in [
        "pbsvertexcolortransparent.wgsl",
        "pbsvertexcolortransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            ["forward_transparent_cull_back"],
            "{material}"
        );
    }

    for material in [
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert_eq!(
            pass_directives(&src),
            [
                "forward_transparent_cull_front",
                "forward_transparent_cull_back"
            ],
            "{material}"
        );
    }

    Ok(())
}

#[test]
fn pbs_displace_alpha_clip_matches_unity_threshold_equality() -> io::Result<()> {
    for material in [
        "pbsdisplace.wgsl",
        "pbsdisplacespecular.wgsl",
        "pbsdisplacetransparent.wgsl",
        "pbsdisplacespeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("&& c.a < mat._AlphaClip"),
            "{material} must match Unity `clip(c.a - _AlphaClip)` equality behavior"
        );
        assert!(
            !src.contains("&& c.a <= mat._AlphaClip"),
            "{material} must not reject alpha exactly equal to `_AlphaClip`"
        );
    }
    Ok(())
}

#[test]
fn pbs_vertex_color_transparent_roots_keep_source_alpha_and_emission() -> io::Result<()> {
    for material in [
        "pbsvertexcolortransparent.wgsl",
        "pbsvertexcolortransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("&& albedo.a < mat._AlphaClip"),
            "{material} must match Unity `clip(albedo.a - _AlphaClip)` equality behavior"
        );
        assert!(
            !src.contains("&& albedo.a <= mat._AlphaClip"),
            "{material} must not reject alpha exactly equal to `_AlphaClip`"
        );
        assert!(
            src.contains("var emission = mat._EmissionColor.rgb;"),
            "{material} must include constant emission color even without an emission texture"
        );
        assert!(
            !src.contains("dot(emission_color, emission_color) > 1e-8"),
            "{material} must not suppress tiny nonzero emission colors"
        );
    }
    Ok(())
}

#[test]
fn pbsvoronoicrystal_keeps_global_uv_transform() -> io::Result<()> {
    let src = material_source("pbsvoronoicrystal.wgsl")?;

    for required in [
        "_Global_ST: vec4<f32>",
        "let global_uv = uvu::apply_st(uv, mat._Global_ST);",
        "vor::voronoi_full(global_uv * scale, scale, mat._AnimationOffset)",
    ] {
        assert!(
            src.contains(required),
            "pbsvoronoicrystal.wgsl must contain `{required}`"
        );
    }
    assert!(
        !src.contains("vor::voronoi_full(uv * scale"),
        "pbsvoronoicrystal.wgsl must apply `_Global_ST` before Voronoi sampling"
    );

    Ok(())
}

#[test]
fn pbs_displace_roots_keep_source_authored_one_sided_normals() -> io::Result<()> {
    for material in [
        "pbsdisplacespecular.wgsl",
        "pbsdisplacetransparent.wgsl",
        "pbsdisplacespeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        for forbidden in [
            "@builtin(front_facing)",
            "ts_n.z = -ts_n.z",
            "psamp::two_sided_geometric_normal",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must not apply dual-sided normal handling through `{forbidden}`"
            );
        }
    }
    Ok(())
}

#[test]
fn pbs_distance_lerp_roots_keep_source_zero_uv_and_raw_displacement_direction() -> io::Result<()> {
    for material in [
        "pbsdistancelerp.wgsl",
        "pbsdistancelerpspecular.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("let uv_main = vec2<f32>(0.0);"),
            "{material} must sample material textures at the source-authored zero UV"
        );
        for forbidden in [
            "_MainTex_ST: vec4<f32>",
            "uvu::apply_st(uv0, mat._MainTex_ST)",
            "@location(2) uv0",
            "normalize(n.xyz)",
            "normalize(mat._DisplacementDirection.xyz)",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must not use `{forbidden}`"
            );
        }
        assert!(
            src.contains("pdist::vertex_main("),
            "{material} must route displacement direction selection through the DistanceLerp module"
        );
    }

    let module = module_source("pbs/families/distance_lerp.wgsl")?;
    assert!(
        module.contains(
            "let direction = select(object_n, override_direction, override_direction_enabled);"
        ) && !module.contains("normalize(override_direction)")
            && !module.contains("normalize(object_n)"),
        "distance_lerp.wgsl must preserve raw displacement direction magnitude"
    );
    Ok(())
}

#[test]
fn selected_pbs_material_roots_keep_material_offset_pass_metadata() -> io::Result<()> {
    for material in [
        "pbsdistancelerp.wgsl",
        "pbsdistancelerpspecular.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdualsided.wgsl",
        "pbsdualsidedspecular.wgsl",
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
        "pbsintersect.wgsl",
        "pbsintersectspecular.wgsl",
    ] {
        let src = material_source(material)?;
        let mut pass_count = 0usize;
        for line in src
            .lines()
            .filter(|line| line.trim_start().starts_with("//#pass "))
        {
            pass_count += 1;
            assert!(
                line.contains("offset=material(0,0)"),
                "{material} pass directive must preserve material-driven Unity Offset state: {line}"
            );
        }
        assert!(pass_count > 0, "{material} must declare at least one pass");
    }
    Ok(())
}

#[test]
fn pbs_material_roots_use_shared_sampling_and_mask_helpers() -> io::Result<()> {
    for material in ["pbscolorsplat.wgsl", "pbscolorsplatspecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("psamp::unpack_packed_normal_xy("),
            "{material} must use the shared packed-normal unpack helper"
        );
        assert!(
            !src.contains("fn unpack_normal_xy"),
            "{material} must not redeclare packed-normal unpacking"
        );
    }

    for material in ["pbscolormask.wgsl", "pbscolormaskspecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("splat::color_mask_weights(mask)") && src.contains("mcolor::blend4_vec4("),
            "{material} must use shared color-mask weighting and blend helpers"
        );
    }

    for material in [
        "pbsdualsided.wgsl",
        "pbsdualsidedspecular.wgsl",
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("psamp::sample_optional_two_sided_world_normal("),
            "{material} must use the shared two-sided normal sampling helper"
        );
        for forbidden in [
            "nd::decode_ts_normal_with_placeholder_sample",
            "pnorm::orthonormal_tbn",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must delegate `{forbidden}` through pbs::sampling"
            );
        }
    }

    for material in [
        "pbsdistancelerp.wgsl",
        "pbsdistancelerpspecular.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("psamp::sample_optional_two_sided_world_normal("),
            "{material} must use the shared two-sided normal sampling helper"
        );
        for forbidden in [
            "nd::decode_ts_normal_with_placeholder_sample",
            "pnorm::orthonormal_tbn",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must delegate `{forbidden}` through pbs::sampling"
            );
        }
    }

    let splat = module_source("pbs/splat.wgsl")?;
    assert!(
        splat.contains("fn color_mask_weights(mask: vec4<f32>) -> vec4<f32>")
            && splat.contains("mask * clamp(1.0 / sum, 0.0, 1.0)"),
        "pbs::splat must expose the Unity color-mask blend-weight policy"
    );

    Ok(())
}

#[test]
fn pbs_transparent_roots_use_premultiplied_transparent_lighting() -> io::Result<()> {
    for material in [
        "pbsdisplacetransparent.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdualsidedtransparent.wgsl",
        "pbsintersect.wgsl",
        "pbsrimtransparent.wgsl",
        "pbsrimtransparentzwrite.wgsl",
        "pbsslicetransparent.wgsl",
        "pbstriplanartransparent.wgsl",
        "pbsvertexcolortransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("plight::shade_metallic_transparent_clustered("),
            "{material} must use Unity-style premultiplied metallic transparency"
        );
        assert!(
            !src.contains("plight::shade_metallic_clustered("),
            "{material} must not return straight-alpha metallic lighting"
        );
    }

    for material in [
        "pbsdisplacespeculartransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
        "pbsintersectspecular.wgsl",
        "pbsrimtransparentspecular.wgsl",
        "pbsrimtransparentzwritespecular.wgsl",
        "pbsslicetransparentspecular.wgsl",
        "pbstriplanartransparentspecular.wgsl",
        "pbsvertexcolortransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("plight::shade_specular_transparent_clustered("),
            "{material} must use Unity-style premultiplied specular transparency"
        );
        assert!(
            !src.contains("plight::shade_specular_clustered("),
            "{material} must not return straight-alpha specular lighting"
        );
    }

    let lighting = module_source("pbs/lighting.wgsl")?;
    assert!(
        lighting.contains("brdf::unity_premultiplied_alpha(s.alpha, one_minus_reflectivity)")
            && lighting
                .contains("brdf::unity_premultiplied_alpha(s.alpha, s.one_minus_reflectivity)"),
        "PBS transparent helpers must write Unity's reflectivity-adjusted premultiplied alpha"
    );

    Ok(())
}

#[test]
fn generic_pbs_premultiply_variants_use_unity_transparent_lighting() -> io::Result<()> {
    let metallic = material_source("pbsmetallic.wgsl")?;
    for required in [
        "fn alpha_premultiply_enabled() -> bool",
        "if (alpha_premultiply_enabled())",
        "plight::shade_metallic_transparent_clustered(",
        "return vec4<f32>(color, s.alpha);",
    ] {
        assert!(
            metallic.contains(required),
            "pbsmetallic.wgsl must contain `{required}`"
        );
    }
    for forbidden in [
        "fn apply_premultiply(",
        "return select(color, color * alpha, alpha_premultiply_enabled());",
        "return vec4<f32>(apply_premultiply(color, s.alpha), s.alpha);",
    ] {
        assert!(
            !metallic.contains(forbidden),
            "pbsmetallic.wgsl must not premultiply final lit RGB with `{forbidden}`"
        );
    }

    let specular = material_source("pbsspecular.wgsl")?;
    for required in [
        "fn alpha_premultiply_enabled() -> bool",
        "if (alpha_premultiply_enabled())",
        "plight::shade_specular_transparent_clustered(",
        "return vec4<f32>(color, s.alpha);",
    ] {
        assert!(
            specular.contains(required),
            "pbsspecular.wgsl must contain `{required}`"
        );
    }
    for forbidden in [
        "fn apply_premultiply(",
        "return select(color, color * alpha, alpha_premultiply_enabled());",
        "return vec4<f32>(apply_premultiply(color, s.alpha), s.alpha);",
    ] {
        assert!(
            !specular.contains(forbidden),
            "pbsspecular.wgsl must not premultiply final lit RGB with `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn light_radiance_conversion_reaches_directional_and_punctual_paths() -> io::Result<()> {
    let birp = module_source("lighting/birp.wgsl")?;
    assert!(
        birp.contains("fn light_radiance(light: ft::GpuLight) -> vec3<f32> {")
            && birp.contains("return srgb_light_to_linear(light.color * light.intensity);"),
        "BiRP light module must apply light intensity before light color transfer conversion"
    );
    assert!(
        birp.contains("fn direct_light_scale() -> f32 {\n    return INTENSITY_BOOST;\n}"),
        "BiRP light module must expose the shared direct-light scalar boost helper"
    );
    assert!(
        birp.contains("fn distance_visibility(dist: f32, range: f32) -> f32")
            && birp.contains("return lut * range_fade(t);")
            && birp.contains("return distance_visibility(dist, range) * INTENSITY_BOOST;"),
        "punctual distance attenuation must keep the existing intensity boost while exposing unboosted visibility"
    );
    assert!(
        birp.contains("fn spot_angle_attenuation(light: ft::GpuLight, l: vec3<f32>) -> f32"),
        "BiRP light module must expose the shared spot angle attenuation helper"
    );
    assert!(
        birp.contains("let tan2_theta = max(1.0 - rho2, 0.0) / rho2;")
            && birp.contains("let r2 = clamp(tan2_theta * light.spot_angle_scale, 0.0, 1.0);")
            && birp.contains("return squared_edge_fade(r2);"),
        "spot angle attenuation must use projected radial quadratic falloff"
    );

    let pbs_brdf = module_source("pbs/brdf.wgsl")?;
    assert!(
        pbs_brdf.contains("out.attenuation = bl::direct_light_scale();"),
        "PBS directional lights must use the shared scalar boost"
    );
    assert!(
        pbs_brdf.contains("out.attenuation = distance_attenuation(dist, light.range);")
            && pbs_brdf.contains("let spot_atten = bl::spot_angle_attenuation(light, out.l);")
            && pbs_brdf.contains(
                "out.attenuation = spot_atten * distance_attenuation(dist, light.range);"
            )
            && pbs_brdf.contains("return bl::light_radiance(light) * attenuation * n_dot_l;"),
        "PBS point and spot lights must keep attenuation scalar and use shared light radiance"
    );

    let xiexe = module_source("xiexe/toon2/lighting.wgsl")?;
    assert!(
        xiexe.contains("bl::light_radiance(light),")
            && xiexe.contains("bl::direct_light_scale() * cookies::multiplier(light, world_pos),"),
        "Xiexe directional lights must use shared linear radiance, scalar boost, and cookie attenuation"
    );
    assert!(
        xiexe.contains("var visibility = bl::distance_visibility(dist, light.range);")
            && xiexe.contains("visibility = visibility * bl::spot_angle_attenuation(light, l);")
            && xiexe.contains("let attenuation = visibility * bl::direct_light_scale();")
            && xiexe.contains(
                "return xb::LightSample(l, bl::light_radiance(light), attenuation, visibility, false);"
            ),
        "Xiexe point and spot lights must keep boosted attenuation scalar, unboosted visibility, and shared light radiance"
    );

    for material in ["toonstandard.wgsl", "toonwater.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("attenuation = bl::direct_light_scale();"),
            "{material} directional lights must use the shared scalar boost"
        );
        assert!(
            src.contains("attenuation = brdf::distance_attenuation(dist, light.range);")
                && src
                    .contains("attenuation = attenuation * bl::spot_angle_attenuation(light, l);")
                && src.contains("let radiance = bl::light_radiance(light) * attenuation;"),
            "{material} point and spot lights must keep attenuation scalar and use shared light radiance"
        );
        for forbidden in [
            "direct_light_intensity",
            "punctual_attenuation",
            "light.intensity * distance_attenuation",
            "light.intensity * brdf::distance_attenuation",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must not apply light intensity as a linear attenuation scalar through `{forbidden}`"
            );
        }
    }

    for (name, src) in [
        ("lighting/birp.wgsl", birp.as_str()),
        ("pbs/brdf.wgsl", pbs_brdf.as_str()),
        ("xiexe/toon2/lighting.wgsl", xiexe.as_str()),
    ] {
        for forbidden in [
            "direct_light_intensity",
            "punctual_attenuation",
            "light.intensity * distance_attenuation",
            "light.intensity * brdf::distance_attenuation",
        ] {
            assert!(
                !src.contains(forbidden),
                "{name} must not apply light intensity as a linear attenuation scalar through `{forbidden}`"
            );
        }
    }

    Ok(())
}

#[test]
fn spot_lights_do_not_use_arbitrary_smoothstep_cone_fade() -> io::Result<()> {
    for (name, src) in [
        ("pbs/brdf.wgsl", module_source("pbs/brdf.wgsl")?),
        (
            "xiexe/toon2/lighting.wgsl",
            module_source("xiexe/toon2/lighting.wgsl")?,
        ),
        ("toonstandard.wgsl", material_source("toonstandard.wgsl")?),
        ("toonwater.wgsl", material_source("toonwater.wgsl")?),
    ] {
        assert!(
            !src.contains("spot_cos_half_angle + 0.1")
                && !src.contains("smoothstep(light.spot_cos_half_angle"),
            "{name} must route spot cone fade through the shared BiRP helper"
        );
    }

    Ok(())
}

#[test]
fn toon_standard_and_water_use_unity_toon_brdf_composition() -> io::Result<()> {
    let toon_brdf = module_source("material/toon_brdf.wgsl")?;
    for required in [
        "fn energy_conserved_diffuse(",
        "fn direct_light(",
        "return radiance * (diff_color + spec_color * specular_step) * diffuse_step;",
        "fn indirect_light(",
        "let specular_tint = mix(spec_color, vec3<f32>(grazing_term), fresnel_term);",
    ] {
        assert!(
            toon_brdf.contains(required),
            "toon_brdf.wgsl must contain `{required}`"
        );
    }

    for material in ["toonstandard.wgsl", "toonwater.wgsl"] {
        let src = material_source(material)?;
        for required in [
            "#import renderide::core::texture_sampling as ts",
            "#import renderide::lighting::reflection_probes as rprobe",
            "tbrdf::energy_conserved_diffuse(",
            "tbrdf::direct_light(",
            "tbrdf::indirect_light(",
        ] {
            assert!(
                src.contains(required),
                "{material} must contain `{required}`"
            );
        }
    }

    let water = material_source("toonwater.wgsl")?;
    for required in [
        "fn unity_time_x() -> f32",
        "fn unity_time_y() -> f32",
        "fn unity_sin_time_w() -> f32",
        "sds::scene_world_y_at_uv(refracted_uv, view_layer) + object_y",
        "let smoothness = clamp(spec_s.a * mat._Glossiness, 0.0, 1.0);",
        "//#pass type=forward zwrite=off",
        "ts::sample_tex_2d(_ReflectionTex, _ReflectionTex_sampler, screen_uv, mat._ReflectionTex_LodBias)",
    ] {
        assert!(
            water.contains(required),
            "toonwater.wgsl must contain `{required}`"
        );
    }
    assert!(
        !water.contains("color = color + refl * (1.0 - smoothness)")
            && !water.contains("mat._SmoothnessTextureChannel > 0.5"),
        "toonwater.wgsl must not keep the old additive reflection or albedo-alpha smoothness paths"
    );

    Ok(())
}

#[test]
fn standard_pbs_roots_use_unity_standard_packed_channels() -> io::Result<()> {
    let standard = module_source("pbs/standard.wgsl")?;
    for required in [
        "fn standard_alpha(",
        "return color_alpha;",
        "return color_alpha * texture_alpha;",
        "fn clip_standard_alpha(",
        "if (enabled && alpha < cutoff) {",
        "fn occlusion_from_sample(sample: f32, strength: f32) -> f32",
        "return mix(1.0, sample, clamp(strength, 0.0, 1.0));",
    ] {
        assert!(
            standard.contains(required),
            "pbs/standard.wgsl must contain `{required}`"
        );
    }

    let metallic = material_source("pbsmetallic.wgsl")?;
    for required in [
        "_GlossMapScale: f32",
        "_OcclusionStrength: f32",
        "#import renderide::pbs::standard as pstd",
        "return pbs_kw(PBSMETALLIC_KW_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A);",
        "let base_alpha = pstd::standard_alpha(mat._Color.a, albedo_sample.a, smoothness_from_albedo_alpha());",
        "pstd::clip_standard_alpha(base_alpha, mat._Cutoff, alpha_test_enabled());",
        "let smoothness_scale = mat._GlossMapScale;",
        "smoothness = mg.a * smoothness_scale;",
        "smoothness = albedo_sample.a * smoothness_scale;",
        "ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).g",
        "pstd::occlusion_from_sample(occlusion_sample, mat._OcclusionStrength);",
        "psurf::metallic_with_geometric_normal(",
        "world_n,",
    ] {
        assert!(
            metallic.contains(required),
            "pbsmetallic.wgsl must contain `{required}`"
        );
    }
    assert!(
        !metallic.contains("mat._SmoothnessTextureChannel > 0.5")
            && !metallic.contains("unity_standard_clip_alpha"),
        "pbsmetallic.wgsl must follow the Unity Standard keyword path for albedo-alpha smoothness"
    );

    let specular = material_source("pbsspecular.wgsl")?;
    for required in [
        "#import renderide::pbs::standard as pstd",
        "let base_alpha = pstd::standard_alpha(mat._Color.a, albedo_sample.a, smoothness_from_albedo_alpha());",
        "pstd::clip_standard_alpha(base_alpha, mat._Cutoff, alpha_test_enabled());",
        "ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).g",
        "pstd::occlusion_from_sample(occlusion_sample, mat._OcclusionStrength);",
        "psurf::specular_with_geometric_normal(",
        "world_n,",
    ] {
        assert!(
            specular.contains(required),
            "pbsspecular.wgsl must contain `{required}`"
        );
    }
    assert!(
        !specular.contains("unity_standard_clip_alpha"),
        "pbsspecular.wgsl must clip against the visible filtered albedo alpha"
    );

    Ok(())
}

#[test]
fn standard_pbs_roots_enforce_unity_default_for_unsent_parameters() -> io::Result<()> {
    for material in ["pbsmetallic.wgsl", "pbsspecular.wgsl"] {
        let src = material_source(material)?;
        for required in [
            "//#mat_default _GlossMapScale float 1.0",
            "//#mat_default _OcclusionStrength float 1.0",
            "let smoothness_scale = mat._GlossMapScale;",
            "pstd::occlusion_from_sample(occlusion_sample, mat._OcclusionStrength);",
        ] {
            assert!(
                src.contains(required),
                "{material} must contain `{required}`"
            );
        }
        assert!(
            !src.contains("// let smoothness_scale = mat._GlossMapScale;")
                && !src.contains("// let occlusion_strength = mat._OcclusionStrength;"),
            "{material} must use material-default metadata instead of commented shader fallbacks"
        );
    }
    Ok(())
}

#[test]
fn furfx_and_toon_roots_declare_unity_defaults_for_unsent_fields() -> io::Result<()> {
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        if !label.contains("/furfx") {
            continue;
        }
        let src = source_file(&path)?;
        let required: &[&str] = if src.contains("renderide::fur::classic_selfshadow") {
            &[
                "//#mat_default _RimColor vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceGlobal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceLocal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _EdgeFade float 0.15",
                "//#mat_default _SkinAlpha float 0.5",
                "//#mat_default _Reflection float 0.0",
                "//#mat_default _ShadowStrength float 1.0",
            ]
        } else if src.contains("renderide::fur::classic_advanced") {
            &[
                "//#mat_default _RimColor vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceGlobal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceLocal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _EdgeFade float 0.15",
                "//#mat_default _SkinAlpha float 0.5",
                "//#mat_default _Reflection float 0.0",
            ]
        } else if src.contains("renderide::fur::classic_basic") {
            &[
                "//#mat_default _ForceGlobal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceLocal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _EdgeFade float 0.15",
                "//#mat_default _SkinAlpha float 0.5",
            ]
        } else if src.contains("renderide::fur::modern") {
            &[
                "//#mat_default _RimColor vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceGlobal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceLocal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _BonusAmbient vec4 0.0 0.0 0.0 1.0",
                "//#mat_default _RimColor vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ReflColor vec4 1.0 1.0 1.0 1.0",
                "//#mat_default _ForceGlobal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _ForceLocal vec4 0.0 0.0 0.0 0.0",
                "//#mat_default _SkinAlpha float 0.5",
                "//#mat_default _Reflection float 0.0",
                "//#mat_default _ReflMinLevel float 0.0",
            ]
        } else {
            continue;
        };
        for directive in required {
            assert!(
                src.contains(directive),
                "{label} must declare `{directive}`"
            );
        }
    }

    for (material, required) in [
        (
            "toonstandard.wgsl",
            [
                "//#mat_default _SpecularHighlights float 1.0",
                "//#mat_default _GlossyReflections float 1.0",
            ],
        ),
        (
            "toonwater.wgsl",
            [
                "//#mat_default _SpecularHighlights float 1.0",
                "//#mat_default _SmoothnessTextureChannel float 0.0",
            ],
        ),
    ] {
        let src = material_source(material)?;
        for directive in required {
            assert!(
                src.contains(directive),
                "{material} must declare `{directive}`"
            );
        }
    }
    Ok(())
}

#[test]
fn early_material_parity_gaps_stay_closed() -> io::Result<()> {
    let depth_projection = material_source("depthprojection.wgsl")?;
    assert!(
        !depth_projection.contains("@location(1) n:"),
        "depthprojection.wgsl must not request normals; the source vertex input uses position and UV only"
    );

    let fogbox = material_source("fogboxvolume.wgsl")?;
    assert!(
        !fogbox.contains("clamp_volume_source_rgb(apply_saturation"),
        "fogboxvolume.wgsl must not clamp RGB unless SATURATE_COLOR is selected"
    );
    assert!(
        fogbox.contains(
            "return rg::retain_globals_additive(apply_saturation(mat._BaseColor + acc));"
        ),
        "fogboxvolume.wgsl must preserve unsaturated HDR accumulation output"
    );

    let material_sample = module_source("material/sample.wgsl")?;
    for required in [
        "let mapped = uvu::polar_mapping(raw_uv, st, polar_power);",
        "textureSampleGrad(tex, samp, mapped.uv, mapped.ddx_uv, mapped.ddy_uv)",
    ] {
        assert!(
            material_sample.contains(required),
            "material/sample.wgsl must contain `{required}`"
        );
    }

    let fresnellerp = material_source("fresnellerp.wgsl")?;
    assert!(
        fresnellerp
            .contains("let mapped = uvu::polar_mapping(uv, mat._LerpTex_ST, mat._LerpPolarPow);"),
        "fresnellerp.wgsl must use Unity-style polar gradients for _LerpTex"
    );
    assert!(
        !fresnellerp
            .contains("uvu::apply_st(uvu::polar_uv(uv, mat._LerpPolarPow), mat._LerpTex_ST)"),
        "fresnellerp.wgsl must not sample polar _LerpTex without gradient repair"
    );

    Ok(())
}

#[test]
fn modern_furfx_roots_use_unity_property_names_and_noise_alpha() -> io::Result<()> {
    let modern = module_source("fur/modern.wgsl")?;
    for required in [
        "_BumpMap_ST: vec4<f32>",
        "_BumpMap_LodBias: f32",
        "var _BumpMap: texture_2d<f32>",
        "furc::alpha_clip(noise, mat._Cutoff);",
    ] {
        assert!(
            modern.contains(required),
            "fur/modern.wgsl must contain `{required}`"
        );
    }
    for forbidden in ["_NormalMap", "_EdgeFade", "classic_shell_alpha"] {
        assert!(
            !modern.contains(forbidden),
            "fur/modern.wgsl must not contain `{forbidden}`"
        );
    }

    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = source_file(&path)?;
        if !src.contains("renderide::fur::modern") {
            continue;
        }
        let uses_bump_map = src.contains("fur::fragment_base(input)");
        if uses_bump_map {
            assert!(
                src.contains("//#texture_default _BumpMap bump"),
                "{label} must declare Unity's _BumpMap texture fallback"
            );
        } else {
            assert!(
                !src.contains("//#texture_default _BumpMap bump"),
                "{label} must not declare an unused _BumpMap texture fallback"
            );
        }
        for forbidden in ["_NormalMap", "_EdgeFade"] {
            assert!(
                !src.contains(forbidden),
                "{label} must not contain `{forbidden}`"
            );
        }
    }

    Ok(())
}

#[test]
fn pbs_indirect_specular_energy_respects_zero_f0() -> io::Result<()> {
    let brdf = module_source("pbs/brdf.wgsl")?;

    for required in [
        "let f90 = vec3<f32>(f90_from_f0(clamped_f0));",
        "return clamped_f0 * (dfg.y - dfg.x) + f90 * dfg.x;",
    ] {
        assert!(
            brdf.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    assert!(
        !brdf.contains(
            "return mix(vec3<f32>(dfg.x), vec3<f32>(dfg.y), clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0)));"
        ),
        "pbs/brdf.wgsl must not use the old implicit f90=1 DFG formula"
    );
    Ok(())
}

#[test]
fn pbs_roughness_keeps_indirect_mirror_path_unclamped() -> io::Result<()> {
    let sampling_src = source_file(manifest_dir().join("shaders/modules/pbs/sampling.wgsl"))?;
    assert!(
        sampling_src.contains("return clamp(1.0 - smoothness, 0.0, 1.0);"),
        "PBS smoothness conversion must keep perceptual roughness at 0 for mirror-smooth indirect reflections"
    );
    assert!(
        !sampling_src.contains("return clamp(1.0 - smoothness, 0.045, 1.0);"),
        "PBS smoothness conversion must not apply the direct-light roughness floor globally"
    );

    let brdf_src = source_file(manifest_dir().join("shaders/modules/pbs/brdf.wgsl"))?;
    for required in [
        "const MIN_ALPHA: f32 = 0.002;",
        "fn direct_alpha_from_perceptual_roughness(",
        "return max(clamped * clamped, MIN_ALPHA);",
        "fn direct_perceptual_roughness(",
        "fn eval_direct_specular_lobe(",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    let lighting_src = source_file(manifest_dir().join("shaders/modules/pbs/lighting.wgsl"))?;
    for required in [
        "let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);",
        "let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);",
        "let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);",
        "let indirect_dfg = brdf::sample_ibl_dfg_lut(indirect_roughness, n_dot_v);",
    ] {
        assert!(
            lighting_src.contains(required),
            "pbs/lighting.wgsl must contain `{required}`"
        );
    }

    for path in wgsl_files_recursive("shaders/materials")? {
        let src = source_file(&path)?;
        for forbidden in [
            "clamp(1.0 - smoothness, 0.045, 1.0)",
            "clamp(1.0 - clamp(smoothness, 0.0, 1.0), 0.045, 1.0)",
        ] {
            assert!(
                !src.contains(forbidden),
                "{} must not contain the global PBS roughness floor `{forbidden}`",
                file_label(&path)
            );
        }
    }

    Ok(())
}

#[test]
fn pbs_direct_diffuse_uses_fresnel_transmission() -> io::Result<()> {
    let brdf_src = module_source("pbs/brdf.wgsl")?;
    for required in [
        "fn max_component(v: vec3<f32>) -> f32",
        "fn direct_diffuse_fresnel_transmission(f: vec3<f32>, f0: vec3<f32>) -> f32",
        "return clamp((1.0 - f_peak) / max(1.0 - f0_peak, 1e-4), 0.0, 1.0);",
        "* direct_diffuse_fresnel_transmission(direct_lobe.f, f0)",
        "* fd_burley(direct_lobe.n_dot_v, direct_lobe.n_dot_l, direct_lobe.l_dot_h, diffuse_roughness);",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    assert!(
        !brdf_src.contains("let fd = diffuse_color * fd_lambert();"),
        "PBS direct lighting must not bypass Fresnel diffuse transmission"
    );
    Ok(())
}

#[test]
fn pbs_direct_diffuse_uses_burley_rough_diffuse() -> io::Result<()> {
    let brdf_src = module_source("pbs/brdf.wgsl")?;
    for required in [
        "fn fd_burley(n_dot_v: f32, n_dot_l: f32, l_dot_h: f32, perceptual_roughness: f32) -> f32",
        "let fd90 = 0.5 + 2.0 * roughness * loh * loh;",
        "let light_scatter = f_schlick_scalar(1.0, fd90, n_dot_l);",
        "let view_scatter = f_schlick_scalar(1.0, fd90, n_dot_v);",
        "return light_scatter * view_scatter * (1.0 / PI);",
        "specular_roughness: f32,\n    diffuse_roughness: f32,",
        "let alpha = direct_alpha_from_perceptual_roughness(perceptual_roughness);",
        "fd_burley(direct_lobe.n_dot_v, direct_lobe.n_dot_l, direct_lobe.l_dot_h, diffuse_roughness)",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain rough-diffuse term `{required}`"
        );
    }

    let lighting_src = module_source("pbs/lighting.wgsl")?;
    for required in [
        "aa_roughness,\n                s.roughness,\n                s.metallic,",
        "aa_roughness,\n                s.roughness,\n                s.base_color,",
        "brdf::diffuse_only_metallic(light, world_pos, s.normal, view_dir, s.roughness, s.base_color, s.metallic)",
        "brdf::diffuse_only_specular(\n                light,\n                world_pos,\n                s.normal,\n                view_dir,\n                s.roughness,",
    ] {
        assert!(
            lighting_src.contains(required),
            "pbs/lighting.wgsl must keep direct specular AA roughness separate from diffuse roughness; missing `{required}`"
        );
    }

    Ok(())
}

#[test]
fn pbs_direct_specular_lobe_is_shared() -> io::Result<()> {
    let brdf_src = module_source("pbs/brdf.wgsl")?;
    for required in [
        "struct DirectSpecularEval",
        "fn eval_direct_specular_lobe(",
        "let alpha = direct_alpha_from_perceptual_roughness(perceptual_roughness);",
        "let f = f_schlick(f0, f90_from_f0(f0), v_dot_h);",
        "let d = d_ggx(n_dot_h, alpha);",
        "let vis = v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);",
        "let fr = max(vec3<f32>(0.0), (d * vis) * f * energy_compensation);",
        "let direct_lobe = eval_direct_specular_lobe(n, ls.l, v, specular_roughness, f0, energy_compensation);",
        "direct_lobe.specular_brdf",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must share direct specular lobe term `{required}`"
        );
    }
    Ok(())
}

#[test]
fn xiexe_pbs_and_fur_stay_on_shared_modern_brdf() -> io::Result<()> {
    for (module, required_terms) in [
        (
            "pbs/lighting.wgsl",
            [
                "#import renderide::pbs::brdf as brdf",
                "brdf::direct_radiance_metallic(",
                "brdf::direct_radiance_specular(",
                "brdf::indirect_specular_visibility(",
            ],
        ),
        (
            "xiexe/toon2/lighting.wgsl",
            [
                "#import renderide::pbs::brdf as brdf",
                "brdf::eval_direct_specular_lobe(",
                "brdf::fd_burley(",
                "brdf::indirect_specular_visibility(",
            ],
        ),
        (
            "fur/lighting.wgsl",
            [
                "#import renderide::pbs::brdf as brdf",
                "brdf::direct_radiance_specular(",
                "brdf::indirect_diffuse_specular(",
                "brdf::indirect_specular_visibility(",
            ],
        ),
    ] {
        let src = module_source(module)?;
        for required in required_terms {
            assert!(
                src.contains(required),
                "{module} must keep modern PBS BRDF term `{required}`"
            );
        }
    }

    let forbidden_terms = [
        "#import renderide::material::toon_brdf",
        "tbrdf::",
        "brdf::fd_lambert(",
        "let d_term = brdf::d_ggx(",
        "let v_term = brdf::v_smith_ggx_correlated(",
        "let f_term = brdf::f_schlick(",
        "d_term * v_term * f_term",
        "brdf::MIN_ALPHA",
        "let specular_occlusion = brdf::specular_ao_lagarde",
        "clamp(perceptual_roughness, 0.045, 1.0)",
        "clamp(s.roughness, 0.045, 1.0)",
        "clamp(1.0 - smoothness, 0.045, 1.0)",
    ];
    let mut offenders = Vec::new();

    for root in [
        "shaders/modules/pbs",
        "shaders/modules/xiexe",
        "shaders/modules/fur",
        "shaders/materials",
    ] {
        for path in wgsl_files_recursive(root)? {
            let label = file_label(&path);
            if label.ends_with("shaders/modules/pbs/brdf.wgsl") || !modern_brdf_family_label(&label)
            {
                continue;
            }

            let src = source_file(&path)?;
            for forbidden in forbidden_terms {
                if src.contains(forbidden) {
                    offenders.push(format!("{label}: {forbidden}"));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "Xiexe, PBS, and Fur shaders must not reintroduce older local BRDF paths:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Returns true when a source path belongs to the shader families that should use the modern PBS BRDF.
fn modern_brdf_family_label(label: &str) -> bool {
    label.starts_with("shaders/modules/pbs/")
        || label.starts_with("shaders/modules/xiexe/")
        || label.starts_with("shaders/modules/fur/")
        || label.starts_with("shaders/materials/pbs")
        || label.starts_with("shaders/materials/paintpbs")
        || label.starts_with("shaders/materials/xstoon")
        || label.starts_with("shaders/materials/furfx")
}

#[test]
fn classic_furfx_modules_keep_source_parity_details() -> io::Result<()> {
    let common = module_source("fur/common.wgsl")?;
    for required in [
        "@location(9) base_world_pos: vec3<f32>,",
        "let base_world_pos = mv::world_position(draw, pos).xyz;",
        "out.shell_noise_uv = uv0 + shell_noise_offset;",
        "out.base_world_pos = base_world_pos;",
        "var color = tex_rgb;\n    color = color - shadow_rgb * hair_coloring;\n    color = color - vec3<f32>(pow(1.0 - fur_multiplier, 4.0) * hair_shading);\n    color = color * tint_rgb;",
    ] {
        assert!(
            common.contains(required),
            "fur/common.wgsl must preserve classic FurFX parity detail `{required}`"
        );
    }
    assert!(
        !common.contains("out.shell_noise_uv = uvu::apply_st(uv0 + shell_noise_offset, noise_st);"),
        "classic FurFX shell shadow UVs must not apply _NoiseTex_ST"
    );

    for module in [
        "fur/classic_basic.wgsl",
        "fur/classic_advanced.wgsl",
        "fur/classic_selfshadow.wgsl",
    ] {
        let src = module_source(module)?;
        for required in [
            "input.base_world_pos",
            "furc::alpha_clip(1.0, mat._Cutoff);",
        ] {
            assert!(
                src.contains(required),
                "{module} must preserve classic FurFX parity detail `{required}`"
            );
        }
    }

    for module in ["fur/classic_advanced.wgsl", "fur/classic_selfshadow.wgsl"] {
        let src = module_source(module)?;
        assert!(
            src.contains("rg::view_dir_for_world_pos(input.base_world_pos, input.view_layer);"),
            "{module} must evaluate rim/reflection from the base mesh world position"
        );
    }

    Ok(())
}

#[test]
fn fur_lighting_uses_full_pbs_brdf_stack() -> io::Result<()> {
    let fur_lighting = module_source("fur/lighting.wgsl")?;
    for required in [
        "let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);",
        "direct = direct + brdf::direct_radiance_specular(",
        "aa_roughness,\n                s.roughness,",
        "let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);",
        "let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);",
        "let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, f0);",
        "rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled)",
        "let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);",
        "brdf::indirect_specular_energy_from_dfg(indirect_dfg, f0, indirect_specular_enabled)",
        "brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, f0)",
        "let ambient = brdf::indirect_diffuse_specular(",
        "let indirect_specular = rprobe::indirect_specular_with_energy(",
        "specular_energy * specular_visibility",
    ] {
        assert!(
            fur_lighting.contains(required),
            "fur/lighting.wgsl must use PBS BRDF feature `{required}`"
        );
    }
    assert!(
        !fur_lighting.contains("let specular_occlusion = brdf::specular_ao_lagarde"),
        "Fur lighting must route specular AO through PBS multi-bounce visibility"
    );

    let fur_common = module_source("fur/common.wgsl")?;
    assert!(
        fur_common.contains("return clamp(sqrt(2.0 / (max(shininess, 0.0) + 2.0)), 0.0, 1.0);"),
        "Fur shininess conversion must keep mirror-smooth indirect roughness available"
    );
    assert!(
        !fur_common.contains(", 0.02, 1.0)"),
        "Fur shininess conversion must not bake in a direct-light roughness floor"
    );

    for path in wgsl_files_recursive("shaders/modules/fur")? {
        let label = file_label(&path);
        if label.ends_with("fur/lighting.wgsl") || label.ends_with("fur/common.wgsl") {
            continue;
        }
        let src = source_file(&path)?;
        if !src.contains("psurf::specular_with_geometric_normal(") {
            continue;
        }
        assert!(
            src.contains("furl::shade_specular_clustered("),
            "{label} must route core FurFX lighting through fur::lighting"
        );
        for forbidden in [
            "brdf::d_ggx",
            "brdf::v_smith_ggx_correlated",
            "brdf::f_schlick",
            "rprobe::indirect_specular_with_energy",
            "brdf::specular_ao_lagarde",
        ] {
            assert!(
                !src.contains(forbidden),
                "{label} must not duplicate PBS BRDF feature `{forbidden}`"
            );
        }
    }

    Ok(())
}

#[test]
fn shared_pbs_surface_roots_pass_geometric_normals() -> io::Result<()> {
    let sampling = module_source("pbs/sampling.wgsl")?;
    assert!(
        sampling.contains(
            "fn two_sided_geometric_normal(world_n: vec3<f32>, front_facing: bool) -> vec3<f32>"
        ),
        "pbs/sampling.wgsl must expose a two-sided geometric normal helper for horizon occlusion"
    );

    let mut offenders = Vec::new();
    for root in ["shaders/materials", "shaders/modules/fur"] {
        for path in wgsl_files_recursive(root)? {
            let src = source_file(&path)?;
            if !src.contains("renderide::pbs::surface as psurf") {
                continue;
            }
            if src.contains("psurf::metallic(") || src.contains("psurf::specular(") {
                offenders.push(file_label(&path));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "shared PBS surface roots must pass a separate geometric normal for reflection-probe horizon occlusion:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

#[test]
fn pbs_indirect_ao_uses_multibounce_visibility() -> io::Result<()> {
    let brdf_src = module_source("pbs/brdf.wgsl")?;
    for required in [
        "fn multi_bounce_visibility(visibility: f32, albedo: vec3<f32>) -> vec3<f32>",
        "let a = 2.0404 * clamped_albedo - vec3<f32>(0.3324);",
        "let b = -4.7951 * clamped_albedo + vec3<f32>(0.6417);",
        "let c = 2.7552 * clamped_albedo + vec3<f32>(0.6903);",
        "fn indirect_diffuse_visibility(visibility: f32, diffuse_color: vec3<f32>) -> vec3<f32>",
        "return multi_bounce_visibility(visibility, diffuse_color);",
        "fn indirect_specular_visibility(",
        "let single_bounce = specular_ao_lagarde(n_dot_v, visibility, perceptual_roughness);",
        "return multi_bounce_visibility(single_bounce, f0);",
        "let visibility = indirect_diffuse_visibility(occlusion, diffuse_color);",
        "return ambient * diffuse_color * energy_scale * visibility;",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    let lighting_src = module_source("pbs/lighting.wgsl")?;
    for required in [
        "let specular_visibility =\n        brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, specular_color);",
        "let specular_visibility =\n        brdf::indirect_specular_visibility(n_dot_v, s.occlusion, indirect_roughness, s.specular_color);",
        "specular_energy * specular_visibility",
    ] {
        assert!(
            lighting_src.contains(required),
            "pbs/lighting.wgsl must contain `{required}`"
        );
    }
    assert!(
        !lighting_src.contains("let specular_occlusion = brdf::specular_ao_lagarde"),
        "PBS clustered lighting should route specular AO through multi-bounce visibility"
    );
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

/// Verifies PBSLerp roots use their source-authored alpha clip property names and defaults.
#[test]
fn pbs_lerp_uses_alpha_clip_property_and_defaults() -> io::Result<()> {
    let metallic = material_source("pbslerp.wgsl")?;
    for required in [
        "//#mat_default _AlphaClip float 0.5",
        "//#mat_default _Glossiness1 float 0.5",
        "_AlphaClip: f32",
        "c.a < mat._AlphaClip",
    ] {
        assert!(
            metallic.contains(required),
            "pbslerp.wgsl must contain `{required}`"
        );
    }
    for rejected in ["_Cutoff: f32", "mat._Cutoff", "c.a <= mat._AlphaClip"] {
        assert!(
            !metallic.contains(rejected),
            "pbslerp.wgsl must not contain `{rejected}`"
        );
    }

    let specular = material_source("pbslerpspecular.wgsl")?;
    for required in [
        "//#mat_default _AlphaClip float 0.5",
        "_AlphaClip: f32",
        "c.a < mat._AlphaClip",
    ] {
        assert!(
            specular.contains(required),
            "pbslerpspecular.wgsl must contain `{required}`"
        );
    }
    for rejected in ["_Cutoff: f32", "mat._Cutoff", "c.a <= mat._AlphaClip"] {
        assert!(
            !specular.contains(rejected),
            "pbslerpspecular.wgsl must not contain `{rejected}`"
        );
    }

    Ok(())
}

/// Verifies PBSMetallic keeps Unity's secondary texture UV selector.
#[test]
fn pbs_metallic_uses_uvsec_for_detail_uvs() -> io::Result<()> {
    let src = material_source("pbsmetallic.wgsl")?;
    for required in [
        "//#mat_default _UVSec float 0.0",
        "_UVSec: f32",
        "@location(5) uv1: vec2<f32>",
        "pstd::detail_uv(uv0, uv1, mat._UVSec, mat._DetailAlbedoMap_ST)",
        "mv::world_uv2_vertex_main(instance_index, view_idx, pos, n, t, uv0, uv1)",
        "mv::world_uv2_vertex_main(instance_index, 0u, pos, n, t, uv0, uv1)",
    ] {
        assert!(
            src.contains(required),
            "pbsmetallic.wgsl must contain `{required}`"
        );
    }
    assert!(
        !src.contains("pstd::detail_uv(uv0, uv1, 0.0"),
        "pbsmetallic.wgsl must not hard-code UV0 for secondary textures"
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

/// Verifies alpha clip paths in this material slice preserve Unity `clip(x)` equality behavior.
#[test]
fn pbs_materials_81_to_90_use_strict_alpha_clip_thresholds() -> io::Result<()> {
    for material in [
        "pbslerp.wgsl",
        "pbslerpspecular.wgsl",
        "pbsmultiuv.wgsl",
        "pbsmultiuvspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("c.a < mat._AlphaClip"),
            "{material} must discard only when alpha is below `_AlphaClip`"
        );
        assert!(
            !src.contains("c.a <= mat._AlphaClip"),
            "{material} must not reject alpha exactly equal to `_AlphaClip`"
        );
        assert!(
            !src.contains("c.a <= mat._Cutoff"),
            "{material} must not use the wrong alpha clip property"
        );
    }

    let standard = module_source("pbs/standard.wgsl")?;
    assert!(
        standard.contains("if (enabled && alpha < cutoff)"),
        "pbs/standard.wgsl must match Unity `clip(alpha - cutoff)` equality behavior"
    );
    assert!(
        !standard.contains("if (enabled && alpha <= cutoff)"),
        "pbs/standard.wgsl must not reject alpha exactly equal to cutoff"
    );
    Ok(())
}

/// Material roots using the shared PBS lighting module should not also carry their own clustered loop.
#[test]
fn shared_pbs_lighting_roots_do_not_duplicate_clustered_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = source_file(&path)?;
        if !src.contains("renderide::pbs::lighting") {
            continue;
        }

        for forbidden in [
            "#import renderide::ibl::sh2_ambient",
            "#import renderide::pbs::brdf",
            "#import renderide::pbs::cluster",
            "fn clustered_direct_lighting",
            "pcls::cluster_id_from_frag",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!(
                    "{} still contains `{forbidden}`",
                    file_label(&path)
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "materials importing renderide::pbs::lighting must delegate clustered PBS lighting:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

#[test]
fn pbs_dualsided_emission_is_keyword_gated_only() -> io::Result<()> {
    for file_name in ["pbsdualsidedspecular.wgsl", "pbsdualsidedtransparent.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("var emission = mat._EmissionColor.rgb;"),
            "{file_name} must start emission from _EmissionColor"
        );
        assert!(
            !src.contains("dot(emission_color, emission_color)"),
            "{file_name} must not skip emission by inspecting _EmissionColor at runtime"
        );
    }
    Ok(())
}

/// Standard PBS parallax must project the view vector into the material's tangent frame before
/// offsetting UVs so height maps behave consistently as lighting and camera state become active.

#[test]
fn pbs_standard_parallax_uses_tangent_space_view_dir() -> io::Result<()> {
    let module_src = source_file(manifest_dir().join("shaders/modules/pbs/parallax.wgsl"))?;
    for required in [
        "#define_import_path renderide::pbs::parallax",
        "rg::view_dir_for_world_pos(world_pos, view_layer)",
        "pnorm::orthonormal_tbn(world_n, world_t)",
        "dot(world_view, tbn[0])",
        "dot(world_view, tbn[1])",
        "dot(world_view, tbn[2])",
        "UNITY_PARALLAX_VIEW_Z_BIAS: f32 = 0.42",
        "height_sample * height_scale - height_scale * 0.5",
    ] {
        assert!(
            module_src.contains(required),
            "parallax module should contain `{required}`"
        );
    }

    let standard_src = module_source("pbs/standard.wgsl")?;
    for required in [
        "#import renderide::pbs::parallax as ppar",
        "ts::sample_tex_2d(parallax_map, parallax_sampler, uv, parallax_lod_bias).g",
        "ppar::unity_parallax_offset(h, parallax, world_pos, world_n, world_t, view_layer)",
    ] {
        assert!(
            standard_src.contains(required),
            "pbs/standard.wgsl should contain `{required}`"
        );
    }

    for file_name in ["pbsmetallic.wgsl", "pbsspecular.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("#import renderide::pbs::standard as pstd")
                && src.contains("pstd::apply_parallax("),
            "{file_name} should route Standard parallax through the shared standard module"
        );

        for forbidden in [
            "view_dir.xy / max(abs(view_dir.z), 0.25)",
            "rg::view_dir_for_world_pos(world_pos, view_layer)",
            "fn uv_with_parallax(",
        ] {
            assert!(
                !src.contains(forbidden),
                "{file_name} should not contain the old world-space parallax expression `{forbidden}`"
            );
        }
    }

    Ok(())
}

/// Standard PBS-like roots should delegate clustered Standard lighting to the shared PBS module.
#[test]
fn standard_material_roots_do_not_duplicate_clustered_pbs_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = source_file(&path)?;
        if label == "shaders/materials/toonstandard.wgsl"
            || label == "shaders/materials/toonwater.wgsl"
        {
            continue;
        }

        for forbidden in [
            "#import renderide::ibl::sh2_ambient",
            "#import renderide::pbs::cluster",
            "cluster_id_from_frag",
            "direct_radiance_metallic",
            "direct_radiance_specular",
            "indirect_diffuse_metallic",
            "indirect_diffuse_specular",
            "indirect_specular",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "standard PBS-like material roots must delegate clustered PBS lighting:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}
