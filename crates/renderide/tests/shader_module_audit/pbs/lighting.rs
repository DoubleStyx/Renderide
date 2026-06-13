//! PBS lighting and BRDF source audits.

use super::super::*;

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
fn birp_range_fade_uses_sextic_smoothing() -> io::Result<()> {
    let birp = module_source("lighting/birp.wgsl")?;
    assert!(
        birp.contains(
            "fn range_fade(t: f32) -> f32 {\n    let t2 = t * t;\n    return squared_edge_fade(t2 * t2 * t2);\n}"
        ),
        "BiRP punctual range smoothing must use a sextic fade input"
    );

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
            && birp.contains("return squared_edge_fade(r2 * r2 * r2);")
            && !birp.contains("return clamp(1.0 - r2, 0.0, 1.0);")
            && !birp.contains("return squared_edge_fade(r2);"),
        "spot angle attenuation must use the cubic projected radial Unity BiRP-style curve"
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
        "fn filter_perceptual_roughness(\n    perceptual_roughness: f32,\n    shading_normal: vec3<f32>,\n    geometric_normal: vec3<f32>,",
        "let normal_map_variance = max(shading_variance - geometric_variance, 0.0);",
        "let normal_filtered = filter_normal_map_perceptual_roughness(perceptual_roughness, normal_map_variance);",
        "return filter_geometric_perceptual_roughness(normal_filtered, geometric_variance);",
        "fn eval_direct_specular_lobe(",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    let lighting_src = source_file(manifest_dir().join("shaders/modules/pbs/lighting.wgsl"))?;
    for required in [
        "fn direct_energy_compensation(",
        "let direct_roughness = brdf::direct_perceptual_roughness(perceptual_roughness);",
        "let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);",
        "let filtered_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal, s.geometric_normal);",
        "fn indirect_specular_energy(",
        "let indirect_dfg = brdf::sample_ibl_dfg_lut(perceptual_roughness, n_dot_v);",
    ] {
        assert!(
            lighting_src.contains(required),
            "pbs/lighting.wgsl must contain `{required}`"
        );
    }
    assert!(
        !lighting_src.contains("filter_perceptual_roughness(s.roughness, s.geometric_normal)"),
        "PBS specular AA must pass both shading and geometric normals"
    );

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
fn pbs_shifts_view_normals_for_specular_brdf_and_ibl() -> io::Result<()> {
    let brdf_src = module_source("pbs/brdf.wgsl")?;
    for required in [
        "const MIN_VIEW_FACING_N_DOT_V: f32 = 1e-4;",
        "fn view_facing_normal(n: vec3<f32>, v: vec3<f32>) -> vec3<f32>",
        "return safe_normalize(unit_n + unit_v * (MIN_VIEW_FACING_N_DOT_V - n_dot_v), unit_v);",
        "let brdf_n = view_facing_normal(n, view);",
        "let n_dot_v = max(dot(brdf_n, view), MIN_VIEW_FACING_N_DOT_V);",
        "let n_dot_h = clamp(dot(brdf_n, h), 0.0, 1.0);",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain view-facing normal term `{required}`"
        );
    }
    assert!(
        !brdf_src.contains("let n_dot_v = max(dot(n, v), 1e-4);"),
        "direct PBS BRDF evaluation must not clamp a back-facing shading normal to epsilon"
    );

    let lighting_src = module_source("pbs/lighting.wgsl")?;
    assert_eq!(
        lighting_src
            .matches("let specular_normal = brdf::view_facing_normal(s.normal, view_dir);")
            .count(),
        4,
        "PBS opaque and transparent metallic/specular paths must shift the specular normal"
    );
    assert_eq!(
        lighting_src
            .matches("let n_dot_v = clamp(dot(specular_normal, view_dir), 0.0, 1.0);")
            .count(),
        4,
        "PBS DFG and specular AO lookup must use the shifted specular normal"
    );
    assert_eq!(
        lighting_src
            .matches("world_pos,\n        specular_normal,\n        s.geometric_normal,")
            .count(),
        4,
        "PBS reflection-probe specular direction must use the shifted specular normal"
    );
    assert_eq!(
        lighting_src
            .matches("rprobe::indirect_diffuse(world_pos, s.normal, view_layer, options.include_directional)")
            .count(),
        4,
        "PBS diffuse ambient should remain on the original shading normal"
    );

    Ok(())
}

#[test]
fn standard_pbs_detail_normals_use_tangent_space_blend() -> io::Result<()> {
    let standard = module_source("pbs/standard.wgsl")?;
    for required in [
        "let detail_blend = normalize(vec3<f32>(ts_n.xy + ts_detail.xy, ts_n.z * ts_detail.z));",
        "ts_n = mix(ts_n, detail_blend, clamp(detail_mask, 0.0, 1.0));",
    ] {
        assert!(
            standard.contains(required),
            "pbs/standard.wgsl must blend detail normals with `{required}`"
        );
    }
    assert!(
        !standard.contains("ts_n.xy + ts_detail.xy * detail_mask, ts_n.z"),
        "Standard PBS detail normals must not mask only the XY channels"
    );
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
        "filtered_roughness,\n                s.roughness,\n                s.metallic,",
        "filtered_roughness,\n                s.roughness,\n                s.base_color,",
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
        "fn indirect_specular_visibility(",
        "return brdf::indirect_specular_visibility(n_dot_v, occlusion, perceptual_roughness, f0);",
        "let specular_visibility = indirect_specular_visibility(",
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
