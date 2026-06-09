//! Fur, Toon, and Xiexe audits that share the PBS lighting stack.

use super::super::*;
use super::helpers::modern_brdf_family_label;

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
fn furfx_shell_force_uses_projection_local_offset() -> io::Result<()> {
    let common = module_source("fur/common.wgsl")?;
    for required in [
        "#import renderide::frame::globals as rg",
        "fn projected_local_force(force_local: vec4<f32>, view_idx: u32) -> vec3<f32>",
        "let clamped_force = clamp(force_local, vec4<f32>(-1.0), vec4<f32>(1.0));",
        "return (rg::projection_for_view(view_idx) * clamped_force).xyz;",
        "let shell_offset = n.xyz * fur_length * fur_multiplier * hair_hardness;",
        "let local_force_offset = projected_local_force(force_local, view_idx) * force_scale;",
        "let shell_model_pos = pos.xyz + shell_offset + local_force_offset;",
        "let shell_pos = vec4<f32>(shell_model_pos, pos.w);",
        "let global_force_offset = global_force * force_scale;",
        "let world_p = shell_world_pos + global_force_offset;",
    ] {
        assert!(
            common.contains(required),
            "fur/common.wgsl must preserve FurFX shell force detail `{required}`"
        );
    }

    for forbidden in [
        "mv::model_vector(draw, clamp(force_local.xyz",
        "let force_offset = (global_force + local_force)",
        "let shell_pos = vec4<f32>(pos.xyz + shell_offset, pos.w);",
    ] {
        assert!(
            !common.contains(forbidden),
            "fur/common.wgsl must not keep old shell force path `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn furfx_roots_do_not_use_shader_variant_bits() -> io::Result<()> {
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        if !label.contains("/furfx") {
            continue;
        }
        let src = source_file(&path)?;
        assert!(
            !src.contains("_RenderideVariantBits"),
            "{label} must not decode shader variant bits for FurFX shell behavior"
        );
    }

    Ok(())
}

#[test]
fn fur_lighting_uses_full_pbs_brdf_stack() -> io::Result<()> {
    let fur_lighting = module_source("fur/lighting.wgsl")?;
    for required in [
        "let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.geometric_normal);",
        "direct = direct + brdf::direct_radiance_specular(",
        "aa_roughness,\n                s.roughness,",
        "let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);",
        "let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);",
        "let energy_compensation = brdf::energy_compensation_from_dfg(direct_dfg, f0);",
        "rprobe::has_indirect_specular(view_layer, options.glossy_reflections_enabled)",
        "let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.geometric_normal);",
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
    assert!(
        !fur_lighting.contains("filter_perceptual_roughness(s.roughness, s.normal)"),
        "Fur specular AA must derive roughness from geometric normals, not normal-map-perturbed shading normals"
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
