//! Shader source audits for this behavior family.

use super::*;

const XSTOON2_MATERIAL_ROOTS: &[&str] = &[
    "xstoon2.0.wgsl",
    "xstoon2.0-cutout.wgsl",
    "xstoon2.0-cutouta2c.wgsl",
    "xstoon2.0-cutouta2c-outlined.wgsl",
    "xstoon2.0-cutouta2cmasked.wgsl",
    "xstoon2.0-dithered.wgsl",
    "xstoon2.0-dithered-outlined.wgsl",
    "xstoon2.0-fade.wgsl",
    "xstoon2.0-outlined.wgsl",
    "xstoon2.0-transparent.wgsl",
    "xstoon2.0_outlined.wgsl",
];

const XSTOON2_OUTLINED_MATERIAL_ROOTS: &[&str] = &[
    "xstoon2.0-cutouta2c-outlined.wgsl",
    "xstoon2.0-dithered-outlined.wgsl",
    "xstoon2.0-outlined.wgsl",
    "xstoon2.0_outlined.wgsl",
];

const XIEXE_NON_CORE_EXTENSION_IDENTIFIERS: &[&str] = &[
    "_DetailNormalMap",
    "_DetailMask",
    "_DetailNormalMapScale",
    "_ReflectionMode",
    "_ClearCoat",
    "_ReflectionBlendMode",
    "_BakedCubemap",
    "_ClearcoatStrength",
    "_ClearcoatSmoothness",
    "_ScaleWithLight",
    "_EmissionToDiffuse",
    "_ScaleWithLightSensitivity",
    "_SpecMode",
    "_SpecularStyle",
    "_SpecularMap",
    "_AnisotropicAX",
    "_AnisotropicAY",
    "_HalftoneDotSize",
    "_HalftoneDotAmount",
    "_HalftoneLineAmount",
    "_UVSetDetNormal",
    "_UVSetDetMask",
    "_UVSetSpecular",
    "_AdvMode",
    "_TilingMode",
];

#[test]
fn xiexe_transparent_keeps_premultiplied_transparent_pass_directive() -> io::Result<()> {
    let src = material_source("xstoon2.0-transparent.wgsl")?;
    assert!(
        src.contains("//#pass type=forward name=forward_premultiplied_transparent"),
        "xstoon2.0-transparent.wgsl must use the source-authored premultiplied transparent pass"
    );
    assert!(
        !src.contains("//#pass type=forward\n"),
        "xstoon2.0-transparent.wgsl must not alias the opaque forward pass"
    );
    let main_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/main.wgsl"))?;
    assert!(
        !main_src.contains("rgb = rgb * alpha"),
        "XSToon transparent must not premultiply the entire lit RGB result like straight alpha"
    );
    Ok(())
}

#[test]
fn xiexe_matcap_uses_stereo_center_view_dir() -> io::Result<()> {
    let globals_src = source_file(manifest_dir().join("shaders/modules/frame/globals.wgsl"))?;
    assert!(
        globals_src.contains("fn stereo_center_view_dir_for_world_pos("),
        "globals.wgsl must expose a stereo-center view direction helper for eye-stable effects"
    );
    assert!(
        globals_src
            .contains("(frame.camera_world_pos.xyz + frame.camera_world_pos_right.xyz) * 0.5"),
        "stereo-center view direction must average the left and right camera positions in multiview"
    );

    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;
    assert!(
        lighting_src.contains(
            "let stereo_view_dir = rg::stereo_center_view_dir_for_world_pos(world_pos, view_layer);"
        ),
        "Xiexe matcap sampling must derive its view direction from the stereo-center camera"
    );
    assert!(
        lighting_src.contains("let uv = matcap_uv(stereo_view_dir, normal);"),
        "Xiexe matcap sampling must use the stereo-center view direction for matcap UVs"
    );
    assert!(
        !lighting_src.contains("let uv = matcap_uv(view_dir, normal);"),
        "Xiexe matcap UVs must not use the per-eye lighting view direction"
    );
    assert!(
        lighting_src.contains("spec = spec * (ambient + dominant_light_col_atten * 0.5);"),
        "Xiexe matcaps must receive the host-compatible light-scaling term"
    );
    for forbidden in [
        "reflection_is_multiplicative",
        "baked_cubemap_enabled",
        "reflection_disabled",
        "_ReflectionBlendMode",
        "_BakedCubemap",
        "apply_reflection_blend",
        "reflection_blend_weight",
    ] {
        assert!(
            !lighting_src.contains(forbidden),
            "Xiexe lighting must not retain non-core extension `{forbidden}`"
        );
    }
    Ok(())
}

#[test]
fn xiexe_sources_exclude_non_core_extension_identifiers() -> io::Result<()> {
    for material in XSTOON2_MATERIAL_ROOTS
        .iter()
        .copied()
        .chain(["xstoonstenciler.wgsl"])
    {
        let src = material_source(material)?;
        for forbidden in XIEXE_NON_CORE_EXTENSION_IDENTIFIERS {
            assert!(
                !src.contains(forbidden),
                "{material} must not declare non-core extension `{forbidden}`"
            );
        }
    }

    for module in [
        "xiexe/toon2/base.wgsl",
        "xiexe/toon2/surface.wgsl",
        "xiexe/toon2/alpha.wgsl",
        "xiexe/toon2/lighting.wgsl",
        "xiexe/toon2/outline.wgsl",
        "xiexe/toon2/variant_bits.wgsl",
        "xiexe/toon2/main.wgsl",
    ] {
        let src = module_source(module)?;
        for forbidden in XIEXE_NON_CORE_EXTENSION_IDENTIFIERS {
            assert!(
                !src.contains(forbidden),
                "{module} must not retain non-core extension `{forbidden}`"
            );
        }
    }

    Ok(())
}

#[test]
fn xiexe_primary_direct_specular_uses_ggx_pbr_core() -> io::Result<()> {
    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;

    for required in [
        "let specular_reflectance = brdf::metallic_f0(s.diffuse_color, s.metallic);",
        "let roughness = clamp(1.0 - remap_specular_area(xb::mat._SpecularArea), 0.0, 1.0);",
        "let aa_roughness = brdf::filter_perceptual_roughness(roughness, s.normal);",
        "fn primary_direct_specular_terms(s: xb::SurfaceData, view_dir: vec3<f32>) -> DirectSpecularTerms {",
        "let direct_roughness = brdf::direct_perceptual_roughness(roughness);",
        "let dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);",
        "let energy_compensation = brdf::energy_compensation_from_dfg(dfg, specular_reflectance);",
        "fn direct_specular_ggx(",
        "let direct_lobe = brdf::eval_direct_specular_lobe(",
        "var specular = direct_lobe.specular_brdf;",
        "let radiance = light.color * light.attenuation * direct_lobe.n_dot_l;",
        "terms.aa_roughness",
        "max(0.0, xb::mat._SpecularIntensity)",
        "xb::mat._SpecularAlbedoTint",
        "clamp(albedo_tint, 0.0, 1.0)",
    ] {
        assert!(
            lighting_src.contains(required),
            "Xiexe primary direct specular must use PBS GGX term `{required}`"
        );
    }

    for forbidden in [
        "fn direct_specular_xstoon2(",
        "fn xiexe_specular_reflectance(",
        "fn primary_specular_roughness(",
        "0.16 * reflectivity * reflectivity",
        "exp2((-5.55473 * ldh) - (6.98316 * ldh))",
        "let reflection = v_term * d_term * 3.14159265;",
        "let alpha = max(perceptual_roughness * perceptual_roughness, brdf::MIN_ALPHA);",
        "let d_term = brdf::d_ggx(",
        "let v_term = brdf::v_smith_ggx_correlated(",
        "let f_term = brdf::f_schlick(",
        "d_term * v_term * f_term * energy_compensation",
        "smooth_specular",
        "xb::mat._SpecularIntensity * 0.001",
        "s.specular_mask",
        "clearcoat_direct_specular",
        "clearcoat_roughness",
    ] {
        assert!(
            !lighting_src.contains(forbidden),
            "Xiexe primary direct specular must not contain `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn xiexe_roots_declare_unity_defaults_for_nonzero_core_fields() -> io::Result<()> {
    for material in XSTOON2_MATERIAL_ROOTS.iter().copied() {
        let src = material_source(material)?;
        for directive in [
            "//#mat_default _RimCubemapTint float 0.0",
            "//#mat_default _SpecularAlbedoTint float 1.0",
            "//#mat_default _Saturation float 1.0",
            "//#mat_default _BumpScale float 1.0",
            "//#mat_default _Reflectivity float 1.0",
            "//#mat_default _RimAttenEffect float 1.0",
            "//#mat_default _RimRange float 0.7",
            "//#mat_default _RimThreshold float 0.1",
            "//#mat_default _RimSharpness float 0.1",
            "//#mat_default _SpecularArea float 0.5",
            "//#mat_default _ShadowSharpness float 0.5",
            "//#mat_default _ShadowRimRange float 0.7",
            "//#mat_default _ShadowRimThreshold float 0.1",
            "//#mat_default _ShadowRimSharpness float 0.3",
            "//#mat_default _OutlineWidth float 1.0",
            "//#mat_default _SSDistortion float 1.0",
            "//#mat_default _SSPower float 1.0",
            "//#mat_default _SSScale float 1.0",
        ] {
            assert!(
                src.contains(directive),
                "{material} must declare Unity default `{directive}`"
            );
        }
    }
    Ok(())
}

#[test]
fn xiexe_outlined_roots_declare_outline_mask_texture_default() -> io::Result<()> {
    for material in XSTOON2_OUTLINED_MATERIAL_ROOTS.iter().copied() {
        let src = material_source(material)?;
        assert!(
            src.contains("//#texture_default _OutlineMask white"),
            "{material} must keep the Unity white fallback for `_OutlineMask`"
        );
    }
    Ok(())
}

#[test]
fn xiexe_pbr_reflections_use_pbs_probe_energy_terms() -> io::Result<()> {
    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;

    for required in [
        "return rprobe::indirect_diffuse(world_pos, s.normal, view_layer, true);",
        "let indirect_enabled = rprobe::has_indirect_specular(view_layer, xvb::reflection_uses_pbr_for_layout(keyword_layout));",
        "let roughness = brdf::filter_perceptual_roughness(clamp(perceptual_roughness, 0.0, 1.0), normal);",
        "let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);",
        "let specular_energy = brdf::indirect_specular_energy_from_dfg(dfg, specular_reflectance, indirect_enabled);",
        "let specular_visibility =\n        brdf::indirect_specular_visibility(n_dot_v, occlusion_scalar(s), roughness, specular_reflectance);",
        "let spec = rprobe::indirect_specular_with_energy(",
        "specular_energy * specular_visibility",
        "let specular_reflectance = brdf::metallic_f0(s.diffuse_color, s.metallic);",
        "spec = mix(spec, spec * dominant_ramp, roughness);",
        "col + reflection * clamp(s.reflectivity_mask, 0.0, 1.0)",
    ] {
        assert!(
            lighting_src.contains(required),
            "Xiexe PBR reflections must contain `{required}`"
        );
    }

    assert!(
        !lighting_src.contains("xiexe_specular_reflectance"),
        "Indirect specular must not call the removed `xiexe_specular_reflectance` helper"
    );
    assert!(
        !lighting_src.contains("let specular_occlusion = brdf::specular_ao_lagarde"),
        "Xiexe PBR reflections must route specular AO through PBS multi-bounce visibility"
    );
    assert!(
        !lighting_src.contains("clamp(perceptual_roughness, 0.045, 1.0)")
            && !lighting_src.contains("clamp(s.roughness, 0.045, 1.0)"),
        "Xiexe indirect reflections must not apply the direct-light roughness floor"
    );

    let pbr_branch_pos = lighting_src
        .find("let indirect_enabled = rprobe::has_indirect_specular(view_layer, xvb::reflection_uses_pbr_for_layout(keyword_layout));")
        .expect("Xiexe PBR reflection branch must query probe availability");
    let pbr_return_pos = lighting_src[pbr_branch_pos..]
        .find("return spec;")
        .map(|offset| pbr_branch_pos + offset)
        .expect("Xiexe PBR reflection branch must return its specular result");
    let pbr_branch = &lighting_src[pbr_branch_pos..pbr_return_pos];
    assert!(
        !pbr_branch.contains("raw_indirect_specular"),
        "Xiexe PBR reflection branch must not multiply raw probe radiance by hand-rolled Fresnel"
    );

    Ok(())
}

#[test]
fn reflection_probe_specular_samples_manual_cubemap_array_atlas() -> io::Result<()> {
    let probe_src = module_source("lighting/reflection_probes.wgsl")?;

    for required in [
        "#import renderide::ibl::cubemap_filter as cube_filter",
        "cube_filter::sample_trilinear_base(",
        "atlas_index * 6u,",
        "let sample_dir = box_project_dir(probe, world_pos, dir, perceptual_roughness);",
    ] {
        assert!(
            probe_src.contains(required),
            "lighting/reflection_probes.wgsl must contain `{required}`"
        );
    }
    assert!(
        !probe_src.contains("REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED")
            && !probe_src.contains("textureSampleLevel(\n        rg::reflection_probe_specular"),
        "reflection probe specular sampling must avoid hard-coded atlas V inversion and hardware cube sampling"
    );

    Ok(())
}

#[test]
fn reflection_probe_specular_applies_horizon_occlusion() -> io::Result<()> {
    let probe_src = module_source("lighting/reflection_probes.wgsl")?;
    for required in [
        "fn horizon_specular_occlusion(",
        "let dir = dominant_reflection_dir(n, v, perceptual_roughness);",
        "let base_n = horizon_normal(n, geometric_n);",
        "let horizon = clamp(1.0 + dot(dir, base_n), 0.0, 1.0);",
        "return horizon * horizon;",
        "let horizon_occlusion = horizon_specular_occlusion(n, geometric_n, v, perceptual_roughness);",
        "return radiance * specular_energy * clamp(specular_occlusion, 0.0, 1.0) * horizon_occlusion;",
    ] {
        assert!(
            probe_src.contains(required),
            "reflection probes must apply horizon specular occlusion; missing `{required}`"
        );
    }

    let pbs_lighting = module_source("pbs/lighting.wgsl")?;
    assert!(
        pbs_lighting.contains("s.normal,\n        s.geometric_normal,\n        view_dir,"),
        "PBS indirect specular must pass the base normal into reflection-probe horizon occlusion"
    );

    let xiexe_lighting = module_source("xiexe/toon2/lighting.wgsl")?;
    assert!(
        xiexe_lighting.contains("normal,\n        s.raw_normal,\n        view_dir,"),
        "Xiexe indirect specular must pass the raw surface normal into reflection-probe horizon occlusion"
    );
    assert!(
        xiexe_lighting.contains("let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);")
            && xiexe_lighting.contains("rprobe::raw_indirect_specular_with_horizon(world_pos, s.normal, s.raw_normal, view_dir, indirect_roughness, true, view_layer)"),
        "Xiexe environment tint must use filtered roughness and horizon-occluded raw probe radiance"
    );

    Ok(())
}

#[test]
fn xiexe_indirect_diffuse_uses_pbs_energy_split() -> io::Result<()> {
    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;
    for required in [
        "let indirect_specular_reflectance = brdf::metallic_f0(s.diffuse_color, s.metallic);",
        "let indirect_specular_energy = brdf::indirect_specular_energy_from_dfg(",
        "let indirect_diffuse_energy_scale =\n        brdf::indirect_diffuse_energy_scale(indirect_specular_energy, indirect_specular_enabled);",
        "fn indirect_diffuse_visibility(s: xb::SurfaceData) -> vec3<f32>",
        "let visibility = brdf::indirect_diffuse_visibility(scalar_occlusion, s.albedo.rgb);",
        "return min(vec3<f32>(1.0), colored_occlusion * visibility / vec3<f32>(scalar_occlusion));",
        "diffuse = diffuse + s.albedo.rgb * ambient * indirect_diffuse_energy_scale * indirect_diffuse_visibility(s);",
    ] {
        assert!(
            lighting_src.contains(required),
            "Indirect-diffuse path must use PBSMetallic-style energy split; missing `{required}`"
        );
    }
    assert!(
        !lighting_src.contains("diffuse = diffuse * s.occlusion;"),
        "The combined `diffuse * occlusion` step must be removed; AO now modulates indirect diffuse only"
    );
    assert!(
        !lighting_src.contains("diffuse = diffuse + s.albedo.rgb * ambient;"),
        "Indirect diffuse must include the PBSMetallic energy scale, not raw `albedo * ambient`"
    );
    Ok(())
}

#[test]
fn xiexe_direct_diffuse_uses_burley_with_ramp_tint() -> io::Result<()> {
    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;
    for required in [
        "fn direct_diffuse_fresnel_transmission(",
        "let f = brdf::f_schlick(",
        "return brdf::direct_diffuse_fresnel_transmission(f, specular_reflectance);",
        "fn direct_diffuse_brdf(",
        "return brdf::fd_burley(",
        "let diffuse_transmission = direct_diffuse_fresnel_transmission(",
        "let diffuse_brdf = direct_diffuse_brdf(s.normal, light.direction, view_dir, s.roughness);",
        "s.albedo.rgb * diffuse_transmission * diffuse_brdf * light.color * light.attenuation * ramp",
    ] {
        assert!(
            lighting_src.contains(required),
            "Per-light direct diffuse must preserve ramp styling inside PBS Fresnel transmission and Burley diffuse; missing `{required}`"
        );
    }
    assert!(
        !lighting_src.contains("s.albedo.rgb * ramp * light_col_atten"),
        "The legacy un-normalized direct-diffuse accumulator must be removed"
    );
    // Guard the regression: missing `light.attenuation` makes diffuse π× dimmer than
    // PBSMetallic and washes out the toon shadow ramp.
    assert!(
        !lighting_src.contains("s.albedo.rgb * brdf::fd_lambert() * light.color * ramp"),
        "Direct diffuse must include `light.attenuation` and must not use Lambert directly"
    );
    assert!(
        !lighting_src
            .contains("s.albedo.rgb * brdf::fd_lambert() * light.color * light.attenuation * ramp"),
        "Direct diffuse must not bypass the PBS Fresnel transmission envelope or Burley diffuse"
    );
    Ok(())
}

#[test]
fn xiexe_surface_keeps_indirect_roughness_unfloored() -> io::Result<()> {
    let surface_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/surface.wgsl"))?;
    assert!(
        surface_src.contains("roughness = clamp(roughness * (1.7 - 0.7 * roughness), 0.0, 1.0);"),
        "Xiexe surface roughness must allow mirror-smooth indirect reflections"
    );
    assert!(
        !surface_src
            .contains("roughness = clamp(roughness * (1.7 - 0.7 * roughness), 0.045, 1.0);"),
        "Xiexe surface roughness must not bake in the direct-light roughness floor"
    );

    let base_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/base.wgsl"))?;
    assert!(
        base_src.contains("/// Remapped roughness clamped to `[0.0, 1.0]`."),
        "Xiexe roughness contract should document the unfloored indirect range"
    );
    Ok(())
}

#[test]
fn xiexe_generic_stems_resolve_alpha_mode_from_variant_bits() -> io::Result<()> {
    let base_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/base.wgsl"))?;
    assert!(
        declares_u32_field(&base_src, "_RenderideVariantBits"),
        "xiexe_toon2_base.wgsl must expose `_RenderideVariantBits` as a u32 field"
    );
    for forbidden in ["Cutout", "AlphaBlend", "Transparent"] {
        assert!(
            !declares_f32_field(&base_src, forbidden),
            "xiexe_toon2_base.wgsl must not retain the legacy `{forbidden}` f32 keyword field"
        );
    }
    assert!(
        !base_src.contains("fn resolved_alpha_mode("),
        "xiexe_toon2_base.wgsl must not retain the legacy `resolved_alpha_mode` helper"
    );

    let variant_bits_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/variant_bits.wgsl"))?;
    for (constant, bit) in [
        ("XTOON_KW_ALPHABLEND", 0),
        ("XTOON_KW_CUTOUT", 1),
        ("XTOON_KW_EMISSION_MAP", 2),
        ("XTOON_KW_MATCAP", 3),
        ("XTOON_KW_NORMAL_MAP", 4),
        ("XTOON_KW_OCCLUSION_METALLIC", 5),
        ("XTOON_KW_RAMPMASK_OUTLINEMASK_THICKNESS", 6),
        ("XTOON_KW_TRANSPARENT", 7),
        ("XTOON_KW_VERTEX_COLOR_ALBEDO", 8),
    ] {
        assert!(
            variant_bits_src.contains(&format!("const {constant}: u32 = 1u << {bit}u;")),
            "{constant} must match Froox's sorted UniqueKeywords bit order"
        );
    }
    for required in [
        "const XTOON_KEYWORD_LAYOUT_GENERIC: u32 = 0u;",
        "const XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT: u32 = 1u;",
        "fn static_vertexlight_layout(keyword_layout: u32) -> bool",
        "return keyword_layout == XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT;",
        "fn resolved_alpha_mode_from_bits_for_layout(static_alpha_mode: u32, keyword_layout: u32) -> u32",
        "fn resolved_alpha_mode_from_bits(static_alpha_mode: u32) -> u32",
        "kw_Cutout_for_layout(keyword_layout)",
        "return xb::ALPHA_CUTOUT;",
        "kw_Transparent_for_layout(keyword_layout)",
        "return xb::ALPHA_TRANSPARENT;",
        "kw_AlphaBlend_for_layout(keyword_layout)",
        "return xb::ALPHA_FADE;",
    ] {
        assert!(
            variant_bits_src.contains(required),
            "xiexe_toon2_variant_bits.wgsl must contain `{required}`"
        );
    }

    for file_name in ["xstoon2.0.wgsl", "xstoon2.0_outlined.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("xvb::resolved_alpha_mode_from_bits(XIEE_ALPHA_MODE)"),
            "{file_name} must route the generic Xiexe alpha mode through the variant bitmask"
        );
        assert!(
            !src.contains("xb::resolved_alpha_mode("),
            "{file_name} must not retain the legacy keyword-driven alpha-mode resolver"
        );
    }

    let fixed_outlined = material_source("xstoon2.0-outlined.wgsl")?;
    assert!(
        !fixed_outlined.contains("resolved_alpha_mode_from_bits"),
        "xstoon2.0-outlined.wgsl maps to the fixed XSToon2.0 Outlined source and must not route alpha through generic variant bits"
    );
    assert!(
        fixed_outlined.contains("view_layer, XIEE_ALPHA_MODE"),
        "xstoon2.0-outlined.wgsl must keep the fixed opaque alpha mode"
    );
    Ok(())
}

#[test]
fn xiexe_static_stems_use_static_vertexlight_keyword_layout() -> io::Result<()> {
    let variant_bits_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/variant_bits.wgsl"))?;
    for required in [
        "fn normal_map_enabled_for_layout(keyword_layout: u32) -> bool",
        "fn emission_map_enabled_for_layout(keyword_layout: u32) -> bool",
        "fn matcap_enabled_for_layout(keyword_layout: u32) -> bool",
        "fn reflection_uses_pbr_for_layout(keyword_layout: u32) -> bool",
    ] {
        assert!(
            variant_bits_src.contains(required),
            "xiexe_toon2_variant_bits.wgsl must contain `{required}`"
        );
    }

    for file_name in [
        "xstoon2.0-cutout.wgsl",
        "xstoon2.0-cutouta2c.wgsl",
        "xstoon2.0-cutouta2cmasked.wgsl",
        "xstoon2.0-cutouta2c-outlined.wgsl",
        "xstoon2.0-dithered-outlined.wgsl",
    ] {
        let src = material_source(file_name)?;
        assert!(
            src.contains(
                "const XIEE_KEYWORD_LAYOUT: u32 = xvb::XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT;"
            ),
            "{file_name} must select the static XSToon keyword layout"
        );
        assert!(
            src.contains("xs::fragment_forward_for_layout(")
                && src.contains("XIEE_ALPHA_MODE, XIEE_KEYWORD_LAYOUT"),
            "{file_name} must pass the static keyword layout into the forward fragment path"
        );
    }

    for file_name in [
        "xstoon2.0-cutouta2c-outlined.wgsl",
        "xstoon2.0-dithered-outlined.wgsl",
    ] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("xo::fragment_outline_for_layout("),
            "{file_name} must pass the static keyword layout into the outline fragment path"
        );
    }

    Ok(())
}

#[test]
fn xiexe_outline_vertex_color_preserves_outline_color_for_vertex_color_albedo() -> io::Result<()> {
    let outline_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/outline.wgsl"))?;
    assert!(
        outline_src.contains("out.color = vec4<f32>(xb::mat._OutlineColor.rgb, 1.0);"),
        "Xiexe outline vertices must pass `_OutlineColor.rgb` through vertex color so vertex-color albedo variants preserve outline tinting"
    );

    let surface_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/surface.wgsl"))?;
    assert!(
        surface_src.contains("albedo = vec4<f32>(albedo.rgb * color.rgb, albedo.a);"),
        "Xiexe surface sampling must continue applying vertex color when the vertex-color albedo keyword is enabled"
    );

    Ok(())
}

#[test]
fn xiexe_a2c_has_single_sample_dither_fallback() -> io::Result<()> {
    let alpha_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/alpha.wgsl"))?;
    for required in [
        "rg::frame_sample_count() <= 1u",
        "if (coverage <= d)",
        "if (coverage <= xb::bayer_threshold(frag_xy))",
        "textureSample(xb::_CutoutMask, xb::_CutoutMask_sampler, uv_primary).r",
    ] {
        assert!(
            alpha_src.contains(required),
            "Xiexe A2C single-sample fallback must contain `{required}`"
        );
    }
    assert!(
        !alpha_src.contains("textureSampleLevel(xb::_CutoutMask"),
        "Xiexe cutout masks must not force base-mip sampling"
    );

    let globals_src = source_file(manifest_dir().join("shaders/modules/frame/globals.wgsl"))?;
    for required in [
        "fn frame_sample_count() -> u32",
        "ft::FRAME_TAIL_SAMPLE_COUNT_MASK",
        "return 1u;",
    ] {
        assert!(
            globals_src.contains(required),
            "frame globals must expose sample count decoding through `{required}`"
        );
    }

    Ok(())
}
