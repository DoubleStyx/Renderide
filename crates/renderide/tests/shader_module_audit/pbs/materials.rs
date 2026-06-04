//! PBS material-root parity audits.

use super::super::*;

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
            "//#mat_default _BumpScale float 1.0",
            "//#mat_default _Parallax float 0.02",
            "//#mat_default _Color vec4 1.0 1.0 1.0 1.0",
            "//#mat_default _DetailNormalMapScale float 1.0",
            "//#mat_default _EmissionColor vec4 0.0 0.0 0.0 1.0",
            "//#mat_default _Cutoff float 0.5",
            "//#mat_default _Glossiness float 0.5",
            "mat._Parallax,",
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
        "pdet::detail_uv(uv0, uv1, mat._UVSec, mat._DetailAlbedoMap_ST)",
        "mv::world_uv2_vertex_main(instance_index, view_idx, pos, n, t, uv0, uv1)",
        "mv::world_uv2_vertex_main(instance_index, 0u, pos, n, t, uv0, uv1)",
    ] {
        assert!(
            src.contains(required),
            "pbsmetallic.wgsl must contain `{required}`"
        );
    }
    assert!(
        !src.contains("pdet::detail_uv(uv0, uv1, 0.0"),
        "pbsmetallic.wgsl must not hard-code UV0 for secondary textures"
    );
    Ok(())
}

/// Verifies PBS detail albedo uses Unity's linear `unity_ColorSpaceDouble` value.
#[test]
fn pbs_detail_albedo_uses_unity_linear_color_space_double() -> io::Result<()> {
    let detail = module_source("pbs/detail.wgsl")?;
    assert!(
        detail.contains("const COLOR_SPACE_DOUBLE: f32 = 4.59479380;"),
        "pbs/detail.wgsl must use Unity's linear `unity_ColorSpaceDouble` detail multiplier"
    );
    assert!(
        !detail.contains("4.67199902667"),
        "pbs/detail.wgsl must not use exact sRGB midpoint inversion for Unity detail albedo"
    );
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
