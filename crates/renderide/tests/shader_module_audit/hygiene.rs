//! Shader source audits for this behavior family.

use super::*;

#[test]
fn file_labels_use_forward_slashes_for_cross_platform_audits() {
    assert_eq!(
        normalize_file_label(r"shaders\modules\draw\per_draw.wgsl"),
        "shaders/modules/draw/per_draw.wgsl"
    );
}

/// Nested WGSL modules must remain discoverable and uniquely addressable by naga-oil.

#[test]
fn shader_modules_have_unique_import_paths() -> io::Result<()> {
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut offenders = Vec::new();

    for path in wgsl_files_recursive("shaders/modules")? {
        let src = source_file(&path)?;
        let Some(import_path) = define_import_path(&src) else {
            offenders.push(format!("{} has no #define_import_path", file_label(&path)));
            continue;
        };
        if let Some((_, first_path)) = seen
            .iter()
            .find(|(seen_import_path, _)| seen_import_path == import_path)
        {
            offenders.push(format!(
                "{} duplicates import path {import_path} from {first_path}",
                file_label(&path)
            ));
        }
        seen.push((import_path.to_string(), file_label(&path)));
    }

    assert!(
        offenders.is_empty(),
        "shader module import paths must be present and unique:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Material roots should route per-draw view-projection selection through `renderide::mesh::vertex`.
#[test]
fn material_roots_do_not_duplicate_view_projection_selection() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = source_file(&path)?;
        for forbidden in ["view_proj_left", "view_proj_right"] {
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
        "materials importing renderide::mesh::vertex must delegate view-projection selection:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Shared modules should centralize raw per-draw matrix field access in the mesh vertex module.

#[test]
fn shader_modules_centralize_view_projection_selection() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/modules")? {
        let label = file_label(&path);
        if matches!(
            label.as_str(),
            "shaders/modules/draw/types.wgsl"
                | "shaders/modules/draw/per_draw.wgsl"
                | "shaders/modules/mesh/vertex.wgsl"
        ) {
            continue;
        }

        let src = source_file(&path)?;
        for forbidden in ["view_proj_left", "view_proj_right"] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "only draw types, per_draw, and mesh::vertex should touch raw view-projection fields:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Material roots should consume shared helper modules instead of reintroducing helper copies.
#[test]
fn material_roots_do_not_redeclare_shared_helpers() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = source_file(&path)?;

        for forbidden in [
            "fn alpha_over",
            "fn intersection_lerp",
            "fn inside_rect",
            "fn outside_rect",
            "fn pick_uv(",
            "fn unpack_normal_xy",
            "fn roughness_from_smoothness",
            "fn safe_normalize",
            "fn shade_distance_field",
            "fn view_angle_fresnel",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "material roots should import shared helper modules instead of redeclaring them:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

#[test]
fn pbs_family_roots_use_shared_shader_modules() -> io::Result<()> {
    for material in ["pbsmultiuv.wgsl", "pbsmultiuvspecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("uvu::apply_st_uv4(") && !src.contains("fn pick_uv("),
            "{material} must route UV0-UV3 selection through core::uv"
        );
    }

    for material in ["pbsintersect.wgsl", "pbsintersectspecular.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("renderide::pbs::families::intersect as pint")
                && src.contains("pint::intersection_lerp(")
                && !src.contains("fn intersection_lerp("),
            "{material} must route scene-depth intersection math through the intersect family module"
        );
    }

    for material in [
        "pbsdisplace.wgsl",
        "pbsdisplacespecular.wgsl",
        "pbsdisplacetransparent.wgsl",
        "pbsdisplacespeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("-> pdisp::VertexOutput")
                && src.contains("pdisp::vertex_output(")
                && !src.contains("struct VertexOutput")
                && !src.contains("fn sample_normal_world("),
            "{material} must use the displacement module for vertex payloads and PBS sampling helpers for normals"
        );
    }

    for material in [
        "pbsdistancelerp.wgsl",
        "pbsdistancelerpspecular.wgsl",
        "pbsdistancelerptransparent.wgsl",
        "pbsdistancelerpspeculartransparent.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("-> pdist::VertexOutput")
                && src.contains("pdist::vertex_main(")
                && !src.contains("struct VertexOutput")
                && !src.contains("fn sample_normal_world("),
            "{material} must use the DistanceLerp family module for vertex payloads and PBS sampling helpers for normals"
        );
    }

    for material in [
        "pbsslice.wgsl",
        "pbsslicespecular.wgsl",
        "pbsslicetransparent.wgsl",
        "pbsslicetransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("pslice::sample_world_normal(")
                && !src.contains("fn sample_normal_world(")
                && !src.contains("renderide::pbs::normal as pnorm")
                && !src.contains("renderide::core::normal_decode as nd"),
            "{material} must route slice normal-map blending through the Slice family module"
        );
    }

    for material in [
        "pbstriplanar.wgsl",
        "pbstriplanarspecular.wgsl",
        "pbstriplanartransparent.wgsl",
        "pbstriplanartransparentspecular.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("ptri::vertex_main(")
                && src.contains("ptri::resolve_world_normal(")
                && !src.contains("mv::model_vector(")
                && !src.contains("renderide::draw::per_draw as pd")
                && !src.contains("renderide::mesh::vertex as mv"),
            "{material} must use the triplanar family module for vertex payloads and object-space normal remapping"
        );
    }

    Ok(())
}

#[test]
fn volume_unlit_clamps_emitted_source_rgb() -> io::Result<()> {
    let volume_box = module_source("material/volume_box.wgsl")?;
    assert!(
        volume_box.contains("fn clamp_volume_source_rgb(color: vec4<f32>) -> vec4<f32>")
            && volume_box.contains("clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(1.0))"),
        "volume_box must expose a helper that clamps emitted volume source RGB"
    );

    let src = material_source("volumeunlit.wgsl")?;
    assert!(
        src.contains("vol::clamp_volume_source_rgb("),
        "volumeunlit.wgsl must clamp emitted source RGB before additive retention"
    );

    Ok(())
}

#[test]
fn alpha_clip_paths_do_not_force_base_mip_sampling() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders")? {
        let src = source_file(&path)?;
        for forbidden in [
            "renderide::material::alpha_clip_sample",
            "texture_alpha_base_mip",
            "texture_rgba_base_mip",
            "mask_luminance_mul_base_mip",
            "sample_rgba_lod0",
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
        "alpha clipping must use the same filtered texture samples as visible color:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

#[test]
fn blur_poisson_path_avoids_dynamic_sample_array_indexing() -> io::Result<()> {
    let blur = material_source("blur.wgsl")?;
    assert!(
        blur.contains("#import renderide::post::poisson_blur as pb")
            && blur.contains("if (kw_POISSON_DISC())")
            && blur.contains("pb::sample_poisson_blur("),
        "blur.wgsl must route Poisson blur through a real control-flow branch"
    );
    for forbidden in [
        "let offset = select(",
        "fm::poisson_blur_offset(",
        "POISSON_2D_SAMPLES",
    ] {
        assert!(
            !blur.contains(forbidden),
            "blur.wgsl must not use eager Poisson/circular selection through `{forbidden}`"
        );
    }

    let filter_math = module_source("post/filter_math.wgsl")?;
    for forbidden in [
        "POISSON_2D_SAMPLE_COUNT",
        "POISSON_2D_SAMPLES",
        "poisson_blur_offset",
    ] {
        assert!(
            !filter_math.contains(forbidden),
            "filter_math.wgsl must not retain Poisson sample-array helpers"
        );
    }

    let poisson = module_source("post/poisson_blur.wgsl")?;
    assert!(
        poisson.contains("#define_import_path renderide::post::poisson_blur"),
        "poisson_blur.wgsl must be a composable shader module"
    );
    assert_eq!(
        poisson.matches("c = c + sample_poisson_tap").count(),
        128,
        "poisson_blur.wgsl must keep all 128 Poisson taps explicitly unrolled"
    );
    for forbidden in [
        "array<vec2<f32>",
        "POISSON_2D_SAMPLES",
        "[idx]",
        "[i]",
        "poisson_blur_offset",
    ] {
        assert!(
            !poisson.contains(forbidden),
            "poisson_blur.wgsl must not use dynamic sample-array indexing through `{forbidden}`"
        );
    }
    Ok(())
}

#[test]
fn grab_filter_roots_use_shared_filter_common_helpers() -> io::Result<()> {
    for material in [
        "blur.wgsl",
        "channelmatrix.wgsl",
        "gamma.wgsl",
        "getdepth.wgsl",
        "grayscale.wgsl",
        "hsv.wgsl",
        "invert.wgsl",
        "lut.wgsl",
        "lut_perobject.wgsl",
        "pixelate.wgsl",
        "posterize.wgsl",
        "refract.wgsl",
        "threshold.wgsl",
    ] {
        let src = material_source(material)?;
        assert!(
            src.contains("renderide::post::filter_common as fc"),
            "{material} must import the shared filter_common module"
        );
        for forbidden in [
            "gp::sample_scene_color(gp::frag_screen_uv",
            "uirc::should_clip_rect_kw",
            "rg::retain_globals_additive",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must delegate `{forbidden}` through filter_common"
            );
        }
    }
    Ok(())
}

#[test]
fn threshold_filter_preserves_signed_transition() -> io::Result<()> {
    let src = material_source("threshold.wgsl")?;
    assert!(
        src.contains("select(-1e-6, 1e-6, mat._Transition >= 0.0)")
            && !src.contains("max(abs(mat._Transition), 1e-6)"),
        "threshold.wgsl must keep negative `_Transition` values inverted like the Unity source"
    );
    Ok(())
}

#[test]
fn refraction_filter_roots_use_shared_refraction_helpers() -> io::Result<()> {
    for material in ["blur.wgsl", "refract.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("renderide::post::filter_refraction as fr"),
            "{material} must import the shared filter_refraction module"
        );
        for forbidden in [
            "nd::decode_ts_normal_with_placeholder_sample",
            "pnorm::orthonormal_tbn",
        ] {
            assert!(
                !src.contains(forbidden),
                "{material} must delegate `{forbidden}` through filter_refraction"
            );
        }
    }

    let refract = material_source("refract.wgsl")?;
    for forbidden in ["fn refract_offset", "fn refracted_screen_uv"] {
        assert!(
            !refract.contains(forbidden),
            "refract.wgsl must delegate `{forbidden}` through filter_refraction"
        );
    }
    Ok(())
}

#[test]
fn gradient_skybox_roots_use_shared_evaluator() -> io::Result<()> {
    let roots = [
        (
            "materials/gradientskybox.wgsl",
            material_source("gradientskybox.wgsl")?,
        ),
        (
            "passes/backend/skybox_gradientskybox.wgsl",
            source_file(manifest_dir().join("shaders/passes/backend/skybox_gradientskybox.wgsl"))?,
        ),
    ];

    for (label, src) in roots {
        assert!(
            src.contains("renderide::skybox::gradient as skygrad")
                && src.contains("skygrad::gradient_sky_color(")
                && src.contains("mat._BaseColor")
                && src.contains("mat._Params")
                && !src.contains("fn gradient_color("),
            "{label} must route GradientSkybox evaluation through the shared skybox module",
        );
    }

    let module = source_file(manifest_dir().join("shaders/modules/skybox/gradient.wgsl"))?;
    for required in [
        "#define_import_path renderide::skybox::gradient",
        "fn gradient_sky_color(",
        "dirs_spread_values: array<vec4<f32>, 16>",
        "param_values: array<vec4<f32>, 16>",
        "let ray = normalize(ray_in);",
    ] {
        assert!(
            module.contains(required),
            "skybox/gradient.wgsl must contain `{required}`",
        );
    }

    Ok(())
}

#[test]
fn normal_decode_scales_xy_before_reconstructing_z() -> io::Result<()> {
    let src = source_file(manifest_dir().join("shaders/modules/core/normal_decode.wgsl"))?;
    let xy_scale = src
        .find("let xy = (raw.xy * 2.0 - 1.0) * scale;")
        .expect("normal decode must scale tangent XY before Z reconstruction");
    let z_reconstruct = src
        .find("let z = sqrt(max(1.0 - dot(xy, xy), 0.0));")
        .expect("normal decode must reconstruct Z from the scaled XY vector");

    assert!(
        xy_scale < z_reconstruct,
        "normal decode must apply `_NormalScale` / `_BumpScale` before reconstructing Z"
    );
    assert!(
        !src.contains("scale-after-Z"),
        "normal decode comments must not describe the old scale-after-Z behavior"
    );
    Ok(())
}

/// Pass shaders using the fullscreen module should not duplicate fullscreen-triangle bit math.

#[test]
fn shared_fullscreen_roots_do_not_duplicate_fullscreen_triangle_setup() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/passes")? {
        let src = source_file(&path)?;
        if !src.contains("renderide::core::fullscreen") {
            continue;
        }

        for forbidden in ["<< 1u", "vec2(-1.0, -1.0)", "vec2(3.0, -1.0)"] {
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
        "passes importing renderide::core::fullscreen must delegate fullscreen-triangle setup:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}
