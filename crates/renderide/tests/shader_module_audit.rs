//! Source audits for WGSL module factoring invariants.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Returns the renderide crate directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Recursively returns all WGSL files below `relative_dir`.
fn wgsl_files_recursive(relative_dir: &str) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_wgsl_files(&manifest_dir().join(relative_dir), &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_wgsl_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_wgsl_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            out.push(path);
        }
    }
    Ok(())
}

fn file_label(path: &Path) -> String {
    normalize_file_label(
        path.strip_prefix(manifest_dir())
            .unwrap_or(path)
            .display()
            .to_string(),
    )
}

fn normalize_file_label(label: impl AsRef<str>) -> String {
    label.as_ref().replace('\\', "/")
}

fn define_import_path(src: &str) -> Option<&str> {
    src.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix("#define_import_path")
            .map(str::trim)
            .filter(|path| !path.is_empty())
    })
}

fn material_source(file_name: &str) -> io::Result<String> {
    fs::read_to_string(manifest_dir().join("shaders/materials").join(file_name))
}

fn declares_f32_field(src: &str, field_name: &str) -> bool {
    src.lines().any(|line| {
        let trimmed = line.trim();
        let Some((name, ty)) = trimmed.split_once(':') else {
            return false;
        };
        name.trim() == field_name && ty.trim_start().starts_with("f32")
    })
}

fn all_texture_samples_guarded_by_keyword(src: &str, texture_name: &str, keyword: &str) -> bool {
    let sample = format!("textureSample({texture_name},");
    let guard = format!("uvu::kw_enabled(mat.{keyword})");
    let mut saw_sample = false;

    for (sample_pos, _) in src.match_indices(&sample) {
        saw_sample = true;
        let before_sample = &src[..sample_pos];
        let Some(guard_pos) = before_sample.rfind(&guard) else {
            return false;
        };
        if before_sample[guard_pos..].contains('}') {
            return false;
        }
    }

    saw_sample
}

fn normal_sampling_guarded_by_keyword(src: &str) -> bool {
    let Some(call_pos) = src.find("sample_optional_world_normal(") else {
        return false;
    };
    let call = &src[call_pos..];
    let Some(call_end) = call.find(");") else {
        return false;
    };
    call[..call_end].contains("uvu::kw_enabled(mat._NORMALMAP)")
}

#[test]
fn file_labels_use_forward_slashes_for_cross_platform_audits() {
    assert_eq!(
        normalize_file_label(r"shaders\modules\per_draw.wgsl"),
        "shaders/modules/per_draw.wgsl"
    );
}

/// Nested WGSL modules must remain discoverable and uniquely addressable by naga-oil.
#[test]
fn shader_modules_have_unique_import_paths() -> io::Result<()> {
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut offenders = Vec::new();

    for path in wgsl_files_recursive("shaders/modules")? {
        let src = fs::read_to_string(&path)?;
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

/// Material roots using the shared PBS lighting module should not also carry their own clustered loop.
#[test]
fn shared_pbs_lighting_roots_do_not_duplicate_clustered_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = fs::read_to_string(&path)?;
        if !src.contains("renderide::pbs::lighting") {
            continue;
        }

        for forbidden in [
            "#import renderide::sh2_ambient",
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

/// PBS rim variants mirror Unity's optional-texture keywords: maps only override fallback material
/// values when their corresponding multi-compile keyword is active.
#[test]
fn pbs_rim_shaders_preserve_unity_texture_keywords() -> io::Result<()> {
    struct Case<'a> {
        file_name: &'a str,
        workflow_keyword: &'a str,
        workflow_texture: &'a str,
        forbidden_unity_fallback_mul: &'a [&'a str],
    }

    let cases = [
        Case {
            file_name: "pbsrim.wgsl",
            workflow_keyword: "_METALLICMAP",
            workflow_texture: "_MetallicMap",
            forbidden_unity_fallback_mul: &["mat._Metallic * mg", "mat._Glossiness * mg"],
        },
        Case {
            file_name: "pbsrimspecular.wgsl",
            workflow_keyword: "_SPECULARMAP",
            workflow_texture: "_SpecularMap",
            forbidden_unity_fallback_mul: &["mat._SpecularColor * spec_s"],
        },
    ];

    for case in cases {
        let src = material_source(case.file_name)?;
        for keyword in [
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            case.workflow_keyword,
            "_OCCLUSION",
        ] {
            assert!(
                declares_f32_field(&src, keyword),
                "{} must declare Unity keyword uniform {keyword}",
                case.file_name
            );
        }

        for (texture, keyword) in [
            ("_MainTex", "_ALBEDOTEX"),
            ("_EmissionMap", "_EMISSIONTEX"),
            (case.workflow_texture, case.workflow_keyword),
            ("_OcclusionMap", "_OCCLUSION"),
        ] {
            assert!(
                all_texture_samples_guarded_by_keyword(&src, texture, keyword),
                "{} must sample {texture} only when {keyword} is enabled",
                case.file_name
            );
        }

        assert!(
            normal_sampling_guarded_by_keyword(&src),
            "{} must sample _NormalMap through the _NORMALMAP keyword",
            case.file_name
        );

        for forbidden in case.forbidden_unity_fallback_mul {
            assert!(
                !src.contains(forbidden),
                "{} must not multiply workflow maps by fallback material values (`{forbidden}`)",
                case.file_name
            );
        }
    }

    Ok(())
}

/// Material roots should route per-draw view-projection selection through `renderide::mesh::vertex`.
#[test]
fn material_roots_do_not_duplicate_view_projection_selection() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = fs::read_to_string(&path)?;
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
            "shaders/modules/per_draw.wgsl" | "shaders/modules/mesh/vertex.wgsl"
        ) {
            continue;
        }

        let src = fs::read_to_string(&path)?;
        for forbidden in ["view_proj_left", "view_proj_right"] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "only per_draw and mesh::vertex should touch raw view-projection fields:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Standard PBS-like roots should delegate clustered Standard lighting to the shared PBS module.
#[test]
fn standard_material_roots_do_not_duplicate_clustered_pbs_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = fs::read_to_string(&path)?;
        if label == "shaders/materials/toonstandard.wgsl"
            || label == "shaders/materials/toonwater.wgsl"
        {
            continue;
        }

        for forbidden in [
            "#import renderide::sh2_ambient",
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

/// Material roots should consume shared helper modules instead of reintroducing helper copies.
#[test]
fn material_roots_do_not_redeclare_shared_helpers() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = fs::read_to_string(&path)?;

        for forbidden in [
            "fn alpha_over",
            "fn inside_rect",
            "fn outside_rect",
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

/// Pass shaders using the fullscreen module should not duplicate fullscreen-triangle bit math.
#[test]
fn shared_fullscreen_roots_do_not_duplicate_fullscreen_triangle_setup() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/passes")? {
        let src = fs::read_to_string(&path)?;
        if !src.contains("renderide::fullscreen") {
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
        "passes importing renderide::fullscreen must delegate fullscreen-triangle setup:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}
