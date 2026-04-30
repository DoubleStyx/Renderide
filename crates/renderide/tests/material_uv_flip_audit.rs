//! Audit: every WGSL material that samples a **host-uploaded** texture must apply the engine's
//! V-flip convention. Resonite/Unity stores textures bottom-up and the renderer reads them
//! through wgpu's top-down `textureSample` origin, so each sampling site must compose with one
//! of the centralised flip helpers (`apply_st` / `flip_v` from `renderide::uv_utils`), with the
//! `xiexe_toon2` shared module (which itself uses `apply_st`), or with an inline `1.0 - <uv>.y`
//! flip in the shader itself.
//!
//! This guard exists because the historical convention has been easy to forget when adding new
//! shaders; a missed flip silently inverts a texture instead of failing the build. The test
//! parses the source text — fast, no GPU, no naga compile.
//!
//! Scope: only material-owned textures count. A shader is in scope iff it declares a
//! `@group(1) @binding(N) var <name>: texture_*<...>` (per-material resource). Sampling reads
//! against renderer-internal textures (`@group(0)` frame globals such as `rg::scene_depth`)
//! produced by the renderer itself are top-down and must **not** be V-flipped.
//!
//! Acceptable flip evidence per material:
//! - imports `renderide::uv_utils` (PBS / Unlit / overlay / triplanar / matcap / projection360 /
//!   reflection / testblend / texturedebug / nosamplers …)
//! - imports `renderide::xiexe::toon2` (the xstoon2.0 family)
//! - contains an inline `1.0 - <something>.y` flip (text shaders, triplanar uv build)

use std::path::{Path, PathBuf};

/// Sampling intrinsics that touch a `texture_*` resource. Used to decide whether a shader has
/// any sampling at all; the audit further requires a `@group(1)` texture binding to be in scope.
const SAMPLE_INTRINSICS: &[&str] = &[
    "textureSample(",
    "textureSampleLevel(",
    "textureSampleBias(",
    "textureSampleGrad(",
    "textureSampleCompare(",
    "textureSampleCompareLevel(",
    "textureLoad(",
    "textureGather(",
];

/// Detects a per-material texture binding (`@group(1) @binding(N) var <name>: texture_*<...>`).
/// Whitespace between attributes and identifiers is tolerated.
fn declares_group1_texture(src: &str) -> bool {
    src.lines().any(|line| {
        let l = line.trim();
        if !l.starts_with("@group(1)") {
            return false;
        }
        // `texture_2d`, `texture_3d`, `texture_cube`, `texture_2d_array`, `texture_external`, etc.
        l.contains("texture_")
    })
}

/// Markers that prove a material applies the engine's V-flip convention at every sampling site.
fn shader_applies_v_flip(src: &str) -> bool {
    let imports_uv_utils = src.contains("renderide::uv_utils");
    let imports_xiexe_toon2 =
        src.contains("renderide::xiexe::toon2") || src.contains("xiexe_toon2");
    let inline_v_flip = inline_v_flip_present(src);
    imports_uv_utils || imports_xiexe_toon2 || inline_v_flip
}

/// Detects an inline `1.0 - <expr>.y` pattern (with optional whitespace), tolerating `.y` on a
/// vec swizzle (e.g. `tiled.y`, `uv.y`, `out.uv.y`). Conservative: requires the literal `1.0`
/// followed by whitespace, `-`, whitespace, then any non-newline characters, then `.y`.
fn inline_v_flip_present(src: &str) -> bool {
    src.lines().any(|line| {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("///") {
            return false;
        }
        if !line.contains("1.0") || !line.contains(".y") {
            return false;
        }
        // Simple heuristic: a `1.0 -` (or `1.0-`) followed later in the line by `.y`. This catches
        // every existing inline flip site (text shaders, triplanar UV build, matcap derived UVs)
        // without false-positives like `uv.y * 1.0` because we require `1.0` to come first.
        let one_pos = line.find("1.0");
        let dash_pos = line[one_pos.unwrap_or(0)..]
            .find('-')
            .map(|p| p + one_pos.unwrap_or(0));
        let dot_y_pos = line.find(".y");
        match (one_pos, dash_pos, dot_y_pos) {
            (Some(one), Some(dash), Some(dy)) => one < dash && dash < dy,
            _ => false,
        }
    })
}

fn shader_samples_textures(src: &str) -> bool {
    SAMPLE_INTRINSICS.iter().any(|s| src.contains(s))
}

fn materials_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("shaders/materials")
}

fn modules_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("shaders/modules")
}

fn wgsl_files_in(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("wgsl") {
            files.push(path);
        }
    }
    Ok(files)
}

fn count_occurrences(src: &str, needle: &str) -> usize {
    src.match_indices(needle).count()
}

fn is_wgsl_ident_byte(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

fn storage_orientation_stems(src: &str) -> Vec<String> {
    const SUFFIX: &str = "_StorageVInverted";
    let bytes = src.as_bytes();
    let mut stems = Vec::new();
    for (suffix_start, _) in src.match_indices(SUFFIX) {
        let mut stem_start = suffix_start;
        while stem_start > 0 && is_wgsl_ident_byte(bytes[stem_start - 1]) {
            stem_start -= 1;
        }
        if stem_start < suffix_start {
            stems.push(src[stem_start..suffix_start].to_owned());
        }
    }
    stems.sort();
    stems.dedup();
    stems
}

fn declared_storage_orientation_stems(src: &str) -> Vec<String> {
    let mut stems = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || !trimmed.contains("_StorageVInverted") {
            continue;
        }
        let Some((name, ty)) = trimmed.split_once(':') else {
            continue;
        };
        if !ty.trim_start().starts_with("f32") {
            continue;
        }
        let name = name.trim().trim_end_matches(',');
        let Some(stem) = name.strip_suffix("_StorageVInverted") else {
            continue;
        };
        stems.push(stem.to_owned());
    }
    stems.sort();
    stems.dedup();
    stems
}

fn group1_texture_names(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("@group(1)") || !trimmed.contains("texture_") {
            continue;
        }
        let Some(var_start) = trimmed.find("var ") else {
            continue;
        };
        let after_var = &trimmed[var_start + "var ".len()..];
        let Some((name, _)) = after_var.split_once(':') else {
            continue;
        };
        names.push(name.trim().to_owned());
    }
    names.sort();
    names.dedup();
    names
}

fn non_comment_lines(src: &str) -> impl Iterator<Item = &str> {
    src.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
}

fn storage_orientation_flag_used_by_shader_code(src: &str, stem: &str) -> bool {
    let mat_ref = format!("mat.{stem}_StorageVInverted");
    let module_mat_ref = format!("xb::mat.{stem}_StorageVInverted");
    non_comment_lines(src).any(|line| line.contains(&mat_ref) || line.contains(&module_mat_ref))
}

fn call_snippets<'a>(src: &'a str, callee: &str) -> Vec<&'a str> {
    let mut snippets = Vec::new();
    let mut offset = 0usize;
    while let Some(relative_start) = src[offset..].find(callee) {
        let start = offset + relative_start;
        let end = src[start..].find(';').map_or_else(
            || (start + 400).min(src.len()),
            |relative_end| start + relative_end,
        );
        snippets.push(&src[start..end]);
        offset = start + callee.len();
    }
    snippets
}

fn plain_apply_st_uses_st(src: &str, stem: &str) -> bool {
    let mat_st = format!("mat.{stem}_ST");
    let module_mat_st = format!("xb::mat.{stem}_ST");
    call_snippets(src, "apply_st(")
        .iter()
        .any(|snippet| snippet.contains(&mat_st) || snippet.contains(&module_mat_st))
}

fn storage_aware_apply_st_uses_flag(src: &str, stem: &str) -> bool {
    let mat_st = format!("mat.{stem}_ST");
    let module_mat_st = format!("xb::mat.{stem}_ST");
    let mat_flag = format!("mat.{stem}_StorageVInverted");
    let module_mat_flag = format!("xb::mat.{stem}_StorageVInverted");
    call_snippets(src, "apply_st_for_storage(")
        .iter()
        .any(|snippet| {
            (snippet.contains(&mat_st) || snippet.contains(&module_mat_st))
                && (snippet.contains(&mat_flag) || snippet.contains(&module_mat_flag))
        })
}

#[test]
fn every_sampling_material_applies_v_flip() -> Result<(), Box<dyn std::error::Error>> {
    let dir = materials_dir();
    let mut offenders: Vec<String> = Vec::new();
    let mut audited = 0usize;
    for path in wgsl_files_in(&dir)? {
        let src = std::fs::read_to_string(&path)?;
        if !shader_samples_textures(&src) {
            continue;
        }
        if !declares_group1_texture(&src) {
            // Shader only touches frame-global / per-draw textures (e.g. depth pyramid). The
            // V-flip convention only applies to host-uploaded `@group(1)` textures.
            continue;
        }
        audited += 1;
        if !shader_applies_v_flip(&src) {
            offenders.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }

    assert!(audited > 0, "audit found no sampling materials in {dir:?}");

    if !offenders.is_empty() {
        offenders.sort();
        panic!(
            "{} material shader(s) sample textures without applying the V-flip convention. \
             Each must import `renderide::uv_utils` (use `apply_st`/`flip_v`), import \
             `renderide::xiexe::toon2`, or add an inline `1.0 - <uv>.y` flip:\n  - {}",
            offenders.len(),
            offenders.join("\n  - ")
        );
    }
    Ok(())
}

#[test]
fn inline_v_flip_detection_self_check() {
    // Positive cases — patterns the audit must accept.
    assert!(inline_v_flip_present(
        "    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);\n"
    ));
    assert!(inline_v_flip_present(
        "    return vec2<f32>(tiled.x, 1.0 - tiled.y);\n"
    ));
    // Negative — `.y` alone is not a flip.
    assert!(!inline_v_flip_present(
        "    out.uv = vec2<f32>(uv.x, uv.y);\n"
    ));
    // Negative — `1.0` without a following `.y` flip on the same line.
    assert!(!inline_v_flip_present("    let x = uv.x * 1.0;\n"));
    // Negative — comment-only line is ignored.
    assert!(!inline_v_flip_present(
        "    // historical: 1.0 - uv.y was here\n"
    ));
}

#[test]
fn material_storage_orientation_flags_are_used_in_shader_code()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = materials_dir();
    let mut offenders: Vec<String> = Vec::new();
    for path in wgsl_files_in(&dir)? {
        let src = std::fs::read_to_string(&path)?;
        for stem in declared_storage_orientation_stems(&src) {
            if !storage_orientation_flag_used_by_shader_code(&src, &stem) {
                offenders.push(format!(
                    "{} declares {stem}_StorageVInverted but never uses it",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }

    if !offenders.is_empty() {
        offenders.sort();
        panic!(
            "storage orientation flags must feed shader UV/direction helpers:\n  - {}",
            offenders.join("\n  - ")
        );
    }
    Ok(())
}

#[test]
fn explicit_storage_orientation_stems_do_not_use_plain_apply_st()
-> Result<(), Box<dyn std::error::Error>> {
    let mut offenders: Vec<String> = Vec::new();
    for dir in [materials_dir(), modules_dir()] {
        for path in wgsl_files_in(&dir)? {
            let src = std::fs::read_to_string(&path)?;
            for stem in storage_orientation_stems(&src) {
                let flag = format!("{stem}_StorageVInverted");
                if plain_apply_st_uses_st(&src, &stem) {
                    offenders.push(format!(
                        "{} uses plain apply_st for {stem} despite {flag}",
                        path.file_name().unwrap().to_string_lossy()
                    ));
                }
            }
        }
    }

    if !offenders.is_empty() {
        offenders.sort();
        panic!(
            "explicit storage orientation fields must use apply_st_for_storage to avoid double/missing V compensation:\n  - {}",
            offenders.join("\n  - ")
        );
    }
    Ok(())
}

#[test]
fn multiuv_textures_use_matching_storage_orientation_flags()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = materials_dir();
    let mut offenders: Vec<String> = Vec::new();
    for file_name in ["pbsmultiuv.wgsl", "pbsmultiuvspecular.wgsl"] {
        let path = dir.join(file_name);
        let src = std::fs::read_to_string(&path)?;
        for texture in group1_texture_names(&src) {
            let flag = format!("{texture}_StorageVInverted");
            if !declared_storage_orientation_stems(&src).contains(&texture) {
                offenders.push(format!("{file_name} declares {texture} without {flag}"));
                continue;
            }
            if !storage_aware_apply_st_uses_flag(&src, &texture) {
                offenders.push(format!(
                    "{file_name} does not feed {flag} into apply_st_for_storage for {texture}_ST"
                ));
            }
        }
    }

    if !offenders.is_empty() {
        offenders.sort();
        panic!(
            "MultiUV texture bindings must each have matching storage-aware V-flip plumbing:\n  - {}",
            offenders.join("\n  - ")
        );
    }
    Ok(())
}

#[test]
fn xiexe_storage_orientation_flags_are_consumed_by_surface_module()
-> Result<(), Box<dyn std::error::Error>> {
    let modules = modules_dir();
    let mut combined = String::new();
    for path in wgsl_files_in(&modules)? {
        combined.push_str(&std::fs::read_to_string(&path)?);
        combined.push('\n');
    }

    for flag in [
        "_MainTex_StorageVInverted",
        "_BumpMap_StorageVInverted",
        "_DetailNormalMap_StorageVInverted",
        "_MetallicGlossMap_StorageVInverted",
        "_EmissionMap_StorageVInverted",
        "_OcclusionMap_StorageVInverted",
        "_ThicknessMap_StorageVInverted",
        "_ReflectivityMask_StorageVInverted",
        "_SpecularMap_StorageVInverted",
    ] {
        assert!(
            count_occurrences(&combined, flag) >= 2,
            "{flag} should be declared in xiexe_toon2_base and consumed by xiexe_toon2_surface"
        );
    }
    Ok(())
}
