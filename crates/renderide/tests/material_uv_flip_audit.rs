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
    Path::new(manifest).join("shaders/source/materials")
}

#[test]
fn every_sampling_material_applies_v_flip() {
    let dir = materials_dir();
    let mut offenders: Vec<String> = Vec::new();
    let mut audited = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read materials dir") {
        let entry = entry.expect("read materials entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("wgsl") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read shader source");
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

    assert!(
        audited > 0,
        "audit found no sampling materials in {:?}",
        dir
    );

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
