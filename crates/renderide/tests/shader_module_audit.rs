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

fn source_file(path: impl AsRef<Path>) -> io::Result<String> {
    fs::read_to_string(path).map(normalize_line_endings)
}

fn normalize_line_endings(src: String) -> String {
    if src.contains('\r') {
        src.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        src
    }
}

#[test]
fn source_normalization_accepts_windows_line_endings() {
    assert_eq!(
        normalize_line_endings("line one\r\nline two\rline three".to_owned()),
        "line one\nline two\nline three"
    );
}

fn material_source(file_name: &str) -> io::Result<String> {
    source_file(manifest_dir().join("shaders/materials").join(file_name))
}

fn module_source(file_name: &str) -> io::Result<String> {
    source_file(manifest_dir().join("shaders/modules").join(file_name))
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

fn declares_u32_field(src: &str, field_name: &str) -> bool {
    src.lines().any(|line| {
        let trimmed = line.trim();
        let Some((name, ty)) = trimmed.split_once(':') else {
            return false;
        };
        name.trim() == field_name && ty.trim_start().starts_with("u32")
    })
}

#[test]
fn unlit_uses_reserved_variant_bits_instead_of_keyword_uniform_fields() -> io::Result<()> {
    let src = material_source("unlit.wgsl")?;
    assert!(src.contains("_RenderideVariantBits: u32"));

    for field_name in [
        "_ALPHATEST",
        "_ALPHATEST_ON",
        "_ALPHABLEND_ON",
        "_COLOR",
        "_MASK_TEXTURE_CLIP",
        "_MASK_TEXTURE_MUL",
        "_MUL_ALPHA_INTENSITY",
        "_MUL_RGB_BY_ALPHA",
        "_OFFSET_TEXTURE",
        "_POLARUV",
        "_RIGHT_EYE_ST",
        "_TEXTURE",
        "_TEXTURE_NORMALMAP",
        "_VERTEX_LINEAR_COLOR",
        "_VERTEX_SRGB_COLOR",
        "_VERTEXCOLORS",
    ] {
        assert!(
            !declares_f32_field(&src, field_name),
            "{field_name} must be decoded from _RenderideVariantBits instead of packed as f32"
        );
    }

    for (constant_name, bit_index) in [
        ("UNLIT_KW_ALPHATEST", 0),
        ("UNLIT_KW_COLOR", 1),
        ("UNLIT_KW_MASK_TEXTURE_CLIP", 2),
        ("UNLIT_KW_MASK_TEXTURE_MUL", 3),
        ("UNLIT_KW_MUL_ALPHA_INTENSITY", 4),
        ("UNLIT_KW_MUL_RGB_BY_ALPHA", 5),
        ("UNLIT_KW_OFFSET_TEXTURE", 6),
        ("UNLIT_KW_POLARUV", 7),
        ("UNLIT_KW_RIGHT_EYE_ST", 8),
        ("UNLIT_KW_TEXTURE", 9),
        ("UNLIT_KW_TEXTURE_NORMALMAP", 10),
        ("UNLIT_KW_VERTEX_LINEAR_COLOR", 11),
        ("UNLIT_KW_VERTEX_SRGB_COLOR", 12),
        ("UNLIT_KW_VERTEXCOLORS", 13),
    ] {
        assert!(
            src.contains(&format!("const {constant_name}: u32 = 1u << {bit_index}u;")),
            "{constant_name} must match the Froox sorted UniqueKeywords bit order"
        );
    }

    assert!(src.contains("tex_color = tex_color * mat._Color;"));
    assert!(src.contains("color = mat._Color;"));
    Ok(())
}

#[test]
fn unlit_polar_variants_use_unity_derivative_selection() -> io::Result<()> {
    let unlit = material_source("unlit.wgsl")?;
    assert!(
        unlit
            .contains("let mapped = uvu::polar_mapping(in.uv, main_st, max(mat._PolarPow, 1e-4));")
            && unlit.contains("ddx_uv = mapped.ddx_uv;")
            && unlit.contains("ddy_uv = mapped.ddy_uv;")
            && unlit.contains("textureSampleGrad(_Tex, _Tex_sampler, uv_main, ddx_uv, ddy_uv)"),
        "Unlit must use the shared Unity polar derivative-selection helper before textureSampleGrad"
    );
    assert!(
        !unlit.contains("let polar = uvu::polar_uv(in.uv"),
        "Unlit must not reconstruct polar derivatives with raw dpdx/dpdy"
    );

    let polar = material_source("unlitpolarmapping.wgsl")?;
    assert!(
        polar.contains(
            "let mapped = uvu::polar_mapping(uv_in, mat._MainTex_ST, max(mat._Pow, 1e-4));"
        ) && polar.contains(
            "textureSampleGrad(_MainTex, _MainTex_sampler, mapped.uv, mapped.ddx_uv, mapped.ddy_uv)"
        ),
        "UnlitPolarMapping must use the shared Unity polar derivative-selection helper"
    );
    assert!(
        !polar.contains("dpdx(polar_st)") && !polar.contains("dpdy(polar_st)"),
        "UnlitPolarMapping must not use raw derivatives across the polar seam"
    );
    Ok(())
}

#[test]
fn unlitdistancelerp_matches_sorted_keyword_bits_and_fragment_parity() -> io::Result<()> {
    let src = material_source("unlitdistancelerp.wgsl")?;
    for (constant_name, bit_index) in [
        ("UNLITDISTANCELERP_KW_ALPHATEST", 0),
        ("UNLITDISTANCELERP_KW_VERTEXCOLORS", 1),
        ("UNLITDISTANCELERP_KW_LOCAL_SPACE", 2),
        ("UNLITDISTANCELERP_KW_WORLD_SPACE", 3),
    ] {
        assert!(
            src.contains(&format!("const {constant_name}: u32 = 1u << {bit_index}u;")),
            "{constant_name} must match the Froox sorted UniqueKeywords bit order"
        );
    }
    assert!(
        src.contains("UNLITDISTANCELERP_SPACE_GROUP")
            && src.contains("(mat._RenderideVariantBits & UNLITDISTANCELERP_SPACE_GROUP) == 0u")
            && src.contains("return true;"),
        "UnlitDistanceLerp must default the WORLD_SPACE/LOCAL_SPACE group to WORLD_SPACE"
    );
    for forbidden in [
        "near = near * in.color",
        "far = far * in.color",
        "select(1.0, in.color.a",
    ] {
        assert!(
            !src.contains(forbidden),
            "UnlitDistanceLerp Unity fragment does not apply `_VERTEXCOLORS`; found `{forbidden}`"
        );
    }
    Ok(())
}

fn count_font_atlas_lod_bias_samples(src: &str) -> usize {
    src.match_indices("ts::sample_tex_2d(")
        .filter(|(sample_pos, _)| {
            let call = &src[*sample_pos..];
            let call_end = call.find(");").unwrap_or(call.len());
            call[..call_end].contains("_FontAtlas")
        })
        .count()
}

#[path = "shader_module_audit/hygiene.rs"]
mod hygiene;
#[path = "shader_module_audit/material_defaults.rs"]
mod material_defaults;
#[path = "shader_module_audit/pbs.rs"]
mod pbs;
#[path = "shader_module_audit/tangent_basis.rs"]
mod tangent_basis;
#[path = "shader_module_audit/text.rs"]
mod text;
#[path = "shader_module_audit/xiexe_and_probes.rs"]
mod xiexe_and_probes;
