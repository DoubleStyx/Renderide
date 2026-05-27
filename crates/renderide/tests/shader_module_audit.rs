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

#[test]
fn auto_exposure_histogram_meters_linear_luminance() -> io::Result<()> {
    let src = source_file(
        manifest_dir()
            .join("shaders/passes/compute")
            .join("auto_exposure_histogram.wgsl"),
    )?;

    assert!(
        src.contains("fn linear_luminance_for_bin"),
        "auto-exposure must reconstruct linear luminance from retained histogram bins"
    );
    assert!(
        src.contains("linear_luminance_sum += f32(bin_count) * linear_luminance_for_bin(i);"),
        "auto-exposure must average retained linear luminance, not retained bin indices"
    );
    assert!(
        src.contains("avg_lum = log2(max(avg_linear_lum, MIN_AVERAGE_LUMINANCE));"),
        "auto-exposure must convert the linear average back to EV before adaptation"
    );
    assert!(
        !src.contains("sum / (f32(count) * 63.0)"),
        "auto-exposure must not compute average luminance from raw histogram bin indices"
    );
    assert!(
        !src.contains("sum += f32(bin_count) * f32(i);"),
        "auto-exposure must not accumulate histogram bin indices as luminance"
    );

    Ok(())
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
fn wireframe_helpers_keep_unity_distance_conventions() -> io::Result<()> {
    let src = module_source("mesh/wireframe.wgsl")?;
    assert!(
        src.contains("fn unity_world_edge_distances")
            && src.contains("return world_edge_distances(barycentric, world_pos) * 0.5;"),
        "common Wireframe world-space mode must keep the source shader's half-altitude edge distances"
    );
    assert!(
        src.contains("fn unity_screen_edge_distances")
            && src.contains("return screen_edge_distances(barycentric) * 2.0;"),
        "common Wireframe screen-space mode must use Unity's doubled clip-to-screen distance scale"
    );
    assert!(
        src.contains("fn line_lerp_from_distances")
            && src.contains("let distance = min_edge_distance(distances);")
            && src.contains("return coverage_from_distance(distance, thickness);"),
        "common Wireframe roots must keep the source shader's dist3 min + fwidth line-lerp shape"
    );
    assert!(
        !src.contains("abs(det) <= 1e-12"),
        "non-screenspace Wireframe must not drop close-up valid triangles through an absolute determinant cutoff"
    );
    assert!(
        src.contains("WIREFRAME_GRAM_DETERMINANT_RELATIVE_EPSILON")
            && src
                .contains("gram_scale * gram_scale * WIREFRAME_GRAM_DETERMINANT_RELATIVE_EPSILON")
            && src.contains("if (!(det > det_floor))"),
        "non-screenspace Wireframe must use a scale-relative determinant floor for world derivative inversion"
    );

    let line_stream_start = src
        .find("fn line_stream_edge_distances")
        .expect("line stream edge helper");
    let next_helper = src[line_stream_start..]
        .find("fn world_gradient_length")
        .expect("following helper")
        + line_stream_start;
    let line_stream_helper = &src[line_stream_start..next_helper];
    assert!(
        line_stream_helper.contains("distances.x")
            && line_stream_helper.contains("distances.z")
            && line_stream_helper.contains("WIREFRAME_FALLBACK_DISTANCE")
            && !line_stream_helper.contains("distances.y"),
        "XSToon wireframe override must match the two-segment LineStream topology and skip the closing edge"
    );

    Ok(())
}

#[test]
fn common_wireframe_roots_use_unity_world_edge_lerp() -> io::Result<()> {
    for file_name in [
        "wireframe.wgsl",
        "wireframedoublesided.wgsl",
        "wireframeunlittransition.wgsl",
    ] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("wf::unity_edge_lerp("),
            "{file_name} must use the source-compatible world-space wire distance helper"
        );
    }

    Ok(())
}

#[test]
fn unlit_polar_variants_use_unity_derivative_selection() -> io::Result<()> {
    let unlit = material_source("unlit.wgsl")?;
    assert!(
        unlit.contains("let mapped = uvu::polar_mapping(in.uv, main_st, mat._PolarPow);")
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
            "let mapped = uvu::polar_mapping(uv_in, mat._MainTex_ST, mat._Pow);"
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
    assert!(
        src.contains("#import renderide::core::texture_sampling as ts")
            && src.contains("_NearTex_LodBias: f32")
            && src.contains("_FarTex_LodBias: f32")
            && src.contains(
                "ts::sample_tex_2d(_NearTex, _NearTex_sampler, near_uv, mat._NearTex_LodBias)"
            )
            && src.contains(
                "ts::sample_tex_2d(_FarTex, _FarTex_sampler, far_uv, mat._FarTex_LodBias)"
            ),
        "UnlitDistanceLerp must apply host mip bias to the filtered near/far samples"
    );
    assert!(
        src.contains("select(-1e-6, 1e-6, mat._Transition >= 0.0)")
            && !src.contains("max(abs(mat._Transition), 1e-6)"),
        "UnlitDistanceLerp must preserve the sign of `_Transition`"
    );
    Ok(())
}

#[test]
fn ui_unlit_mask_uv_matches_source_transform_order() -> io::Result<()> {
    let src = material_source("ui_unlit.wgsl")?;
    assert!(
        src.contains("let uv_main = uvu::apply_st(in.uv, mat._MainTex_ST);")
            && src.contains("let uv_mask = uvu::apply_st(uv_main, mat._MaskTex_ST);")
            && !src.contains("let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);"),
        "ui_unlit.wgsl must transform mask UVs from the already transformed main UV"
    );
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

#[path = "shader_module_audit/camera360.rs"]
mod camera360;
#[path = "shader_module_audit/hygiene.rs"]
mod hygiene;
#[path = "shader_module_audit/material_defaults.rs"]
mod material_defaults;
#[path = "shader_module_audit/pbs.rs"]
mod pbs;
#[path = "shader_module_audit/skybox_implicit_defaults.rs"]
mod skybox_implicit_defaults;
#[path = "shader_module_audit/tangent_basis.rs"]
mod tangent_basis;
#[path = "shader_module_audit/text.rs"]
mod text;
#[path = "shader_module_audit/xiexe_and_probes.rs"]
mod xiexe_and_probes;
