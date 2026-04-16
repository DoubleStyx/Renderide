//! Composes `shaders/source/modules/*.wgsl` with [`naga_oil`] (`#import`), validates with naga, and
//! writes `OUT_DIR/embedded_shaders.rs` (WGSL source embedded as Rust string literals).
//! Also writes flat `shaders/target/*.wgsl` for inspection and diffing; the **crate does not**
//! `include_str!` those paths — embedding avoids compile failures when `shaders/target/` is missing
//! or stale relative to the build script output.
//!
//! Every `*.wgsl` file under `shaders/source/modules/` is registered as a composable module (sorted
//! by path for deterministic builds). Add a new shared module by dropping a file there with a
//! `#define_import_path` matching your `#import` in materials.
//!
//! Material sources under `shaders/source/materials/*.wgsl`: the **file stem** must match
//! `normalize_unity_shader_lookup_key` in the renderide crate (Unity shader **asset** name, e.g.
//! `UI_TextUnlit` → `ui_textunlit` → `ui_textunlit_default` / `ui_textunlit_multiview`).
//! Build **always** emits composed targets `{stem}_default` and `{stem}_multiview` for every material file;
//! each source must handle multiview (typically `#ifdef MULTIVIEW` around `@builtin(view_index)` and
//! view-projection selection in a single `vs_main`, or separate `vs_main` blocks) as documented below.
//!
//! ## Multiview variants (`*_default` / `*_multiview`)
//!
//! Sources use `#ifdef MULTIVIEW` … `#else` … `#endif`. In naga-oil, `#ifdef NAME` is true when
//! `NAME` exists in the compose [`ShaderDefValue`] map **regardless of value** — so
//! `MULTIVIEW` → `ShaderDefValue::Bool(false)` still
//! enables the `#ifdef` branch. For the non-multiview target, **omit** `MULTIVIEW` from `shader_defs`
//! entirely; for the multiview target, set `MULTIVIEW` to [`ShaderDefValue::Bool(true)`].
//!
//! Alternatively, WGSL could use `#if MULTIVIEW == true` with both `Bool(true)` and `Bool(false)`
//! in the map; the omit-key approach matches existing `#ifdef` in source materials without edits.
//!
//! ## Vendored OpenXR loader (Windows)
//!
//! For `windows` targets, copies **one** `openxr_loader.dll` from
//! `../../third_party/openxr_loader/openxr_loader_windows-*/` matching [`CARGO_CFG_TARGET_ARCH`](https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts)
//! into the same **artifact directory** Cargo uses for this build: `target/<PROFILE>/` when
//! [`TARGET`](https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts)
//! equals [`HOST`](https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts),
//! and `target/<TARGET>/<PROFILE>/` when cross-compiling (`--target`). Non-Windows targets skip this
//! (Linux uses the system loader at run time).

/// Khronos `openxr_loader_windows-*` subfolder names for each Rust target arch (shared with `openxr_windows_arch.rs`).
mod openxr_win {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/openxr_windows_arch.rs"
    ));
}

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderLanguage,
    ShaderType,
};

/// Shader defs for material sources that use `#ifdef MULTIVIEW`.
///
/// When `enable` is false, returns an empty map so `#ifdef MULTIVIEW` is not taken. When true,
/// inserts `MULTIVIEW` as [`ShaderDefValue::Bool(true)`].
fn multiview_shader_defs(enable: bool) -> HashMap<String, ShaderDefValue> {
    let mut defs = HashMap::new();
    if enable {
        defs.insert("MULTIVIEW".to_string(), ShaderDefValue::Bool(true));
    }
    defs
}

#[derive(Clone, Debug)]
struct BuildBlendComponent {
    src_factor: &'static str,
    dst_factor: &'static str,
    operation: &'static str,
}

#[derive(Clone, Debug)]
struct BuildPassDirective {
    name: String,
    vertex_entry: String,
    fragment_entry: String,
    depth_compare: &'static str,
    depth_write: bool,
    cull_mode: &'static str,
    blend: Option<(BuildBlendComponent, BuildBlendComponent)>,
    write_mask: &'static str,
    depth_bias_slope_scale: f32,
    depth_bias_constant: i32,
    material_state: &'static str,
}

impl BuildPassDirective {
    fn new(name: String) -> Self {
        Self {
            name,
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            depth_compare: "wgpu::CompareFunction::GreaterEqual",
            depth_write: true,
            cull_mode: "None",
            blend: None,
            write_mask: "wgpu::ColorWrites::COLOR",
            depth_bias_slope_scale: 0.0,
            depth_bias_constant: 0,
            material_state: "crate::materials::MaterialPassState::Static",
        }
    }
}

fn parse_bool_like(value: &str, label: &str, file: &str, line: usize) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => true,
        "0" | "false" | "off" | "no" => false,
        _ => panic!("{file}:{line}: invalid {label} value `{value}`"),
    }
}

fn compare_token(value: &str, file: &str, line: usize) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "never" => "wgpu::CompareFunction::Never",
        "less" => "wgpu::CompareFunction::Less",
        "equal" => "wgpu::CompareFunction::Equal",
        "less_equal" | "lessequal" | "lequal" => "wgpu::CompareFunction::LessEqual",
        "greater" => "wgpu::CompareFunction::Greater",
        "not_equal" | "notequal" => "wgpu::CompareFunction::NotEqual",
        "greater_equal" | "greaterequal" | "gequal" => "wgpu::CompareFunction::GreaterEqual",
        "always" => "wgpu::CompareFunction::Always",
        _ => panic!("{file}:{line}: invalid depth compare `{value}`"),
    }
}

fn cull_token(value: &str, file: &str, line: usize) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" | "off" | "0" => "None",
        "front" => "Some(wgpu::Face::Front)",
        "back" => "Some(wgpu::Face::Back)",
        _ => panic!("{file}:{line}: invalid cull mode `{value}`"),
    }
}

fn color_writes_token(value: &str, file: &str, line: usize) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "all" | "rgba" => "wgpu::ColorWrites::ALL",
        "color" | "rgb" => "wgpu::ColorWrites::COLOR",
        "none" | "off" => "crate::materials::COLOR_WRITES_NONE",
        _ => panic!("{file}:{line}: invalid color write mask `{value}`"),
    }
}

fn material_pass_state_token(value: &str, file: &str, line: usize) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "static" | "none" => "crate::materials::MaterialPassState::Static",
        "base" | "forward_base" | "forwardbase" | "unity_forward_base" => {
            "crate::materials::MaterialPassState::UnityForwardBase"
        }
        "add" | "forward_add" | "forwardadd" | "delta" | "unity_forward_add" => {
            "crate::materials::MaterialPassState::UnityForwardAdd"
        }
        _ => panic!("{file}:{line}: invalid material pass state `{value}`"),
    }
}

fn blend_factor_token(value: &str, file: &str, line: usize) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "zero" | "0" => "wgpu::BlendFactor::Zero",
        "one" | "1" => "wgpu::BlendFactor::One",
        "src" | "src_color" | "srccolor" => "wgpu::BlendFactor::Src",
        "one_minus_src" | "one_minus_src_color" | "oneminussrc" | "oneminussrccolor" => {
            "wgpu::BlendFactor::OneMinusSrc"
        }
        "dst" | "dst_color" | "dstcolor" => "wgpu::BlendFactor::Dst",
        "one_minus_dst" | "one_minus_dst_color" | "oneminusdst" | "oneminusdstcolor" => {
            "wgpu::BlendFactor::OneMinusDst"
        }
        "src_alpha" | "srcalpha" => "wgpu::BlendFactor::SrcAlpha",
        "one_minus_src_alpha" | "oneminussrcalpha" => "wgpu::BlendFactor::OneMinusSrcAlpha",
        "dst_alpha" | "dstalpha" => "wgpu::BlendFactor::DstAlpha",
        "one_minus_dst_alpha" | "oneminusdstalpha" => "wgpu::BlendFactor::OneMinusDstAlpha",
        "constant" | "constant_color" | "constantcolor" => "wgpu::BlendFactor::Constant",
        "one_minus_constant" | "one_minus_constant_color" | "oneminusconstant" => {
            "wgpu::BlendFactor::OneMinusConstant"
        }
        "src_alpha_saturated" | "srcalphasaturated" => "wgpu::BlendFactor::SrcAlphaSaturated",
        _ => panic!("{file}:{line}: invalid blend factor `{value}`"),
    }
}

fn blend_operation_token(value: &str, file: &str, line: usize) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "add" => "wgpu::BlendOperation::Add",
        "subtract" | "sub" => "wgpu::BlendOperation::Subtract",
        "reverse_subtract" | "revsub" => "wgpu::BlendOperation::ReverseSubtract",
        "min" => "wgpu::BlendOperation::Min",
        "max" => "wgpu::BlendOperation::Max",
        _ => panic!("{file}:{line}: invalid blend operation `{value}`"),
    }
}

fn parse_blend_component(
    parts: &[&str],
    index: &mut usize,
    first_value: &str,
    file: &str,
    line: usize,
) -> BuildBlendComponent {
    if *index + 2 >= parts.len() {
        panic!("{file}:{line}: blend component needs src,dst,op");
    }
    let src = blend_factor_token(first_value, file, line);
    *index += 1;
    let dst = blend_factor_token(parts[*index], file, line);
    *index += 1;
    let op = blend_operation_token(parts[*index], file, line);
    BuildBlendComponent {
        src_factor: src,
        dst_factor: dst,
        operation: op,
    }
}

fn parse_pass_directives(source: &str, file: &str) -> Vec<BuildPassDirective> {
    let mut passes = Vec::new();
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#pass") else {
            continue;
        };
        let rest = rest.trim();
        let (name, body) = rest.split_once(':').unwrap_or_else(|| {
            panic!("{file}:{line_no}: pass directive must be `//#pass name: key=value`")
        });
        let mut pass = BuildPassDirective::new(name.trim().to_string());
        let mut color_blend: Option<BuildBlendComponent> = None;
        let mut alpha_blend: Option<BuildBlendComponent> = None;
        let mut blend_disabled = false;

        let parts = body
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        let mut i = 0usize;
        while i < parts.len() {
            let (key, value) = parts[i].split_once('=').unwrap_or_else(|| {
                panic!("{file}:{line_no}: expected key=value in `{}`", parts[i])
            });
            match key.trim().to_ascii_lowercase().as_str() {
                "vs" | "vertex" => pass.vertex_entry = value.trim().to_string(),
                "fs" | "fragment" => pass.fragment_entry = value.trim().to_string(),
                "depth" | "ztest" => pass.depth_compare = compare_token(value, file, line_no),
                "zwrite" | "depth_write" => {
                    pass.depth_write = parse_bool_like(value, "zwrite", file, line_no)
                }
                "cull" => pass.cull_mode = cull_token(value, file, line_no),
                "write" | "writes" | "color_write" | "colorwrites" => {
                    pass.write_mask = color_writes_token(value, file, line_no)
                }
                "bias" | "depth_bias" => {
                    pass.depth_bias_constant = value.trim().parse().unwrap_or_else(|_| {
                        panic!("{file}:{line_no}: invalid depth bias `{value}`")
                    });
                }
                "slope" | "slope_bias" => {
                    pass.depth_bias_slope_scale = value.trim().parse().unwrap_or_else(|_| {
                        panic!("{file}:{line_no}: invalid slope bias `{value}`")
                    });
                }
                "material" | "material_state" => {
                    pass.material_state = material_pass_state_token(value, file, line_no);
                }
                "blend" => {
                    if value.trim().eq_ignore_ascii_case("none") {
                        blend_disabled = true;
                        color_blend = None;
                        alpha_blend = None;
                    } else if value.trim().eq_ignore_ascii_case("alpha") {
                        color_blend = Some(BuildBlendComponent {
                            src_factor: "wgpu::BlendFactor::SrcAlpha",
                            dst_factor: "wgpu::BlendFactor::OneMinusSrcAlpha",
                            operation: "wgpu::BlendOperation::Add",
                        });
                        alpha_blend = Some(BuildBlendComponent {
                            src_factor: "wgpu::BlendFactor::One",
                            dst_factor: "wgpu::BlendFactor::OneMinusSrcAlpha",
                            operation: "wgpu::BlendOperation::Add",
                        });
                    } else {
                        blend_disabled = false;
                        color_blend =
                            Some(parse_blend_component(&parts, &mut i, value, file, line_no));
                    }
                }
                "alpha" => {
                    blend_disabled = false;
                    alpha_blend = Some(parse_blend_component(&parts, &mut i, value, file, line_no));
                }
                _ => panic!("{file}:{line_no}: unknown pass key `{key}`"),
            }
            i += 1;
        }

        if !blend_disabled {
            if let Some(color) = color_blend {
                let alpha = alpha_blend.unwrap_or_else(|| color.clone());
                pass.blend = Some((color, alpha));
                pass.write_mask = "wgpu::ColorWrites::ALL";
            } else if let Some(alpha) = alpha_blend {
                pass.blend = Some((
                    BuildBlendComponent {
                        src_factor: "wgpu::BlendFactor::One",
                        dst_factor: "wgpu::BlendFactor::Zero",
                        operation: "wgpu::BlendOperation::Add",
                    },
                    alpha,
                ));
                pass.write_mask = "wgpu::ColorWrites::ALL";
            }
        }

        passes.push(pass);
    }
    passes
}

fn blend_component_literal(c: &BuildBlendComponent) -> String {
    format!(
        "wgpu::BlendComponent {{ src_factor: {}, dst_factor: {}, operation: {} }}",
        c.src_factor, c.dst_factor, c.operation
    )
}

fn pass_literal(pass: &BuildPassDirective) -> String {
    let blend = match &pass.blend {
        Some((color, alpha)) => format!(
            "Some(wgpu::BlendState {{ color: {}, alpha: {} }})",
            blend_component_literal(color),
            blend_component_literal(alpha)
        ),
        None => "None".to_string(),
    };
    format!(
        "crate::materials::MaterialPassDesc {{ name: {name:?}, vertex_entry: {vs:?}, fragment_entry: {fs:?}, depth_compare: {depth}, depth_write: {zwrite}, cull_mode: {cull}, blend: {blend}, write_mask: {write}, depth_bias_slope_scale: {slope:?}, depth_bias_constant: {bias}, material_state: {material_state} }}",
        name = pass.name.as_str(),
        vs = pass.vertex_entry.as_str(),
        fs = pass.fragment_entry.as_str(),
        depth = pass.depth_compare,
        zwrite = pass.depth_write,
        cull = pass.cull_mode,
        blend = blend,
        write = pass.write_mask,
        slope = pass.depth_bias_slope_scale,
        bias = pass.depth_bias_constant,
        material_state = pass.material_state,
    )
}

/// Validates `module`, writes WGSL to `out_path`, returns the same string for embedding in Rust.
fn validate_and_write_wgsl(
    module: &naga::Module,
    label: &str,
    out_path: &std::path::Path,
    expect_view_index: Option<bool>,
    passes: &[BuildPassDirective],
) -> String {
    if passes.is_empty() {
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
        let has_fs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Fragment && e.name == "fs_main");
        if !has_vs || !has_fs {
            panic!(
                "{label}: expected entry points vs_main and fs_main (vertex={has_vs} fragment={has_fs})",
            );
        }
    } else {
        for pass in passes {
            let has_vs = module.entry_points.iter().any(|e| {
                e.stage == naga::ShaderStage::Vertex && e.name == pass.vertex_entry.as_str()
            });
            let has_fs = module.entry_points.iter().any(|e| {
                e.stage == naga::ShaderStage::Fragment && e.name == pass.fragment_entry.as_str()
            });
            if !has_vs || !has_fs {
                panic!(
                    "{label}: pass `{}` expected entry points {} and {} (vertex={has_vs} fragment={has_fs})",
                    pass.name, pass.vertex_entry, pass.fragment_entry
                );
            }
        }
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("validate {label}: {e}"));
    let wgsl = naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .unwrap_or_else(|e| panic!("wgsl out {label}: {e}"));
    if let Some(want) = expect_view_index {
        let has = wgsl.contains("@builtin(view_index)");
        if want != has {
            panic!(
                "{label}: expected @builtin(view_index) {} in output (multiview shader_defs contract)",
                if want { "present" } else { "absent" }
            );
        }
    }
    fs::write(out_path, &wgsl).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    wgsl
}

/// Escapes `s` as a Rust `str` literal token (same as `format!("{s:?}")`).
fn rust_string_literal_token(s: &str) -> String {
    format!("{s:?}")
}

/// Loads every `*.wgsl` under `shaders/source/modules/` relative to `manifest_dir`.
///
/// Returns `(file_path, source)` where `file_path` uses forward slashes (e.g.
/// `shaders/source/modules/globals.wgsl`) for [`ComposableModuleDescriptor::file_path`].
fn discover_shader_modules(manifest_dir: &Path) -> Vec<(String, String)> {
    let modules_dir = manifest_dir.join("shaders/source/modules");
    let mut paths: Vec<PathBuf> = fs::read_dir(&modules_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", modules_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    paths.sort();

    let mut modules = Vec::with_capacity(paths.len());
    for path in paths {
        let source =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let rel = path.strip_prefix(manifest_dir).unwrap_or_else(|_| {
            panic!(
                "module path {} is not under manifest {}",
                path.display(),
                manifest_dir.display()
            )
        });
        let file_path = rel.to_string_lossy().replace('\\', "/");
        modules.push((file_path, source));
    }

    if modules.is_empty() {
        panic!(
            "no *.wgsl modules under {} (naga-oil imports will fail)",
            modules_dir.display()
        );
    }

    modules
}

fn register_composable_modules(composer: &mut Composer, modules: &[(String, String)]) {
    for (file_path, source) in modules {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: source.as_str(),
                file_path: file_path.as_str(),
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("add composable module {file_path}: {e}"));
    }
}

fn compose_material(
    modules: &[(String, String)],
    material_source: &str,
    material_file_path: &str,
    shader_defs: HashMap<String, ShaderDefValue>,
) -> naga::Module {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    register_composable_modules(&mut composer, modules);
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: material_source,
            file_path: material_file_path,
            shader_type: ShaderType::Wgsl,
            shader_defs,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {material_file_path}: {e}"))
}

/// Picks the lexicographically last `openxr_loader_windows-*` directory so newer SDK versions win.
fn find_latest_openxr_windows_package_dir(third_party_openxr: &Path) -> Option<PathBuf> {
    let rd = fs::read_dir(third_party_openxr).ok()?;
    let mut candidates: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("openxr_loader_windows-"))
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next_back()
}

/// Directory where Cargo places binaries for the current package build (`renderide.exe`, etc.).
///
/// Cargo uses `target/<PROFILE>/` for the default host target and `target/<TARGET>/<PROFILE>/` when
/// `--target` selects a different triple; copying the loader only to `target/<PROFILE>/` misses
/// cross-compiled outputs (e.g. `x86_64-pc-windows-gnu`).
fn cargo_artifact_profile_dir(cargo_target_dir: &Path, profile: &str) -> PathBuf {
    let target = std::env::var("TARGET").expect("TARGET");
    let host = std::env::var("HOST").expect("HOST");
    if target == host {
        cargo_target_dir.join(profile)
    } else {
        cargo_target_dir.join(target).join(profile)
    }
}

/// Copies the Khronos OpenXR loader DLL next to the build output for Windows targets only.
fn copy_vendored_openxr_loader_windows(manifest_dir: &Path) {
    let Ok(target_os) = std::env::var("CARGO_CFG_TARGET_OS") else {
        return;
    };
    if target_os != "windows" {
        return;
    }

    let Ok(arch) = std::env::var("CARGO_CFG_TARGET_ARCH") else {
        println!("cargo:warning=openxr_loader: CARGO_CFG_TARGET_ARCH unset");
        return;
    };

    let Some(subdir) = openxr_win::khronos_windows_subdir_for_arch(&arch) else {
        println!("cargo:warning=openxr_loader: no vendored Khronos folder for target arch {arch}");
        return;
    };

    let workspace_dir = manifest_dir.join("../..");
    let third_party = workspace_dir.join("third_party/openxr_loader");
    println!("cargo:rerun-if-changed={}", third_party.display());

    let Some(pkg_root) = find_latest_openxr_windows_package_dir(&third_party) else {
        println!(
            "cargo:warning=openxr_loader: no openxr_loader_windows-* under {}",
            third_party.display()
        );
        return;
    };

    let src = pkg_root.join(subdir).join("openxr_loader.dll");
    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        println!(
            "cargo:warning=openxr_loader: missing vendored DLL at {}",
            src.display()
        );
        return;
    }

    let cargo_target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("../../target"));
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());

    let dest_dir = cargo_artifact_profile_dir(&cargo_target_dir, &profile);
    if let Err(e) = fs::create_dir_all(&dest_dir) {
        println!(
            "cargo:warning=openxr_loader: mkdir {} failed: {e}",
            dest_dir.display()
        );
        return;
    }
    let dest = dest_dir.join("openxr_loader.dll");
    if let Err(e) = fs::copy(&src, &dest) {
        println!(
            "cargo:warning=openxr_loader: copy {} -> {} failed: {e}",
            src.display(),
            dest.display()
        );
    }
}

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    copy_vendored_openxr_loader_windows(&manifest_dir);
    let materials_dir = manifest_dir.join("shaders/source/materials");
    let target_dir = manifest_dir.join("shaders/target");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed=shaders/source");
    println!("cargo:rerun-if-changed=build.rs");

    fs::create_dir_all(materials_dir.parent().expect("materials parent")).expect("mkdir");
    fs::create_dir_all(&target_dir).expect("mkdir target");

    let shader_modules = discover_shader_modules(&manifest_dir);

    let mut embedded_arms = String::new();
    let mut embedded_pass_arms = String::new();
    let mut output_stems: Vec<String> = Vec::new();

    let mut material_paths: Vec<PathBuf> = fs::read_dir(&materials_dir)
        .expect("read materials")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "wgsl").unwrap_or(false))
        .collect();
    material_paths.sort();

    for path in &material_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("material stem");

        let material_source =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let material_file_path = path.to_str().expect("utf8 path");
        let pass_directives = parse_pass_directives(&material_source, material_file_path);

        let variants = [
            (format!("{stem}_default"), false),
            (format!("{stem}_multiview"), true),
        ];
        for (target_stem, multiview) in variants {
            let defs = multiview_shader_defs(multiview);
            let module =
                compose_material(&shader_modules, &material_source, material_file_path, defs);
            let label = format!("{target_stem} (MULTIVIEW={multiview})");
            let out_path = target_dir.join(format!("{target_stem}.wgsl"));
            let wgsl = validate_and_write_wgsl(
                &module,
                &label,
                &out_path,
                Some(multiview),
                &pass_directives,
            );
            let lit = rust_string_literal_token(&wgsl);
            embedded_arms.push_str(&format!("        \"{target_stem}\" => Some({lit}),\n"));
            if !pass_directives.is_empty() {
                let pass_literals = pass_directives
                    .iter()
                    .map(pass_literal)
                    .collect::<Vec<_>>()
                    .join(",\n            ");
                embedded_pass_arms.push_str(&format!(
                    "        \"{target_stem}\" => &[\n            {pass_literals},\n        ],\n"
                ));
            }
            output_stems.push(target_stem);
        }
    }

    let stems_list = output_stems
        .iter()
        .map(|s| format!("    \"{s}\","))
        .collect::<Vec<_>>()
        .join("\n");

    let embedded_rs = format!(
        r#"// Generated by `build.rs` — do not edit.

/// Flattened WGSL for `stem` (also written under `shaders/target/{{stem}}.wgsl` at build time).
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {{
    match stem {{
{embedded_arms}        _ => None,
    }}
}}

/// Declared render passes for `stem`, parsed from `//#pass` directives in the source WGSL.
pub fn embedded_target_passes(stem: &str) -> &'static [crate::materials::MaterialPassDesc] {{
    match stem {{
{embedded_pass_arms}        _ => &[],
    }}
}}

/// Stems under `shaders/target/*.wgsl` present at build time.
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{stems}
];
"#,
        embedded_arms = embedded_arms,
        embedded_pass_arms = embedded_pass_arms,
        stems = stems_list
    );

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs).expect("write embedded_shaders.rs");
}
