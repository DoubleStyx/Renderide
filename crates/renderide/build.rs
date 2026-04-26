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
//! ## Non-material shaders (`post/`, `backend/`, `compute/`, `present/`)
//!
//! Every `.wgsl` under these directories is composed and registered. The build script composes
//! each source twice — once without `MULTIVIEW` and once with `MULTIVIEW = Bool(true)` — and
//! compares the flattened WGSL:
//!
//! - If the two outputs are byte-identical, the shader does not vary on multiview and a single
//!   target `{stem}.wgsl` is emitted with no suffix.
//! - Otherwise both `{stem}_default.wgsl` and `{stem}_multiview.wgsl` are emitted, matching the
//!   materials convention.
//!
//! Runtime code loads any target via [`crate::embedded_shaders::embedded_target_wgsl`]. Compute
//! sources pass `validate_view_index = false` because WGSL grammar forbids `@builtin(view_index)`
//! outside fragment entry points.
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
//! ## Vendored `OpenXR` loader (Windows)
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

use hashbrown::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use naga::back::wgsl::WriterFlags;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue, ShaderLanguage,
    ShaderType,
};
use thiserror::Error;

/// Errors from shader discovery, composition, validation, and generated code I/O.
#[derive(Debug, Error)]
enum BuildError {
    /// User-facing message (directive parse, validation, naga errors).
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Required Cargo environment variable missing.
    #[error("missing build environment variable `{0}`")]
    MissingEnv(&'static str),
}

fn env_var(name: &'static str) -> Result<String, BuildError> {
    #[expect(
        clippy::map_err_ignore,
        reason = "MissingEnv carries the variable name; `VarError` provides no additional detail"
    )]
    std::env::var(name).map_err(|_| BuildError::MissingEnv(name))
}

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

/// One declared pass: the [`PassKind`] tag and the fragment entry point it sits above.
#[derive(Clone, Debug)]
struct BuildPassDirective {
    /// Path to the [`crate::materials::PassKind`] variant (e.g. `"Forward"`).
    kind_variant: &'static str,
    /// Fragment entry point name the `//#pass` tag sits above.
    fragment_entry: String,
    /// Vertex entry point for this pass. Defaults to `vs_main`; overridden via `vs=...` on the tag.
    vertex_entry: String,
}

/// Maps a `//#pass <kind>` value to the matching [`crate::materials::PassKind`] variant name.
fn pass_kind_variant(value: &str, file: &str, line: usize) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "forward" => Ok("Forward"),
        "outline" => Ok("Outline"),
        "stencil" => Ok("Stencil"),
        "depth_prepass" | "depthprepass" | "prepass" => Ok("DepthPrepass"),
        "overlay_front" | "overlayfront" | "front" => Ok("OverlayFront"),
        "overlay_behind" | "overlaybehind" | "behind" => Ok("OverlayBehind"),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: unknown `//#pass` kind `{value}`"
        ))),
    }
}

/// Finds the first `@fragment` entry point declared after `start_line` (0-based).
///
/// Naga composes to one line per function signature, so the entry point definition lives either on
/// the immediately following non-blank line (`@fragment fn name(...) ...`) or split across the
/// `@fragment` line followed by `fn name(...)`. Both layouts are handled.
fn next_fragment_entry_after(
    source_lines: &[&str],
    start_line: usize,
    file: &str,
    directive_line_no: usize,
) -> Result<String, BuildError> {
    let mut saw_attribute = false;
    for line in &source_lines[start_line..] {
        let trimmed = line.trim_start();
        if !saw_attribute {
            if trimmed.starts_with("//") || trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("@fragment") {
                let rest = rest.trim_start();
                if let Some(name) = parse_fn_name(rest) {
                    return Ok(name);
                }
                saw_attribute = true;
                continue;
            }
            return Err(BuildError::Message(format!(
                "{file}:{directive_line_no}: `//#pass` tag must immediately precede an `@fragment` entry point"
            )));
        }
        if trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }
        if let Some(name) = parse_fn_name(trimmed) {
            return Ok(name);
        }
        return Err(BuildError::Message(format!(
            "{file}:{directive_line_no}: expected `fn <name>(...)` after `@fragment` attribute"
        )));
    }
    Err(BuildError::Message(format!(
        "{file}:{directive_line_no}: `//#pass` tag has no following `@fragment` entry point"
    )))
}

/// Parses `fn <name>(...)` out of a line, returning `<name>` if present.
fn parse_fn_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("fn ")?.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    Some(rest[..end].to_string())
}

fn parse_pass_directives(source: &str, file: &str) -> Result<Vec<BuildPassDirective>, BuildError> {
    let lines: Vec<&str> = source.lines().collect();
    let mut passes = Vec::new();
    for (line_idx, line) in lines.iter().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#pass") else {
            continue;
        };
        let body = rest.trim();
        if body.is_empty() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#pass` tag requires a kind (e.g. `//#pass forward`)"
            )));
        }
        let mut tokens = body.split_whitespace();
        let kind_value = tokens.next().unwrap_or("");
        let kind_variant = pass_kind_variant(kind_value, file, line_no)?;
        let mut vertex_entry = "vs_main".to_string();
        for token in tokens {
            let (key, value) = token.split_once('=').ok_or_else(|| {
                BuildError::Message(format!(
                    "{file}:{line_no}: expected `key=value` after kind in `//#pass`, got `{token}`"
                ))
            })?;
            match key.trim().to_ascii_lowercase().as_str() {
                "vs" | "vertex" => vertex_entry = value.trim().to_string(),
                _ => {
                    return Err(BuildError::Message(format!(
                        "{file}:{line_no}: unknown `//#pass` override `{key}` (only `vs=` is allowed)"
                    )));
                }
            }
        }
        let fragment_entry = next_fragment_entry_after(&lines, line_idx + 1, file, line_no)?;
        passes.push(BuildPassDirective {
            kind_variant,
            fragment_entry,
            vertex_entry,
        });
    }
    Ok(passes)
}

fn pass_literal(pass: &BuildPassDirective) -> String {
    if pass.vertex_entry == "vs_main" {
        format!(
            "crate::materials::pass_from_kind(crate::materials::PassKind::{kind}, {fs:?})",
            kind = pass.kind_variant,
            fs = pass.fragment_entry.as_str(),
        )
    } else {
        format!(
            "crate::materials::MaterialPassDesc {{ vertex_entry: {vs:?}, ..crate::materials::pass_from_kind(crate::materials::PassKind::{kind}, {fs:?}) }}",
            kind = pass.kind_variant,
            fs = pass.fragment_entry.as_str(),
            vs = pass.vertex_entry.as_str(),
        )
    }
}

/// Checks that `module` declares the entry points required by `passes` (or the default
/// `vs_main` + `fs_main` pair when no `//#material` directives are present). Compute-only
/// shaders pass an empty `passes` slice and must declare at least one compute entry point —
/// enforced separately by [`validate_compute_entry_point`].
fn validate_entry_points(
    module: &naga::Module,
    label: &str,
    passes: &[BuildPassDirective],
) -> Result<(), BuildError> {
    if passes.is_empty() {
        let has_compute = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Compute);
        if has_compute {
            return Ok(());
        }
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == "vs_main");
        // Non-material raster shaders (post/backend/present) may declare any number of
        // `@fragment` entry points — pipelines pick which one to compile via
        // [`wgpu::FragmentState::entry_point`]. The build script only needs to confirm at
        // least one fragment stage exists alongside `vs_main`.
        let has_any_fs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Fragment);
        if !has_vs || !has_any_fs {
            return Err(BuildError::Message(format!(
                "{label}: expected a vs_main vertex entry point and at least one @fragment \
                 entry point (vertex={has_vs} fragment={has_any_fs})",
            )));
        }
        return Ok(());
    }
    for pass in passes {
        let has_vs = module
            .entry_points
            .iter()
            .any(|e| e.stage == naga::ShaderStage::Vertex && e.name == pass.vertex_entry.as_str());
        let has_fs = module.entry_points.iter().any(|e| {
            e.stage == naga::ShaderStage::Fragment && e.name == pass.fragment_entry.as_str()
        });
        if !has_vs || !has_fs {
            return Err(BuildError::Message(format!(
                "{label}: pass `{}` expected entry points {} and {} (vertex={has_vs} fragment={has_fs})",
                pass.kind_variant, pass.vertex_entry, pass.fragment_entry
            )));
        }
    }
    Ok(())
}

/// Canonical Unity pipeline-state property names that must NEVER appear in a material's
/// `@group(1) @binding(0)` uniform struct.
///
/// Pipeline state (blend, depth compare/write, cull, stencil, color mask, depth bias) is consumed
/// by [`crate::materials::MaterialPipelineCacheKey`] via
/// [`crate::materials::MaterialBlendMode`] + [`crate::materials::MaterialRenderState`], never by
/// shader code. Embedding any of these names in a shader uniform wastes uniform space and blurs
/// the boundary between "what the shader needs" and "what the pipeline needs".
///
/// Keep this list in sync with `MaterialPipelinePropertyIds::new` in
/// `src/materials/material_passes.rs`.
const PIPELINE_STATE_PROPERTY_NAMES: &[&str] = &[
    "_SrcBlend",
    "_SrcBlendBase",
    "_SrcBlendAdd",
    "_DstBlend",
    "_DstBlendBase",
    "_DstBlendAdd",
    "_ZWrite",
    "_ZTest",
    "_Cull",
    "_Stencil",
    "_StencilComp",
    "_StencilOp",
    "_StencilFail",
    "_StencilZFail",
    "_StencilReadMask",
    "_StencilWriteMask",
    "_ColorMask",
    "_OffsetFactor",
    "_OffsetUnits",
];

/// Rejects any material whose `@group(1) @binding(0)` uniform struct declares a member named in
/// [`PIPELINE_STATE_PROPERTY_NAMES`]. Run after composition so imports and module merges don't
/// hide a leak.
fn validate_no_pipeline_state_uniform_fields(
    module: &naga::Module,
    label: &str,
) -> Result<(), BuildError> {
    for (_, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else {
            continue;
        };
        if binding.group != 1 || binding.binding != 0 {
            continue;
        }
        if !matches!(var.space, naga::AddressSpace::Uniform) {
            continue;
        }
        let ty = &module.types[var.ty];
        let naga::TypeInner::Struct { ref members, .. } = ty.inner else {
            continue;
        };
        for member in members {
            let Some(name) = member.name.as_deref() else {
                continue;
            };
            if PIPELINE_STATE_PROPERTY_NAMES.contains(&name) {
                let struct_name = ty.name.as_deref().unwrap_or("<unnamed>");
                return Err(BuildError::Message(format!(
                    "{label}: material uniform struct `{struct_name}` declares pipeline-state \
                     field `{name}` at @group(1) @binding(0). Pipeline-state properties \
                     (blend, depth, cull, stencil, color mask, depth bias) flow through \
                     MaterialBlendMode + MaterialRenderState and are baked into \
                     MaterialPipelineCacheKey; they must never appear in a shader uniform. \
                     Remove the field from the WGSL struct."
                )));
            }
        }
    }
    Ok(())
}

/// Validates `module` with naga and flattens it back to WGSL. Returns the WGSL string without
/// writing it to disk — callers decide whether/where to persist it.
fn module_to_wgsl(module: &naga::Module, label: &str) -> Result<String, BuildError> {
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| BuildError::Message(format!("validate {label}: {e}")))?;
    naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .map_err(|e| BuildError::Message(format!("wgsl out {label}: {e}")))
}

/// Escapes `s` as a Rust `str` literal token (same as `format!("{s:?}")`).
fn rust_string_literal_token(s: &str) -> String {
    format!("{s:?}")
}

/// Loads every `*.wgsl` under `shaders/source/modules/` relative to `manifest_dir`.
///
/// Returns `(file_path, source)` where `file_path` uses forward slashes (e.g.
/// `shaders/source/modules/globals.wgsl`) for [`ComposableModuleDescriptor::file_path`].
fn discover_shader_modules(manifest_dir: &Path) -> Result<Vec<(String, String)>, BuildError> {
    let modules_dir = manifest_dir.join("shaders/source/modules");
    let mut paths: Vec<PathBuf> = fs::read_dir(&modules_dir)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", modules_dir.display())))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    paths.sort();

    let mut modules = Vec::with_capacity(paths.len());
    for path in paths {
        let source = fs::read_to_string(&path)
            .map_err(|e| BuildError::Message(format!("read {}: {e}", path.display())))?;
        let rel = path.strip_prefix(manifest_dir).map_err(|e| {
            BuildError::Message(format!(
                "module path {} is not under manifest {}: {e}",
                path.display(),
                manifest_dir.display()
            ))
        })?;
        let file_path = rel.to_string_lossy().replace('\\', "/");
        modules.push((file_path, source));
    }

    if modules.is_empty() {
        return Err(BuildError::Message(format!(
            "no *.wgsl modules under {} (naga-oil imports will fail)",
            modules_dir.display()
        )));
    }

    Ok(modules)
}

fn register_composable_modules(
    composer: &mut Composer,
    modules: &[(String, String)],
) -> Result<(), BuildError> {
    for (file_path, source) in modules {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source: source.as_str(),
                file_path: file_path.as_str(),
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .map_err(|e| BuildError::Message(format!("add composable module {file_path}: {e}")))?;
    }
    Ok(())
}

fn compose_material(
    modules: &[(String, String)],
    material_source: &str,
    material_file_path: &str,
    shader_defs: HashMap<String, ShaderDefValue>,
) -> Result<naga::Module, BuildError> {
    let mut composer = Composer::default().with_capabilities(Capabilities::all());
    register_composable_modules(&mut composer, modules)?;
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: material_source,
            file_path: material_file_path,
            shader_type: ShaderType::Wgsl,
            // naga_oil 0.22 `NagaModuleDescriptor::shader_defs` is `std::collections::HashMap` (not hashbrown).
            shader_defs: std::collections::HashMap::from_iter(shader_defs),
            ..Default::default()
        })
        .map_err(|e| BuildError::Message(format!("compose {material_file_path}: {e}")))
}

/// Picks the lexicographically last `openxr_loader_windows-*` directory so newer SDK versions win.
fn find_latest_openxr_windows_package_dir(third_party_openxr: &Path) -> Option<PathBuf> {
    let rd = fs::read_dir(third_party_openxr).ok()?;
    let mut candidates: Vec<PathBuf> = rd
        .filter_map(std::result::Result::ok)
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
fn cargo_artifact_profile_dir(
    cargo_target_dir: &Path,
    profile: &str,
) -> Result<PathBuf, BuildError> {
    let target = env_var("TARGET")?;
    let host = env_var("HOST")?;
    if target == host {
        Ok(cargo_target_dir.join(profile))
    } else {
        Ok(cargo_target_dir.join(target).join(profile))
    }
}

/// Copies the Khronos `OpenXR` loader DLL next to the build output for Windows targets only.
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

    let Ok(dest_dir) = cargo_artifact_profile_dir(&cargo_target_dir, &profile) else {
        println!("cargo:warning=openxr_loader: TARGET/HOST unset");
        return;
    };
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

/// Derives the Cargo artifact profile directory from `OUT_DIR`.
///
/// Cargo always sets `OUT_DIR = .../target/<profile-dir>/build/<pkg>-<hash>/out` — walking up
/// three components recovers `.../target/<profile-dir>/` even when [`PROFILE`](https://doc.rust-lang.org/cargo/reference/environment-variables.html)
/// is `debug` for a custom profile that inherits from `dev` (like this workspace's `dev-fast`).
fn artifact_dir_from_out_dir(out_dir: &Path) -> Option<PathBuf> {
    out_dir.ancestors().nth(3).map(std::path::Path::to_path_buf)
}

/// Copies the XR action manifest and per-profile binding tables into the artifact directory so the
/// runtime can load them alongside the binary (same convention as `config.toml`).
///
/// Source files live at `crates/renderide/assets/xr/` and are mirrored to
/// `target/<profile-dir>/xr/` with `actions.toml` at the root and `bindings/*.toml` below.
/// `cargo:rerun-if-changed` is emitted for the source directory so TOML edits trigger a rebuild
/// copy.
fn copy_xr_assets_to_artifact_dir(manifest_dir: &Path, out_dir: &Path) {
    let src_root = manifest_dir.join("assets/xr");
    println!("cargo:rerun-if-changed={}", src_root.display());
    if !src_root.is_dir() {
        return;
    }

    let Some(dest_root_parent) = artifact_dir_from_out_dir(out_dir) else {
        println!("cargo:warning=xr_assets: cannot derive artifact dir from OUT_DIR");
        return;
    };
    let dest_root = dest_root_parent.join("xr");
    let dest_bindings = dest_root.join("bindings");
    if let Err(e) = fs::create_dir_all(&dest_bindings) {
        println!(
            "cargo:warning=xr_assets: mkdir {} failed: {e}",
            dest_bindings.display()
        );
        return;
    }

    let src_actions = src_root.join("actions.toml");
    let dest_actions = dest_root.join("actions.toml");
    if let Err(e) = fs::copy(&src_actions, &dest_actions) {
        println!(
            "cargo:warning=xr_assets: copy {} -> {} failed: {e}",
            src_actions.display(),
            dest_actions.display()
        );
    }

    let src_bindings = src_root.join("bindings");
    let Ok(entries) = fs::read_dir(&src_bindings) else {
        println!(
            "cargo:warning=xr_assets: read_dir {} failed",
            src_bindings.display()
        );
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Some(file_name) = path.file_name() else {
            continue;
        };
        let dest = dest_bindings.join(file_name);
        if let Err(e) = fs::copy(&path, &dest) {
            println!(
                "cargo:warning=xr_assets: copy {} -> {} failed: {e}",
                path.display(),
                dest.display()
            );
        }
    }
}

fn main() {
    if let Err(e) = run() {
        #[expect(
            clippy::print_stderr,
            reason = "build script: errors route to cargo stderr"
        )]
        {
            eprintln!("renderide build.rs: {e:#}");
        }
        std::process::exit(1);
    }
}

/// Per-source composition. Composes the source once without the `MULTIVIEW` shader def and
/// once with it. If the two flattened WGSL outputs are byte-identical, the shader does not vary
/// on multiview and a **single** target `{stem}.wgsl` is emitted. Otherwise both
/// `{stem}_default.wgsl` and `{stem}_multiview.wgsl` are emitted.
///
/// `validate_view_index` only applies to the fan-out case. When `true`, the multiview variant
/// must contain `@builtin(view_index)` and the default variant must not — this catches sources
/// that declare themselves multiview-aware but failed to guard the `view_index` builtin.
/// Compute-stage shaders must pass `false` because WGSL grammar forbids `@builtin(view_index)`
/// outside fragment entry points.
/// Per-subdirectory validation toggles for [`compose_and_emit_variants`].
struct ComposeValidationOpts {
    /// Enforce that the multiview-fan-out variant carries `@builtin(view_index)` and the default
    /// variant doesn't (skip for compute shaders, which can't carry the builtin).
    validate_view_index: bool,
    /// Reject WGSL with no `//#pass` directives (enabled for the materials subdirectory only).
    require_pass_directive: bool,
}

fn compose_and_emit_variants(
    shader_modules: &[(String, String)],
    source_path: &Path,
    target_dir: &Path,
    opts: ComposeValidationOpts,
    embedded_arms: &mut String,
    embedded_pass_arms: &mut String,
    output_stems: &mut Vec<String>,
) -> Result<(), BuildError> {
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| BuildError::Message(format!("invalid stem: {}", source_path.display())))?;
    let source = fs::read_to_string(source_path)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", source_path.display())))?;
    let file_path = source_path.to_str().ok_or_else(|| {
        BuildError::Message(format!(
            "shader path must be UTF-8: {}",
            source_path.display()
        ))
    })?;
    let pass_directives = parse_pass_directives(&source, file_path)?;
    if opts.require_pass_directive && pass_directives.is_empty() {
        return Err(BuildError::Message(format!(
            "{file_path}: material WGSL must declare at least one //#pass directive (e.g. //#pass forward)"
        )));
    }

    let default_module = compose_material(
        shader_modules,
        &source,
        file_path,
        multiview_shader_defs(false),
    )?;
    let multiview_module = compose_material(
        shader_modules,
        &source,
        file_path,
        multiview_shader_defs(true),
    )?;
    validate_entry_points(
        &default_module,
        &format!("{stem} (MULTIVIEW=false)"),
        &pass_directives,
    )?;
    validate_entry_points(
        &multiview_module,
        &format!("{stem} (MULTIVIEW=true)"),
        &pass_directives,
    )?;
    validate_no_pipeline_state_uniform_fields(
        &default_module,
        &format!("{stem} (MULTIVIEW=false)"),
    )?;
    validate_no_pipeline_state_uniform_fields(
        &multiview_module,
        &format!("{stem} (MULTIVIEW=true)"),
    )?;
    let default_wgsl = module_to_wgsl(&default_module, &format!("{stem} (MULTIVIEW=false)"))?;
    let multiview_wgsl = module_to_wgsl(&multiview_module, &format!("{stem} (MULTIVIEW=true)"))?;

    if default_wgsl == multiview_wgsl {
        let target_stem = stem.to_string();
        let out_path = target_dir.join(format!("{target_stem}.wgsl"));
        fs::write(&out_path, &default_wgsl)?;
        emit_embedded_arms(
            embedded_arms,
            embedded_pass_arms,
            &target_stem,
            &default_wgsl,
            &pass_directives,
        );
        output_stems.push(target_stem);
    } else {
        for (target_stem, wgsl, multiview) in [
            (format!("{stem}_default"), &default_wgsl, false),
            (format!("{stem}_multiview"), &multiview_wgsl, true),
        ] {
            if opts.validate_view_index {
                let has = wgsl.contains("@builtin(view_index)");
                if multiview != has {
                    return Err(BuildError::Message(format!(
                        "{target_stem}: expected @builtin(view_index) {} in output (multiview shader_defs contract)",
                        if multiview { "present" } else { "absent" }
                    )));
                }
            }
            let out_path = target_dir.join(format!("{target_stem}.wgsl"));
            fs::write(&out_path, wgsl)?;
            emit_embedded_arms(
                embedded_arms,
                embedded_pass_arms,
                &target_stem,
                wgsl,
                &pass_directives,
            );
            output_stems.push(target_stem);
        }
    }
    Ok(())
}

/// Appends one `match` arm to `embedded_arms` (and optionally `embedded_pass_arms`) for a
/// composed shader target. Factored out so the single-variant and fan-out paths share the
/// registry-emission code without diverging.
fn emit_embedded_arms(
    embedded_arms: &mut String,
    embedded_pass_arms: &mut String,
    target_stem: &str,
    wgsl: &str,
    pass_directives: &[BuildPassDirective],
) {
    use std::fmt::Write as _;
    let lit = rust_string_literal_token(wgsl);
    let _ = writeln!(embedded_arms, "        \"{target_stem}\" => Some({lit}),");
    if !pass_directives.is_empty() {
        let pass_literals = pass_directives
            .iter()
            .map(pass_literal)
            .collect::<Vec<_>>()
            .join(",\n            ");
        let _ = writeln!(
            embedded_pass_arms,
            "        \"{target_stem}\" => const {{ &[\n            {pass_literals},\n        ] }},"
        );
    }
}

/// Lists every `.wgsl` file directly under `dir`, sorted lexicographically for deterministic
/// build output.
fn list_wgsl_files(dir: &Path) -> Result<Vec<PathBuf>, BuildError> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| BuildError::Message(format!("read {}: {e}", dir.display())))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn run() -> Result<(), BuildError> {
    let manifest_dir = PathBuf::from(env_var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env_var("OUT_DIR")?);
    copy_vendored_openxr_loader_windows(&manifest_dir);
    copy_xr_assets_to_artifact_dir(&manifest_dir, &out_dir);
    let source_root = manifest_dir.join("shaders/source");
    let target_dir = manifest_dir.join("shaders/target");

    println!("cargo:rerun-if-changed=shaders/source");
    println!("cargo:rerun-if-changed=build.rs");

    fs::create_dir_all(&source_root)?;
    fs::create_dir_all(&target_dir)?;

    let shader_modules = discover_shader_modules(&manifest_dir)?;

    let mut embedded_arms = String::new();
    let mut embedded_pass_arms = String::new();
    let mut material_stems: Vec<String> = Vec::new();
    let mut post_stems: Vec<String> = Vec::new();
    let mut backend_stems: Vec<String> = Vec::new();
    let mut compute_stems: Vec<String> = Vec::new();
    let mut present_stems: Vec<String> = Vec::new();

    let scans: [(&str, &mut Vec<String>, bool, bool); 5] = [
        ("materials", &mut material_stems, true, true),
        ("post", &mut post_stems, true, false),
        ("backend", &mut backend_stems, true, false),
        ("compute", &mut compute_stems, false, false),
        ("present", &mut present_stems, true, false),
    ];
    for (subdir, stems_out, validate_view_index, require_pass_directive) in scans {
        let dir = source_root.join(subdir);
        if !dir.is_dir() {
            continue;
        }
        for path in list_wgsl_files(&dir)? {
            compose_and_emit_variants(
                &shader_modules,
                &path,
                &target_dir,
                ComposeValidationOpts {
                    validate_view_index,
                    require_pass_directive,
                },
                &mut embedded_arms,
                &mut embedded_pass_arms,
                stems_out,
            )?;
        }
    }

    let stems_list = |stems: &[String]| {
        stems
            .iter()
            .map(|s| format!("    \"{s}\","))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let material_stems_list = stems_list(&material_stems);
    let post_stems_list = stems_list(&post_stems);
    let backend_stems_list = stems_list(&backend_stems);
    let compute_stems_list = stems_list(&compute_stems);
    let present_stems_list = stems_list(&present_stems);

    let embedded_rs = format!(
        r#"// Generated by `build.rs` — do not edit.

/// Flattened WGSL for `stem` (also written under `shaders/target/{{stem}}.wgsl` at build time).
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_wgsl(stem: &str) -> Option<&'static str> {{
    match stem {{
{embedded_arms}        _ => None,
    }}
}}

/// Declared render passes for `stem`, parsed from `//#pass` directives in the source WGSL.
#[expect(clippy::too_many_lines, reason = "match arm per embedded shader target; scales with shader count")]
pub fn embedded_target_passes(stem: &str) -> &'static [crate::materials::MaterialPassDesc] {{
    match stem {{
{embedded_pass_arms}        _ => &[],
    }}
}}

/// Material target stems (composed from `shaders/source/materials/*.wgsl`). These follow the AAA
/// `@group(0)` frame-uniform convention validated by `materials::wgsl_reflect`.
pub const COMPILED_MATERIAL_STEMS: &[&str] = &[
{material_stems}
];

/// Post-processing target stems (composed from `shaders/source/post/*.wgsl`). These use custom
/// `@group(0)` bind layouts (per-pass texture/sampler/uniforms) and are **not** subject to the
/// material frame-uniform reflection check.
pub const COMPILED_POST_STEMS: &[&str] = &[
{post_stems}
];

/// Backend target stems (composed from `shaders/source/backend/*.wgsl`). Fragment/blit shaders
/// that feed backend compute → render conversions (e.g. depth resolve blit).
pub const COMPILED_BACKEND_STEMS: &[&str] = &[
{backend_stems}
];

/// Compute target stems (composed from `shaders/source/compute/*.wgsl`). WGSL forbids
/// `@builtin(view_index)` in compute entry points, so the build script does not enforce the
/// multiview view-index contract for this directory.
pub const COMPILED_COMPUTE_STEMS: &[&str] = &[
{compute_stems}
];

/// Present target stems (composed from `shaders/source/present/*.wgsl`). Desktop swapchain
/// blit shaders; not part of any multiview fan-out.
pub const COMPILED_PRESENT_STEMS: &[&str] = &[
{present_stems}
];
"#,
        embedded_arms = embedded_arms,
        embedded_pass_arms = embedded_pass_arms,
        material_stems = material_stems_list,
        post_stems = post_stems_list,
        backend_stems = backend_stems_list,
        compute_stems = compute_stems_list,
        present_stems = present_stems_list,
    );

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs)?;
    Ok(())
}
