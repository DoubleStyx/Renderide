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
pub(crate) enum BuildError {
    /// User-facing message (directive parse, validation, naga errors).
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Required Cargo environment variable missing.
    #[error("missing build environment variable `{0}`")]
    MissingEnv(&'static str),
}

pub(crate) fn env_var(name: &'static str) -> Result<String, BuildError> {
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

/// Maps a shader stem (which may include `.` or `-`, e.g. `xstoon2.0-cutout`) to a
/// SCREAMING_SNAKE_CASE Rust identifier suitable for the per-stem WGSL constant name.
///
/// Unambiguous encoding: ASCII alphanumeric pass through (uppercased), `_` is preserved,
/// `.` becomes `_DOT_`, `-` becomes `_DASH_`. This keeps `xstoon2.0-outlined` and
/// `xstoon2.0_outlined` distinct.
fn stem_to_const_ident(stem: &str) -> String {
    let mut out = String::with_capacity(stem.len());
    for c in stem.chars() {
        match c {
            c if c.is_ascii_alphanumeric() => out.push(c.to_ascii_uppercase()),
            '_' => out.push('_'),
            '.' => out.push_str("_DOT_"),
            '-' => out.push_str("_DASH_"),
            _ => out.push('_'),
        }
    }
    out
}

/// Loads every `*.wgsl` under `shaders/source/modules/` relative to `manifest_dir`.
///
/// Returns `(file_path, source)` where `file_path` uses forward slashes (e.g.
/// `shaders/source/modules/globals.wgsl`) for [`ComposableModuleDescriptor::file_path`].
/// Modules are returned in dependency order: each module's `#import` targets appear before
/// it. Within a single dependency level the order is alphabetical for deterministic builds.
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

    topo_sort_shader_modules(&modules)
}

/// Topologically sorts shader modules so each `#import` target is registered before its
/// importer. naga-oil 0.22's `add_composable_module` resolves imports eagerly, so the
/// alphabetical traversal of the modules directory must be re-ordered when import edges
/// run against alphabetic order (e.g. `xiexe_toon2.wgsl` imports `xiexe_toon2::base` from
/// `xiexe_toon2_base.wgsl` which sorts after it).
///
/// Within a single dependency depth the original alphabetical order is preserved for
/// determinism.
fn topo_sort_shader_modules(
    modules: &[(String, String)],
) -> Result<Vec<(String, String)>, BuildError> {
    // Map `#define_import_path` → file index.
    let mut path_to_idx: HashMap<String, usize> = HashMap::default();
    let mut imports_per_module: Vec<Vec<String>> = Vec::with_capacity(modules.len());
    for (i, (file_path, source)) in modules.iter().enumerate() {
        let define = parse_define_import_path(source).ok_or_else(|| {
            BuildError::Message(format!(
                "module {file_path} has no `#define_import_path` directive",
            ))
        })?;
        if let Some(prev) = path_to_idx.insert(define.clone(), i) {
            return Err(BuildError::Message(format!(
                "duplicate `#define_import_path {define}` in {file_path} and {}",
                modules[prev].0,
            )));
        }
        imports_per_module.push(parse_import_paths(source));
    }

    // Edges: child → parent (depends_on); record in-degree and adjacency.
    let mut in_degree = vec![0usize; modules.len()];
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); modules.len()];
    for (i, imports) in imports_per_module.iter().enumerate() {
        for import_path in imports {
            // Skip imports satisfied by external/builtin modules (e.g. unknown paths) —
            // naga-oil itself surfaces a helpful error if those are actually unresolved.
            if let Some(&j) = path_to_idx.get(import_path) {
                if i == j {
                    continue;
                }
                children_of[j].push(i);
                in_degree[i] += 1;
            }
        }
    }

    // Kahn's algorithm with alphabetic tie-break (modules are pre-sorted by path).
    let mut ready: Vec<usize> = (0..modules.len()).filter(|&i| in_degree[i] == 0).collect();
    let mut sorted = Vec::with_capacity(modules.len());
    while let Some(idx) = ready.first().copied() {
        ready.remove(0);
        sorted.push(idx);
        for &child in &children_of[idx] {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                // Insert maintaining alphabetic order.
                let pos = ready
                    .binary_search_by(|&j| modules[j].0.cmp(&modules[child].0))
                    .unwrap_or_else(|e| e);
                ready.insert(pos, child);
            }
        }
    }
    if sorted.len() != modules.len() {
        let unresolved: Vec<&str> = (0..modules.len())
            .filter(|i| !sorted.contains(i))
            .map(|i| modules[i].0.as_str())
            .collect();
        return Err(BuildError::Message(format!(
            "shader-module import graph has a cycle; unresolved: {unresolved:?}",
        )));
    }

    Ok(sorted.into_iter().map(|i| modules[i].clone()).collect())
}

/// Parses the first `#define_import_path <path>` directive from a WGSL source.
fn parse_define_import_path(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("#define_import_path") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// Parses every `#import <path>` (including aliased `#import <path> as <alias>` forms) from
/// a WGSL source. Multi-line block-import syntax is not used in this codebase, so the line
/// scanner suffices.
fn parse_import_paths(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("#import") else {
            continue;
        };
        let rest = rest.trim();
        // Forms: `#import path`, `#import path as alias`, `#import path::{a, b}`.
        let path = rest
            .split_whitespace()
            .next()
            .map(|p| p.trim_end_matches('{').to_string());
        if let Some(p) = path {
            // Drop trailing `::{...` if a brace-list followed without whitespace.
            let p = p.split("::{").next().unwrap_or(&p).to_string();
            if !p.is_empty() {
                out.push(p);
            }
        }
    }
    out
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

/// Accumulators populated as each shader source is composed and emitted.
///
/// `embedded_arms` and `embedded_pass_arms` are the bodies of the registry-lookup `match` blocks;
/// `embedded_consts` collects per-stem `pub const <STEM>_WGSL: &str = ...` declarations so static-stem
/// callers can reference shader source as a compile-time `&'static str` without a runtime lookup.
struct EmbeddedShaderEmitters<'a> {
    embedded_arms: &'a mut String,
    embedded_pass_arms: &'a mut String,
    embedded_consts: &'a mut String,
    output_stems: &'a mut Vec<String>,
}

fn compose_and_emit_variants(
    shader_modules: &[(String, String)],
    source_path: &Path,
    target_dir: &Path,
    opts: ComposeValidationOpts,
    emitters: &mut EmbeddedShaderEmitters<'_>,
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
            emitters.embedded_arms,
            emitters.embedded_pass_arms,
            emitters.embedded_consts,
            &target_stem,
            &default_wgsl,
            &pass_directives,
        );
        emitters.output_stems.push(target_stem);
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
                emitters.embedded_arms,
                emitters.embedded_pass_arms,
                emitters.embedded_consts,
                &target_stem,
                wgsl,
                &pass_directives,
            );
            emitters.output_stems.push(target_stem);
        }
    }
    Ok(())
}

/// Appends one `match` arm to `embedded_arms` (and optionally `embedded_pass_arms`) for a
/// composed shader target, plus a `pub const <STEM_UPPER>_WGSL: &str = ...;` to
/// `embedded_consts` so static-stem callers can reference shader source without a runtime
/// lookup. Factored out so the single-variant and fan-out paths share the registry-emission
/// code without diverging.
fn emit_embedded_arms(
    embedded_arms: &mut String,
    embedded_pass_arms: &mut String,
    embedded_consts: &mut String,
    target_stem: &str,
    wgsl: &str,
    pass_directives: &[BuildPassDirective],
) {
    use std::fmt::Write as _;
    let lit = rust_string_literal_token(wgsl);
    let _ = writeln!(embedded_arms, "        \"{target_stem}\" => Some({lit}),");
    let const_ident = stem_to_const_ident(target_stem);
    let _ = writeln!(
        embedded_consts,
        "/// Composed WGSL for the `{target_stem}` embedded shader target.\npub const {const_ident}_WGSL: &str = {lit};"
    );
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

/// Per-subdirectory composed shader output: stem lists for each scanned source category,
/// plus the registry-emission accumulators that will feed the generated `embedded_shaders.rs`.
struct ComposedShaders {
    material_stems: Vec<String>,
    post_stems: Vec<String>,
    backend_stems: Vec<String>,
    compute_stems: Vec<String>,
    present_stems: Vec<String>,
    embedded_arms: String,
    embedded_pass_arms: String,
    embedded_consts: String,
}

fn compose_all_shaders(
    shader_modules: &[(String, String)],
    source_root: &Path,
    target_dir: &Path,
) -> Result<ComposedShaders, BuildError> {
    let mut out = ComposedShaders {
        material_stems: Vec::new(),
        post_stems: Vec::new(),
        backend_stems: Vec::new(),
        compute_stems: Vec::new(),
        present_stems: Vec::new(),
        embedded_arms: String::new(),
        embedded_pass_arms: String::new(),
        embedded_consts: String::new(),
    };
    let scans: [(&str, &mut Vec<String>, bool, bool); 5] = [
        ("materials", &mut out.material_stems, true, true),
        ("post", &mut out.post_stems, true, false),
        ("backend", &mut out.backend_stems, true, false),
        ("compute", &mut out.compute_stems, false, false),
        ("present", &mut out.present_stems, true, false),
    ];
    for (subdir, stems_out, validate_view_index, require_pass_directive) in scans {
        let dir = source_root.join(subdir);
        if !dir.is_dir() {
            continue;
        }
        for path in list_wgsl_files(&dir)? {
            let mut emitters = EmbeddedShaderEmitters {
                embedded_arms: &mut out.embedded_arms,
                embedded_pass_arms: &mut out.embedded_pass_arms,
                embedded_consts: &mut out.embedded_consts,
                output_stems: stems_out,
            };
            compose_and_emit_variants(
                shader_modules,
                &path,
                target_dir,
                ComposeValidationOpts {
                    validate_view_index,
                    require_pass_directive,
                },
                &mut emitters,
            )?;
        }
    }
    Ok(out)
}

fn render_embedded_shaders_rs(c: &ComposedShaders) -> String {
    let stems_list = |stems: &[String]| {
        stems
            .iter()
            .map(|s| format!("    \"{s}\","))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        r#"// Generated by `build.rs` — do not edit.

{embedded_consts}
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
        embedded_arms = c.embedded_arms,
        embedded_pass_arms = c.embedded_pass_arms,
        embedded_consts = c.embedded_consts,
        material_stems = stems_list(&c.material_stems),
        post_stems = stems_list(&c.post_stems),
        backend_stems = stems_list(&c.backend_stems),
        compute_stems = stems_list(&c.compute_stems),
        present_stems = stems_list(&c.present_stems),
    )
}

pub(crate) fn compile(manifest_dir: &Path, out_dir: &Path) -> Result<(), BuildError> {
    let source_root = manifest_dir.join("shaders/source");
    let target_dir = manifest_dir.join("shaders/target");

    println!("cargo:rerun-if-changed=shaders/source");

    fs::create_dir_all(&source_root)?;
    fs::create_dir_all(&target_dir)?;

    let shader_modules = discover_shader_modules(manifest_dir)?;
    let composed = compose_all_shaders(&shader_modules, &source_root, &target_dir)?;
    let embedded_rs = render_embedded_shaders_rs(&composed);

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs)?;
    Ok(())
}
