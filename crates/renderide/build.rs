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
//! ## Post-processing shaders (`shaders/source/post/*.wgsl`)
//!
//! Each non-`_mono` / non-`_multiview` post source is composed twice exactly like materials and
//! emitted as `{stem}_default` / `{stem}_multiview` so post passes can look up their variant via
//! [`crate::embedded_shaders::embedded_target_wgsl`]. Sources with `_mono` or `_multiview` in the
//! stem are treated as pre-composed and skipped (they are loaded directly via `include_str!` by
//! their owning pass — used by passes that have not yet adopted the build-time `#ifdef MULTIVIEW`
//! pattern).
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
            cull_mode: "Some(wgpu::Face::Back)",
            blend: None,
            write_mask: "wgpu::ColorWrites::COLOR",
            depth_bias_slope_scale: 0.0,
            depth_bias_constant: 0,
            material_state: "crate::materials::MaterialPassState::Static",
        }
    }
}

fn parse_bool_like(value: &str, label: &str, file: &str, line: usize) -> Result<bool, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid {label} value `{value}`"
        ))),
    }
}

fn compare_token(value: &str, file: &str, line: usize) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "never" => Ok("wgpu::CompareFunction::Never"),
        "less" => Ok("wgpu::CompareFunction::Less"),
        "equal" => Ok("wgpu::CompareFunction::Equal"),
        "less_equal" | "lessequal" | "lequal" => Ok("wgpu::CompareFunction::LessEqual"),
        "greater" => Ok("wgpu::CompareFunction::Greater"),
        "not_equal" | "notequal" => Ok("wgpu::CompareFunction::NotEqual"),
        "greater_equal" | "greaterequal" | "gequal" => Ok("wgpu::CompareFunction::GreaterEqual"),
        "always" => Ok("wgpu::CompareFunction::Always"),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid depth compare `{value}`"
        ))),
    }
}

fn cull_token(value: &str, file: &str, line: usize) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" | "off" | "0" => Ok("None"),
        "front" => Ok("Some(wgpu::Face::Front)"),
        "back" => Ok("Some(wgpu::Face::Back)"),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid cull mode `{value}`"
        ))),
    }
}

fn color_writes_token(value: &str, file: &str, line: usize) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "all" | "rgba" => Ok("wgpu::ColorWrites::ALL"),
        "color" | "rgb" => Ok("wgpu::ColorWrites::COLOR"),
        "none" | "off" => Ok("crate::materials::COLOR_WRITES_NONE"),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid color write mask `{value}`"
        ))),
    }
}

fn material_pass_state_token(
    value: &str,
    file: &str,
    line: usize,
) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "static" | "none" => Ok("crate::materials::MaterialPassState::Static"),
        "base" | "forward_base" | "forwardbase" | "unity_forward_base" => {
            Ok("crate::materials::MaterialPassState::UnityForwardBase")
        }
        "add" | "forward_add" | "forwardadd" | "delta" | "unity_forward_add" => {
            Ok("crate::materials::MaterialPassState::UnityForwardAdd")
        }
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid material pass state `{value}`"
        ))),
    }
}

fn blend_factor_token(value: &str, file: &str, line: usize) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "zero" | "0" => Ok("wgpu::BlendFactor::Zero"),
        "one" | "1" => Ok("wgpu::BlendFactor::One"),
        "src" | "src_color" | "srccolor" => Ok("wgpu::BlendFactor::Src"),
        "one_minus_src" | "one_minus_src_color" | "oneminussrc" | "oneminussrccolor" => {
            Ok("wgpu::BlendFactor::OneMinusSrc")
        }
        "dst" | "dst_color" | "dstcolor" => Ok("wgpu::BlendFactor::Dst"),
        "one_minus_dst" | "one_minus_dst_color" | "oneminusdst" | "oneminusdstcolor" => {
            Ok("wgpu::BlendFactor::OneMinusDst")
        }
        "src_alpha" | "srcalpha" => Ok("wgpu::BlendFactor::SrcAlpha"),
        "one_minus_src_alpha" | "oneminussrcalpha" => Ok("wgpu::BlendFactor::OneMinusSrcAlpha"),
        "dst_alpha" | "dstalpha" => Ok("wgpu::BlendFactor::DstAlpha"),
        "one_minus_dst_alpha" | "oneminusdstalpha" => Ok("wgpu::BlendFactor::OneMinusDstAlpha"),
        "constant" | "constant_color" | "constantcolor" => Ok("wgpu::BlendFactor::Constant"),
        "one_minus_constant" | "one_minus_constant_color" | "oneminusconstant" => {
            Ok("wgpu::BlendFactor::OneMinusConstant")
        }
        "src_alpha_saturated" | "srcalphasaturated" => Ok("wgpu::BlendFactor::SrcAlphaSaturated"),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid blend factor `{value}`"
        ))),
    }
}

/// Comma-split `//#pass` body tokens and mutable scan position for blend key parsing.
struct BlendDirectiveParseCursor<'a> {
    parts: &'a [&'a str],
    index: &'a mut usize,
}

/// Source location for shader build script errors.
struct BlendDirectiveSite<'a> {
    file: &'a str,
    line_no: usize,
}

/// Mutable blend state while parsing one `//#pass` directive.
struct PassBlendDirectiveState<'a> {
    blend_disabled: &'a mut bool,
    color_blend: &'a mut Option<BuildBlendComponent>,
    alpha_blend: &'a mut Option<BuildBlendComponent>,
}

fn blend_operation_token(value: &str, file: &str, line: usize) -> Result<&'static str, BuildError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "add" => Ok("wgpu::BlendOperation::Add"),
        "subtract" | "sub" => Ok("wgpu::BlendOperation::Subtract"),
        "reverse_subtract" | "revsub" => Ok("wgpu::BlendOperation::ReverseSubtract"),
        "min" => Ok("wgpu::BlendOperation::Min"),
        "max" => Ok("wgpu::BlendOperation::Max"),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: invalid blend operation `{value}`"
        ))),
    }
}

fn parse_blend_component(
    cursor: &mut BlendDirectiveParseCursor<'_>,
    first_value: &str,
    site: BlendDirectiveSite<'_>,
) -> Result<BuildBlendComponent, BuildError> {
    let BlendDirectiveSite { file, line_no } = site;
    if *cursor.index + 2 >= cursor.parts.len() {
        return Err(BuildError::Message(format!(
            "{file}:{line_no}: blend component needs src,dst,op"
        )));
    }
    let src = blend_factor_token(first_value, file, line_no)?;
    *cursor.index += 1;
    let dst = blend_factor_token(cursor.parts[*cursor.index], file, line_no)?;
    *cursor.index += 1;
    let op = blend_operation_token(cursor.parts[*cursor.index], file, line_no)?;
    Ok(BuildBlendComponent {
        src_factor: src,
        dst_factor: dst,
        operation: op,
    })
}

/// Applies blend-related key/value pairs and updates `blend_disabled` / blend component state.
fn apply_pass_blend_or_alpha_key(
    key: &str,
    value: &str,
    cursor: &mut BlendDirectiveParseCursor<'_>,
    site: BlendDirectiveSite<'_>,
    state: &mut PassBlendDirectiveState<'_>,
) -> Result<(), BuildError> {
    let BlendDirectiveSite { file, line_no } = site;
    match key {
        "blend" => {
            if value.trim().eq_ignore_ascii_case("none") {
                *state.blend_disabled = true;
                *state.color_blend = None;
                *state.alpha_blend = None;
            } else if value.trim().eq_ignore_ascii_case("alpha") {
                *state.color_blend = Some(BuildBlendComponent {
                    src_factor: "wgpu::BlendFactor::SrcAlpha",
                    dst_factor: "wgpu::BlendFactor::OneMinusSrcAlpha",
                    operation: "wgpu::BlendOperation::Add",
                });
                *state.alpha_blend = Some(BuildBlendComponent {
                    src_factor: "wgpu::BlendFactor::One",
                    dst_factor: "wgpu::BlendFactor::OneMinusSrcAlpha",
                    operation: "wgpu::BlendOperation::Add",
                });
            } else {
                *state.blend_disabled = false;
                *state.color_blend = Some(parse_blend_component(
                    cursor,
                    value,
                    BlendDirectiveSite { file, line_no },
                )?);
            }
        }
        "alpha" => {
            *state.blend_disabled = false;
            *state.alpha_blend = Some(parse_blend_component(
                cursor,
                value,
                BlendDirectiveSite { file, line_no },
            )?);
        }
        _ => {}
    }
    Ok(())
}

fn finalize_pass_blend_state(
    pass: &mut BuildPassDirective,
    blend_disabled: bool,
    color_blend: Option<BuildBlendComponent>,
    alpha_blend: Option<BuildBlendComponent>,
) {
    if blend_disabled {
        return;
    }
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

fn parse_one_pass_directive(
    file: &str,
    line_no: usize,
    name: &str,
    body: &str,
) -> Result<BuildPassDirective, BuildError> {
    let mut pass = BuildPassDirective::new(name.to_string());
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
        let (key, value) = parts[i].split_once('=').ok_or_else(|| {
            BuildError::Message(format!(
                "{file}:{line_no}: expected key=value in `{}`",
                parts[i]
            ))
        })?;
        let key_lc = key.trim().to_ascii_lowercase();
        match key_lc.as_str() {
            "vs" | "vertex" => pass.vertex_entry = value.trim().to_string(),
            "fs" | "fragment" => pass.fragment_entry = value.trim().to_string(),
            "depth" | "ztest" => pass.depth_compare = compare_token(value, file, line_no)?,
            "zwrite" | "depth_write" => {
                pass.depth_write = parse_bool_like(value, "zwrite", file, line_no)?;
            }
            "cull" => pass.cull_mode = cull_token(value, file, line_no)?,
            "write" | "writes" | "color_write" | "colorwrites" => {
                pass.write_mask = color_writes_token(value, file, line_no)?;
            }
            "bias" | "depth_bias" => {
                pass.depth_bias_constant = value.trim().parse().map_err(|e| {
                    BuildError::Message(format!(
                        "{file}:{line_no}: invalid depth bias `{value}`: {e}"
                    ))
                })?;
            }
            "slope" | "slope_bias" => {
                pass.depth_bias_slope_scale = value.trim().parse().map_err(|e| {
                    BuildError::Message(format!(
                        "{file}:{line_no}: invalid slope bias `{value}`: {e}"
                    ))
                })?;
            }
            "material" | "material_state" => {
                pass.material_state = material_pass_state_token(value, file, line_no)?;
            }
            "blend" | "alpha" => apply_pass_blend_or_alpha_key(
                key_lc.as_str(),
                value,
                &mut BlendDirectiveParseCursor {
                    parts: parts.as_slice(),
                    index: &mut i,
                },
                BlendDirectiveSite { file, line_no },
                &mut PassBlendDirectiveState {
                    blend_disabled: &mut blend_disabled,
                    color_blend: &mut color_blend,
                    alpha_blend: &mut alpha_blend,
                },
            )?,
            _ => {
                return Err(BuildError::Message(format!(
                    "{file}:{line_no}: unknown pass key `{key}`"
                )));
            }
        }
        i += 1;
    }

    finalize_pass_blend_state(&mut pass, blend_disabled, color_blend, alpha_blend);
    Ok(pass)
}

fn parse_pass_directives(source: &str, file: &str) -> Result<Vec<BuildPassDirective>, BuildError> {
    let mut passes = Vec::new();
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#pass") else {
            continue;
        };
        let rest = rest.trim();
        let (name, body) = rest.split_once(':').ok_or_else(|| {
            BuildError::Message(format!(
                "{file}:{line_no}: pass directive must be `//#pass name: key=value`"
            ))
        })?;
        passes.push(parse_one_pass_directive(file, line_no, name.trim(), body)?);
    }
    Ok(passes)
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
    out_path: &Path,
    expect_view_index: Option<bool>,
    passes: &[BuildPassDirective],
) -> Result<String, BuildError> {
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
            return Err(BuildError::Message(format!(
                "{label}: expected entry points vs_main and fs_main (vertex={has_vs} fragment={has_fs})",
            )));
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
                return Err(BuildError::Message(format!(
                    "{label}: pass `{}` expected entry points {} and {} (vertex={has_vs} fragment={has_fs})",
                    pass.name, pass.vertex_entry, pass.fragment_entry
                )));
            }
        }
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| BuildError::Message(format!("validate {label}: {e}")))?;
    let wgsl = naga::back::wgsl::write_string(module, &info, WriterFlags::EXPLICIT_TYPES)
        .map_err(|e| BuildError::Message(format!("wgsl out {label}: {e}")))?;
    if let Some(want) = expect_view_index {
        let has = wgsl.contains("@builtin(view_index)");
        if want != has {
            return Err(BuildError::Message(format!(
                "{label}: expected @builtin(view_index) {} in output (multiview shader_defs contract)",
                if want { "present" } else { "absent" }
            )));
        }
    }
    fs::write(out_path, &wgsl)?;
    Ok(wgsl)
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

/// Per-source composition: emits `{stem}_default` and `{stem}_multiview` variants for `path`,
/// validating each and appending entries to the embedded shader registry.
///
/// `validate_view_index` controls whether the build script asserts that the composed WGSL
/// contains `@builtin(view_index)` for the multiview variant only. Materials and post-processing
/// shaders both use multiview view-index selection so this is `true` for them; future entry
/// points that intentionally compose multiview without `view_index` (e.g. layered geometry or
/// `view_count` drawcall fan-out) can pass `false`.
fn compose_and_emit_variants(
    shader_modules: &[(String, String)],
    source_path: &Path,
    target_dir: &Path,
    validate_view_index: bool,
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

    let variants = [
        (format!("{stem}_default"), false),
        (format!("{stem}_multiview"), true),
    ];
    for (target_stem, multiview) in variants {
        let defs = multiview_shader_defs(multiview);
        let module = compose_material(shader_modules, &source, file_path, defs)?;
        let label = format!("{target_stem} (MULTIVIEW={multiview})");
        let out_path = target_dir.join(format!("{target_stem}.wgsl"));
        let expect_view_index = validate_view_index.then_some(multiview);
        let wgsl = validate_and_write_wgsl(
            &module,
            &label,
            &out_path,
            expect_view_index,
            &pass_directives,
        )?;
        let lit = rust_string_literal_token(&wgsl);
        use std::fmt::Write as _;
        let _ = writeln!(embedded_arms, "        \"{target_stem}\" => Some({lit}),");
        if !pass_directives.is_empty() {
            let pass_literals = pass_directives
                .iter()
                .map(pass_literal)
                .collect::<Vec<_>>()
                .join(",\n            ");
            let _ = writeln!(
                embedded_pass_arms,
                "        \"{target_stem}\" => &[\n            {pass_literals},\n        ],"
            );
        }
        output_stems.push(target_stem);
    }
    Ok(())
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

/// Returns `true` when `path`'s stem ends with `_mono` or `_multiview`.
///
/// Such files are treated as pre-composed by the build script (they are loaded directly via
/// `include_str!` by their owning pass) and are skipped by the post-shader composition loop.
fn is_legacy_precomposed_post_stem(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    stem.ends_with("_mono") || stem.ends_with("_multiview")
}

fn run() -> Result<(), BuildError> {
    let manifest_dir = PathBuf::from(env_var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env_var("OUT_DIR")?);
    copy_vendored_openxr_loader_windows(&manifest_dir);
    copy_xr_assets_to_artifact_dir(&manifest_dir, &out_dir);
    let materials_dir = manifest_dir.join("shaders/source/materials");
    let post_dir = manifest_dir.join("shaders/source/post");
    let target_dir = manifest_dir.join("shaders/target");

    println!("cargo:rerun-if-changed=shaders/source");
    println!("cargo:rerun-if-changed=build.rs");

    let materials_parent = materials_dir
        .parent()
        .ok_or_else(|| BuildError::Message("shaders/source/materials has no parent".into()))?;
    fs::create_dir_all(materials_parent)?;
    fs::create_dir_all(&target_dir)?;

    let shader_modules = discover_shader_modules(&manifest_dir)?;

    let mut embedded_arms = String::new();
    let mut embedded_pass_arms = String::new();
    let mut material_stems: Vec<String> = Vec::new();
    let mut post_stems: Vec<String> = Vec::new();

    for path in list_wgsl_files(&materials_dir)? {
        compose_and_emit_variants(
            &shader_modules,
            &path,
            &target_dir,
            true,
            &mut embedded_arms,
            &mut embedded_pass_arms,
            &mut material_stems,
        )?;
    }

    if post_dir.is_dir() {
        for path in list_wgsl_files(&post_dir)? {
            if is_legacy_precomposed_post_stem(&path) {
                continue;
            }
            compose_and_emit_variants(
                &shader_modules,
                &path,
                &target_dir,
                true,
                &mut embedded_arms,
                &mut embedded_pass_arms,
                &mut post_stems,
            )?;
        }
    }

    let material_stems_list = material_stems
        .iter()
        .map(|s| format!("    \"{s}\","))
        .collect::<Vec<_>>()
        .join("\n");
    let post_stems_list = post_stems
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
"#,
        embedded_arms = embedded_arms,
        embedded_pass_arms = embedded_pass_arms,
        material_stems = material_stems_list,
        post_stems = post_stems_list
    );

    let gen_path = out_dir.join("embedded_shaders.rs");
    fs::write(&gen_path, embedded_rs)?;
    Ok(())
}
