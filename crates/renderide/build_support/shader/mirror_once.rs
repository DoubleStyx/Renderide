//! Build-time WGSL rewrite that emulates `MirrorOnce` material texture addressing.
//!
//! WebGPU exposes repeat, mirror-repeat, and clamp-to-edge sampler address modes, but not
//! mirror-clamp addressing. Material shaders therefore receive per-texture wrap bits in their
//! reflected group-1 uniform block, and this pass rewrites host texture sampling to mirror the
//! adjacent negative tile before clamping to the edge.

use std::collections::{BTreeMap, BTreeSet};

use super::error::BuildError;

/// Suffix used by the runtime uniform packer for per-texture wrap bits.
const WRAP_MODE_BITS_SUFFIX: &str = "_WrapModeBits";
/// Helper WGSL appended to material targets that sample mirror-once textures.
const MIRROR_ONCE_HELPERS: &str = r#"
const RENDERIDE_WRAP_MODE_MIRROR_ONCE_U: u32 = 1u;
const RENDERIDE_WRAP_MODE_MIRROR_ONCE_V: u32 = 2u;
const RENDERIDE_WRAP_MODE_MIRROR_ONCE_W: u32 = 4u;

struct RenderideWrappedGrad2D {
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
}

fn renderide_mirror_once_coord(coord: f32) -> f32 {
    return clamp(abs(coord), 0.0, 1.0);
}

fn renderide_mirror_once_grad_scale(coord: f32) -> f32 {
    if (coord < -1.0 || coord > 1.0) {
        return 0.0;
    }
    if (coord < 0.0) {
        return -1.0;
    }
    return 1.0;
}

fn renderide_mirror_once_2d(uv: vec2<f32>, wrap_mode_bits: u32) -> vec2<f32> {
    var wrapped = uv;
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_U) != 0u) {
        wrapped.x = renderide_mirror_once_coord(wrapped.x);
    }
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_V) != 0u) {
        wrapped.y = renderide_mirror_once_coord(wrapped.y);
    }
    return wrapped;
}

fn renderide_mirror_once_3d(uvw: vec3<f32>, wrap_mode_bits: u32) -> vec3<f32> {
    var wrapped = uvw;
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_U) != 0u) {
        wrapped.x = renderide_mirror_once_coord(wrapped.x);
    }
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_V) != 0u) {
        wrapped.y = renderide_mirror_once_coord(wrapped.y);
    }
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_W) != 0u) {
        wrapped.z = renderide_mirror_once_coord(wrapped.z);
    }
    return wrapped;
}

fn renderide_mirror_once_grad_2d(
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    wrap_mode_bits: u32,
) -> RenderideWrappedGrad2D {
    var wrapped = RenderideWrappedGrad2D(uv, ddx_uv, ddy_uv);
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_U) != 0u) {
        let scale_x = renderide_mirror_once_grad_scale(uv.x);
        wrapped.uv.x = renderide_mirror_once_coord(uv.x);
        wrapped.ddx_uv.x = ddx_uv.x * scale_x;
        wrapped.ddy_uv.x = ddy_uv.x * scale_x;
    }
    if ((wrap_mode_bits & RENDERIDE_WRAP_MODE_MIRROR_ONCE_V) != 0u) {
        let scale_y = renderide_mirror_once_grad_scale(uv.y);
        wrapped.uv.y = renderide_mirror_once_coord(uv.y);
        wrapped.ddx_uv.y = ddx_uv.y * scale_y;
        wrapped.ddy_uv.y = ddy_uv.y * scale_y;
    }
    return wrapped;
}

fn renderide_mirroronce_sample_2d(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    return textureSample(tex, samp, renderide_mirror_once_2d(uv, wrap_mode_bits));
}

fn renderide_mirroronce_sample_bias_2d(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    bias: f32,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    return textureSampleBias(tex, samp, renderide_mirror_once_2d(uv, wrap_mode_bits), bias);
}

fn renderide_mirroronce_sample_level_2d(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    level: f32,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    return textureSampleLevel(tex, samp, renderide_mirror_once_2d(uv, wrap_mode_bits), level);
}

fn renderide_mirroronce_sample_grad_2d(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    let wrapped = renderide_mirror_once_grad_2d(uv, ddx_uv, ddy_uv, wrap_mode_bits);
    return textureSampleGrad(tex, samp, wrapped.uv, wrapped.ddx_uv, wrapped.ddy_uv);
}

fn renderide_mirroronce_sample_3d(
    tex: texture_3d<f32>,
    samp: sampler,
    uvw: vec3<f32>,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    return textureSample(tex, samp, renderide_mirror_once_3d(uvw, wrap_mode_bits));
}

fn renderide_mirroronce_sample_bias_3d(
    tex: texture_3d<f32>,
    samp: sampler,
    uvw: vec3<f32>,
    bias: f32,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    return textureSampleBias(tex, samp, renderide_mirror_once_3d(uvw, wrap_mode_bits), bias);
}

fn renderide_mirroronce_sample_level_3d(
    tex: texture_3d<f32>,
    samp: sampler,
    uvw: vec3<f32>,
    level: f32,
    wrap_mode_bits: u32,
) -> vec4<f32> {
    return textureSampleLevel(tex, samp, renderide_mirror_once_3d(uvw, wrap_mode_bits), level);
}
"#;

/// Texture dimensionality relevant to mirror-once coordinate rewriting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WrapTextureDimension {
    /// `texture_2d<f32>`.
    D2,
    /// `texture_3d<f32>`.
    D3,
}

/// One reflected group-1 texture global that can receive material wrap metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
struct WrapTextureGlobal {
    /// WGSL variable name, including any naga-oil suffix.
    variable_name: String,
    /// Uniform field name without naga-oil suffix.
    wrap_field_name: String,
    /// Texture dimensionality.
    dimension: WrapTextureDimension,
}

/// One parsed function parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FunctionParam {
    /// Parameter identifier.
    name: String,
    /// Parameter type text.
    ty: String,
}

/// One parsed top-level function span.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FunctionSpan {
    /// Function identifier.
    name: String,
    /// Start byte of the full function declaration.
    start: usize,
    /// End byte of the full function body.
    end: usize,
    /// Start byte of the parameter list contents.
    params_start: usize,
    /// End byte of the parameter list contents.
    params_end: usize,
    /// Start byte of the function body contents.
    body_start: usize,
    /// End byte of the function body contents.
    body_end: usize,
    /// Parsed parameters.
    params: Vec<FunctionParam>,
}

/// One texture parameter that receives an inserted wrap-bits argument.
#[derive(Clone, Debug, Eq, PartialEq)]
struct TextureParam {
    /// Original parameter index.
    index: usize,
    /// Parameter identifier.
    name: String,
    /// Inserted wrap-bits parameter identifier.
    bits_name: String,
    /// Texture dimensionality.
    dimension: WrapTextureDimension,
}

/// Signature metadata for a function whose calls must receive inserted wrap bits.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FunctionRewrite {
    /// Texture parameters in original signature order.
    texture_params: Vec<TextureParam>,
}

/// Per-function context used while rewriting function bodies.
struct RewriteContext<'a> {
    /// Group-1 texture globals keyed by final WGSL variable name.
    globals: &'a BTreeMap<String, WrapTextureGlobal>,
    /// Transformed functions keyed by function name.
    function_rewrites: &'a BTreeMap<String, FunctionRewrite>,
    /// Uniform variable used to read inserted wrap fields.
    uniform_var: &'a str,
    /// Current function texture parameters keyed by parameter name.
    texture_params: BTreeMap<String, TextureParam>,
}

/// Adds `MirrorOnce` emulation to a flattened material WGSL target.
pub(super) fn rewrite_material_mirror_once_wgsl(
    wgsl: &str,
    label: &str,
) -> Result<String, BuildError> {
    let Some((uniform_var, uniform_ty)) = group1_material_uniform(wgsl) else {
        return Ok(wgsl.to_string());
    };
    let globals = group1_wrap_texture_globals(wgsl);
    if globals.is_empty() {
        return Ok(wgsl.to_string());
    }

    let mut with_fields = insert_wrap_fields(wgsl, &uniform_ty, globals.values(), label)?;
    let functions = parse_functions(&with_fields)?;
    let function_rewrites = function_rewrites(&functions);
    with_fields = rewrite_functions(
        &with_fields,
        &functions,
        &function_rewrites,
        &globals,
        &uniform_var,
    );
    with_fields.push_str(MIRROR_ONCE_HELPERS);
    Ok(with_fields)
}

/// Returns the group-1 material uniform variable and struct type.
fn group1_material_uniform(wgsl: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = wgsl.lines().collect();
    for pair in lines.windows(2) {
        if !pair[0].contains("@group(1)") || !pair[0].contains("@binding(0)") {
            continue;
        }
        let line = pair[1].trim();
        let rest = line.strip_prefix("var<uniform> ")?;
        let (name, ty) = rest.strip_suffix(';')?.split_once(':')?;
        return Some((name.trim().to_string(), ty.trim().to_string()));
    }
    None
}

/// Returns all wrap-aware group-1 texture globals keyed by final WGSL variable name.
fn group1_wrap_texture_globals(wgsl: &str) -> BTreeMap<String, WrapTextureGlobal> {
    let lines: Vec<&str> = wgsl.lines().collect();
    let mut globals = BTreeMap::new();
    for pair in lines.windows(2) {
        if !pair[0].contains("@group(1)") || pair[0].contains("@binding(0)") {
            continue;
        }
        let line = pair[1].trim();
        let Some(rest) = line.strip_prefix("var ") else {
            continue;
        };
        let Some((name, ty)) = rest.strip_suffix(';').and_then(|rest| rest.split_once(':')) else {
            continue;
        };
        let dimension = match ty.trim() {
            "texture_2d<f32>" => WrapTextureDimension::D2,
            "texture_3d<f32>" => WrapTextureDimension::D3,
            _ => continue,
        };
        let variable_name = name.trim().to_string();
        let base_name = unmangled_identifier(&variable_name).to_string();
        globals.insert(
            variable_name.clone(),
            WrapTextureGlobal {
                variable_name,
                wrap_field_name: format!("{base_name}{WRAP_MODE_BITS_SUFFIX}"),
                dimension,
            },
        );
    }
    globals
}

/// Strips the naga-oil module suffix from an identifier.
fn unmangled_identifier(identifier: &str) -> &str {
    identifier
        .split_once("X_naga_oil_mod_")
        .map_or(identifier, |(base, _)| base)
}

/// Inserts wrap-bit fields into the material uniform struct.
fn insert_wrap_fields<'a>(
    wgsl: &str,
    uniform_ty: &str,
    globals: impl Iterator<Item = &'a WrapTextureGlobal>,
    label: &str,
) -> Result<String, BuildError> {
    let fields = globals
        .map(|global| global.wrap_field_name.as_str())
        .collect::<BTreeSet<_>>();
    let Some((body_start, body_end)) = struct_body_span(wgsl, uniform_ty) else {
        return Err(BuildError::Message(format!(
            "{label}: group-1 material uniform struct `{uniform_ty}` was not found"
        )));
    };

    let body = &wgsl[body_start..body_end];
    let mut missing = Vec::new();
    for field in fields {
        if !body.contains(&format!("{field}:")) {
            missing.push(field.to_string());
        }
    }
    if missing.is_empty() {
        return Ok(wgsl.to_string());
    }

    let mut out = String::with_capacity(wgsl.len() + missing.len() * 40);
    out.push_str(&wgsl[..body_end]);
    for field in missing {
        out.push_str("    ");
        out.push_str(&field);
        out.push_str(": u32,\n");
    }
    out.push_str(&wgsl[body_end..]);
    Ok(out)
}

/// Returns the body span for a named struct, excluding braces.
fn struct_body_span(wgsl: &str, struct_name: &str) -> Option<(usize, usize)> {
    let needle = format!("struct {struct_name}");
    let start = wgsl.find(&needle)?;
    let open = wgsl[start..].find('{')? + start;
    let close = matching_delimiter(wgsl, open, '{', '}')?;
    Some((open + 1, close))
}

/// Parses all top-level WGSL functions.
fn parse_functions(wgsl: &str) -> Result<Vec<FunctionSpan>, BuildError> {
    let mut functions = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = wgsl[pos..].find("fn ") {
        let start = pos + rel;
        if start > 0 && is_identifier_byte(wgsl.as_bytes()[start - 1]) {
            pos = start + 3;
            continue;
        }
        let name_start = start + 3;
        let Some((name, name_end)) = read_identifier(wgsl, name_start) else {
            pos = start + 3;
            continue;
        };
        let Some(params_open) = skip_ws(wgsl, name_end).filter(|idx| wgsl.as_bytes()[*idx] == b'(')
        else {
            pos = name_end;
            continue;
        };
        let params_close = matching_delimiter(wgsl, params_open, '(', ')').ok_or_else(|| {
            BuildError::Message(format!(
                "unclosed parameter list for WGSL function `{name}`"
            ))
        })?;
        let body_open = wgsl[params_close..]
            .find('{')
            .map(|rel| params_close + rel)
            .ok_or_else(|| {
                BuildError::Message(format!("missing body for WGSL function `{name}`"))
            })?;
        let body_close = matching_delimiter(wgsl, body_open, '{', '}').ok_or_else(|| {
            BuildError::Message(format!("unclosed body for WGSL function `{name}`"))
        })?;
        let params = parse_params(&wgsl[params_open + 1..params_close]);
        functions.push(FunctionSpan {
            name: name.to_string(),
            start,
            end: body_close + 1,
            params_start: params_open + 1,
            params_end: params_close,
            body_start: body_open + 1,
            body_end: body_close,
            params,
        });
        pos = body_close + 1;
    }
    Ok(functions)
}

/// Builds function rewrite metadata.
fn function_rewrites(functions: &[FunctionSpan]) -> BTreeMap<String, FunctionRewrite> {
    functions
        .iter()
        .filter_map(|function| {
            let texture_params = function
                .params
                .iter()
                .enumerate()
                .filter_map(|(index, param)| {
                    let dimension = texture_dimension(&param.ty)?;
                    Some(TextureParam {
                        index,
                        name: param.name.clone(),
                        bits_name: format!("{}_wrap_mode_bits", param.name),
                        dimension,
                    })
                })
                .collect::<Vec<_>>();
            (!texture_params.is_empty())
                .then(|| (function.name.clone(), FunctionRewrite { texture_params }))
        })
        .collect()
}

/// Rewrites function signatures and bodies.
fn rewrite_functions(
    wgsl: &str,
    functions: &[FunctionSpan],
    function_rewrites: &BTreeMap<String, FunctionRewrite>,
    globals: &BTreeMap<String, WrapTextureGlobal>,
    uniform_var: &str,
) -> String {
    let mut out = String::with_capacity(wgsl.len() + functions.len() * 16);
    let mut cursor = 0usize;
    for function in functions {
        out.push_str(&wgsl[cursor..function.params_start]);
        if let Some(rewrite) = function_rewrites.get(&function.name) {
            out.push_str(&rewritten_params(&function.params, rewrite));
        } else {
            out.push_str(&wgsl[function.params_start..function.params_end]);
        }
        out.push_str(&wgsl[function.params_end..function.body_start]);
        let texture_params = function_rewrites
            .get(&function.name)
            .map(|rewrite| {
                rewrite
                    .texture_params
                    .iter()
                    .map(|param| (param.name.clone(), param.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let ctx = RewriteContext {
            globals,
            function_rewrites,
            uniform_var,
            texture_params,
        };
        out.push_str(&rewrite_body(
            &wgsl[function.body_start..function.body_end],
            &ctx,
        ));
        out.push_str(&wgsl[function.body_end..function.end]);
        cursor = function.end;
    }
    out.push_str(&wgsl[cursor..]);
    out
}

/// Returns a function parameter list with inserted wrap-bit parameters.
fn rewritten_params(params: &[FunctionParam], rewrite: &FunctionRewrite) -> String {
    let texture_by_index = rewrite
        .texture_params
        .iter()
        .map(|param| (param.index, param))
        .collect::<BTreeMap<_, _>>();
    let mut out = Vec::with_capacity(params.len() + rewrite.texture_params.len());
    for (index, param) in params.iter().enumerate() {
        out.push(format!("{}: {}", param.name, param.ty));
        if let Some(texture_param) = texture_by_index.get(&index) {
            out.push(format!("{}: u32", texture_param.bits_name));
        }
    }
    out.join(", ")
}

/// Rewrites one function body.
fn rewrite_body(body: &str, ctx: &RewriteContext<'_>) -> String {
    let with_texture_samples = rewrite_texture_builtin_calls(body, ctx);
    rewrite_user_function_calls(&with_texture_samples, ctx)
}

/// Rewrites WGSL `textureSample*` builtins for known group-1 or propagated texture parameters.
fn rewrite_texture_builtin_calls(body: &str, ctx: &RewriteContext<'_>) -> String {
    rewrite_named_calls(body, |name, args| {
        let sample_kind = TextureBuiltinKind::from_name(name)?;
        let first_arg = args.first()?.trim();
        let (dimension, bits) = texture_bits_expr(first_arg, ctx)?;
        rewrite_texture_builtin_call(sample_kind, dimension, args, &bits)
    })
}

/// Rewrites calls to functions whose texture parameters gained wrap-bit arguments.
fn rewrite_user_function_calls(body: &str, ctx: &RewriteContext<'_>) -> String {
    rewrite_named_calls(body, |name, args| {
        let rewrite = ctx.function_rewrites.get(name)?;
        Some(rewrite_user_function_call(name, args, rewrite, ctx))
    })
}

/// Rewrites matching named calls in a WGSL snippet.
fn rewrite_named_calls(
    body: &str,
    mut rewrite: impl FnMut(&str, &[String]) -> Option<String>,
) -> String {
    let mut out = String::with_capacity(body.len());
    let mut cursor = 0usize;
    while let Some((name_start, name, open)) = next_call(body, cursor) {
        let Some(close) = matching_delimiter(body, open, '(', ')') else {
            break;
        };
        let args = split_args(&body[open + 1..close]);
        if let Some(replacement) = rewrite(name, &args) {
            out.push_str(&body[cursor..name_start]);
            out.push_str(&replacement);
            cursor = close + 1;
        } else {
            out.push_str(&body[cursor..=open]);
            cursor = open + 1;
        }
    }
    out.push_str(&body[cursor..]);
    out
}

/// Returns the next identifier call in `body` starting at `cursor`.
fn next_call(body: &str, mut cursor: usize) -> Option<(usize, &str, usize)> {
    let bytes = body.as_bytes();
    while cursor < bytes.len() {
        if !is_identifier_start(bytes[cursor]) {
            cursor += 1;
            continue;
        }
        let start = cursor;
        cursor += 1;
        while cursor < bytes.len() && is_identifier_byte(bytes[cursor]) {
            cursor += 1;
        }
        let name = &body[start..cursor];
        let open = skip_ws(body, cursor)?;
        if body.as_bytes()[open] == b'(' {
            return Some((start, name, open));
        }
    }
    None
}

/// WGSL texture builtin kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextureBuiltinKind {
    /// `textureSample`.
    Sample,
    /// `textureSampleBias`.
    Bias,
    /// `textureSampleLevel`.
    Level,
    /// `textureSampleGrad`.
    Grad,
}

impl TextureBuiltinKind {
    /// Converts a WGSL builtin identifier into a sample kind.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "textureSample" => Some(Self::Sample),
            "textureSampleBias" => Some(Self::Bias),
            "textureSampleLevel" => Some(Self::Level),
            "textureSampleGrad" => Some(Self::Grad),
            _ => None,
        }
    }
}

/// Builds a wrapped replacement for a texture builtin call.
fn rewrite_texture_builtin_call(
    kind: TextureBuiltinKind,
    dimension: WrapTextureDimension,
    args: &[String],
    bits: &str,
) -> Option<String> {
    let helper = match (kind, dimension, args.len()) {
        (TextureBuiltinKind::Sample, WrapTextureDimension::D2, 3) => {
            "renderide_mirroronce_sample_2d"
        }
        (TextureBuiltinKind::Sample, WrapTextureDimension::D2, 4) => {
            return Some(format!(
                "textureSample({}, {}, renderide_mirror_once_2d({}, {}), {})",
                args[0], args[1], args[2], bits, args[3]
            ));
        }
        (TextureBuiltinKind::Sample, WrapTextureDimension::D3, 3) => {
            "renderide_mirroronce_sample_3d"
        }
        (TextureBuiltinKind::Bias, WrapTextureDimension::D2, 4) => {
            "renderide_mirroronce_sample_bias_2d"
        }
        (TextureBuiltinKind::Bias, WrapTextureDimension::D3, 4) => {
            "renderide_mirroronce_sample_bias_3d"
        }
        (TextureBuiltinKind::Level, WrapTextureDimension::D2, 4) => {
            "renderide_mirroronce_sample_level_2d"
        }
        (TextureBuiltinKind::Level, WrapTextureDimension::D3, 4) => {
            "renderide_mirroronce_sample_level_3d"
        }
        (TextureBuiltinKind::Grad, WrapTextureDimension::D2, 5) => {
            "renderide_mirroronce_sample_grad_2d"
        }
        _ => return None,
    };
    let mut all_args = args.to_vec();
    all_args.push(bits.to_string());
    Some(format!("{helper}({})", all_args.join(", ")))
}

/// Builds a rewritten user-function call with inserted wrap-bit arguments.
fn rewrite_user_function_call(
    name: &str,
    args: &[String],
    rewrite: &FunctionRewrite,
    ctx: &RewriteContext<'_>,
) -> String {
    let texture_by_index = rewrite
        .texture_params
        .iter()
        .map(|param| (param.index, param))
        .collect::<BTreeMap<_, _>>();
    let mut out_args = Vec::with_capacity(args.len() + rewrite.texture_params.len());
    for (index, arg) in args.iter().enumerate() {
        out_args.push(arg.clone());
        if let Some(texture_param) = texture_by_index.get(&index) {
            out_args.push(
                texture_bits_expr_for_dimension(arg.trim(), texture_param.dimension, ctx)
                    .unwrap_or_else(|| "0u".to_string()),
            );
        }
    }
    format!("{name}({})", out_args.join(", "))
}

/// Returns the dimension and bits expression for a sampled texture expression.
fn texture_bits_expr(
    texture_expr: &str,
    ctx: &RewriteContext<'_>,
) -> Option<(WrapTextureDimension, String)> {
    if let Some(global) = ctx.globals.get(texture_expr) {
        return Some((
            global.dimension,
            format!("{}.{}", ctx.uniform_var, global.wrap_field_name),
        ));
    }
    if let Some(param) = ctx.texture_params.get(texture_expr) {
        return Some((param.dimension, param.bits_name.clone()));
    }
    None
}

/// Returns the bits expression for a texture argument with an expected dimension.
fn texture_bits_expr_for_dimension(
    texture_expr: &str,
    dimension: WrapTextureDimension,
    ctx: &RewriteContext<'_>,
) -> Option<String> {
    let (actual_dimension, bits) = texture_bits_expr(texture_expr, ctx)?;
    (actual_dimension == dimension).then_some(bits)
}

/// Parses a comma-separated function parameter list.
fn parse_params(src: &str) -> Vec<FunctionParam> {
    split_args(src)
        .into_iter()
        .filter_map(|part| {
            let (name, ty) = part.split_once(':')?;
            Some(FunctionParam {
                name: name.trim().to_string(),
                ty: ty.trim().to_string(),
            })
        })
        .collect()
}

/// Splits comma-separated arguments while respecting nested delimiters.
fn split_args(src: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut start = 0usize;
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut angle = 0i32;
    for (idx, ch) in src.char_indices() {
        match ch {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '<' => angle += 1,
            '>' => angle -= 1,
            ',' if paren == 0 && bracket == 0 && angle == 0 => {
                push_arg(src, start, idx, &mut args);
                start = idx + 1;
            }
            _ => {}
        }
    }
    push_arg(src, start, src.len(), &mut args);
    args
}

/// Pushes one trimmed argument when it is non-empty.
fn push_arg(src: &str, start: usize, end: usize, args: &mut Vec<String>) {
    let arg = src[start..end].trim();
    if !arg.is_empty() {
        args.push(arg.to_string());
    }
}

/// Returns the texture dimension represented by a type string.
fn texture_dimension(ty: &str) -> Option<WrapTextureDimension> {
    match ty.trim() {
        "texture_2d<f32>" => Some(WrapTextureDimension::D2),
        "texture_3d<f32>" => Some(WrapTextureDimension::D3),
        _ => None,
    }
}

/// Returns the byte index of the matching delimiter.
fn matching_delimiter(src: &str, open: usize, open_ch: char, close_ch: char) -> Option<usize> {
    let mut depth = 0i32;
    for (rel, ch) in src[open..].char_indices() {
        if ch == open_ch {
            depth += 1;
        } else if ch == close_ch {
            depth -= 1;
            if depth == 0 {
                return Some(open + rel);
            }
        }
    }
    None
}

/// Skips whitespace and returns the next byte index.
fn skip_ws(src: &str, mut idx: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    (idx < bytes.len()).then_some(idx)
}

/// Reads an identifier at `idx`.
fn read_identifier(src: &str, idx: usize) -> Option<(&str, usize)> {
    let bytes = src.as_bytes();
    if idx >= bytes.len() || !is_identifier_start(bytes[idx]) {
        return None;
    }
    let mut end = idx + 1;
    while end < bytes.len() && is_identifier_byte(bytes[end]) {
        end += 1;
    }
    Some((&src[idx..end], end))
}

/// Returns whether `byte` can start a WGSL identifier.
fn is_identifier_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

/// Returns whether `byte` can appear in a WGSL identifier.
fn is_identifier_byte(byte: u8) -> bool {
    is_identifier_start(byte) || byte.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a minimal flattened material WGSL source with one sampled 2D texture.
    fn material_shader_source(sample_expr: &str) -> String {
        format!(
            r#"
struct Material {{
    color: vec4<f32>,
}}

@group(1) @binding(0)
var<uniform> material: Material;

@group(1) @binding(1)
var _MainTex: texture_2d<f32>;

@group(1) @binding(2)
var _MainTex_sampler: sampler;

fn fragment_main(uv: vec2<f32>) -> vec4<f32> {{
    return {sample_expr};
}}
"#
        )
    }

    /// Verifies that texture samples gain MirrorOnce metadata and mirror-clamp math.
    #[test]
    fn rewrite_injects_mirror_once_coordinate_emulation() {
        let source = material_shader_source("textureSample(_MainTex, _MainTex_sampler, uv)");
        let rewritten = rewrite_material_mirror_once_wgsl(&source, "test").unwrap();

        assert!(rewritten.contains("_MainTex_WrapModeBits: u32"));
        assert!(rewritten.contains(
            "renderide_mirroronce_sample_2d(_MainTex, _MainTex_sampler, uv, material._MainTex_WrapModeBits)"
        ));
        assert!(rewritten.contains("fn renderide_mirror_once_coord(coord: f32) -> f32"));
        assert!(rewritten.contains("return clamp(abs(coord), 0.0, 1.0);"));
        assert!(rewritten.contains("return -1.0;"));
        assert!(!rewritten.contains("return coord + 1.0;"));
        assert!(!rewritten.contains("return coord - 1.0;"));
    }

    /// Verifies that offset samples still mirror coordinates before using the builtin overload.
    #[test]
    fn rewrite_offset_sample_uses_mirror_once_coordinate_emulation() {
        let source = material_shader_source(
            "textureSample(_MainTex, _MainTex_sampler, uv, vec2<i32>(1, -1))",
        );
        let rewritten = rewrite_material_mirror_once_wgsl(&source, "test").unwrap();

        assert!(rewritten.contains(
            "textureSample(_MainTex, _MainTex_sampler, renderide_mirror_once_2d(uv, material._MainTex_WrapModeBits), vec2<i32>(1, -1))"
        ));
    }
}
