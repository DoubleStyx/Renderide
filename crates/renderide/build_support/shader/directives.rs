//! WGSL source directive parsing.

use super::error::BuildError;

/// Build-side `wgpu::Features` selector declared by `//#wgpu_feature`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildWgpuFeature {
    /// Fragment shader barycentric coordinates.
    ShaderBarycentrics,
}

impl BuildWgpuFeature {
    /// Parses a `//#wgpu_feature` token.
    fn parse(value: &str, file: &str, line: usize) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "shader_barycentrics" | "shader-barycentrics" => Ok(Self::ShaderBarycentrics),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: unknown `//#wgpu_feature` token `{value}` (allowed: shader_barycentrics)"
            ))),
        }
    }

    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::ShaderBarycentrics => "wgpu::Features::SHADER_BARYCENTRICS",
        }
    }
}

/// One required wgpu feature directive attached to a WGSL source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WgpuFeatureDirective {
    /// Required feature bit for the composed target.
    pub feature: BuildWgpuFeature,
}

/// Texture fallback token declared by `//#texture_default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TextureDefaultKind {
    /// Unity `"white"` default texture.
    White,
    /// Unity `"black"` default texture.
    Black,
    /// Unity `"gray"` / `"grey"` default texture.
    Gray,
    /// Unity `"bump"` default texture.
    Bump,
    /// Unity `"red"` default texture.
    Red,
    /// Empty Unity texture default (`""`), resolved by the runtime as Unity's gray placeholder.
    Empty,
}

impl TextureDefaultKind {
    /// Parses a source directive token.
    fn parse(value: &str, file: &str, line: usize) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "white" => Ok(Self::White),
            "black" => Ok(Self::Black),
            "gray" | "grey" => Ok(Self::Gray),
            "bump" => Ok(Self::Bump),
            "red" => Ok(Self::Red),
            "empty" => Ok(Self::Empty),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: unknown `//#texture_default` token `{value}` (allowed: white, black, gray, grey, bump, red, empty)"
            ))),
        }
    }

    /// Rust variant name used in generated embedded metadata.
    const fn rust_variant(self) -> &'static str {
        match self {
            Self::White => "White",
            Self::Black => "Black",
            Self::Gray => "Gray",
            Self::Bump => "Bump",
            Self::Red => "Red",
            Self::Empty => "Empty",
        }
    }
}

/// One texture fallback directive attached to a material WGSL source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TextureDefaultDirective {
    /// Reflected host texture property name, e.g. `_MainTex`.
    pub property: String,
    /// Unity default token for the texture slot.
    pub kind: TextureDefaultKind,
}

/// Material uniform fallback value kind declared by `//#mat_default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MaterialDefaultKind {
    /// Unity float property default.
    Float,
    /// Unity vector/color property default.
    Vec4,
}

/// Material uniform fallback value declared by `//#mat_default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MaterialDefaultValue {
    /// Unity property default kind.
    pub kind: MaterialDefaultKind,
    /// Unity property default bits. Float defaults use only the first element.
    pub bits: [u32; 4],
}

impl MaterialDefaultValue {
    /// Creates a float material default from raw `f32` bits.
    pub(super) const fn float_bits(bits: u32) -> Self {
        Self {
            kind: MaterialDefaultKind::Float,
            bits: [bits, 0, 0, 0],
        }
    }

    /// Creates a vec4 material default from raw `f32` bits.
    pub(super) const fn vec4_bits(bits: [u32; 4]) -> Self {
        Self {
            kind: MaterialDefaultKind::Vec4,
            bits,
        }
    }

    /// Rust expression used in generated embedded metadata.
    fn rust_literal(self) -> String {
        match self.kind {
            MaterialDefaultKind::Float => {
                format!(
                    "EmbeddedMaterialDefaultValue::float({value})",
                    value = rust_f32_from_bits(self.bits[0])
                )
            }
            MaterialDefaultKind::Vec4 => {
                let values = self
                    .bits
                    .iter()
                    .map(|bits| rust_f32_from_bits(*bits))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("EmbeddedMaterialDefaultValue::vec4([{values}])")
            }
        }
    }
}

/// Renders raw `f32` bits as a readable Rust expression.
fn rust_f32_from_bits(bits: u32) -> String {
    format!(
        "f32::from_bits(0x{upper:04x}_{lower:04x})",
        upper = bits >> 16,
        lower = bits & 0xffff
    )
}

/// One material uniform fallback directive attached to a material WGSL source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MaterialDefaultDirective {
    /// Reflected host material property name, e.g. `_GlossMapScale`.
    pub property: String,
    /// Unity property default value for the uniform field.
    pub value: MaterialDefaultValue,
}

/// Material pass role declared by `//#pass type=...`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildPassType {
    /// Normal raster material pass.
    Forward,
    /// Source-authored depth-only prepass.
    DepthPrepass,
}

impl BuildPassType {
    /// Converts a source token to a pass type.
    fn parse(value: &str, file: &str, line: usize) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "forward" => Ok(Self::Forward),
            "depth_prepass" | "depthprepass" | "prepass" => Ok(Self::DepthPrepass),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: unknown `//#pass` type `{value}` (allowed: forward, depth_prepass)"
            ))),
        }
    }

    /// Rust `PassType` variant name used in generated embedded metadata.
    const fn rust_variant(self) -> &'static str {
        match self {
            Self::Forward => "Forward",
            Self::DepthPrepass => "DepthPrepass",
        }
    }

    /// Default debug label for this pass type.
    const fn default_name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::DepthPrepass => "depth_prepass",
        }
    }
}

/// Build-side `wgpu::CompareFunction` selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildDepthCompare {
    /// Main reverse-Z forward compare.
    Main,
    /// Always pass depth test.
    Always,
    /// Less-than depth test.
    Less,
    /// Greater-equal depth test.
    GreaterEqual,
}

impl BuildDepthCompare {
    /// Parses a depth-compare token.
    fn parse(value: &str, file: &str, line: usize, key: &str) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "main" | "forward" | "default" => Ok(Self::Main),
            "always" => Ok(Self::Always),
            "less" => Ok(Self::Less),
            "greater_equal" | "greaterequal" | "gequal" | "g_equal" => Ok(Self::GreaterEqual),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: `//#pass` `{key}` expects a depth compare (main, always, less, greater_equal), got `{value}`"
            ))),
        }
    }

    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::Main => "crate::gpu::MAIN_FORWARD_DEPTH_COMPARE",
            Self::Always => "wgpu::CompareFunction::Always",
            Self::Less => "wgpu::CompareFunction::Less",
            Self::GreaterEqual => "wgpu::CompareFunction::GreaterEqual",
        }
    }
}

/// Build-side cull-mode selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildCullMode {
    /// Cull back faces.
    Back,
    /// Cull front faces.
    Front,
    /// Disable culling.
    Off,
}

impl BuildCullMode {
    /// Parses a cull token.
    fn parse(value: &str, file: &str, line: usize, key: &str) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "back" => Ok(Self::Back),
            "front" => Ok(Self::Front),
            "off" | "none" => Ok(Self::Off),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: `//#pass` `{key}` expects a cull mode (back, front, off), got `{value}`"
            ))),
        }
    }

    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::Back => "Some(wgpu::Face::Back)",
            Self::Front => "Some(wgpu::Face::Front)",
            Self::Off => "None",
        }
    }
}

/// Build-side color-write selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildColorWrites {
    /// Write all color channels.
    Rgba,
    /// Write RGB only.
    Rgb,
    /// Write no color channels.
    None,
}

impl BuildColorWrites {
    /// Parses a color-mask token.
    fn parse(value: &str, file: &str, line: usize, key: &str) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rgba" | "all" | "15" => Ok(Self::Rgba),
            "rgb" | "color" | "7" => Ok(Self::Rgb),
            "0" | "none" | "off" => Ok(Self::None),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: `//#pass` `{key}` expects a color mask (rgba, rgb, 0), got `{value}`"
            ))),
        }
    }

    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::Rgba => "wgpu::ColorWrites::ALL",
            Self::Rgb => "wgpu::ColorWrites::COLOR",
            Self::None => "crate::materials::COLOR_WRITES_NONE",
        }
    }
}

/// Build-side static blend selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildBlend {
    /// Disable blending.
    Off,
    /// Unity straight-alpha blending.
    Alpha,
    /// Unity additive blending.
    Additive,
    /// Unity premultiplied-alpha blending.
    Premultiplied,
    /// Unity overlay color/no-op plus max-alpha blending.
    Overlay,
}

impl BuildBlend {
    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::Off => "None",
            Self::Alpha => "Some(crate::materials::PASS_BLEND_SRC_ALPHA_ONE_MINUS_SRC_ALPHA)",
            Self::Additive => "Some(crate::materials::PASS_BLEND_ONE_ONE)",
            Self::Premultiplied => "Some(crate::materials::PASS_BLEND_ONE_ONE_MINUS_SRC_ALPHA)",
            Self::Overlay => "Some(crate::materials::PASS_BLEND_OVERLAY_NOOP_COLOR_MAX_ALPHA)",
        }
    }
}

/// Build-side material blend materialization mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildMaterialPassState {
    /// Static blend state.
    Static,
    /// Material-driven forward blend state.
    Forward,
    /// Material-driven transparent blend state.
    TransparentForward,
    /// Material-driven overlay blend state.
    Overlay,
    /// Material-driven filter blend state.
    Filter,
}

impl BuildMaterialPassState {
    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::Static => "crate::materials::MaterialPassState::Static",
            Self::Forward => "crate::materials::MaterialPassState::Forward",
            Self::TransparentForward => "crate::materials::MaterialPassState::TransparentForward",
            Self::Overlay => "crate::materials::MaterialPassState::Overlay",
            Self::Filter => "crate::materials::MaterialPassState::Filter",
        }
    }
}

/// Build-side per-field material render-state override policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BuildRenderStatePolicy {
    /// Whether `_ColorMask` overrides the pass color write mask.
    pub(super) color_mask: bool,
    /// Whether `_ZWrite` overrides the pass depth-write flag.
    pub(super) depth_write: bool,
    /// Whether `_ZTest` overrides the pass depth compare function.
    pub(super) depth_compare: bool,
    /// Whether `_Cull` overrides the pass cull mode.
    pub(super) cull: bool,
    /// Whether `_Stencil*` properties override the pass stencil state.
    pub(super) stencil: bool,
    /// Whether `_OffsetFactor` / `_OffsetUnits` override the pass depth bias.
    pub(super) depth_offset: bool,
}

impl BuildRenderStatePolicy {
    /// Material properties may override every supported render-state field.
    pub(super) const ALL_MATERIAL: Self = Self {
        color_mask: true,
        depth_write: true,
        depth_compare: true,
        cull: true,
        stencil: true,
        depth_offset: true,
    };

    /// Rust expression used in generated embedded metadata.
    fn rust_literal(self) -> String {
        format!(
            "crate::materials::MaterialRenderStatePolicy {{ color_mask: {color_mask}, depth_write: {depth_write}, depth_compare: {depth_compare}, cull: {cull}, stencil: {stencil}, depth_offset: {depth_offset} }}",
            color_mask = self.color_mask,
            depth_write = self.depth_write,
            depth_compare = self.depth_compare,
            cull = self.cull,
            stencil = self.stencil,
            depth_offset = self.depth_offset,
        )
    }
}

/// Parsed source metadata for one pass before the fragment entry point is attached.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BuildPassDraft {
    /// Material pass type.
    pass_type: BuildPassType,
    /// Debug label for logs / pipeline names.
    name: String,
    /// Vertex entry point for this pass.
    vertex_entry: String,
    /// Whether this pass enables hardware alpha-to-coverage.
    alpha_to_coverage: bool,
    /// Depth comparison fallback.
    depth_compare: BuildDepthCompare,
    /// `_ZTest` enum layout used when host material state overrides this pass.
    depth_compare_domain: BuildDepthCompareDomain,
    /// Whether this pass writes depth by default.
    depth_write: bool,
    /// Authored cull fallback.
    cull_mode: BuildCullMode,
    /// Authored blend fallback.
    blend: BuildBlend,
    /// Authored color write fallback.
    write_mask: BuildColorWrites,
    /// Static reverse-Z slope depth bias emitted from Unity `Offset factor`.
    depth_bias_slope_scale_bits: u32,
    /// Static reverse-Z constant depth bias emitted from Unity `Offset units`.
    depth_bias_constant: i32,
    /// Material blend-state materialization mode.
    material_state: BuildMaterialPassState,
    /// Per-field material render-state override policy.
    render_state_policy: BuildRenderStatePolicy,
}

impl BuildPassDraft {
    /// Returns default metadata for a pass type.
    fn for_type(pass_type: BuildPassType) -> Self {
        match pass_type {
            BuildPassType::Forward => Self {
                pass_type,
                name: pass_type.default_name().to_string(),
                vertex_entry: "vs_main".to_string(),
                alpha_to_coverage: false,
                depth_compare: BuildDepthCompare::Main,
                depth_compare_domain: BuildDepthCompareDomain::FrooxZTest,
                depth_write: true,
                cull_mode: BuildCullMode::Back,
                blend: BuildBlend::Off,
                write_mask: BuildColorWrites::Rgb,
                depth_bias_slope_scale_bits: 0.0f32.to_bits(),
                depth_bias_constant: 0,
                material_state: BuildMaterialPassState::Forward,
                render_state_policy: BuildRenderStatePolicy::ALL_MATERIAL,
            },
            BuildPassType::DepthPrepass => {
                let mut policy = BuildRenderStatePolicy::ALL_MATERIAL;
                policy.color_mask = false;
                policy.depth_write = false;
                Self {
                    pass_type,
                    name: pass_type.default_name().to_string(),
                    vertex_entry: "vs_main".to_string(),
                    alpha_to_coverage: false,
                    depth_compare: BuildDepthCompare::Main,
                    depth_compare_domain: BuildDepthCompareDomain::FrooxZTest,
                    depth_write: true,
                    cull_mode: BuildCullMode::Back,
                    blend: BuildBlend::Off,
                    write_mask: BuildColorWrites::None,
                    depth_bias_slope_scale_bits: 0.0f32.to_bits(),
                    depth_bias_constant: 0,
                    material_state: BuildMaterialPassState::Static,
                    render_state_policy: policy,
                }
            }
        }
    }
}

/// `_ZTest` enum layout selected by a material pass directive.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum BuildDepthCompareDomain {
    /// FrooxEngine `ZTest` layout used by host material-provider fields.
    #[default]
    FrooxZTest,
    /// Unity `CompareFunction` layout used by BiRP shader properties.
    UnityCompareFunction,
}

impl BuildDepthCompareDomain {
    /// Rust expression used in generated embedded metadata.
    const fn rust_literal(self) -> &'static str {
        match self {
            Self::FrooxZTest => "crate::materials::MaterialDepthCompareDomain::FrooxZTest",
            Self::UnityCompareFunction => {
                "crate::materials::MaterialDepthCompareDomain::UnityCompareFunction"
            }
        }
    }
}

/// One declared pass: parsed metadata and the fragment entry point it sits above.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BuildPassDirective {
    /// Material pass type.
    pub pass_type: BuildPassType,
    /// Debug label for logs / pipeline names.
    pub name: String,
    /// Fragment entry point name the `//#pass` tag sits above.
    pub fragment_entry: String,
    /// Vertex entry point for this pass. Defaults to `vs_main`; overridden via `vs=...`.
    pub vertex_entry: String,
    /// Whether this pass enables hardware alpha-to-coverage.
    pub alpha_to_coverage: bool,
    /// `_ZTest` enum layout used when host material state overrides this pass.
    pub depth_compare_domain: BuildDepthCompareDomain,
    /// Depth comparison fallback.
    pub depth_compare: BuildDepthCompare,
    /// Whether this pass writes depth by default.
    pub depth_write: bool,
    /// Authored cull fallback.
    pub cull_mode: BuildCullMode,
    /// Authored blend fallback.
    pub blend: BuildBlend,
    /// Authored color write fallback.
    pub write_mask: BuildColorWrites,
    /// Static reverse-Z slope depth bias emitted from Unity `Offset factor`.
    pub depth_bias_slope_scale_bits: u32,
    /// Static reverse-Z constant depth bias emitted from Unity `Offset units`.
    pub depth_bias_constant: i32,
    /// Material blend-state materialization mode.
    pub material_state: BuildMaterialPassState,
    /// Per-field material render-state override policy.
    pub render_state_policy: BuildRenderStatePolicy,
}

/// Parses `fn <name>(...)` out of a line.
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

/// Finds the first `@fragment` entry point declared after `start_line`.
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

/// Parses material pass directives from WGSL source.
pub(super) fn parse_pass_directives(
    source: &str,
    file: &str,
) -> Result<Vec<BuildPassDirective>, BuildError> {
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
                "{file}:{line_no}: `//#pass` tag requires `type=forward` or `type=depth_prepass`"
            )));
        }
        let mut tokens = body.split_whitespace();
        let first = tokens.next().unwrap_or("");
        let Some((first_key, first_value)) = first.split_once('=') else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#pass` uses explicit key=value metadata; start with `type=forward` or `type=depth_prepass`, got `{first}`"
            )));
        };
        if !first_key.trim().eq_ignore_ascii_case("type") {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: first `//#pass` token must be `type=...`, got `{first}`"
            )));
        }
        let mut draft =
            BuildPassDraft::for_type(BuildPassType::parse(first_value.trim(), file, line_no)?);
        for token in tokens {
            let (key, value) = token.split_once('=').ok_or_else(|| {
                BuildError::Message(format!(
                    "{file}:{line_no}: expected `key=value` after `type=...` in `//#pass`, got `{token}`"
                ))
            })?;
            match key.trim().to_ascii_lowercase().as_str() {
                "type" => {
                    return Err(BuildError::Message(format!(
                        "{file}:{line_no}: `//#pass` accepts exactly one `type=...` token"
                    )));
                }
                "name" => draft.name = parse_pass_name(value.trim(), file, line_no)?,
                "vs" | "vertex" => draft.vertex_entry = value.trim().to_string(),
                "a2c" | "alpha_to_coverage" => {
                    draft.alpha_to_coverage =
                        parse_bool_value(value.trim(), file, line_no, key.trim())?;
                }
                "blend" => parse_blend_value(value.trim(), file, line_no, &mut draft)?,
                "zwrite" | "z_write" | "depth_write" | "depthwrite" => {
                    parse_zwrite_value(value.trim(), file, line_no, &mut draft)?;
                }
                "ztest" | "z_test" | "depth_compare" | "depthcompare" => {
                    parse_ztest_value(value.trim(), file, line_no, &mut draft)?;
                }
                "cull" => parse_cull_value(value.trim(), file, line_no, &mut draft)?,
                "color_mask" | "colormask" | "write_mask" | "writemask" => {
                    parse_color_mask_value(value.trim(), file, line_no, &mut draft)?;
                }
                "stencil" => parse_stencil_value(value.trim(), file, line_no, &mut draft)?,
                "offset" => parse_offset_value(value.trim(), file, line_no, &mut draft)?,
                _ => {
                    return Err(BuildError::Message(format!(
                        "{file}:{line_no}: unknown `//#pass` key `{key}` (allowed: type, name, vs, a2c, blend, zwrite, ztest, cull, color_mask, stencil, offset)"
                    )));
                }
            }
        }
        let fragment_entry = next_fragment_entry_after(&lines, line_idx + 1, file, line_no)?;
        passes.push(BuildPassDirective {
            pass_type: draft.pass_type,
            name: draft.name,
            fragment_entry,
            vertex_entry: draft.vertex_entry,
            alpha_to_coverage: draft.alpha_to_coverage,
            depth_compare_domain: draft.depth_compare_domain,
            depth_compare: draft.depth_compare,
            depth_write: draft.depth_write,
            cull_mode: draft.cull_mode,
            blend: draft.blend,
            write_mask: draft.write_mask,
            depth_bias_slope_scale_bits: draft.depth_bias_slope_scale_bits,
            depth_bias_constant: draft.depth_bias_constant,
            material_state: draft.material_state,
            render_state_policy: draft.render_state_policy,
        });
    }
    Ok(passes)
}

/// Parses and validates a pass debug label.
fn parse_pass_name(value: &str, file: &str, line: usize) -> Result<String, BuildError> {
    if value.is_empty()
        || value
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
    {
        return Err(BuildError::Message(format!(
            "{file}:{line}: `//#pass` name must contain only ASCII letters, digits, `_`, or `-`, got `{value}`"
        )));
    }
    Ok(value.to_string())
}

/// Parses the pass blend metadata token.
fn parse_blend_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    match value.to_ascii_lowercase().as_str() {
        "off" | "none" => {
            draft.blend = BuildBlend::Off;
            draft.material_state = BuildMaterialPassState::Static;
        }
        "alpha" | "src_alpha" | "srcalpha" => {
            draft.blend = BuildBlend::Alpha;
            draft.write_mask = BuildColorWrites::Rgba;
            draft.material_state = BuildMaterialPassState::Static;
        }
        "additive" | "one_one" | "oneone" => {
            draft.blend = BuildBlend::Additive;
            draft.write_mask = BuildColorWrites::Rgba;
            draft.material_state = BuildMaterialPassState::Static;
        }
        "premul" | "premultiplied" | "premultiplied_alpha" => {
            draft.blend = BuildBlend::Premultiplied;
            draft.write_mask = BuildColorWrites::Rgba;
            draft.material_state = BuildMaterialPassState::Static;
        }
        "material" => {
            draft.blend = BuildBlend::Off;
            draft.material_state = BuildMaterialPassState::Forward;
        }
        "material_filter" | "filter" => {
            draft.blend = BuildBlend::Off;
            draft.material_state = BuildMaterialPassState::Filter;
        }
        "material_overlay" | "overlay" => {
            draft.blend = BuildBlend::Overlay;
            draft.write_mask = BuildColorWrites::Rgba;
            draft.material_state = BuildMaterialPassState::Overlay;
        }
        "transparent_material" | "material_transparent" => {
            draft.blend = BuildBlend::Premultiplied;
            draft.write_mask = BuildColorWrites::Rgba;
            draft.depth_write = false;
            draft.material_state = BuildMaterialPassState::TransparentForward;
        }
        _ => {
            return Err(BuildError::Message(format!(
                "{file}:{line}: `//#pass` blend expects off, alpha, additive, premul, material, material_filter, material_overlay, or transparent_material, got `{value}`"
            )));
        }
    }
    Ok(())
}

/// Parses the pass ZWrite metadata token.
fn parse_zwrite_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    if let Some(inner) = material_inner(value, "material") {
        draft.depth_write = parse_bool_like(inner, file, line, "zwrite")?;
        draft.render_state_policy.depth_write = true;
        return Ok(());
    }
    draft.depth_write = parse_bool_like(value, file, line, "zwrite")?;
    draft.render_state_policy.depth_write = false;
    Ok(())
}

/// Parses the pass ZTest metadata token.
fn parse_ztest_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    if let Some(inner) =
        material_inner(value, "material_froox").or_else(|| material_inner(value, "material"))
    {
        draft.depth_compare = BuildDepthCompare::parse(inner, file, line, "ztest")?;
        draft.depth_compare_domain = BuildDepthCompareDomain::FrooxZTest;
        draft.render_state_policy.depth_compare = true;
        return Ok(());
    }
    if let Some(inner) = material_inner(value, "material_unity") {
        draft.depth_compare = BuildDepthCompare::parse(inner, file, line, "ztest")?;
        draft.depth_compare_domain = BuildDepthCompareDomain::UnityCompareFunction;
        draft.render_state_policy.depth_compare = true;
        return Ok(());
    }
    draft.depth_compare = BuildDepthCompare::parse(value, file, line, "ztest")?;
    draft.depth_compare_domain = BuildDepthCompareDomain::FrooxZTest;
    draft.render_state_policy.depth_compare = false;
    Ok(())
}

/// Parses the pass culling metadata token.
fn parse_cull_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    if let Some(inner) = material_inner(value, "material") {
        draft.cull_mode = BuildCullMode::parse(inner, file, line, "cull")?;
        draft.render_state_policy.cull = true;
        return Ok(());
    }
    draft.cull_mode = BuildCullMode::parse(value, file, line, "cull")?;
    draft.render_state_policy.cull = false;
    Ok(())
}

/// Parses the pass color-mask metadata token.
fn parse_color_mask_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    if let Some(inner) = material_inner(value, "material") {
        draft.write_mask = BuildColorWrites::parse(inner, file, line, "color_mask")?;
        draft.render_state_policy.color_mask = true;
        return Ok(());
    }
    draft.write_mask = BuildColorWrites::parse(value, file, line, "color_mask")?;
    draft.render_state_policy.color_mask = false;
    Ok(())
}

/// Parses the pass stencil metadata token.
fn parse_stencil_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    match value.to_ascii_lowercase().as_str() {
        "material" => draft.render_state_policy.stencil = true,
        "off" | "none" => draft.render_state_policy.stencil = false,
        _ => {
            return Err(BuildError::Message(format!(
                "{file}:{line}: `//#pass` stencil expects material or off, got `{value}`"
            )));
        }
    }
    Ok(())
}

/// Parses the pass offset metadata token.
fn parse_offset_value(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    if let Some(inner) = material_inner(value, "material") {
        parse_offset_pair(inner, file, line, draft)?;
        draft.render_state_policy.depth_offset = true;
        return Ok(());
    }
    parse_offset_pair(value, file, line, draft)?;
    draft.render_state_policy.depth_offset = false;
    Ok(())
}

/// Parses a Unity `Offset factor, units` pair.
fn parse_offset_pair(
    value: &str,
    file: &str,
    line: usize,
    draft: &mut BuildPassDraft,
) -> Result<(), BuildError> {
    let (factor, units) = value.split_once(',').ok_or_else(|| {
        BuildError::Message(format!(
            "{file}:{line}: `//#pass` offset expects `factor,units`, got `{value}`"
        ))
    })?;
    let factor = parse_f32_value(factor.trim(), file, line, "offset")?;
    let units = parse_f32_value(units.trim(), file, line, "offset")?;
    draft.depth_bias_slope_scale_bits = reverse_z_offset_factor(factor).to_bits();
    draft.depth_bias_constant = unity_offset_units(units).saturating_neg();
    Ok(())
}

/// Returns the content of `prefix(...)` metadata values.
fn material_inner<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    value
        .strip_prefix(prefix)?
        .strip_prefix('(')?
        .strip_suffix(')')
        .map(str::trim)
}

/// Parses boolean-like state values.
fn parse_bool_like(value: &str, file: &str, line: usize, key: &str) -> Result<bool, BuildError> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: `//#pass` `{key}` expects on/off, got `{value}`"
        ))),
    }
}

/// Validates that a directive property token can map to a reflected WGSL identifier.
fn validate_directive_property(
    property: &str,
    file: &str,
    line: usize,
    directive: &str,
) -> Result<(), BuildError> {
    if property.is_empty()
        || property
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || c == '_'))
    {
        return Err(BuildError::Message(format!(
            "{file}:{line}: `//#{directive}` property must be a WGSL-compatible identifier, got `{property}`"
        )));
    }
    Ok(())
}

/// Parses texture fallback directives from WGSL source.
pub(super) fn parse_texture_default_directives(
    source: &str,
    file: &str,
) -> Result<Vec<TextureDefaultDirective>, BuildError> {
    let mut defaults = Vec::new();
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#texture_default") else {
            continue;
        };
        let mut tokens = rest.split_whitespace();
        let Some(property) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#texture_default` requires a texture property name"
            )));
        };
        let Some(default_token) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#texture_default` requires a Unity default token"
            )));
        };
        if tokens.next().is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#texture_default` accepts exactly two arguments"
            )));
        }
        validate_directive_property(property, file, line_no, "texture_default")?;
        if defaults
            .iter()
            .any(|d: &TextureDefaultDirective| d.property == property)
        {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: duplicate `//#texture_default` for `{property}`"
            )));
        }
        defaults.push(TextureDefaultDirective {
            property: property.to_string(),
            kind: TextureDefaultKind::parse(default_token, file, line_no)?,
        });
    }
    Ok(defaults)
}

/// Parses required wgpu feature directives from WGSL source.
pub(super) fn parse_wgpu_feature_directives(
    source: &str,
    file: &str,
) -> Result<Vec<WgpuFeatureDirective>, BuildError> {
    let mut features = Vec::new();
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#wgpu_feature") else {
            continue;
        };
        let mut tokens = rest.split_whitespace();
        let Some(feature_token) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#wgpu_feature` requires a feature token"
            )));
        };
        if tokens.next().is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#wgpu_feature` accepts exactly one argument"
            )));
        }
        let feature = BuildWgpuFeature::parse(feature_token, file, line_no)?;
        if features
            .iter()
            .any(|d: &WgpuFeatureDirective| d.feature == feature)
        {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: duplicate `//#wgpu_feature` for `{feature_token}`"
            )));
        }
        features.push(WgpuFeatureDirective { feature });
    }
    Ok(features)
}

/// Parses material uniform fallback directives from WGSL source.
pub(super) fn parse_material_default_directives(
    source: &str,
    file: &str,
) -> Result<Vec<MaterialDefaultDirective>, BuildError> {
    let mut defaults = Vec::new();
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#mat_default") else {
            continue;
        };
        let mut tokens = rest.split_whitespace();
        let Some(property) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#mat_default` requires a material property name"
            )));
        };
        let Some(default_kind) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#mat_default` requires a default kind (`float` or `vec4`)"
            )));
        };
        validate_directive_property(property, file, line_no, "mat_default")?;
        if defaults
            .iter()
            .any(|d: &MaterialDefaultDirective| d.property == property)
        {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: duplicate `//#mat_default` for `{property}`"
            )));
        }
        let value = parse_material_default_value(default_kind, tokens, file, line_no)?;
        defaults.push(MaterialDefaultDirective {
            property: property.to_string(),
            value,
        });
    }
    Ok(defaults)
}

/// Parses a typed material default payload.
fn parse_material_default_value<'a>(
    default_kind: &str,
    mut tokens: impl Iterator<Item = &'a str>,
    file: &str,
    line: usize,
) -> Result<MaterialDefaultValue, BuildError> {
    match default_kind.trim().to_ascii_lowercase().as_str() {
        "float" | "f32" => {
            let Some(value) = tokens.next() else {
                return Err(BuildError::Message(format!(
                    "{file}:{line}: `//#mat_default` float requires one f32 value"
                )));
            };
            if tokens.next().is_some() {
                return Err(BuildError::Message(format!(
                    "{file}:{line}: `//#mat_default` float accepts exactly one f32 value"
                )));
            }
            Ok(MaterialDefaultValue::float_bits(
                parse_mat_default_f32_value(value, file, line)?.to_bits(),
            ))
        }
        "vec4" | "float4" => {
            let mut values = [0u32; 4];
            for bits in &mut values {
                let Some(value) = tokens.next() else {
                    return Err(BuildError::Message(format!(
                        "{file}:{line}: `//#mat_default` vec4 requires four f32 values"
                    )));
                };
                *bits = parse_mat_default_f32_value(value, file, line)?.to_bits();
            }
            if tokens.next().is_some() {
                return Err(BuildError::Message(format!(
                    "{file}:{line}: `//#mat_default` vec4 accepts exactly four f32 values"
                )));
            }
            Ok(MaterialDefaultValue::vec4_bits(values))
        }
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: unknown `//#mat_default` kind `{default_kind}` (allowed: float, f32, vec4, float4)"
        ))),
    }
}

/// Parses a directive boolean value.
fn parse_bool_value(value: &str, file: &str, line: usize, key: &str) -> Result<bool, BuildError> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(BuildError::Message(format!(
            "{file}:{line}: `//#pass` override `{key}` expects a boolean value, got `{value}`"
        ))),
    }
}

/// Parses a finite `f32` pass override.
fn parse_f32_value(value: &str, file: &str, line: usize, key: &str) -> Result<f32, BuildError> {
    let parsed = value.parse::<f32>().map_err(|e| {
        BuildError::Message(format!(
            "{file}:{line}: `//#pass` override `{key}` expects a finite f32 value, got `{value}`: {e}"
        ))
    })?;
    if !parsed.is_finite() {
        return Err(BuildError::Message(format!(
            "{file}:{line}: `//#pass` override `{key}` expects a finite f32 value, got `{value}`"
        )));
    }
    Ok(parsed)
}

/// Parses a finite `f32` material default value.
fn parse_mat_default_f32_value(value: &str, file: &str, line: usize) -> Result<f32, BuildError> {
    let parsed = value.parse::<f32>().map_err(|e| {
        BuildError::Message(format!(
            "{file}:{line}: `//#mat_default` expects finite f32 values, got `{value}`: {e}"
        ))
    })?;
    if !parsed.is_finite() {
        return Err(BuildError::Message(format!(
            "{file}:{line}: `//#mat_default` expects finite f32 values, got `{value}`"
        )));
    }
    Ok(parsed)
}

/// Rounds and saturates Unity `Offset units` into wgpu's constant depth-bias integer.
fn unity_offset_units(v: f32) -> i32 {
    let rounded = v.round();
    if rounded >= i32::MAX as f32 {
        i32::MAX
    } else if rounded <= i32::MIN as f32 {
        i32::MIN
    } else {
        rounded as i32
    }
}

/// Converts Unity's positive-forward depth slope bias to reverse-Z without preserving negative zero.
fn reverse_z_offset_factor(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { -v }
}

/// Parses an optional `//#source_alias <stem>` directive from a thin shader wrapper.
pub(super) fn parse_source_alias(source: &str, file: &str) -> Result<Option<String>, BuildError> {
    let mut alias = None;
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#source_alias") else {
            continue;
        };
        if alias.is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: duplicate `//#source_alias` directive"
            )));
        }
        let mut tokens = rest.split_whitespace();
        let Some(stem) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#source_alias` requires a source file stem"
            )));
        };
        if tokens.next().is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#source_alias` accepts exactly one source file stem"
            )));
        }
        if stem.contains('/')
            || stem.contains('\\')
            || std::path::Path::new(stem)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wgsl"))
        {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#source_alias` must be a sibling WGSL file stem, got `{stem}`"
            )));
        }
        alias = Some(stem.to_string());
    }
    Ok(alias)
}

/// Renders a generated Rust expression for one pass directive.
pub(super) fn pass_literal(pass: &BuildPassDirective) -> String {
    let slope = f32::from_bits(pass.depth_bias_slope_scale_bits);
    format!(
        "crate::materials::MaterialPassDesc {{ name: {name:?}, pass_type: crate::materials::PassType::{pass_type}, vertex_entry: {vs:?}, fragment_entry: {fs:?}, depth_compare: {depth_compare}, depth_compare_domain: {depth_domain}, depth_write: {depth_write}, cull_mode: {cull_mode}, blend: {blend}, write_mask: {write_mask}, depth_bias_slope_scale: {slope:?}, depth_bias_constant: {depth_bias_constant}, alpha_to_coverage: {alpha_to_coverage}, material_state: {material_state}, render_state_policy: {policy} }}",
        name = pass.name.as_str(),
        pass_type = pass.pass_type.rust_variant(),
        vs = pass.vertex_entry.as_str(),
        fs = pass.fragment_entry.as_str(),
        depth_compare = pass.depth_compare.rust_literal(),
        depth_domain = pass.depth_compare_domain.rust_literal(),
        depth_write = pass.depth_write,
        cull_mode = pass.cull_mode.rust_literal(),
        blend = pass.blend.rust_literal(),
        write_mask = pass.write_mask.rust_literal(),
        depth_bias_constant = pass.depth_bias_constant,
        alpha_to_coverage = pass.alpha_to_coverage,
        material_state = pass.material_state.rust_literal(),
        policy = pass.render_state_policy.rust_literal(),
    )
}

/// Renders a generated Rust expression for one texture default directive.
pub(super) fn texture_default_literal(default: &TextureDefaultDirective) -> String {
    format!(
        "EmbeddedTextureDefault {{ property: {property:?}, kind: EmbeddedTextureDefaultKind::{kind} }}",
        property = default.property.as_str(),
        kind = default.kind.rust_variant()
    )
}

/// Renders a generated Rust expression for one material default directive.
pub(super) fn material_default_literal(default: &MaterialDefaultDirective) -> String {
    format!(
        "EmbeddedMaterialDefault {{ property: {property:?}, value: {value} }}",
        property = default.property.as_str(),
        value = default.value.rust_literal()
    )
}

/// Renders a generated Rust expression for required wgpu features.
pub(super) fn wgpu_features_literal(features: &[WgpuFeatureDirective]) -> String {
    if features.is_empty() {
        return "wgpu::Features::empty()".to_string();
    }
    features
        .iter()
        .map(|feature| feature.feature.rust_literal())
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source-alias wrappers carry exactly one sibling WGSL stem.
    #[test]
    fn source_alias_parses_sibling_stem() -> Result<(), BuildError> {
        let source = "//! wrapper\n//#source_alias blur\n";

        assert_eq!(
            parse_source_alias(source, "blur_perobject.wgsl")?.as_deref(),
            Some("blur")
        );
        Ok(())
    }

    /// Source-alias wrappers reject paths so build output stays deterministic and local.
    #[test]
    fn source_alias_rejects_paths() {
        let err = parse_source_alias("//#source_alias ../blur\n", "bad.wgsl")
            .expect_err("path aliases must be rejected");

        assert!(err.to_string().contains("sibling WGSL file stem"));
    }

    #[test]
    fn texture_default_directives_parse_supported_tokens() -> Result<(), BuildError> {
        let defaults = parse_texture_default_directives(
            r#"
//#texture_default _MainTex white
//#texture_default _EmissionMap black
//#texture_default _DetailAlbedoMap grey
//#texture_default _BumpMap bump
//#texture_default _NoiseTex empty
//#texture_default _MaskTex red
"#,
            "test.wgsl",
        )?;

        assert_eq!(
            defaults,
            [
                TextureDefaultDirective {
                    property: "_MainTex".to_string(),
                    kind: TextureDefaultKind::White,
                },
                TextureDefaultDirective {
                    property: "_EmissionMap".to_string(),
                    kind: TextureDefaultKind::Black,
                },
                TextureDefaultDirective {
                    property: "_DetailAlbedoMap".to_string(),
                    kind: TextureDefaultKind::Gray,
                },
                TextureDefaultDirective {
                    property: "_BumpMap".to_string(),
                    kind: TextureDefaultKind::Bump,
                },
                TextureDefaultDirective {
                    property: "_NoiseTex".to_string(),
                    kind: TextureDefaultKind::Empty,
                },
                TextureDefaultDirective {
                    property: "_MaskTex".to_string(),
                    kind: TextureDefaultKind::Red,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn texture_default_directives_reject_duplicates() {
        let err = parse_texture_default_directives(
            r#"
//#texture_default _MainTex white
//#texture_default _MainTex black
"#,
            "test.wgsl",
        )
        .expect_err("duplicate texture defaults must fail");

        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn texture_default_literal_emits_embedded_struct() {
        assert_eq!(
            texture_default_literal(&TextureDefaultDirective {
                property: "_MainTex".to_string(),
                kind: TextureDefaultKind::White,
            }),
            "EmbeddedTextureDefault { property: \"_MainTex\", kind: EmbeddedTextureDefaultKind::White }"
        );
    }

    #[test]
    fn material_default_directives_parse_float_and_vec4() -> Result<(), BuildError> {
        let defaults = parse_material_default_directives(
            r#"
//#mat_default _GlossMapScale float 1.0
//#mat_default _Tint vec4 0.25 0.5 0.75 1.0
"#,
            "test.wgsl",
        )?;

        assert_eq!(
            defaults,
            [
                MaterialDefaultDirective {
                    property: "_GlossMapScale".to_string(),
                    value: MaterialDefaultValue::float_bits(1.0f32.to_bits()),
                },
                MaterialDefaultDirective {
                    property: "_Tint".to_string(),
                    value: MaterialDefaultValue::vec4_bits([
                        0.25f32.to_bits(),
                        0.5f32.to_bits(),
                        0.75f32.to_bits(),
                        1.0f32.to_bits(),
                    ]),
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn material_default_directives_reject_duplicates() {
        let err = parse_material_default_directives(
            r#"
//#mat_default _GlossMapScale float 1.0
//#mat_default _GlossMapScale float 0.5
"#,
            "test.wgsl",
        )
        .expect_err("duplicate material defaults must fail");

        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn material_default_directives_reject_non_finite_values() {
        let err = parse_material_default_directives(
            "//#mat_default _GlossMapScale float NaN\n",
            "test.wgsl",
        )
        .expect_err("non-finite material defaults must fail");

        assert!(err.to_string().contains("finite f32"));
    }

    #[test]
    fn material_default_literal_emits_embedded_struct() {
        assert_eq!(
            material_default_literal(&MaterialDefaultDirective {
                property: "_GlossMapScale".to_string(),
                value: MaterialDefaultValue::float_bits(1.0f32.to_bits()),
            }),
            "EmbeddedMaterialDefault { property: \"_GlossMapScale\", value: EmbeddedMaterialDefaultValue::float(f32::from_bits(0x3f80_0000)) }"
        );
    }

    #[test]
    fn wgpu_feature_directives_parse_barycentrics() -> Result<(), BuildError> {
        let features =
            parse_wgpu_feature_directives("//#wgpu_feature shader_barycentrics\n", "test.wgsl")?;

        assert_eq!(
            features,
            [WgpuFeatureDirective {
                feature: BuildWgpuFeature::ShaderBarycentrics,
            }]
        );
        assert_eq!(
            wgpu_features_literal(&features),
            "wgpu::Features::SHADER_BARYCENTRICS"
        );
        Ok(())
    }

    #[test]
    fn wgpu_feature_directives_reject_duplicates() {
        let err = parse_wgpu_feature_directives(
            "//#wgpu_feature shader_barycentrics\n//#wgpu_feature shader_barycentrics\n",
            "test.wgsl",
        )
        .expect_err("duplicate feature directives must fail");

        assert!(err.to_string().contains("duplicate"));
    }

    /// Pass directives bind explicit metadata to the following fragment entry point.
    #[test]
    fn pass_directive_extracts_fragment_entry_and_state() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward name=outline vs=vs_outline blend=off cull=front zwrite=material(on) ztest=material_froox(main) offset=material(0,0)
@fragment
fn fs_outline() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "test.wgsl",
        )?;

        let pass = &passes[0];
        assert_eq!(pass.pass_type, BuildPassType::Forward);
        assert_eq!(pass.name, "outline");
        assert_eq!(pass.fragment_entry, "fs_outline");
        assert_eq!(pass.vertex_entry, "vs_outline");
        assert_eq!(pass.blend, BuildBlend::Off);
        assert_eq!(pass.cull_mode, BuildCullMode::Front);
        assert!(pass.render_state_policy.depth_write);
        assert!(!pass.render_state_policy.cull);
        Ok(())
    }

    /// Pass directives can opt into hardware alpha-to-coverage.
    #[test]
    fn pass_directive_extracts_alpha_to_coverage() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward a2c=true
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "test.wgsl",
        )?;

        assert!(passes[0].alpha_to_coverage);
        assert_eq!(passes[0].name, "forward");
        Ok(())
    }

    /// Pass directives can select Unity `CompareFunction` decoding for material `_ZTest`.
    #[test]
    fn pass_directive_extracts_ztest_domain() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward name=stencil ztest=material_unity(main)
@fragment
fn fs_stencil() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "stencil.wgsl",
        )?;

        assert_eq!(
            passes[0].depth_compare_domain,
            BuildDepthCompareDomain::UnityCompareFunction
        );
        assert!(passes[0].render_state_policy.depth_compare);
        assert!(pass_literal(&passes[0]).contains(
            "depth_compare_domain: crate::materials::MaterialDepthCompareDomain::UnityCompareFunction"
        ));
        Ok(())
    }

    /// Explicit metadata replaces former fixed-state cartesian pass aliases.
    #[test]
    fn pass_directive_parses_explicit_fixed_state_metadata() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward name=transparent_rgb blend=alpha zwrite=off ztest=main cull=off color_mask=rgb stencil=off offset=0,0
@fragment
fn fs_circle() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
//#pass type=forward name=volume_front blend=material_overlay zwrite=off ztest=always cull=front color_mask=rgba offset=0,0
@fragment
fn fs_volume() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "test.wgsl",
        )?;

        assert_eq!(passes[0].blend, BuildBlend::Alpha);
        assert_eq!(passes[0].write_mask, BuildColorWrites::Rgb);
        assert_eq!(passes[0].material_state, BuildMaterialPassState::Static);
        assert_eq!(
            passes[0].render_state_policy,
            BuildRenderStatePolicy {
                color_mask: false,
                depth_write: false,
                depth_compare: false,
                cull: false,
                stencil: false,
                depth_offset: false,
            }
        );
        assert_eq!(passes[1].blend, BuildBlend::Overlay);
        assert_eq!(passes[1].material_state, BuildMaterialPassState::Overlay);
        assert_eq!(passes[1].depth_compare, BuildDepthCompare::Always);
        assert!(!passes[1].render_state_policy.depth_compare);
        Ok(())
    }

    #[test]
    fn pass_directive_parses_static_additive_blend() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward blend=additive
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "additive.wgsl",
        )?;

        assert_eq!(passes[0].blend, BuildBlend::Additive);
        assert_eq!(passes[0].write_mask, BuildColorWrites::Rgba);
        assert!(pass_literal(&passes[0]).contains("PASS_BLEND_ONE_ONE"));
        Ok(())
    }

    /// Transparent material state carries premultiplied defaults and still allows material overrides.
    #[test]
    fn pass_directive_parses_transparent_material_state() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward name=forward_transparent blend=transparent_material zwrite=material(off) cull=material(off) color_mask=material(rgba)
@fragment
fn fs_transparent() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "transparent.wgsl",
        )?;

        let pass = &passes[0];
        assert_eq!(pass.blend, BuildBlend::Premultiplied);
        assert_eq!(pass.write_mask, BuildColorWrites::Rgba);
        assert!(!pass.depth_write);
        assert_eq!(
            pass.material_state,
            BuildMaterialPassState::TransparentForward
        );
        assert!(pass.render_state_policy.depth_write);
        assert!(pass.render_state_policy.cull);
        Ok(())
    }

    /// Old cartesian pass tokens are rejected instead of silently selecting presets.
    #[test]
    fn pass_directive_rejects_old_preset_token() {
        let err = parse_pass_directives(
            r#"
//#pass forward_alpha_blend
@fragment
fn fs_fur() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "furfx.wgsl",
        )
        .expect_err("old pass aliases must be rejected");

        assert!(err.to_string().contains("key=value metadata"));
    }

    /// Static Unity pass offsets are converted to reverse-Z wgpu depth-bias defaults.
    #[test]
    fn pass_directive_extracts_static_unity_offset() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward offset=2,2
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "null.wgsl",
        )?;

        assert_eq!(passes[0].depth_bias_slope_scale_bits, (-2.0f32).to_bits());
        assert_eq!(passes[0].depth_bias_constant, -2);
        assert!(!passes[0].render_state_policy.depth_offset);
        Ok(())
    }

    /// Zero Unity slope offset stays a canonical zero in generated pass literals.
    #[test]
    fn pass_directive_canonicalizes_zero_unity_offset_factor() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass type=forward offset=0,1
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "newunlitshader.wgsl",
        )?;

        assert_eq!(passes[0].depth_bias_slope_scale_bits, 0.0f32.to_bits());
        assert_eq!(passes[0].depth_bias_constant, -1);
        assert!(pass_literal(&passes[0]).contains("depth_bias_constant: -1"));
        Ok(())
    }
}
