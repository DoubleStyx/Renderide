//! Material pass directive parsing and generated metadata rendering.

use super::super::error::BuildError;

/// Material pass role declared by `//#pass type=...`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) enum BuildPassType {
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
pub(in super::super) enum BuildDepthCompare {
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
pub(in super::super) enum BuildCullMode {
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
pub(in super::super) enum BuildColorWrites {
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
pub(in super::super) enum BuildBlend {
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
pub(in super::super) enum BuildMaterialPassState {
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
pub(in super::super) struct BuildRenderStatePolicy {
    /// Whether `_ColorMask` overrides the pass color write mask.
    pub(in super::super) color_mask: bool,
    /// Whether `_ZWrite` overrides the pass depth-write flag.
    pub(in super::super) depth_write: bool,
    /// Whether `_ZTest` overrides the pass depth compare function.
    pub(in super::super) depth_compare: bool,
    /// Whether `_Cull` overrides the pass cull mode.
    pub(in super::super) cull: bool,
    /// Whether `_Stencil*` properties override the pass stencil state.
    pub(in super::super) stencil: bool,
    /// Whether `_OffsetFactor` / `_OffsetUnits` override the pass depth bias.
    pub(in super::super) depth_offset: bool,
}

impl BuildRenderStatePolicy {
    /// Material properties may override every supported render-state field.
    pub(in super::super) const ALL_MATERIAL: Self = Self {
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
pub(in super::super) enum BuildDepthCompareDomain {
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
pub(in super::super) struct BuildPassDirective {
    /// Material pass type.
    pub(in super::super) pass_type: BuildPassType,
    /// Debug label for logs / pipeline names.
    pub(in super::super) name: String,
    /// Fragment entry point name the `//#pass` tag sits above.
    pub(in super::super) fragment_entry: String,
    /// Vertex entry point for this pass. Defaults to `vs_main`; overridden via `vs=...`.
    pub(in super::super) vertex_entry: String,
    /// Whether this pass enables hardware alpha-to-coverage.
    pub(in super::super) alpha_to_coverage: bool,
    /// `_ZTest` enum layout used when host material state overrides this pass.
    pub(in super::super) depth_compare_domain: BuildDepthCompareDomain,
    /// Depth comparison fallback.
    pub(in super::super) depth_compare: BuildDepthCompare,
    /// Whether this pass writes depth by default.
    pub(in super::super) depth_write: bool,
    /// Authored cull fallback.
    pub(in super::super) cull_mode: BuildCullMode,
    /// Authored blend fallback.
    pub(in super::super) blend: BuildBlend,
    /// Authored color write fallback.
    pub(in super::super) write_mask: BuildColorWrites,
    /// Static reverse-Z slope depth bias emitted from Unity `Offset factor`.
    pub(in super::super) depth_bias_slope_scale_bits: u32,
    /// Static reverse-Z constant depth bias emitted from Unity `Offset units`.
    pub(in super::super) depth_bias_constant: i32,
    /// Material blend-state materialization mode.
    pub(in super::super) material_state: BuildMaterialPassState,
    /// Per-field material render-state override policy.
    pub(in super::super) render_state_policy: BuildRenderStatePolicy,
}

impl BuildPassDirective {
    /// Returns whether pass state is eligible for the renderer's generic depth prepass.
    pub(in super::super) const fn is_generic_depth_prepass_candidate(&self) -> bool {
        matches!(self.pass_type, BuildPassType::Forward)
            && matches!(self.blend, BuildBlend::Off)
            && self.depth_write
            && !self.alpha_to_coverage
    }
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
pub(in super::super) fn parse_pass_directives(
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

/// Renders a generated Rust expression for one pass directive.
pub(in super::super) fn pass_literal(pass: &BuildPassDirective) -> String {
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

#[cfg(test)]
#[path = "passes/tests.rs"]
mod tests;
