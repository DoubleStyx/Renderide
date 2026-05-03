//! WGSL source directive parsing.

use super::error::BuildError;

/// Material pass kind declared by `//#pass`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuildPassKind {
    /// Main forward material pass.
    Forward,
    /// Main forward material pass with authored two-sided culling.
    ForwardTwoSided,
    /// Transparent forward material pass.
    ForwardTransparent,
    /// Transparent forward material pass with fixed front-face culling.
    ForwardTransparentCullFront,
    /// Transparent forward material pass with fixed back-face culling.
    ForwardTransparentCullBack,
    /// Static transparent RGB-only material pass.
    TransparentRgb,
    /// Outline shell pass.
    Outline,
    /// Stencil-only pass.
    Stencil,
    /// Depth-only prepass.
    DepthPrepass,
    /// Fixed always-on-top alpha overlay pass.
    OverlayAlways,
    /// Overlay front pass.
    OverlayFront,
    /// Overlay-behind pass.
    OverlayBehind,
}

impl BuildPassKind {
    /// Converts a source token to a pass kind.
    fn parse(value: &str, file: &str, line: usize) -> Result<Self, BuildError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "forward" => Ok(Self::Forward),
            "forward_two_sided" | "forwardtwosided" | "two_sided" | "twosided" => {
                Ok(Self::ForwardTwoSided)
            }
            "forward_transparent" | "transparent" => Ok(Self::ForwardTransparent),
            "forward_transparent_cull_front" | "transparent_cull_front" | "transparent_front" => {
                Ok(Self::ForwardTransparentCullFront)
            }
            "forward_transparent_cull_back" | "transparent_cull_back" | "transparent_back" => {
                Ok(Self::ForwardTransparentCullBack)
            }
            "transparent_rgb" | "transparentrgb" => Ok(Self::TransparentRgb),
            "outline" => Ok(Self::Outline),
            "stencil" => Ok(Self::Stencil),
            "depth_prepass" | "depthprepass" | "prepass" => Ok(Self::DepthPrepass),
            "overlay_always" | "overlayalways" | "always_overlay" | "overlay_alpha"
            | "overlayalpha" => Ok(Self::OverlayAlways),
            "overlay_front" | "overlayfront" | "front" => Ok(Self::OverlayFront),
            "overlay_behind" | "overlaybehind" | "behind" => Ok(Self::OverlayBehind),
            _ => Err(BuildError::Message(format!(
                "{file}:{line}: unknown `//#pass` kind `{value}`"
            ))),
        }
    }

    /// Rust `PassKind` variant name used in generated embedded metadata.
    const fn rust_variant(self) -> &'static str {
        match self {
            Self::Forward => "Forward",
            Self::ForwardTwoSided => "ForwardTwoSided",
            Self::ForwardTransparent => "ForwardTransparent",
            Self::ForwardTransparentCullFront => "ForwardTransparentCullFront",
            Self::ForwardTransparentCullBack => "ForwardTransparentCullBack",
            Self::TransparentRgb => "TransparentRgb",
            Self::Outline => "Outline",
            Self::Stencil => "Stencil",
            Self::DepthPrepass => "DepthPrepass",
            Self::OverlayAlways => "OverlayAlways",
            Self::OverlayFront => "OverlayFront",
            Self::OverlayBehind => "OverlayBehind",
        }
    }
}

/// One declared pass: the [`BuildPassKind`] tag and the fragment entry point it sits above.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BuildPassDirective {
    /// Material pass kind.
    pub kind: BuildPassKind,
    /// Fragment entry point name the `//#pass` tag sits above.
    pub fragment_entry: String,
    /// Vertex entry point for this pass. Defaults to `vs_main`; overridden via `vs=...`.
    pub vertex_entry: String,
    /// Whether `vertex_entry` was explicitly authored via `vs=` in the directive.
    pub vertex_entry_explicit: bool,
    /// Whether this pass enables hardware alpha-to-coverage.
    pub alpha_to_coverage: bool,
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
                "{file}:{line_no}: `//#pass` tag requires a kind (e.g. `//#pass forward`)"
            )));
        }
        let mut tokens = body.split_whitespace();
        let kind_value = tokens.next().unwrap_or("");
        let kind = BuildPassKind::parse(kind_value, file, line_no)?;
        let mut vertex_entry = "vs_main".to_string();
        let mut vertex_entry_explicit = false;
        let mut alpha_to_coverage = false;
        for token in tokens {
            let (key, value) = token.split_once('=').ok_or_else(|| {
                BuildError::Message(format!(
                    "{file}:{line_no}: expected `key=value` after kind in `//#pass`, got `{token}`"
                ))
            })?;
            match key.trim().to_ascii_lowercase().as_str() {
                "vs" | "vertex" => {
                    vertex_entry = value.trim().to_string();
                    vertex_entry_explicit = true;
                }
                "a2c" | "alpha_to_coverage" => {
                    alpha_to_coverage = parse_bool_value(value.trim(), file, line_no, key.trim())?;
                }
                _ => {
                    return Err(BuildError::Message(format!(
                        "{file}:{line_no}: unknown `//#pass` override `{key}` (allowed: `vs=`, `a2c=`)"
                    )));
                }
            }
        }
        let fragment_entry = next_fragment_entry_after(&lines, line_idx + 1, file, line_no)?;
        passes.push(BuildPassDirective {
            kind,
            fragment_entry,
            vertex_entry,
            vertex_entry_explicit,
            alpha_to_coverage,
        });
    }
    Ok(passes)
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
        if stem.contains('/') || stem.contains('\\') || stem.ends_with(".wgsl") {
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
    let kind = pass.kind.rust_variant();
    let base = format!(
        "crate::materials::pass_from_kind(crate::materials::PassKind::{kind}, {fs:?})",
        fs = pass.fragment_entry.as_str(),
    );
    match (
        pass.vertex_entry == "vs_main" && !pass.vertex_entry_explicit,
        pass.alpha_to_coverage,
    ) {
        (true, false) => base,
        (false, false) => format!(
            "crate::materials::MaterialPassDesc {{ vertex_entry: {vs:?}, ..{base} }}",
            vs = pass.vertex_entry.as_str(),
        ),
        (true, true) => {
            format!("crate::materials::MaterialPassDesc {{ alpha_to_coverage: true, ..{base} }}")
        }
        (false, true) => format!(
            "crate::materials::MaterialPassDesc {{ vertex_entry: {vs:?}, alpha_to_coverage: true, ..{base} }}",
            vs = pass.vertex_entry.as_str(),
        ),
    }
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

    /// Pass directives bind to the following fragment entry point.
    #[test]
    fn pass_directive_extracts_fragment_entry() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass outline vs=vs_outline
@fragment
fn fs_outline() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "test.wgsl",
        )?;

        assert_eq!(
            passes,
            [BuildPassDirective {
                kind: BuildPassKind::Outline,
                fragment_entry: "fs_outline".to_string(),
                vertex_entry: "vs_outline".to_string(),
                vertex_entry_explicit: true,
                alpha_to_coverage: false,
            }]
        );
        Ok(())
    }

    /// Pass directives can opt into hardware alpha-to-coverage.
    #[test]
    fn pass_directive_extracts_alpha_to_coverage() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass forward a2c=true
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "test.wgsl",
        )?;

        assert_eq!(
            passes,
            [BuildPassDirective {
                kind: BuildPassKind::Forward,
                fragment_entry: "fs_main".to_string(),
                vertex_entry: "vs_main".to_string(),
                vertex_entry_explicit: false,
                alpha_to_coverage: true,
            }]
        );
        Ok(())
    }

    /// Fixed-state Unity pass aliases parse to the generated pass-kind variants used at runtime.
    #[test]
    fn pass_directive_parses_fixed_state_kinds() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass forward_two_sided
@fragment
fn fs_depth_projection() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

//#pass transparent_rgb
@fragment
fn fs_circle() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "test.wgsl",
        )?;

        assert_eq!(
            passes,
            [
                BuildPassDirective {
                    kind: BuildPassKind::ForwardTwoSided,
                    fragment_entry: "fs_depth_projection".to_string(),
                    vertex_entry: "vs_main".to_string(),
                    vertex_entry_explicit: false,
                    alpha_to_coverage: false,
                },
                BuildPassDirective {
                    kind: BuildPassKind::TransparentRgb,
                    fragment_entry: "fs_circle".to_string(),
                    vertex_entry: "vs_main".to_string(),
                    vertex_entry_explicit: false,
                    alpha_to_coverage: false,
                },
            ]
        );
        Ok(())
    }

    /// The single-pass overlay shader needs a fixed `ZTest Always` alpha overlay pass.
    #[test]
    fn pass_directive_accepts_overlay_always() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass overlay_always
@fragment
fn fs_overlay() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "overlay.wgsl",
        )?;

        assert_eq!(passes[0].kind, BuildPassKind::OverlayAlways);
        assert_eq!(
            pass_literal(&passes[0]),
            "crate::materials::pass_from_kind(crate::materials::PassKind::OverlayAlways, \"fs_overlay\")"
        );
        Ok(())
    }

    /// Transparent pass directives map to the corresponding runtime pass variants.
    #[test]
    fn transparent_pass_directives_extract_fragment_entries() -> Result<(), BuildError> {
        let passes = parse_pass_directives(
            r#"
//#pass transparent
@fragment
fn fs_transparent() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
//#pass forward_transparent_cull_front
@fragment
fn fs_back_faces() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
//#pass transparent_cull_back
@fragment
fn fs_front_faces() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#,
            "transparent.wgsl",
        )?;

        assert_eq!(
            passes,
            [
                BuildPassDirective {
                    kind: BuildPassKind::ForwardTransparent,
                    fragment_entry: "fs_transparent".to_string(),
                    vertex_entry: "vs_main".to_string(),
                    vertex_entry_explicit: false,
                    alpha_to_coverage: false,
                },
                BuildPassDirective {
                    kind: BuildPassKind::ForwardTransparentCullFront,
                    fragment_entry: "fs_back_faces".to_string(),
                    vertex_entry: "vs_main".to_string(),
                    vertex_entry_explicit: false,
                    alpha_to_coverage: false,
                },
                BuildPassDirective {
                    kind: BuildPassKind::ForwardTransparentCullBack,
                    fragment_entry: "fs_front_faces".to_string(),
                    vertex_entry: "vs_main".to_string(),
                    vertex_entry_explicit: false,
                    alpha_to_coverage: false,
                },
            ]
        );
        assert_eq!(
            pass_literal(&passes[2]),
            "crate::materials::pass_from_kind(crate::materials::PassKind::ForwardTransparentCullBack, \"fs_front_faces\")"
        );
        Ok(())
    }
}
