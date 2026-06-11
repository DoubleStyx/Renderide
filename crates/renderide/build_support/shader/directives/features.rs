//! Required wgpu feature directive parsing and code generation.

use super::super::error::BuildError;

/// Build-side `wgpu::Features` selector declared by `//#wgpu_feature`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) enum BuildWgpuFeature {
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
}

/// One required wgpu feature directive attached to a WGSL source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct WgpuFeatureDirective {
    /// Required feature bit for the composed target.
    pub feature: BuildWgpuFeature,
}

impl WgpuFeatureDirective {
    /// Returns whether this directive requires fragment shader barycentric coordinates.
    #[cfg(test)]
    pub(in super::super) const fn requires_shader_barycentrics(self) -> bool {
        matches!(self.feature, BuildWgpuFeature::ShaderBarycentrics)
    }
}

/// Parses required wgpu feature directives from WGSL source.
pub(in super::super) fn parse_wgpu_feature_directives(
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(features[0].requires_shader_barycentrics());
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
}
