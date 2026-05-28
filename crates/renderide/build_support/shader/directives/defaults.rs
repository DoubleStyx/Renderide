//! Material and texture default directive parsing and code generation.

use super::super::error::BuildError;

/// Texture fallback token declared by `//#texture_default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) enum TextureDefaultKind {
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
pub(in super::super) struct TextureDefaultDirective {
    /// Reflected host texture property name, e.g. `_MainTex`.
    pub property: String,
    /// Unity default token for the texture slot.
    pub kind: TextureDefaultKind,
}

/// Material uniform fallback value kind declared by `//#mat_default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) enum MaterialDefaultKind {
    /// Unity float property default.
    Float,
    /// Unity vector/color property default.
    Vec4,
}

/// Material uniform fallback value declared by `//#mat_default`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct MaterialDefaultValue {
    /// Unity property default kind.
    pub kind: MaterialDefaultKind,
    /// Unity property default bits. Float defaults use only the first element.
    pub bits: [u32; 4],
}

impl MaterialDefaultValue {
    /// Creates a float material default from raw `f32` bits.
    pub(in super::super) const fn float_bits(bits: u32) -> Self {
        Self {
            kind: MaterialDefaultKind::Float,
            bits: [bits, 0, 0, 0],
        }
    }

    /// Creates a vec4 material default from raw `f32` bits.
    pub(in super::super) const fn vec4_bits(bits: [u32; 4]) -> Self {
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
pub(in super::super) struct MaterialDefaultDirective {
    /// Reflected host material property name, e.g. `_GlossMapScale`.
    pub property: String,
    /// Unity property default value for the uniform field.
    pub value: MaterialDefaultValue,
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
pub(in super::super) fn parse_texture_default_directives(
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

/// Parses material uniform fallback directives from WGSL source.
pub(in super::super) fn parse_material_default_directives(
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

/// Renders a generated Rust expression for one texture default directive.
pub(in super::super) fn texture_default_literal(default: &TextureDefaultDirective) -> String {
    format!(
        "EmbeddedTextureDefault {{ property: {property:?}, kind: EmbeddedTextureDefaultKind::{kind} }}",
        property = default.property.as_str(),
        kind = default.kind.rust_variant()
    )
}

/// Renders a generated Rust expression for one material default directive.
pub(in super::super) fn material_default_literal(default: &MaterialDefaultDirective) -> String {
    format!(
        "EmbeddedMaterialDefault {{ property: {property:?}, value: {value} }}",
        property = default.property.as_str(),
        value = default.value.rust_literal()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
