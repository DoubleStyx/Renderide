//! Host [`ShaderUpload`](crate::shared::ShaderUpload) handling: AssetBundle shader-name extraction and material routing.

use std::fmt;

pub mod route;
pub mod unity_asset;

pub use route::{ResolvedShaderUpload, resolve_shader_upload};

/// Formatter for optional shader variant bitmasks in diagnostic logs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ShaderVariantBitsLog {
    /// Optional shader variant bitmask rendered for logs.
    bits: Option<u32>,
}

impl fmt::Display for ShaderVariantBitsLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.bits {
            Some(bits) => write!(f, "0x{bits:08X}"),
            None => f.write_str("none"),
        }
    }
}

/// Returns a log formatter for an optional shader variant bitmask.
pub(crate) fn shader_variant_bits_log(bits: Option<u32>) -> ShaderVariantBitsLog {
    ShaderVariantBitsLog { bits }
}

/// Tests for shader variant bitmask log formatting.
#[cfg(test)]
mod tests {
    use super::shader_variant_bits_log;

    /// Verifies that present bitmasks use eight-digit uppercase hexadecimal.
    #[test]
    fn shader_variant_bits_log_formats_some_as_padded_hex() {
        assert_eq!(
            shader_variant_bits_log(Some(0xB1)).to_string(),
            "0x000000B1"
        );
    }

    /// Verifies that missing bitmasks use a compact sentinel string.
    #[test]
    fn shader_variant_bits_log_formats_none_as_none() {
        assert_eq!(shader_variant_bits_log(None).to_string(), "none");
    }
}
