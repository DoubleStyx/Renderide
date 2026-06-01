//! Renderer-local shader specialization selectors for material pipeline constants.

/// WGSL override that selects runtime uniform variant bits when zero and static bits otherwise.
pub(crate) const STATIC_VARIANT_BITS_MODE_OVERRIDE: &str = "renderide_static_variant_bits_mode";
/// WGSL override carrying the static shader-specific variant bitmask.
pub(crate) const STATIC_VARIANT_BITS_OVERRIDE: &str = "renderide_static_variant_bits";

const STATIC_VARIANT_BITS_DISABLED: u32 = 0;
const STATIC_VARIANT_BITS_ENABLED: u32 = 1;

/// Pipeline-cache selector for material shader specialization constants.
///
/// This is renderer-local state: it is derived from the already-routed host shader variant bits and
/// never changes the Renderite.Unity / Renderite.Shared wire contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialShaderSpecializationKey {
    static_variant_bits_mode: u32,
    static_variant_bits: u32,
}

impl MaterialShaderSpecializationKey {
    /// Returns a key that keeps shader variant decoding on the runtime material uniform.
    pub(crate) const fn disabled() -> Self {
        Self {
            static_variant_bits_mode: STATIC_VARIANT_BITS_DISABLED,
            static_variant_bits: 0,
        }
    }

    /// Returns a key that exposes the given variant bitmask as pipeline constants.
    pub(crate) const fn from_variant_bits(bits: u32) -> Self {
        Self {
            static_variant_bits_mode: STATIC_VARIANT_BITS_ENABLED,
            static_variant_bits: bits,
        }
    }

    /// Returns a static key when variant bits are known, or the disabled key otherwise.
    pub(crate) const fn from_optional_variant_bits(bits: Option<u32>) -> Self {
        match bits {
            Some(bits) => Self::from_variant_bits(bits),
            None => Self::disabled(),
        }
    }

    /// Returns whether static variant-bit specialization is enabled.
    pub(crate) const fn is_enabled(self) -> bool {
        self.static_variant_bits_mode == STATIC_VARIANT_BITS_ENABLED
    }

    /// Disables the key unless the actual WGSL source declares the specialization override.
    pub(crate) fn for_wgsl_source(self, wgsl_source: &str) -> Self {
        if self.is_enabled() && wgsl_source.contains(STATIC_VARIANT_BITS_OVERRIDE) {
            self
        } else {
            Self::disabled()
        }
    }

    /// Builds the `wgpu` pipeline-constant slice for this key.
    pub(crate) fn pipeline_constants(self) -> MaterialShaderSpecializationConstants {
        if self.is_enabled() {
            MaterialShaderSpecializationConstants {
                entries: [
                    (
                        STATIC_VARIANT_BITS_MODE_OVERRIDE,
                        f64::from(self.static_variant_bits_mode),
                    ),
                    (
                        STATIC_VARIANT_BITS_OVERRIDE,
                        f64::from(self.static_variant_bits),
                    ),
                ],
                len: 2,
            }
        } else {
            MaterialShaderSpecializationConstants {
                entries: [
                    (STATIC_VARIANT_BITS_MODE_OVERRIDE, 0.0),
                    (STATIC_VARIANT_BITS_OVERRIDE, 0.0),
                ],
                len: 0,
            }
        }
    }
}

/// Borrowable fixed storage for `wgpu::PipelineCompilationOptions::constants`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaterialShaderSpecializationConstants {
    entries: [(&'static str, f64); 2],
    len: usize,
}

impl MaterialShaderSpecializationConstants {
    /// Returns the active pipeline constants.
    pub(crate) fn as_slice(&self) -> &[(&'static str, f64)] {
        &self.entries[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MaterialShaderSpecializationKey, STATIC_VARIANT_BITS_MODE_OVERRIDE,
        STATIC_VARIANT_BITS_OVERRIDE,
    };

    #[test]
    fn disabled_key_emits_no_pipeline_constants() {
        let constants = MaterialShaderSpecializationKey::disabled().pipeline_constants();

        assert!(constants.as_slice().is_empty());
    }

    #[test]
    fn static_variant_bits_emit_pipeline_constants() {
        let constants =
            MaterialShaderSpecializationKey::from_variant_bits(0x1020).pipeline_constants();

        assert_eq!(
            constants.as_slice(),
            &[
                (STATIC_VARIANT_BITS_MODE_OVERRIDE, 1.0),
                (STATIC_VARIANT_BITS_OVERRIDE, 0x1020 as f64),
            ]
        );
    }

    #[test]
    fn source_without_override_disables_static_variant_bits() {
        let key = MaterialShaderSpecializationKey::from_variant_bits(0x40)
            .for_wgsl_source("fn fs_main() {}");

        assert_eq!(key, MaterialShaderSpecializationKey::disabled());
    }

    #[test]
    fn source_with_override_preserves_static_variant_bits() {
        let key = MaterialShaderSpecializationKey::from_variant_bits(0x40)
            .for_wgsl_source("override renderide_static_variant_bits: u32 = 0u;");

        assert_eq!(
            key,
            MaterialShaderSpecializationKey::from_variant_bits(0x40)
        );
    }
}
