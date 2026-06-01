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
        if self.is_enabled() && override_names_for_wgsl_source(wgsl_source).is_some() {
            self
        } else {
            Self::disabled()
        }
    }

    /// Builds the `wgpu` pipeline-constant slice for this key and composed WGSL source.
    pub(crate) fn pipeline_constants_for_wgsl_source<'a>(
        self,
        wgsl_source: &'a str,
    ) -> MaterialShaderSpecializationConstants<'a> {
        if self.is_enabled()
            && let Some(names) = override_names_for_wgsl_source(wgsl_source)
        {
            MaterialShaderSpecializationConstants {
                entries: [
                    (names.mode, f64::from(self.static_variant_bits_mode)),
                    (names.bits, f64::from(self.static_variant_bits)),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaterialShaderSpecializationOverrideNames<'a> {
    mode: &'a str,
    bits: &'a str,
}

fn override_names_for_wgsl_source(
    wgsl_source: &str,
) -> Option<MaterialShaderSpecializationOverrideNames<'_>> {
    Some(MaterialShaderSpecializationOverrideNames {
        mode: override_name_for_wgsl_source(wgsl_source, STATIC_VARIANT_BITS_MODE_OVERRIDE)?,
        bits: override_name_for_wgsl_source(wgsl_source, STATIC_VARIANT_BITS_OVERRIDE)?,
    })
}

fn override_name_for_wgsl_source<'a>(wgsl_source: &'a str, base_name: &str) -> Option<&'a str> {
    wgsl_source
        .lines()
        .filter_map(|line| {
            line.trim_start()
                .strip_prefix("override ")
                .and_then(|tail| tail.split_once(':'))
                .map(|(name, _)| name.trim())
        })
        .find(|name| {
            *name == base_name
                || name
                    .strip_prefix(base_name)
                    .is_some_and(|suffix| suffix.starts_with("X_naga_oil_mod_"))
        })
}

/// Borrowable fixed storage for `wgpu::PipelineCompilationOptions::constants`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaterialShaderSpecializationConstants<'a> {
    entries: [(&'a str, f64); 2],
    len: usize,
}

impl MaterialShaderSpecializationConstants<'_> {
    /// Returns the active pipeline constants.
    pub(crate) fn as_slice(&self) -> &[(&str, f64)] {
        &self.entries[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::MaterialShaderSpecializationKey;

    #[test]
    fn disabled_key_emits_no_pipeline_constants() {
        let constants = MaterialShaderSpecializationKey::disabled()
            .pipeline_constants_for_wgsl_source(mangled_override_source());

        assert!(constants.as_slice().is_empty());
    }

    #[test]
    fn static_variant_bits_emit_pipeline_constants() {
        let constants = MaterialShaderSpecializationKey::from_variant_bits(0x1020)
            .pipeline_constants_for_wgsl_source(mangled_override_source());

        assert_eq!(
            constants.as_slice(),
            &[
                ("renderide_static_variant_bits_modeX_naga_oil_mod_TEST", 1.0),
                (
                    "renderide_static_variant_bitsX_naga_oil_mod_TEST",
                    0x1020 as f64
                ),
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
            .for_wgsl_source(mangled_override_source());

        assert_eq!(
            key,
            MaterialShaderSpecializationKey::from_variant_bits(0x40)
        );
    }

    #[test]
    fn static_bits_override_does_not_match_mode_override_prefix() {
        let constants = MaterialShaderSpecializationKey::from_variant_bits(0x40)
            .pipeline_constants_for_wgsl_source(
                "override renderide_static_variant_bits_modeX_naga_oil_mod_TEST: u32 = 0u;",
            );

        assert!(constants.as_slice().is_empty());
    }

    fn mangled_override_source() -> &'static str {
        "\
override renderide_static_variant_bits_modeX_naga_oil_mod_TEST: u32 = 0u;
override renderide_static_variant_bitsX_naga_oil_mod_TEST: u32 = 0u;
"
    }
}
