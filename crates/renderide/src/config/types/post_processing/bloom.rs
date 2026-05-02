//! Physically-based bloom configuration. Persisted as `[post_processing.bloom]`.

use serde::{Deserialize, Serialize};

use crate::config::value::{Clamped, power_of_two_floor};
use crate::labeled_enum;

/// Inclusive bounds for [`BloomSettings::max_mip_dimension`] before the value is rounded down to
/// a power of two. Exposed as a type alias so call sites can name the validated range.
pub type BloomMaxMipDimension =
    Clamped<{ BloomSettings::MIN_MIP_DIMENSION }, { BloomSettings::MAX_MIP_DIMENSION }>;

/// Physically-based bloom configuration.
///
/// Persisted as `[post_processing.bloom]`. Implements the Call of Duty: Advanced Warfare
/// dual-filter technique (13-tap downsample + 3x3 tent upsample) with Karis-average firefly
/// reduction on the first downsample. Runs **pre-tonemap** so it scatters HDR-linear light; the
/// tonemap pass then compresses the combined value. Defaults favor energy-conserving scatter
/// without thresholding, so dim HDR-linear contributions can still participate in the bloom pyramid.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BloomSettings {
    /// Whether bloom runs in the post-processing chain when post-processing is enabled.
    pub enabled: bool,
    /// Baseline scattering strength. Sane range roughly `[0.0, 1.0]`; lower values are subtler.
    /// An intensity of `0.0` gates the pass off even when [`Self::enabled`] is `true`.
    pub intensity: f32,
    /// Extra boost applied to low-frequency (coarse) mips. Valid range `[0.0, 1.0]`. Higher
    /// values produce a more diffused "glow" that spreads further across the image.
    pub low_frequency_boost: f32,
    /// Curvature of the low-frequency boost falloff. Valid range `[0.0, 1.0]`. Higher values
    /// concentrate the boost in the lowest-frequency mips.
    pub low_frequency_boost_curvature: f32,
    /// High-pass cut-off as a fraction of the mip range. `1.0` keeps every mip; smaller values
    /// drop the lowest-frequency (largest) mips entirely, which tightens the scatter radius.
    pub high_pass_frequency: f32,
    /// Soft-knee prefilter threshold applied to the first downsample (in HDR-linear units).
    /// `0.0` disables the prefilter -- physically-based bloom scatters all light, so leave this
    /// at 0 for the realistic path and raise it only for stylized looks.
    pub prefilter_threshold: f32,
    /// Softness of the prefilter knee. Valid range `[0.0, 1.0]`. `0.0` is a hard cutoff.
    pub prefilter_threshold_softness: f32,
    /// How the upsample chain composites into the next mip up (and into the scene).
    pub composite_mode: BloomCompositeMode,
    /// Target height (in pixels) of the largest bloom mip. Each subsequent mip halves the
    /// resolution; smaller values are faster but less wide-spread. Arbitrary values are accepted
    /// in config/UI, then clamped and rounded down to a power of two by
    /// [`Self::effective_max_mip_dimension`]. 512 matches Bevy's default.
    pub max_mip_dimension: u32,
}

impl BloomSettings {
    /// Smallest effective bloom mip-0 edge exposed by renderer settings.
    pub const MIN_MIP_DIMENSION: u32 = 64;

    /// Largest effective bloom mip-0 edge exposed by renderer settings.
    pub const MAX_MIP_DIMENSION: u32 = 2048;

    /// Returns the graph-facing mip-0 edge after clamping and rounding down to a power of two.
    ///
    /// The raw [`Self::max_mip_dimension`] remains a continuous integer for configuration and
    /// HUD editing, but the bloom pyramid is built only from power-of-two dimensions so every
    /// downsample rung remains stable. Built on top of [`crate::config::value::Clamped`] and
    /// [`crate::config::value::power_of_two_floor`] so the algebra is shared with any future
    /// pyramid-style setting.
    pub fn effective_max_mip_dimension(self) -> u32 {
        power_of_two_floor(BloomMaxMipDimension::new(self.max_mip_dimension).get())
    }
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: 0.5,
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 1.0,
            high_pass_frequency: 1.0,
            prefilter_threshold: 0.0,
            prefilter_threshold_softness: 0.0,
            composite_mode: BloomCompositeMode::EnergyConserving,
            max_mip_dimension: 512,
        }
    }
}

labeled_enum! {
    /// Blend rule used when upsampling the bloom pyramid and compositing back onto the scene
    /// color.
    ///
    /// [`Self::EnergyConserving`] uses `out = src * c + dst * (1 - c)`, so total radiance is
    /// preserved -- the scattered light is removed from the base image. [`Self::Additive`] uses
    /// `out = src * c + dst`, which brightens the scene by adding the scattered contribution on
    /// top.
    pub enum BloomCompositeMode: "bloom composite mode" {
        default => EnergyConserving;

        /// Energy-conserving blend (physically-based). Default.
        EnergyConserving => {
            persist: "energy_conserving",
            label: "Energy-Conserving (physical)",
        },
        /// Additive blend (brightens the scene).
        Additive => {
            persist: "additive",
            label: "Additive (stylized)",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{BloomCompositeMode, BloomSettings};

    #[test]
    fn defaults_match_config_contract() {
        let settings = BloomSettings::default();

        assert!(settings.enabled);
        assert_eq!(settings.intensity, 0.5);
        assert_eq!(settings.low_frequency_boost, 0.0);
        assert_eq!(settings.low_frequency_boost_curvature, 1.0);
        assert_eq!(settings.high_pass_frequency, 1.0);
        assert_eq!(settings.prefilter_threshold, 0.0);
        assert_eq!(settings.prefilter_threshold_softness, 0.0);
        assert_eq!(
            settings.composite_mode,
            BloomCompositeMode::EnergyConserving
        );
        assert_eq!(settings.max_mip_dimension, 512);
        assert_eq!(settings.effective_max_mip_dimension(), 512);
    }

    #[test]
    fn composite_mode_default_is_energy_conserving() {
        assert_eq!(
            BloomCompositeMode::default(),
            BloomCompositeMode::EnergyConserving
        );
    }

    #[test]
    fn effective_max_mip_dimension_clamps_and_rounds_down_to_power_of_two() {
        for (raw, effective) in [
            (0, 64),
            (1, 64),
            (63, 64),
            (64, 64),
            (65, 64),
            (127, 64),
            (128, 128),
            (511, 256),
            (512, 512),
            (1023, 512),
            (1024, 1024),
            (2048, 2048),
            (2049, 2048),
            (u32::MAX, 2048),
        ] {
            let settings = BloomSettings {
                max_mip_dimension: raw,
                ..BloomSettings::default()
            };
            assert_eq!(
                settings.effective_max_mip_dimension(),
                effective,
                "raw max_mip_dimension {raw} should resolve to {effective}"
            );
        }
    }
}
