//! Motion blur configuration. Persisted as `[post_processing.motion_blur]`.

use serde::{Deserialize, Serialize};

/// Post-process motion blur configuration.
///
/// Persisted as `[post_processing.motion_blur]`. Motion blur runs on HDR scene color after bloom
/// and before tonemapping. The renderer derives screen-space velocity only while this effect is
/// active, so disabling it removes the velocity pass and blur resolve from the graph.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MotionBlurSettings {
    /// Whether motion blur runs in the post-processing chain when post-processing is enabled.
    pub enabled: bool,
    /// Whether stereo multiview / VR views may run motion blur.
    pub allow_vr: bool,
    /// Shutter opening as a fraction of the frame interval. `0.0` gates the effect off.
    pub shutter_angle: f32,
    /// Number of samples taken along the velocity vector. `0` gates the effect off.
    pub sample_count: u32,
    /// Maximum blur radius in pixels after shutter scaling.
    pub max_velocity_pixels: f32,
}

impl MotionBlurSettings {
    /// Largest sample count exposed to the shader loop.
    pub const MAX_SAMPLE_COUNT: u32 = 16;

    /// Returns `true` when the effect has enough non-zero settings to participate in the graph.
    pub fn is_effectively_enabled(self) -> bool {
        self.enabled
            && self.sample_count > 0
            && self.effective_shutter_angle() > 0.0
            && self.effective_max_velocity_pixels() > 0.0
    }

    /// Returns the sample count clamped to the shader's loop bound.
    pub fn effective_sample_count(self) -> u32 {
        self.sample_count.min(Self::MAX_SAMPLE_COUNT)
    }

    /// Returns the shutter scale clamped to a practical finite range.
    pub fn effective_shutter_angle(self) -> f32 {
        if self.shutter_angle.is_finite() {
            self.shutter_angle.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Returns the maximum blur radius clamped to a practical finite range.
    pub fn effective_max_velocity_pixels(self) -> f32 {
        if self.max_velocity_pixels.is_finite() {
            self.max_velocity_pixels.clamp(0.0, 512.0)
        } else {
            0.0
        }
    }
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_vr: false,
            shutter_angle: 0.5,
            sample_count: 16,
            max_velocity_pixels: 256.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MotionBlurSettings;

    #[test]
    fn defaults_use_lower_runtime_blur_budget() {
        let settings = MotionBlurSettings::default();

        assert_eq!(settings.shutter_angle, 0.35);
        assert_eq!(settings.effective_shutter_angle(), 0.35);
        assert_eq!(settings.sample_count, 8);
        assert_eq!(settings.effective_sample_count(), 8);
        assert_eq!(settings.max_velocity_pixels, 256.0);
        assert_eq!(settings.effective_max_velocity_pixels(), 256.0);
    }
}
