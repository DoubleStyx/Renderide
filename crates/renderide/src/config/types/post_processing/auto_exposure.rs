//! Auto-exposure configuration. Persisted as `[post_processing.auto_exposure]`.

use serde::{Deserialize, Serialize};

/// Auto-exposure configuration.
///
/// Persisted as `[post_processing.auto_exposure]`. The renderer builds a log-luminance histogram
/// from HDR scene color, filters dark and bright percentile tails, and adapts exposure in EV stops
/// toward scene-linear middle gray before bloom and tonemapping. Manual compensation is an offset
/// from the internal middle-gray target, so `0.0` is neutral.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoExposureSettings {
    /// Whether auto-exposure runs in the post-processing chain when post-processing is enabled.
    pub enabled: bool,
    /// Minimum log2 luminance EV included by the histogram.
    pub min_ev: f32,
    /// Maximum log2 luminance EV included by the histogram.
    pub max_ev: f32,
    /// Low percentile cut in `[0, 1]`; darker samples below this cumulative fraction are ignored.
    pub low_percent: f32,
    /// High percentile cut in `[0, 1]`; brighter samples above this cumulative fraction are ignored.
    pub high_percent: f32,
    /// Adaptation speed for positive exposure-EV changes that brighten the image.
    pub speed_brighten: f32,
    /// Adaptation speed for negative exposure-EV changes that darken the image.
    pub speed_darken: f32,
    /// EV distance where adaptation transitions from linear to exponential.
    pub exponential_transition_distance: f32,
    /// Manual EV compensation added after middle-gray metering.
    pub compensation_ev: f32,
}

impl AutoExposureSettings {
    /// EV offset for scene-linear `0.18` middle-gray luminance.
    pub const MIDDLE_GRAY_EV: f32 = -2.473_931_3;
    /// Minimum EV span accepted by the GPU pass.
    pub const MIN_EV_SPAN: f32 = 0.001;
    /// Smallest positive transition distance accepted by the GPU pass.
    pub const MIN_TRANSITION_DISTANCE: f32 = 0.001;

    /// Returns finite, ordered EV bounds with a non-zero span.
    pub fn resolved_ev_range(self) -> (f32, f32) {
        let defaults = Self::default();
        let mut min_ev = finite_or(self.min_ev, defaults.min_ev);
        let mut max_ev = finite_or(self.max_ev, defaults.max_ev);
        if min_ev > max_ev {
            std::mem::swap(&mut min_ev, &mut max_ev);
        }
        if max_ev - min_ev < Self::MIN_EV_SPAN {
            max_ev = min_ev + Self::MIN_EV_SPAN;
        }
        (min_ev, max_ev)
    }

    /// Returns finite, ordered percentile bounds clamped to `[0, 1]`.
    pub fn resolved_filter(self) -> (f32, f32) {
        let defaults = Self::default();
        let mut low = finite_or(self.low_percent, defaults.low_percent).clamp(0.0, 1.0);
        let mut high = finite_or(self.high_percent, defaults.high_percent).clamp(0.0, 1.0);
        if low > high {
            std::mem::swap(&mut low, &mut high);
        }
        (low, high)
    }

    /// Returns a finite non-negative image-brightening adaptation speed.
    pub fn resolved_speed_brighten(self) -> f32 {
        finite_or(self.speed_brighten, Self::default().speed_brighten).max(0.0)
    }

    /// Returns a finite non-negative image-darkening adaptation speed.
    pub fn resolved_speed_darken(self) -> f32 {
        finite_or(self.speed_darken, Self::default().speed_darken).max(0.0)
    }

    /// Returns a finite positive transition distance.
    pub fn resolved_exponential_transition_distance(self) -> f32 {
        finite_or(
            self.exponential_transition_distance,
            Self::default().exponential_transition_distance,
        )
        .max(Self::MIN_TRANSITION_DISTANCE)
    }

    /// Returns finite EV compensation.
    pub fn resolved_compensation_ev(self) -> f32 {
        finite_or(self.compensation_ev, Self::default().compensation_ev)
    }

    /// Returns the absolute target EV used by auto-exposure metering.
    pub fn resolved_target_ev(self) -> f32 {
        Self::MIDDLE_GRAY_EV + self.resolved_compensation_ev()
    }
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            min_ev: -16.0,
            max_ev: 16.0,
            low_percent: 0.25,
            high_percent: 0.75,
            speed_brighten: 3.0,
            speed_darken: 3.0,
            exponential_transition_distance: 1.5,
            compensation_ev: 0.0,
        }
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}
