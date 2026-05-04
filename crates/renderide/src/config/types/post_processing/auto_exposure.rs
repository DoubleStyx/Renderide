//! Auto-exposure configuration. Persisted as `[post_processing.auto_exposure]`.

use serde::{Deserialize, Serialize};

/// Auto-exposure configuration.
///
/// Persisted as `[post_processing.auto_exposure]`. The renderer builds a log-luminance histogram
/// from HDR scene color, filters dark and bright percentile tails, and adapts exposure in EV stops
/// toward middle gray before bloom and tonemapping. Defaults match the Bevy-style histogram path
/// and keep adaptation fast when brightening while darkening more conservatively.
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
    /// Adaptation speed for transitions from dark scenes to bright scenes, in EV stops per second.
    pub speed_brighten: f32,
    /// Adaptation speed for transitions from bright scenes to dark scenes, in EV stops per second.
    pub speed_darken: f32,
    /// EV distance where adaptation transitions from linear to exponential.
    pub exponential_transition_distance: f32,
    /// Manual EV compensation added after metering.
    pub compensation_ev: f32,
}

impl AutoExposureSettings {
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

    /// Returns a finite non-negative dark-to-bright scene adaptation speed.
    pub fn resolved_speed_brighten(self) -> f32 {
        finite_or(self.speed_brighten, Self::default().speed_brighten).max(0.0)
    }

    /// Returns a finite non-negative bright-to-dark scene adaptation speed.
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
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            min_ev: -16.0,
            max_ev: 16.0,
            low_percent: 0.10,
            high_percent: 0.90,
            speed_brighten: 3.0,
            speed_darken: 3.0,
            exponential_transition_distance: 1.5,
            compensation_ev: -3.0,
        }
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

#[cfg(test)]
mod tests {
    use super::AutoExposureSettings;

    #[test]
    fn defaults_match_config_contract() {
        let settings = AutoExposureSettings::default();

        assert!(settings.enabled);
        assert_eq!(settings.min_ev, -16.0);
        assert_eq!(settings.max_ev, 16.0);
        assert_eq!(settings.low_percent, 0.10);
        assert_eq!(settings.high_percent, 0.90);
        assert_eq!(settings.speed_brighten, 3.0);
        assert_eq!(settings.speed_darken, 3.0);
        assert_eq!(settings.exponential_transition_distance, 1.5);
        assert_eq!(settings.compensation_ev, -3.0);
    }

    #[test]
    fn resolved_ev_range_orders_and_expands_degenerate_ranges() {
        let settings = AutoExposureSettings {
            min_ev: 4.0,
            max_ev: 4.0,
            ..Default::default()
        };

        let (min_ev, max_ev) = settings.resolved_ev_range();

        assert_eq!(min_ev, 4.0);
        assert!(max_ev > min_ev);

        let settings = AutoExposureSettings {
            min_ev: 8.0,
            max_ev: -8.0,
            ..Default::default()
        };
        assert_eq!(settings.resolved_ev_range(), (-8.0, 8.0));
    }

    #[test]
    fn resolved_filter_clamps_and_orders_percentiles() {
        let settings = AutoExposureSettings {
            low_percent: 1.2,
            high_percent: -0.3,
            ..Default::default()
        };

        assert_eq!(settings.resolved_filter(), (0.0, 1.0));
    }
}
