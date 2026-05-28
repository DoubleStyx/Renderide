//! Exponential moving averages for the frame-timing HUD's scalar readouts.
//!
//! The HUD's frametime graph keeps **raw** samples so spikes remain visible, but the numeric
//! Frame / CPU / GPU / Host readouts are run through an EMA so they stop jittering frame-to-frame
//! on steady scenes.
//!
//! The EMA is intentionally simple: `value <- value + alpha * (sample - value)`. A fixed
//! `alpha = 0.25` keeps the display responsive while still damping small one-frame oscillations.

/// Fixed EMA smoothing factor for live HUD readouts.
pub const DISPLAY_EMA_ALPHA: f64 = 0.25;

/// Single-channel exponential moving average tracker.
///
/// The first sample seeds the EMA exactly so the displayed value starts from real data instead
/// of converging in from zero. Subsequent samples blend in with a fixed `alpha`.
#[derive(Clone, Copy, Debug)]
pub struct DisplayEmaScalar {
    /// Accumulated EMA value. [`None`] before the first sample.
    value: Option<f64>,
    /// Smoothing factor clamped to the inclusive `[0, 1]` range.
    alpha: f64,
}

impl DisplayEmaScalar {
    /// Creates a tracker with the supplied smoothing factor.
    pub fn new(alpha: f64) -> Self {
        let alpha = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            DISPLAY_EMA_ALPHA
        };
        Self { value: None, alpha }
    }

    /// Folds `sample` into the EMA and returns the new value.
    pub fn update(&mut self, sample: f64) -> f64 {
        let next = match self.value {
            Some(prev) => prev + self.alpha * (sample - prev),
            None => sample,
        };
        self.value = Some(next);
        next
    }

    /// Current EMA value, if any sample has been folded in yet.
    #[cfg(test)]
    pub fn current(&self) -> Option<f64> {
        self.value
    }

    /// Forgets prior samples so the next [`Self::update`] re-seeds the EMA.
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.value = None;
    }
}

impl Default for DisplayEmaScalar {
    fn default() -> Self {
        Self::new(DISPLAY_EMA_ALPHA)
    }
}

/// EMA bundle for frame-timing scalars displayed in the HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimingEma {
    /// EMA of `wall_frame_time_ms`.
    pub frame: DisplayEmaScalar,
    /// EMA of `cpu_frame_ms` (main-thread tick duration).
    pub cpu: DisplayEmaScalar,
    /// EMA of `gpu_frame_ms` (real timestamp readback only).
    pub gpu: DisplayEmaScalar,
    /// EMA of renderer-observed host lockstep turnaround.
    pub host: DisplayEmaScalar,
}

#[cfg(test)]
mod tests {
    use super::{DISPLAY_EMA_ALPHA, DisplayEmaScalar, FrameTimingEma};

    #[test]
    fn first_sample_seeds_exactly() {
        let mut e = DisplayEmaScalar::default();
        assert_eq!(e.update(7.5), 7.5);
        assert_eq!(e.current(), Some(7.5));
    }

    #[test]
    fn constant_input_converges_to_input() {
        let mut e = DisplayEmaScalar::default();
        for _ in 0..200 {
            e.update(16.0);
        }
        let v = e.current().expect("ema");
        assert!((v - 16.0).abs() < 1e-9, "ema={v}");
    }

    #[test]
    fn step_change_is_snappy_but_dampened() {
        let mut e = DisplayEmaScalar::default();
        e.update(10.0);
        let v = e.update(100.0);
        assert!((v - 32.5).abs() < 1e-9, "ema after step was {v}");
    }

    #[test]
    fn reset_reseeds_on_next_update() {
        let mut e = DisplayEmaScalar::default();
        e.update(5.0);
        e.update(5.0);
        e.reset();
        assert_eq!(e.update(99.0), 99.0);
    }

    #[test]
    fn non_finite_alpha_uses_display_default() {
        let mut e = DisplayEmaScalar::new(f64::NAN);
        e.update(10.0);
        assert_eq!(e.update(20.0), 10.0 + DISPLAY_EMA_ALPHA * 10.0);
    }

    #[test]
    fn default_frame_timing_ema_uses_display_alpha_for_all_lanes() {
        let mut ema = FrameTimingEma::default();
        ema.frame.update(10.0);
        ema.cpu.update(10.0);
        ema.gpu.update(10.0);
        ema.host.update(10.0);

        assert_eq!(ema.frame.update(20.0), 10.0 + DISPLAY_EMA_ALPHA * 10.0);
        assert_eq!(ema.cpu.update(20.0), 10.0 + DISPLAY_EMA_ALPHA * 10.0);
        assert_eq!(ema.gpu.update(20.0), 10.0 + DISPLAY_EMA_ALPHA * 10.0);
        assert_eq!(ema.host.update(20.0), 10.0 + DISPLAY_EMA_ALPHA * 10.0);
    }
}
