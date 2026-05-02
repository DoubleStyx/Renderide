//! Exponential moving averages for the frame-timing HUD's scalar readouts.
//!
//! The HUD's frametime graph keeps **raw** samples so spikes remain visible, but the numeric
//! Frame / CPU / GPU readouts are run through an EMA so they stop jittering frame-to-frame on
//! steady scenes. Bevy's `FrameTimeDiagnosticsPlugin` uses the same approach with an `alpha`
//! derived from a virtual history length; see
//! `references_external/Bevy/crates/bevy_diagnostic/src/diagnostic.rs`.
//!
//! The EMA is intentionally simple: `value <- value + alpha * (sample - value)` with `alpha = 2
//! / (history_len + 1)`. A history length of `EMA_HISTORY_LEN = 20` matches the responsiveness
//! of typical MangoHud-style overlays.

/// Virtual history length used to derive the EMA smoothing factor.
///
/// Bevy uses the same derivation (`alpha = 2 / (N + 1)`); `N = 20` settles a step input to ~95%
/// in ~60 frames at 60 fps, which feels responsive without being noisy.
pub const EMA_HISTORY_LEN: usize = 20;

/// Single-channel exponential moving average tracker.
///
/// The first sample seeds the EMA exactly so the displayed value starts from real data instead
/// of converging in from zero. Subsequent samples blend in with a fixed `alpha`.
#[derive(Clone, Copy, Debug)]
pub struct EmaScalar {
    /// Accumulated EMA value. [`None`] before the first sample.
    value: Option<f64>,
    /// Smoothing factor `2 / (history_len + 1)`; precomputed at construction.
    alpha: f64,
}

impl EmaScalar {
    /// Creates a tracker with the given virtual history length (clamped to >= 1).
    pub fn new(history_len: usize) -> Self {
        let history = history_len.max(1) as f64;
        Self {
            value: None,
            alpha: 2.0 / (history + 1.0),
        }
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
    pub fn current(&self) -> Option<f64> {
        self.value
    }

    /// Forgets prior samples so the next [`Self::update`] re-seeds the EMA.
    pub fn reset(&mut self) {
        self.value = None;
    }
}

impl Default for EmaScalar {
    fn default() -> Self {
        Self::new(EMA_HISTORY_LEN)
    }
}

/// EMA bundle for the three frame-timing scalars displayed in the HUD: wall-clock frame time,
/// main-thread CPU frame ms, and GPU frame ms.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimingEma {
    /// EMA of `wall_frame_time_ms`.
    pub frame: EmaScalar,
    /// EMA of `cpu_frame_ms` (main-thread tick duration).
    pub cpu: EmaScalar,
    /// EMA of `gpu_frame_ms` (real timestamp readback or callback-latency fallback).
    pub gpu: EmaScalar,
}

#[cfg(test)]
mod tests {
    use super::{EMA_HISTORY_LEN, EmaScalar};

    #[test]
    fn first_sample_seeds_exactly() {
        let mut e = EmaScalar::new(EMA_HISTORY_LEN);
        assert_eq!(e.update(7.5), 7.5);
        assert_eq!(e.current(), Some(7.5));
    }

    #[test]
    fn constant_input_converges_to_input() {
        let mut e = EmaScalar::new(EMA_HISTORY_LEN);
        for _ in 0..200 {
            e.update(16.0);
        }
        let v = e.current().expect("ema");
        assert!((v - 16.0).abs() < 1e-9, "ema={v}");
    }

    #[test]
    fn spike_is_dampened() {
        let mut e = EmaScalar::new(EMA_HISTORY_LEN);
        for _ in 0..100 {
            e.update(10.0);
        }
        // A single 100ms spike should not push the displayed value past ~20 -- EMA alpha is
        // 2/21 ~= 0.095, so the post-spike value is ~10 + 0.095 * (100 - 10) ~= 18.6.
        let v = e.update(100.0);
        assert!(v < 25.0, "ema after spike was {v}, expected dampened");
        assert!(
            v > 10.0,
            "ema after spike was {v}, expected at least slight rise"
        );
    }

    #[test]
    fn reset_reseeds_on_next_update() {
        let mut e = EmaScalar::new(EMA_HISTORY_LEN);
        e.update(5.0);
        e.update(5.0);
        e.reset();
        assert_eq!(e.update(99.0), 99.0);
    }
}
