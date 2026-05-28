//! Rolling frame-timing samples for the **Frame timing** HUD sparkline and 1-second percentiles.
//!
//! The history always records wall-frame samples for the sparkline. CPU/GPU samples are only
//! recorded when a new primary frame timing generation arrives, so delayed GPU readbacks do not
//! count the same completed frame multiple times.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Maximum retained samples used for the sparkline and rolling stats.
pub const FRAME_TIME_HISTORY_LEN: usize = 240;
/// Rolling window for AAA-style 1% low/high readouts.
pub const FRAME_TIMING_STATS_WINDOW: Duration = Duration::from_secs(1);

/// One raw HUD timing sample.
#[derive(Clone, Copy, Debug)]
pub struct FrameTimingHistorySample {
    /// Time when this sample was captured.
    pub captured_at: Instant,
    /// Wall-clock frame interval in milliseconds.
    pub wall_ms: f64,
    /// New primary timing generation, if a CPU/GPU pair is available.
    pub primary_generation: Option<u64>,
    /// Main-thread active CPU frame work in milliseconds.
    pub cpu_ms: Option<f64>,
    /// Timestamp-backed primary GPU busy time in milliseconds.
    pub gpu_ms: Option<f64>,
    /// Renderer-observed host lockstep turnaround in milliseconds.
    pub host_ms: Option<f64>,
}

/// 1% low/high pair for one HUD lane.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTimingOnePercentStats {
    /// Low-performance 1% value. For FPS this is low FPS; for ms lanes this is slow ms.
    pub low: Option<f64>,
    /// High-performance 1% value. For FPS this is high FPS; for ms lanes this is fast ms.
    pub high: Option<f64>,
}

/// Rolling 1-second stats for the compact frame timing HUD.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTimingHistoryStats {
    /// FPS 1% low/high derived from wall-frame samples.
    pub fps: FrameTimingOnePercentStats,
    /// Frame-ms 1% slow/fast derived from wall-frame samples.
    pub frame_ms: FrameTimingOnePercentStats,
    /// CPU-ms 1% slow/fast from new primary timing generations.
    pub cpu_ms: FrameTimingOnePercentStats,
    /// GPU-ms 1% slow/fast from timestamp-backed new primary timing generations.
    pub gpu_ms: FrameTimingOnePercentStats,
    /// Host-turnaround-ms 1% slow/fast from renderer-observed host submit turnaround.
    pub host_ms: FrameTimingOnePercentStats,
}

/// Rolling frame timing ring used by the HUD.
#[derive(Clone, Debug, Default)]
pub struct FrameTimeHistory {
    samples: VecDeque<FrameTimingHistorySample>,
    last_primary_generation: Option<u64>,
}

impl FrameTimeHistory {
    /// Empty history.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(FRAME_TIME_HISTORY_LEN),
            last_primary_generation: None,
        }
    }

    /// Appends a sample and evicts entries outside the rolling window/capacity.
    pub fn push(&mut self, mut sample: FrameTimingHistorySample) {
        match sample.primary_generation {
            Some(generation) if self.last_primary_generation != Some(generation) => {
                self.last_primary_generation = Some(generation);
            }
            _ => {
                sample.cpu_ms = None;
                sample.gpu_ms = None;
            }
        }

        self.samples.push_back(sample);
        self.evict_old(sample.captured_at);
        while self.samples.len() > FRAME_TIME_HISTORY_LEN {
            self.samples.pop_front();
        }
    }

    /// Clones wall-frame samples oldest-first for the sparkline plot.
    pub fn to_vec(&self) -> Vec<f32> {
        self.samples
            .iter()
            .map(|sample| sample.wall_ms as f32)
            .collect()
    }

    /// Computes rolling 1-second percentile stats.
    pub fn stats(&self) -> FrameTimingHistoryStats {
        let mut wall_ms = Vec::new();
        let mut fps = Vec::new();
        let mut cpu_ms = Vec::new();
        let mut gpu_ms = Vec::new();
        let mut host_ms = Vec::new();

        for sample in &self.samples {
            if let Some(v) = finite_non_negative(sample.wall_ms) {
                wall_ms.push(v);
                if v > f64::EPSILON {
                    fps.push(1000.0 / v);
                }
            }
            push_optional(&mut cpu_ms, sample.cpu_ms);
            push_optional(&mut gpu_ms, sample.gpu_ms);
            push_optional(&mut host_ms, sample.host_ms);
        }

        FrameTimingHistoryStats {
            fps: low_high_fps(fps),
            frame_ms: slow_fast_ms(wall_ms),
            cpu_ms: slow_fast_ms(cpu_ms),
            gpu_ms: slow_fast_ms(gpu_ms),
            host_ms: slow_fast_ms(host_ms),
        }
    }

    fn evict_old(&mut self, now: Instant) {
        while self.samples.front().is_some_and(|sample| {
            now.saturating_duration_since(sample.captured_at) > FRAME_TIMING_STATS_WINDOW
        }) {
            self.samples.pop_front();
        }
    }

    /// Number of stored samples.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` when no samples have been pushed yet.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

fn push_optional(out: &mut Vec<f64>, value: Option<f64>) {
    if let Some(v) = value.and_then(finite_non_negative) {
        out.push(v);
    }
}

fn finite_non_negative(value: f64) -> Option<f64> {
    (value.is_finite() && value >= 0.0).then_some(value)
}

fn low_high_fps(values: Vec<f64>) -> FrameTimingOnePercentStats {
    FrameTimingOnePercentStats {
        low: percentile(values.clone(), 0.01),
        high: percentile(values, 0.99),
    }
}

fn slow_fast_ms(values: Vec<f64>) -> FrameTimingOnePercentStats {
    FrameTimingOnePercentStats {
        low: percentile(values.clone(), 0.99),
        high: percentile(values, 0.01),
    }
}

fn percentile(mut values: Vec<f64>, p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let rank = (p.clamp(0.0, 1.0) * values.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(values.len() - 1);
    values.get(idx).copied()
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{FRAME_TIME_HISTORY_LEN, FrameTimeHistory, FrameTimingHistorySample};

    fn sample(at: Instant, wall_ms: f64, generation: u64) -> FrameTimingHistorySample {
        FrameTimingHistorySample {
            captured_at: at,
            wall_ms,
            primary_generation: Some(generation),
            cpu_ms: Some(wall_ms * 0.5),
            gpu_ms: Some(wall_ms * 0.25),
            host_ms: Some(wall_ms * 0.25),
        }
    }

    #[test]
    fn history_caps_at_configured_length() {
        let mut h = FrameTimeHistory::new();
        let start = Instant::now();
        assert!(h.is_empty());
        for i in 0..(FRAME_TIME_HISTORY_LEN + 10) {
            h.push(sample(
                start + Duration::from_millis(i as u64),
                i as f64,
                i as u64,
            ));
        }
        assert_eq!(h.len(), FRAME_TIME_HISTORY_LEN);
        let v = h.to_vec();
        assert_eq!(v.first().copied(), Some(10.0));
        assert_eq!(v.last().copied(), Some((FRAME_TIME_HISTORY_LEN + 9) as f32));
    }

    #[test]
    fn history_evicts_samples_outside_one_second_window() {
        let mut h = FrameTimeHistory::new();
        let start = Instant::now();
        h.push(sample(start, 16.0, 1));
        h.push(sample(start + Duration::from_millis(500), 17.0, 2));
        h.push(sample(start + Duration::from_millis(1100), 18.0, 3));

        assert_eq!(h.to_vec(), vec![17.0, 18.0]);
    }

    #[test]
    fn rolling_stats_drop_aged_spike() {
        let mut h = FrameTimeHistory::new();
        let start = Instant::now();
        h.push(sample(start, 100.0, 1));
        h.push(sample(start + Duration::from_millis(1100), 10.0, 2));

        let stats = h.stats();
        assert_eq!(stats.frame_ms.low, Some(10.0));
        assert_eq!(stats.frame_ms.high, Some(10.0));
        assert_eq!(stats.fps.low, Some(100.0));
        assert_eq!(stats.fps.high, Some(100.0));
    }

    #[test]
    fn primary_generation_deduplicates_cpu_gpu_stats() {
        let mut h = FrameTimeHistory::new();
        let start = Instant::now();
        h.push(sample(start, 10.0, 1));
        h.push(sample(start + Duration::from_millis(16), 20.0, 1));

        let stats = h.stats();
        assert_eq!(stats.cpu_ms.low, Some(5.0));
        assert_eq!(stats.gpu_ms.low, Some(2.5));
        assert_eq!(stats.frame_ms.low, Some(20.0));
    }
}
