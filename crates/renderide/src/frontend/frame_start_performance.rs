//! Builds optional [`crate::shared::PerformanceState`] for the **`FrameStartData.performance`** IPC
//! field (smoothed FPS and send throttling). This is **not** GPU instrumentation or engine profilers.
//!
//! See [`crate::shared::FrameStartData`] and [`RendererFrontend::pre_frame`](crate::frontend::RendererFrontend::pre_frame).

use std::time::{Duration, Instant};

use crate::shared::PerformanceState;

/// Minimum wall-clock spacing between `Some(PerformanceState)` payloads on `frame_start_data`.
pub(crate) const PERF_SEND_INTERVAL: Duration = Duration::from_secs(1);

/// Exponential moving average blend factor for the `fps` field (`fps` blends toward `immediate_fps`).
pub(crate) const FPS_EMA_ALPHA: f32 = 0.1;

/// Returns whether a new performance payload should be sent, given time since the last send.
pub(crate) fn perf_send_due(elapsed_since_last_send: Option<Duration>, interval: Duration) -> bool {
    elapsed_since_last_send.is_none_or(|d| d >= interval)
}

/// Updates the smoothed FPS estimate from the current instantaneous FPS.
pub(crate) fn next_smoothed_fps(prev: Option<f32>, instant_fps: f32, alpha: f32) -> f32 {
    match prev {
        None => instant_fps,
        Some(s) => alpha.mul_add(instant_fps, (1.0 - alpha) * s),
    }
}

/// Builds an optional [`PerformanceState`] for this frame, updating smoothing and throttle state.
///
/// When this returns [`None`] while `wall_interval_us > 0`, the EMA is still advanced so the next
/// send uses a current smoothed value.
///
/// **Host contract:** the engine only applies a new sample when `FrameStartData.performance` is
/// not null; in-game metrics continue to show the last published values until the next non-null
/// payload.
pub(crate) fn step_frame_performance(
    wall_interval_us: u64,
    last_frame_total_us: u64,
    smoothed_fps: &mut Option<f32>,
    last_perf_send: &mut Option<Instant>,
    now: Instant,
) -> Option<PerformanceState> {
    if wall_interval_us == 0 {
        return None;
    }
    let instant_fps = 1_000_000.0 / wall_interval_us as f32;
    let next_smoothed = next_smoothed_fps(*smoothed_fps, instant_fps, FPS_EMA_ALPHA);
    *smoothed_fps = Some(next_smoothed);

    let elapsed = last_perf_send.map(|t| now.duration_since(t));
    if !perf_send_due(elapsed, PERF_SEND_INTERVAL) {
        return None;
    }
    *last_perf_send = Some(now);
    Some(PerformanceState {
        fps: next_smoothed,
        immediate_fps: instant_fps,
        render_time: last_frame_total_us as f32 / 1_000_000.0,
        ..PerformanceState::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_smoothed_fps_starts_at_instant() {
        assert!((next_smoothed_fps(None, 60.0, 0.1) - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn next_smoothed_fps_ema_blends() {
        let blended = next_smoothed_fps(Some(60.0), 120.0, 0.1);
        assert!((blended - 66.0).abs() < 0.01);
    }

    #[test]
    fn perf_send_due_first_send() {
        assert!(perf_send_due(None, Duration::from_secs(1)));
    }

    #[test]
    fn perf_send_due_not_within_interval() {
        assert!(!perf_send_due(
            Some(Duration::from_millis(100)),
            Duration::from_secs(1)
        ));
    }

    #[test]
    fn perf_send_due_after_interval() {
        assert!(perf_send_due(
            Some(Duration::from_secs(1)),
            Duration::from_secs(1)
        ));
    }

    #[test]
    fn step_frame_performance_first_send_includes_smoothed_fps() {
        let mut smoothed = None;
        let mut last_send = None;
        let now = Instant::now();
        let p = step_frame_performance(16_666, 5_000, &mut smoothed, &mut last_send, now);
        let p = p.expect("first send");
        assert!((p.immediate_fps - 60.0).abs() < 1.0);
        assert!(smoothed.is_some());
    }

    #[test]
    fn step_frame_performance_skips_send_when_same_instant() {
        let mut smoothed = None;
        let mut last_send = None;
        let now = Instant::now();
        let _ = step_frame_performance(16_666, 5_000, &mut smoothed, &mut last_send, now);
        let p2 = step_frame_performance(16_666, 5_000, &mut smoothed, &mut last_send, now);
        assert!(p2.is_none());
    }
}
