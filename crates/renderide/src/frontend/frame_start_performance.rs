//! Builds the [`crate::shared::PerformanceState`] payload carried on every
//! [`crate::shared::FrameStartData`] sent to the host.
//!
//! Contract (matches Renderite.Unity, consumed by `FrooxEngine.PerformanceMetrics`):
//! - `immediate_fps` -- instantaneous, derived from the current tick's wall-clock interval
//!   ([`crate::frontend::RendererFrontend::on_tick_frame_wall_clock`]). No smoothing.
//! - `fps` -- count-based rolling average over a [`FPS_WINDOW`] window: `frame_count /
//!   elapsed_seconds` recomputed once each time the window closes, otherwise the previously
//!   computed value is carried forward unchanged. Mirrors `PerformanceStats.Update` in the
//!   Renderite.Unity reference. Stable for ~[`FPS_WINDOW`] at a time so the host-side
//!   `Sync<float> FPS.Value` change events fire at the window cadence rather than every frame.
//! - `render_time` -- most recently completed GPU submit->idle wall-clock duration in seconds
//!   ([`crate::gpu::GpuContext::last_completed_gpu_render_time_seconds`]); excludes the post-submit
//!   present/vsync block. Reports `-1.0` when no GPU completion callback has fired yet, mirroring the
//!   Renderite.Unity `XRStats.TryGetGPUTimeLastFrame` sentinel.
//! - `rendered_frames_since_last` -- number of completed renderer ticks since the previous
//!   `FrameStartData` send. `1` in lockstep, `> 1` when the renderer ticked multiple times per
//!   host submit (i.e. host is slow and the renderer kept rendering). Drives
//!   `FrooxEngine.PerformanceStats.RenderedFramesSinceLastTick`.
//!
//! A new [`PerformanceState`] is built on every tick where `wall_interval_us > 0` (i.e. starting
//! from the second tick); the host treats a non-null `FrameStartData.performance` as the latest
//! sample, so emitting every frame keeps `immediate_fps` and `render_time` in lock-step with the
//! actual frame loop while the windowed `fps` value stays stable across each window. This is
//! **not** GPU instrumentation; for that, see [`crate::gpu::frame_cpu_gpu_timing`].

use std::time::{Duration, Instant};

use crate::shared::PerformanceState;

/// Window length for the count-based `fps` rolling average. Matches the Renderite.Unity
/// `>= 500` ms threshold inside `PerformanceStats.Update`.
pub(crate) const FPS_WINDOW: Duration = Duration::from_millis(500);

/// Sentinel reported in `render_time` until the first GPU completion callback has fired, matching
/// the Renderite.Unity behavior of `state.renderTime = -1` when `XRStats.TryGetGPUTimeLastFrame`
/// has no sample yet.
pub(crate) const RENDER_TIME_UNAVAILABLE: f32 = -1.0;

/// Mutable performance accumulator that feeds outgoing frame-start payloads.
pub(crate) struct FrameStartPerformanceState {
    last_tick_wall_start: Option<Instant>,
    wall_interval_us_for_perf: u64,
    last_render_time_seconds: f32,
    framerate_window_start: Option<Instant>,
    framerate_counter: u32,
    last_window_fps: f32,
    rendered_frames_since_last: i32,
}

impl Default for FrameStartPerformanceState {
    fn default() -> Self {
        Self {
            last_tick_wall_start: None,
            wall_interval_us_for_perf: 0,
            last_render_time_seconds: RENDER_TIME_UNAVAILABLE,
            framerate_window_start: None,
            framerate_counter: 0,
            last_window_fps: 0.0,
            rendered_frames_since_last: 0,
        }
    }
}

impl FrameStartPerformanceState {
    /// Records wall-clock spacing between app-driver frame ticks and advances the count-based
    /// FPS window.
    ///
    /// Mirrors `PerformanceStats.Update`: the first call starts the window without counting,
    /// subsequent calls increment a frame counter, and once [`FPS_WINDOW`] has elapsed the
    /// window emits `frames / elapsed_seconds` into `last_window_fps` and re-bases off `now`.
    pub(crate) fn on_tick_frame_wall_clock(&mut self, now: Instant) {
        self.wall_interval_us_for_perf = self
            .last_tick_wall_start
            .map_or(0, |t| now.duration_since(t).as_micros() as u64);
        self.last_tick_wall_start = Some(now);

        match self.framerate_window_start {
            None => {
                self.framerate_window_start = Some(now);
                self.framerate_counter = 0;
            }
            Some(start) => {
                self.framerate_counter = self.framerate_counter.saturating_add(1);
                let elapsed = now.duration_since(start);
                if elapsed >= FPS_WINDOW {
                    let elapsed_secs = elapsed.as_secs_f32();
                    if elapsed_secs > 0.0 {
                        self.last_window_fps = self.framerate_counter as f32 / elapsed_secs;
                    }
                    self.framerate_counter = 0;
                    self.framerate_window_start = Some(now);
                }
            }
        }
    }

    /// Stores the most recently completed GPU submit-to-idle interval.
    pub(crate) fn set_last_render_time_seconds(&mut self, render_time_seconds: Option<f32>) {
        self.last_render_time_seconds = render_time_seconds.unwrap_or(RENDER_TIME_UNAVAILABLE);
    }

    /// Increments the renderer-tick counter captured by the next frame-start send.
    pub(crate) fn note_render_tick_complete(&mut self) {
        self.rendered_frames_since_last = self.rendered_frames_since_last.saturating_add(1);
    }

    /// Captures and resets the rendered-frame counter while producing the next performance sample.
    pub(crate) fn step_for_frame_start(&mut self) -> Option<PerformanceState> {
        let rendered_frames_since_last = std::mem::replace(&mut self.rendered_frames_since_last, 0);
        step_frame_performance(
            self.wall_interval_us_for_perf,
            self.last_render_time_seconds,
            self.last_window_fps,
            rendered_frames_since_last,
        )
    }
}

/// Builds a [`PerformanceState`] for this frame.
///
/// Returns [`None`] only on the very first tick (`wall_interval_us == 0`), when no
/// frame-to-frame interval has been measured yet and `immediate_fps` has no defined value.
/// All subsequent ticks return [`Some`], so the host-side `PerformanceMetrics` updates every frame.
///
/// `last_frame_render_time_seconds` should be the value returned by
/// [`crate::gpu::GpuContext::last_completed_gpu_render_time_seconds`] mapped through
/// `unwrap_or(`[`RENDER_TIME_UNAVAILABLE`]`)`.
///
/// `windowed_fps` is the most recently computed value from the count-based [`FPS_WINDOW`] window,
/// or `0.0` before the first window has completed.
///
/// `rendered_frames_since_last` is the renderer-tick count since the previous `FrameStartData`
/// send (the caller should snapshot then reset its counter for the new send window).
pub(crate) fn step_frame_performance(
    wall_interval_us: u64,
    last_frame_render_time_seconds: f32,
    windowed_fps: f32,
    rendered_frames_since_last: i32,
) -> Option<PerformanceState> {
    if wall_interval_us == 0 {
        return None;
    }
    let instant_fps = 1_000_000.0 / wall_interval_us as f32;
    Some(PerformanceState {
        fps: windowed_fps,
        immediate_fps: instant_fps,
        render_time: last_frame_render_time_seconds,
        rendered_frames_since_last,
        ..PerformanceState::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_frame_performance_first_tick_with_zero_interval_returns_none() {
        let p = step_frame_performance(0, 0.005, 0.0, 0);
        assert!(p.is_none());
    }

    #[test]
    fn step_frame_performance_emits_immediate_windowed_and_render_time() {
        let p = step_frame_performance(16_666, 0.005, 60.0, 1)
            .expect("payload built when wall_interval_us > 0");
        assert!((p.immediate_fps - 60.0).abs() < 1.0);
        assert!((p.fps - 60.0).abs() < f32::EPSILON);
        assert!((p.render_time - 0.005).abs() < f32::EPSILON);
    }

    #[test]
    fn step_frame_performance_emits_every_consecutive_call() {
        let a = step_frame_performance(16_666, 0.005, 60.0, 1);
        let b = step_frame_performance(16_666, 0.005, 60.0, 1);
        assert!(a.is_some(), "first non-zero interval must emit");
        assert!(b.is_some(), "subsequent ticks must emit (no throttle)");
    }

    #[test]
    fn step_frame_performance_propagates_render_time_unavailable_sentinel() {
        let p =
            step_frame_performance(16_666, RENDER_TIME_UNAVAILABLE, 0.0, 0).expect("payload built");
        assert_eq!(p.render_time, RENDER_TIME_UNAVAILABLE);
    }

    #[test]
    fn step_frame_performance_propagates_rendered_frames_since_last() {
        let lockstep =
            step_frame_performance(16_666, 0.005, 60.0, 1).expect("lockstep payload built");
        assert_eq!(lockstep.rendered_frames_since_last, 1);
        let decoupled =
            step_frame_performance(16_666, 0.005, 60.0, 7).expect("decoupled payload built");
        assert_eq!(decoupled.rendered_frames_since_last, 7);
    }

    #[test]
    fn windowed_fps_is_zero_before_first_window_completes() {
        let mut state = FrameStartPerformanceState::default();
        let t0 = Instant::now();
        state.on_tick_frame_wall_clock(t0);
        for i in 1..=10 {
            state.on_tick_frame_wall_clock(t0 + Duration::from_millis(i * 10));
        }
        assert_eq!(state.last_window_fps, 0.0);
        let payload = state.step_for_frame_start().expect("payload built");
        assert_eq!(payload.fps, 0.0);
    }

    #[test]
    fn windowed_fps_emits_frames_per_elapsed_seconds_after_window() {
        let mut state = FrameStartPerformanceState::default();
        let t0 = Instant::now();
        state.on_tick_frame_wall_clock(t0);
        // 29 mid-window ticks at ~16.66 ms spacing land just shy of the 500 ms boundary; the
        // 30th tick lands exactly on it and triggers the window close -> 30 frames / 0.5 s = 60 fps.
        for i in 1..30 {
            state.on_tick_frame_wall_clock(t0 + Duration::from_micros(i * 16_666));
        }
        state.on_tick_frame_wall_clock(t0 + Duration::from_millis(500));
        assert!(
            (state.last_window_fps - 60.0).abs() < 0.01,
            "expected 60 fps, got {}",
            state.last_window_fps
        );
    }

    #[test]
    fn windowed_fps_value_is_stable_across_ticks_within_one_window() {
        let mut state = FrameStartPerformanceState::default();
        let t0 = Instant::now();
        state.on_tick_frame_wall_clock(t0);
        // Close the first window with a 30th counted tick at exactly 500 ms.
        for i in 1..30 {
            state.on_tick_frame_wall_clock(t0 + Duration::from_micros(i * 16_666));
        }
        state.on_tick_frame_wall_clock(t0 + Duration::from_millis(500));
        let after_first_window = state.last_window_fps;
        assert!(after_first_window > 0.0);
        // Walk a few mid-window ticks at the same spacing; fps must not change until the next
        // window closes.
        let window_anchor = t0 + Duration::from_millis(500);
        for i in 1..=10 {
            state.on_tick_frame_wall_clock(window_anchor + Duration::from_micros(i * 16_666));
            assert_eq!(state.last_window_fps, after_first_window);
        }
    }

    #[test]
    fn windowed_fps_reports_independent_values_for_back_to_back_windows() {
        let mut state = FrameStartPerformanceState::default();
        let t0 = Instant::now();
        state.on_tick_frame_wall_clock(t0);
        // First window: 60 fps closing at exactly 500 ms.
        for i in 1..30 {
            state.on_tick_frame_wall_clock(t0 + Duration::from_micros(i * 16_666));
        }
        state.on_tick_frame_wall_clock(t0 + Duration::from_millis(500));
        let first_fps = state.last_window_fps;
        assert!((first_fps - 60.0).abs() < 0.01);
        // Second window: 7 mid-window ticks at 66.66 ms spacing, then an 8th tick at exactly
        // 500 ms past the new anchor -> 8 frames / 0.5 s = 16 fps. Independent of the first window.
        let window_anchor = t0 + Duration::from_millis(500);
        for i in 1..8 {
            state.on_tick_frame_wall_clock(window_anchor + Duration::from_micros(i * 66_666));
        }
        state.on_tick_frame_wall_clock(window_anchor + Duration::from_millis(500));
        let second_fps = state.last_window_fps;
        assert!(
            (second_fps - 16.0).abs() < 0.01,
            "expected 16 fps after second window, got {second_fps}"
        );
        assert!(
            second_fps < first_fps / 2.0,
            "second window must drop independently of the first ({first_fps} -> {second_fps})"
        );
    }
}
