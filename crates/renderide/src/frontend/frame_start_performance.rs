//! Builds the [`crate::shared::PerformanceState`] payload carried on every
//! [`crate::shared::FrameStartData`] sent to the host.
//!
//! Contract consumed by `FrooxEngine.PerformanceMetrics`:
//! - `immediate_fps` -- instantaneous, derived from the current tick's wall-clock interval
//!   ([`crate::frontend::RendererFrontend::on_tick_frame_wall_clock`]). No smoothing.
//! - `fps` -- count-based rolling average over a [`FPS_WINDOW`] window: `frame_count /
//!   elapsed_seconds` recomputed once each time the window closes, otherwise the previously
//!   computed value is carried forward unchanged. Stable for ~[`FPS_WINDOW`] at a time so the host-side
//!   `Sync<float> FPS.Value` change events fire at the window cadence rather than every frame.
//! - `render_time` -- most recently completed whole-frame GPU duration in seconds
//!   ([`crate::gpu::GpuContext::last_completed_gpu_render_time_seconds`]), matching the
//!   Unity renderer's `XRStats.TryGetGPUTimeLastFrame` contract. Reports `-1.0` until a full
//!   frame GPU sample has completed.
//! - `rendered_frames_since_last` -- number of completed renderer ticks since the previous
//!   `FrameStartData` send. `1` in lockstep, `> 1` when the renderer ticked multiple times per
//!   host submit (i.e. host is slow and the renderer kept rendering). Drives
//!   `FrooxEngine.PerformanceStats.RenderedFramesSinceLastTick`.
//! - `frame_begin_to_submit_time` -- most recently completed renderer-observed host turnaround:
//!   outgoing `FrameStartData` send to the matching inbound `FrameSubmitData` queue receipt.
//!
//! A new [`PerformanceState`] is built on every tick where `wall_interval_us > 0` (i.e. starting
//! from the second tick); the host treats a non-null `FrameStartData.performance` as the latest
//! sample, so emitting every frame keeps `immediate_fps` and `render_time` in lock-step with the
//! actual frame loop while the windowed `fps` value stays stable across each window. This is
//! **not** GPU instrumentation; for that, see [`crate::gpu::frame_cpu_gpu_timing`].
//!
//! Layout:
//! - [`fps_window`] -- count-based windowed FPS accumulator.
//! - [`asset_integration`] -- per-tick asset-integration sample shape and the running accumulator
//!   folded into each outgoing payload.
//! - [`state`] -- [`FrameStartPerformanceState`] orchestrator that composes the above.

use std::time::Duration;

use crate::shared::PerformanceState;

pub(crate) mod asset_integration;
pub(crate) mod fps_window;
pub(crate) mod state;

pub(crate) use asset_integration::{
    AssetIntegrationPerformanceSample, AssetIntegrationPerformanceState,
};
pub(crate) use state::FrameStartPerformanceState;

/// Window length for the count-based `fps` rolling average.
pub(crate) const FPS_WINDOW: Duration = Duration::from_millis(500);

/// Sentinel reported in `render_time` until the first GPU completion callback has fired.
pub(crate) const RENDER_TIME_UNAVAILABLE: f32 = -1.0;

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
    frame_begin_to_submit_seconds: f32,
    asset_integration: AssetIntegrationPerformanceState,
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
        frame_begin_to_submit_time: frame_begin_to_submit_seconds,
        integration_processing_time: asset_integration.integration_processing_time,
        extra_particle_processing_time: asset_integration.extra_particle_processing_time,
        processed_asset_integrator_tasks: asset_integration.processed_asset_integrator_tasks,
        integration_high_priority_tasks: asset_integration.integration_high_priority_tasks,
        integration_tasks: asset_integration.integration_tasks,
        integration_render_tasks: asset_integration.integration_render_tasks,
        integration_particle_tasks: asset_integration.integration_particle_tasks,
        processing_handle_waits: asset_integration.processing_handle_waits,
        ..PerformanceState::default()
    })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn step_frame_performance_first_tick_with_zero_interval_returns_none() {
        let p = step_frame_performance(
            0,
            0.005,
            0.0,
            0,
            0.0,
            AssetIntegrationPerformanceState::default(),
        );
        assert!(p.is_none());
    }

    #[test]
    fn step_frame_performance_emits_immediate_windowed_and_render_time() {
        let p = step_frame_performance(
            16_666,
            0.005,
            60.0,
            1,
            0.0,
            AssetIntegrationPerformanceState::default(),
        )
        .expect("payload built when wall_interval_us > 0");
        assert!((p.immediate_fps - 60.0).abs() < 1.0);
        assert!((p.fps - 60.0).abs() < f32::EPSILON);
        assert!((p.render_time - 0.005).abs() < f32::EPSILON);
    }

    #[test]
    fn step_frame_performance_emits_every_consecutive_call() {
        let a = step_frame_performance(
            16_666,
            0.005,
            60.0,
            1,
            0.0,
            AssetIntegrationPerformanceState::default(),
        );
        let b = step_frame_performance(
            16_666,
            0.005,
            60.0,
            1,
            0.0,
            AssetIntegrationPerformanceState::default(),
        );
        assert!(a.is_some(), "first non-zero interval must emit");
        assert!(b.is_some(), "subsequent ticks must emit (no throttle)");
    }

    #[test]
    fn step_frame_performance_propagates_render_time_unavailable_sentinel() {
        let p = step_frame_performance(
            16_666,
            RENDER_TIME_UNAVAILABLE,
            0.0,
            0,
            0.0,
            AssetIntegrationPerformanceState::default(),
        )
        .expect("payload built");
        assert_eq!(p.render_time, RENDER_TIME_UNAVAILABLE);
    }

    #[test]
    fn step_frame_performance_propagates_rendered_frames_since_last() {
        let lockstep = step_frame_performance(
            16_666,
            0.005,
            60.0,
            1,
            0.0,
            AssetIntegrationPerformanceState::default(),
        )
        .expect("lockstep payload built");
        assert_eq!(lockstep.rendered_frames_since_last, 1);
        let decoupled = step_frame_performance(
            16_666,
            0.005,
            60.0,
            7,
            0.0,
            AssetIntegrationPerformanceState::default(),
        )
        .expect("decoupled payload built");
        assert_eq!(decoupled.rendered_frames_since_last, 7);
    }

    #[test]
    fn step_frame_performance_propagates_host_turnaround_time() {
        let p = step_frame_performance(
            16_666,
            0.005,
            60.0,
            1,
            0.0125,
            AssetIntegrationPerformanceState::default(),
        )
        .expect("payload built");
        assert!((p.frame_begin_to_submit_time - 0.0125).abs() < f32::EPSILON);
    }

    #[test]
    fn frame_start_performance_accumulates_and_resets_asset_integration_stats() {
        let mut state = FrameStartPerformanceState::default();
        let t0 = Instant::now();
        state.on_tick_frame_wall_clock(t0);
        state.on_tick_frame_wall_clock(t0 + Duration::from_millis(16));
        state.record_asset_integration_stats(AssetIntegrationPerformanceSample {
            integration_elapsed: Duration::from_millis(2),
            particle_elapsed: Duration::from_millis(1),
            processed_tasks: 3,
            high_priority_tasks: 4,
            normal_priority_tasks: 5,
            render_tasks: 6,
            particle_tasks: 7,
            handle_waits: 1,
        });
        state.record_asset_integration_stats(AssetIntegrationPerformanceSample {
            integration_elapsed: Duration::from_millis(3),
            particle_elapsed: Duration::from_millis(2),
            processed_tasks: 8,
            high_priority_tasks: 9,
            normal_priority_tasks: 10,
            render_tasks: 11,
            particle_tasks: 12,
            handle_waits: 2,
        });

        let sample = state
            .step_for_frame_start()
            .expect("payload built after non-zero wall interval");
        assert!((sample.integration_processing_time - 0.005).abs() < f32::EPSILON);
        assert!((sample.extra_particle_processing_time - 0.003).abs() < f32::EPSILON);
        assert_eq!(sample.processed_asset_integrator_tasks, 11);
        assert_eq!(sample.integration_high_priority_tasks, 13);
        assert_eq!(sample.integration_tasks, 15);
        assert_eq!(sample.integration_render_tasks, 17);
        assert_eq!(sample.integration_particle_tasks, 19);
        assert_eq!(sample.processing_handle_waits, 3);

        let reset_sample = state
            .step_for_frame_start()
            .expect("payload built while wall interval remains available");
        assert_eq!(reset_sample.processed_asset_integrator_tasks, 0);
        assert_eq!(reset_sample.integration_high_priority_tasks, 0);
        assert_eq!(reset_sample.integration_tasks, 0);
        assert_eq!(reset_sample.integration_render_tasks, 0);
        assert_eq!(reset_sample.integration_particle_tasks, 0);
        assert_eq!(reset_sample.processing_handle_waits, 0);
    }

    #[test]
    fn windowed_fps_is_zero_before_first_window_completes() {
        let mut state = FrameStartPerformanceState::default();
        let t0 = Instant::now();
        state.on_tick_frame_wall_clock(t0);
        for i in 1..=10 {
            state.on_tick_frame_wall_clock(t0 + Duration::from_millis(i * 10));
        }
        assert_eq!(state.last_window_fps(), 0.0);
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
            (state.last_window_fps() - 60.0).abs() < 0.01,
            "expected 60 fps, got {}",
            state.last_window_fps()
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
        let after_first_window = state.last_window_fps();
        assert!(after_first_window > 0.0);
        // Walk a few mid-window ticks at the same spacing; fps must not change until the next
        // window closes.
        let window_anchor = t0 + Duration::from_millis(500);
        for i in 1..=10 {
            state.on_tick_frame_wall_clock(window_anchor + Duration::from_micros(i * 16_666));
            assert_eq!(state.last_window_fps(), after_first_window);
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
        let first_fps = state.last_window_fps();
        assert!((first_fps - 60.0).abs() < 0.01);
        // Second window: 7 mid-window ticks at 66.66 ms spacing, then an 8th tick at exactly
        // 500 ms past the new anchor -> 8 frames / 0.5 s = 16 fps. Independent of the first window.
        let window_anchor = t0 + Duration::from_millis(500);
        for i in 1..8 {
            state.on_tick_frame_wall_clock(window_anchor + Duration::from_micros(i * 66_666));
        }
        state.on_tick_frame_wall_clock(window_anchor + Duration::from_millis(500));
        let second_fps = state.last_window_fps();
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
