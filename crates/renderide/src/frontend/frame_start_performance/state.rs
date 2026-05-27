//! [`FrameStartPerformanceState`] orchestrator that composes the FPS window
//! and asset-integration accumulator into the outgoing
//! [`crate::shared::PerformanceState`] payload.

use std::time::Instant;

use crate::shared::PerformanceState;

use super::asset_integration::{
    AssetIntegrationPerformanceSample, AssetIntegrationPerformanceState,
};
use super::fps_window::FpsWindow;
use super::{RENDER_TIME_UNAVAILABLE, step_frame_performance};

/// Mutable performance accumulator that feeds outgoing frame-start payloads.
pub(crate) struct FrameStartPerformanceState {
    last_tick_wall_start: Option<Instant>,
    wall_interval_us_for_perf: u64,
    last_render_time_seconds: f32,
    fps_window: FpsWindow,
    rendered_frames_since_last: i32,
    asset_integration: AssetIntegrationPerformanceState,
}

impl Default for FrameStartPerformanceState {
    fn default() -> Self {
        Self {
            last_tick_wall_start: None,
            wall_interval_us_for_perf: 0,
            last_render_time_seconds: RENDER_TIME_UNAVAILABLE,
            fps_window: FpsWindow::default(),
            rendered_frames_since_last: 0,
            asset_integration: AssetIntegrationPerformanceState::default(),
        }
    }
}

impl FrameStartPerformanceState {
    /// Records wall-clock spacing between app-driver frame ticks and advances the count-based
    /// FPS window.
    ///
    /// The first call starts the window without counting, subsequent calls increment a frame
    /// counter, and once [`super::FPS_WINDOW`] has elapsed the window emits
    /// `frames / elapsed_seconds` into the windowed FPS value and re-bases off `now`.
    pub(crate) fn on_tick_frame_wall_clock(&mut self, now: Instant) {
        self.wall_interval_us_for_perf = self
            .last_tick_wall_start
            .map_or(0, |t| now.duration_since(t).as_micros() as u64);
        self.last_tick_wall_start = Some(now);
        self.fps_window.record_tick(now);
    }

    /// Stores the most recently completed whole-frame GPU interval.
    pub(crate) fn set_last_render_time_seconds(&mut self, render_time_seconds: Option<f32>) {
        self.last_render_time_seconds = render_time_seconds.unwrap_or(RENDER_TIME_UNAVAILABLE);
    }

    /// Increments the renderer-tick counter captured by the next frame-start send.
    pub(crate) fn note_render_tick_complete(&mut self) {
        self.rendered_frames_since_last = self.rendered_frames_since_last.saturating_add(1);
    }

    /// Accumulates one cooperative asset-integration drain for the next frame-start payload.
    pub(crate) fn record_asset_integration_stats(
        &mut self,
        sample: AssetIntegrationPerformanceSample,
    ) {
        self.asset_integration.accumulate(sample);
    }

    /// Records one asset-integration wake wait while the renderer is waiting for host submit.
    pub(crate) fn record_asset_integration_handle_wait(&mut self) {
        self.asset_integration.note_handle_wait();
    }

    /// Captures and resets the rendered-frame counter while producing the next performance sample.
    pub(crate) fn step_for_frame_start(&mut self) -> Option<PerformanceState> {
        let rendered_frames_since_last = std::mem::replace(&mut self.rendered_frames_since_last, 0);
        let asset_integration = std::mem::take(&mut self.asset_integration);
        step_frame_performance(
            self.wall_interval_us_for_perf,
            self.last_render_time_seconds,
            self.fps_window.last_fps(),
            rendered_frames_since_last,
            asset_integration,
        )
    }

    /// Most recently emitted windowed FPS value, exposed for tests.
    #[cfg(test)]
    pub(crate) fn last_window_fps(&self) -> f32 {
        self.fps_window.last_fps()
    }
}
