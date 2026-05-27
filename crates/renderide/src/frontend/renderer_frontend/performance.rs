//! Performance-accumulator methods on [`RendererFrontend`].

use std::time::Instant;

use super::super::frame_start_performance::AssetIntegrationPerformanceSample;
use super::RendererFrontend;

impl RendererFrontend {
    /// Records wall-clock spacing for FPS / [`crate::shared::PerformanceState`] before lock-step send.
    pub fn on_tick_frame_wall_clock(&mut self, now: Instant) {
        self.performance.on_tick_frame_wall_clock(now);
    }

    /// Stores the most recently completed whole-frame GPU interval for the next frame-start.
    pub fn set_perf_last_render_time_seconds(&mut self, render_time_seconds: Option<f32>) {
        self.performance
            .set_last_render_time_seconds(render_time_seconds);
    }

    /// Accumulates asset-integration stats for the next outgoing performance payload.
    pub fn record_asset_integration_stats(&mut self, sample: AssetIntegrationPerformanceSample) {
        self.performance.record_asset_integration_stats(sample);
    }

    /// Records one wait on the host/asset integration wake path.
    pub fn record_asset_integration_handle_wait(&mut self) {
        self.performance.record_asset_integration_handle_wait();
    }

    /// Increments the renderer-tick counter feeding frame-start performance data.
    pub fn note_render_tick_complete(&mut self) {
        self.performance.note_render_tick_complete();
    }
}
