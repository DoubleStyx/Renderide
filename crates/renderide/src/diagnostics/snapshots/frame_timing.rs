//! Lightweight per-frame timing for the **Frame timing** ImGui window.
//!
//! The HUD separates wall-frame cadence from active renderer CPU work, timestamp-backed primary
//! GPU work, and renderer-observed host lockstep turnaround. The compact view intentionally leaves
//! those lanes separate; compare the visible Frame/CPU/GPU/Host values directly.
//!
//! Unlike [`super::FrameDiagnosticsSnapshot`], this avoids heavy shader-route and allocator-detail
//! gathering and is safe to populate every displayed frame.

pub mod ema;
pub mod history;

use std::time::{Duration, Instant};

pub use ema::FrameTimingEma;
pub use history::{
    FrameTimeHistory, FrameTimingHistorySample, FrameTimingHistoryStats, FrameTimingOnePercentStats,
};

use crate::gpu::GpuContext;
use crate::gpu::frame_cpu_gpu_timing::GpuMsSource;

use super::frame_diagnostics::{GpuAllocatorHud, HostCpuMemoryHud};

/// Minimal HUD payload: wall-clock cadence, CPU/GPU/host frame ms, memory totals, and rolling
/// history.
#[derive(Clone, Debug, Default)]
pub struct FrameTimingHudSnapshot {
    /// Wall-clock roundtrip between consecutive winit ticks. FPS is `1000 / wall_ms`.
    pub wall_frame_time_ms_smoothed: f64,
    /// CPU per-frame ms: main-thread active renderer work, excluding pacing waits.
    pub cpu_frame_ms_smoothed: Option<f64>,
    /// Timestamp-backed primary GPU busy time in milliseconds.
    pub gpu_frame_ms_smoothed: Option<f64>,
    /// Renderer-observed `FrameStartData` to `FrameSubmitData` turnaround in milliseconds.
    pub host_frame_ms_smoothed: Option<f64>,
    /// Rolling wall-frame samples for the sparkline plot. Raw, not smoothed.
    pub frame_time_history: Vec<f32>,
    /// Rolling 1-second 1% low/high stats.
    pub history_stats: FrameTimingHistoryStats,
    /// Total system RAM in bytes from `sysinfo`.
    pub host_ram_total_bytes: u64,
    /// Used system RAM in bytes from `sysinfo`.
    pub host_ram_used_bytes: u64,
    /// Resident memory of the renderer process in bytes.
    pub process_ram_bytes: Option<u64>,
    /// Live GPU allocator bytes in use.
    pub gpu_allocator_allocated_bytes: Option<u64>,
    /// GPU allocator reserved capacity including allocator fragmentation.
    pub gpu_allocator_reserved_bytes: Option<u64>,
}

/// Inputs for building a frame-timing HUD snapshot.
pub struct FrameTimingHudCapture<'a> {
    /// GPU context that owns frame timing query results and profiler counters.
    pub gpu: &'a GpuContext,
    /// Wall-clock interval between displayed renderer ticks in milliseconds.
    pub wall_frame_time_ms: f64,
    /// Renderer-observed host lockstep turnaround for the most recent primary frame.
    pub host_frame_begin_to_submit: Option<Duration>,
    /// Host/process CPU and RAM snapshot.
    pub host_hud: &'a HostCpuMemoryHud,
    /// GPU allocator totals sampled for the compact HUD.
    pub gpu_allocator: GpuAllocatorHud,
    /// Rolling frame timing history updated by this capture.
    pub history: &'a mut FrameTimeHistory,
    /// EMA state updated by this capture.
    pub ema: &'a mut FrameTimingEma,
    /// Capture timestamp used for rolling one-second stats.
    pub now: Instant,
}

impl FrameTimingHudSnapshot {
    /// Reads GPU timing and folds this tick into the supplied history / EMA state.
    pub fn capture(capture: FrameTimingHudCapture<'_>) -> Self {
        profiling::scope!("hud::build_timing_snapshot");
        let FrameTimingHudCapture {
            gpu,
            wall_frame_time_ms,
            host_frame_begin_to_submit,
            host_hud,
            gpu_allocator,
            history,
            ema,
            now,
        } = capture;
        let primary_work = gpu.primary_frame_work_timing_for_hud();
        let primary_generation = primary_work.map(|v| v.generation);
        let cpu_frame_ms_raw = primary_work.map(|v| v.cpu_frame_ms);
        let gpu_frame_ms_raw = primary_work.and_then(|v| v.gpu_frame_ms);
        let gpu_ms_source = primary_work.and_then(|v| v.gpu_ms_source);
        let host_frame_ms_raw = host_frame_begin_to_submit.map(duration_ms);

        history.push(FrameTimingHistorySample {
            captured_at: now,
            wall_ms: wall_frame_time_ms,
            primary_generation,
            cpu_ms: cpu_frame_ms_raw,
            gpu_ms: authoritative_gpu_ms(gpu_frame_ms_raw, gpu_ms_source),
            host_ms: host_frame_ms_raw,
        });
        let history_stats = history.stats();

        let wall_frame_time_ms_smoothed = ema.frame.update(wall_frame_time_ms);
        let cpu_frame_ms_smoothed = cpu_frame_ms_raw.map(|v| ema.cpu.update(v));
        let gpu_frame_ms_smoothed = gpu_frame_ms_raw.map(|v| ema.gpu.update(v));
        let host_frame_ms_smoothed = host_frame_ms_raw.map(|v| ema.host.update(v));
        Self {
            wall_frame_time_ms_smoothed,
            cpu_frame_ms_smoothed,
            gpu_frame_ms_smoothed,
            host_frame_ms_smoothed,
            frame_time_history: history.to_vec(),
            history_stats,
            host_ram_total_bytes: host_hud.ram_total_bytes,
            host_ram_used_bytes: host_hud.ram_used_bytes,
            process_ram_bytes: host_hud.process_ram_bytes,
            gpu_allocator_allocated_bytes: gpu_allocator.allocated_bytes,
            gpu_allocator_reserved_bytes: gpu_allocator.reserved_bytes,
        }
    }

    /// FPS from smoothed wall-clock interval between redraws.
    pub fn fps_from_wall(&self) -> f64 {
        if self.wall_frame_time_ms_smoothed <= f64::EPSILON {
            0.0
        } else {
            1000.0 / self.wall_frame_time_ms_smoothed
        }
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn authoritative_gpu_ms(
    gpu_frame_ms: Option<f64>,
    gpu_ms_source: Option<GpuMsSource>,
) -> Option<f64> {
    match gpu_ms_source {
        Some(GpuMsSource::FrameBracket) => gpu_frame_ms.and_then(finite_non_negative),
        _ => None,
    }
}

fn finite_non_negative(value: f64) -> Option<f64> {
    (value.is_finite() && value >= 0.0).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::{FrameTimingHudSnapshot, authoritative_gpu_ms};
    use crate::gpu::frame_cpu_gpu_timing::GpuMsSource;

    #[test]
    fn fps_from_wall_matches_inverse_smoothed_ms() {
        let s = FrameTimingHudSnapshot {
            wall_frame_time_ms_smoothed: 16.0,
            cpu_frame_ms_smoothed: Some(2.0),
            gpu_frame_ms_smoothed: Some(1.0),
            ..Default::default()
        };
        assert!((s.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let s = FrameTimingHudSnapshot::default();
        assert_eq!(s.fps_from_wall(), 0.0);
    }

    #[test]
    fn default_has_no_vram_totals() {
        let s = FrameTimingHudSnapshot::default();
        assert_eq!(s.gpu_allocator_allocated_bytes, None);
        assert_eq!(s.gpu_allocator_reserved_bytes, None);
    }

    #[test]
    fn timestamped_gpu_ms_is_authoritative() {
        assert_eq!(
            authoritative_gpu_ms(Some(8.0), Some(GpuMsSource::FrameBracket)),
            Some(8.0)
        );
    }

    #[test]
    fn callback_latency_is_not_authoritative_gpu_ms() {
        assert_eq!(
            authoritative_gpu_ms(Some(20.0), Some(GpuMsSource::CallbackLatency)),
            None
        );
    }
}
