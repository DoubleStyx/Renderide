//! Lightweight per-frame timing for the **Frame timing** ImGui window (FPS, wall interval,
//! CPU/GPU per-frame ms, RAM/VRAM, and a rolling frametime graph -- MangoHud-style overlay).
//!
//! Unlike [`super::FrameDiagnosticsSnapshot`], this avoids the heavy shader-routes / allocator-report
//! gathering and is safe to populate every tick.

use crate::gpu::GpuContext;
use crate::gpu::frame_cpu_gpu_timing::GpuMsSource;

use super::frame_diagnostics::HostCpuMemoryHud;
use crate::diagnostics::ema::FrameTimingEma;
use crate::diagnostics::frame_history::FrameTimeHistory;

/// Minimal HUD payload: wall-clock roundtrip, CPU/GPU per-frame ms, memory totals, and
/// frametime graph.
///
/// Numeric scalars (`*_ms_smoothed`) are run through [`FrameTimingEma`] so the readouts settle
/// instead of jittering each frame. The frametime graph keeps the raw samples in
/// [`Self::frame_time_history`] so spikes remain visible.
#[derive(Clone, Debug, Default)]
pub struct FrameTimingHudSnapshot {
    /// Wall-clock roundtrip between consecutive winit ticks (ms): the time between when one
    /// frame started and the next one started. FPS = `1000.0 / wall_frame_time_ms_smoothed`.
    /// EMA-smoothed for display.
    pub wall_frame_time_ms_smoothed: f64,
    /// Raw wall-clock roundtrip used to drive the sparkline's most recent sample.
    pub wall_frame_time_ms_raw: f64,
    /// CPU per-frame ms (EMA-smoothed): main-thread tick duration from
    /// [`crate::gpu::frame_cpu_gpu_timing::FrameCpuGpuTiming`]. Excludes FPS-gating sleeps,
    /// lockstep waits, and event-loop idles.
    pub cpu_frame_ms_smoothed: Option<f64>,
    /// GPU per-frame ms (EMA-smoothed). Source identified by [`Self::gpu_ms_source`].
    pub gpu_frame_ms_smoothed: Option<f64>,
    /// Origin of the GPU value: real timestamp queries vs callback-latency fallback.
    pub gpu_ms_source: Option<GpuMsSource>,
    /// Rolling frametime samples (ms, oldest-first) for the sparkline plot. Raw -- not smoothed.
    pub frame_time_history: Vec<f32>,
    /// Global host CPU usage 0-100 (sysinfo, throttled).
    pub host_cpu_usage_percent: f32,
    /// Total system RAM in bytes (sysinfo).
    pub host_ram_total_bytes: u64,
    /// Used system RAM in bytes (sysinfo).
    pub host_ram_used_bytes: u64,
    /// Resident memory of the renderer process in bytes (sysinfo; `None` when unavailable).
    pub process_ram_bytes: Option<u64>,
}

impl FrameTimingHudSnapshot {
    /// Reads GPU timing and pairs it with the supplied host / history / EMA state.
    ///
    /// `ema` is updated in place with this tick's samples so steady-state readouts settle.
    pub fn capture(
        gpu: &GpuContext,
        wall_frame_time_ms: f64,
        host: &HostCpuMemoryHud,
        history: &FrameTimeHistory,
        ema: &mut FrameTimingEma,
    ) -> Self {
        profiling::scope!("hud::build_timing_snapshot");
        let (cpu_frame_ms_raw, gpu_frame_ms_raw) = gpu.frame_cpu_gpu_ms_for_hud();
        let gpu_ms_source = gpu.last_gpu_ms_source();
        let wall_frame_time_ms_smoothed = ema.frame.update(wall_frame_time_ms);
        let cpu_frame_ms_smoothed = cpu_frame_ms_raw.map(|v| ema.cpu.update(v));
        let gpu_frame_ms_smoothed = gpu_frame_ms_raw.map(|v| ema.gpu.update(v));
        Self {
            wall_frame_time_ms_smoothed,
            wall_frame_time_ms_raw: wall_frame_time_ms,
            cpu_frame_ms_smoothed,
            gpu_frame_ms_smoothed,
            gpu_ms_source,
            frame_time_history: history.to_vec(),
            host_cpu_usage_percent: host.cpu_usage_percent,
            host_ram_total_bytes: host.ram_total_bytes,
            host_ram_used_bytes: host.ram_used_bytes,
            process_ram_bytes: host.process_ram_bytes,
        }
    }

    /// FPS from smoothed wall-clock interval between redraws. The smoothed value avoids
    /// flickering between, say, 59 and 61 fps when the workload is steady.
    pub fn fps_from_wall(&self) -> f64 {
        if self.wall_frame_time_ms_smoothed <= f64::EPSILON {
            0.0
        } else {
            1000.0 / self.wall_frame_time_ms_smoothed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrameTimingHudSnapshot;

    #[test]
    fn fps_from_wall_matches_inverse_smoothed_ms() {
        let s = FrameTimingHudSnapshot {
            wall_frame_time_ms_smoothed: 16.0,
            wall_frame_time_ms_raw: 16.0,
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
}
