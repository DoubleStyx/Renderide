//! Wall-clock and CPU/GPU per-frame timing fragment of [`super::FrameDiagnosticsSnapshot`].

use crate::gpu::GpuContext;

/// Wall-clock interval and CPU/GPU submit splits for the current diagnostics tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimingFragment {
    /// Wall-clock roundtrip between consecutive winit ticks (ms): the time between when one frame
    /// started and the next one started. FPS = `1000.0 / wall_frame_time_ms`.
    pub wall_frame_time_ms: f64,
    /// CPU per-frame ms: from the start of the winit tick to the moment `Queue::submit` returns
    /// on the driver thread for that tick's last submit. Matches the Frame timing HUD CPU line.
    ///
    /// Comes from the most recent frame whose submit has reached the driver thread, so it may
    /// lag the current tick by one frame.
    pub cpu_frame_ms: Option<f64>,
    /// GPU per-frame ms: from `Queue::submit` returning on the driver thread to the
    /// `on_submitted_work_done` callback firing for that submit. Matches the Frame timing HUD
    /// GPU line.
    ///
    /// Comes from the most recent frame whose completion callback has fired, so it may lag the
    /// current tick by one or more frames.
    pub gpu_frame_ms: Option<f64>,
}

impl FrameTimingFragment {
    /// Builds the timing fragment from the GPU context and the current tick's wall interval.
    pub fn capture(gpu: &GpuContext, wall_frame_time_ms: f64) -> Self {
        let (cpu_frame_ms, gpu_frame_ms) = gpu.frame_cpu_gpu_ms_for_hud();
        Self {
            wall_frame_time_ms,
            cpu_frame_ms,
            gpu_frame_ms,
        }
    }

    /// FPS computed from the wall-clock interval between consecutive redraw events.
    pub fn fps_from_wall(&self) -> f64 {
        if self.wall_frame_time_ms <= f64::EPSILON {
            0.0
        } else {
            1000.0 / self.wall_frame_time_ms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrameTimingFragment;

    #[test]
    fn fps_from_wall_matches_inverse_ms() {
        let f = FrameTimingFragment {
            wall_frame_time_ms: 16.0,
            cpu_frame_ms: Some(2.0),
            gpu_frame_ms: Some(1.0),
        };
        assert!((f.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let f = FrameTimingFragment::default();
        assert_eq!(f.fps_from_wall(), 0.0);
    }
}
