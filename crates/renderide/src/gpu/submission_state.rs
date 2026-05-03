//! Submission, frame timing, and GPU profiling state owned by [`super::GpuContext`].

use std::sync::{Arc, Mutex};

use super::driver_thread::DriverThread;
use super::frame_bracket::FrameBracket;
use super::frame_cpu_gpu_timing::FrameCpuGpuTimingHandle;

/// Long-lived state used when handing recorded command buffers to the driver thread.
pub(super) struct GpuSubmissionState {
    /// Declared first so the driver thread shuts down before timing/profiler handles are dropped.
    pub(super) driver_thread: DriverThread,
    /// Debug HUD CPU/GPU frame timing accumulator.
    pub(super) frame_timing: FrameCpuGpuTimingHandle,
    /// Real-GPU-timestamp factory for the debug HUD's `gpu_frame_ms`. Always present; whether it
    /// produces sessions depends on the adapter feature set ([`FrameBracket::enabled`]).
    pub(super) frame_bracket: FrameBracket,
    /// GPU timestamp profiler for the Tracy timeline.
    pub(super) gpu_profiler: Option<crate::profiling::GpuProfilerHandle>,
    /// Flattened per-pass GPU timings from the most recently drained profiling frame.
    pub(super) latest_gpu_pass_timings: Arc<Mutex<Vec<crate::profiling::GpuPassEntry>>>,
}

impl GpuSubmissionState {
    /// Creates a submission state bundle from already-initialized runtime handles.
    pub(super) fn new(
        driver_thread: DriverThread,
        frame_timing: FrameCpuGpuTimingHandle,
        frame_bracket: FrameBracket,
        gpu_profiler: Option<crate::profiling::GpuProfilerHandle>,
        latest_gpu_pass_timings: Arc<Mutex<Vec<crate::profiling::GpuPassEntry>>>,
    ) -> Self {
        Self {
            driver_thread,
            frame_timing,
            frame_bracket,
            gpu_profiler,
            latest_gpu_pass_timings,
        }
    }
}
