//! Tracked queue submit and per-frame CPU/GPU timing hooks for the debug HUD.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use wgpu::CommandBuffer;

use super::super::frame_cpu_gpu_timing::{make_gpu_done_callback, FrameCpuGpuTimingHandle};

/// Submits render work for this frame; records last submit and GPU idle for the debug HUD timing HUD.
pub(super) fn submit_tracked_frame_commands(
    frame_timing: &FrameCpuGpuTimingHandle,
    queue: &Arc<Mutex<wgpu::Queue>>,
    cmd: CommandBuffer,
) {
    submit_tracked_inner(frame_timing, queue, cmd);
}

/// Same as [`submit_tracked_frame_commands`] but uses an already-locked queue (e.g. debug HUD overlay encode).
pub(super) fn submit_tracked_frame_commands_with_queue(
    frame_timing: &FrameCpuGpuTimingHandle,
    queue: &mut wgpu::Queue,
    cmd: CommandBuffer,
) {
    submit_tracked_inner_with_queue(frame_timing, queue, cmd);
}

fn submit_tracked_inner(
    frame_timing: &FrameCpuGpuTimingHandle,
    queue: &Arc<Mutex<wgpu::Queue>>,
    cmd: CommandBuffer,
) {
    let track = {
        let mut ft = frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        ft.on_before_tracked_submit()
    };
    if let Some((gen, seq)) = track {
        let submit_at = Instant::now();
        let q = queue.lock().unwrap_or_else(|e| e.into_inner());
        q.submit(std::iter::once(cmd));
        let handle = Arc::clone(frame_timing);
        let cb = make_gpu_done_callback(handle, gen, seq, submit_at);
        q.on_submitted_work_done(cb);
    } else {
        queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .submit(std::iter::once(cmd));
    }
}

fn submit_tracked_inner_with_queue(
    frame_timing: &FrameCpuGpuTimingHandle,
    queue: &mut wgpu::Queue,
    cmd: CommandBuffer,
) {
    let track = {
        let mut ft = frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        ft.on_before_tracked_submit()
    };
    if let Some((gen, seq)) = track {
        let submit_at = Instant::now();
        queue.submit(std::iter::once(cmd));
        let handle = Arc::clone(frame_timing);
        let cb = make_gpu_done_callback(handle, gen, seq, submit_at);
        queue.on_submitted_work_done(cb);
    } else {
        queue.submit(std::iter::once(cmd));
    }
}

/// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
pub(super) fn begin_frame_timing(frame_timing: &FrameCpuGpuTimingHandle, frame_start: Instant) {
    frame_timing
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .begin_frame(frame_start);
}

/// Call after all tracked queue submits for this tick (before reading HUD metrics).
pub(super) fn end_frame_timing(frame_timing: &FrameCpuGpuTimingHandle) {
    let mut ft = frame_timing.lock().unwrap_or_else(|e| e.into_inner());
    ft.end_frame();
}

/// CPU time for this tick and the **latest completed** GPU submit→idle ms (may lag).
pub(super) fn frame_cpu_gpu_ms_for_hud(
    frame_timing: &FrameCpuGpuTimingHandle,
) -> (Option<f64>, Option<f64>) {
    let ft = frame_timing.lock().unwrap_or_else(|e| e.into_inner());
    (ft.cpu_until_submit_ms, ft.last_completed_gpu_idle_ms)
}
