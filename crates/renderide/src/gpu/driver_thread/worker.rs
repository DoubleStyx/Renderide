//! Driver thread loop: drains [`super::ring::BoundedRing`] and runs one GPU frame per message.
//!
//! The loop is the only place in the renderer that calls [`wgpu::Queue::submit`] or
//! [`wgpu::SurfaceTexture::present`] for the main render-graph path. Errors are captured
//! into [`super::DriverErrorState`] and surfaced to the main thread at the next
//! [`super::DriverThread::take_pending_error`] call.

use std::sync::Arc;

use super::error::DriverErrorState;
use super::ring::BoundedRing;
use super::submit_batch::{DriverMessage, SubmitBatch};
use super::surface_counters::SurfaceCounters;
use crate::gpu::GpuQueueAccessGate;
use crate::gpu::frame_cpu_gpu_timing::{
    FrameTimingTrack, make_gpu_done_callback, record_frame_bracket_gpu_ms, record_real_submit,
};

/// RAII guard that marks the ring's consumer side dead on drop.
///
/// Drop runs on both clean shutdown (loop break) and panic-driven unwind through
/// [`driver_loop`], so a producer blocked in [`super::ring::BoundedRing::push`] is always
/// released -- preventing the main thread from hanging forever on a crashed driver.
struct ConsumerLivenessGuard<'a> {
    ring: &'a BoundedRing<DriverMessage>,
}

impl Drop for ConsumerLivenessGuard<'_> {
    fn drop(&mut self) {
        self.ring.mark_consumer_dead();
    }
}

/// Thread entry point spawned from [`super::DriverThread::new`].
///
/// Registers itself as `"renderer-driver"` in the active profiler so Tracy groups its
/// spans on a single thread row. Exits on the [`DriverMessage::Shutdown`] sentinel. The
/// [`ConsumerLivenessGuard`] flips the ring's liveness flag on any exit (clean or panic).
pub(super) fn driver_loop(
    ring: Arc<BoundedRing<DriverMessage>>,
    queue: Arc<wgpu::Queue>,
    gpu_queue_access_gate: GpuQueueAccessGate,
    errors: Arc<DriverErrorState>,
    surface_counters: Arc<SurfaceCounters>,
) {
    profiling::register_thread!("renderer-driver");

    let _liveness = ConsumerLivenessGuard { ring: &ring };
    loop {
        {
            profiling::scope!("driver::wait_for_batch");
            let DriverMessage::Submit(batch) = ring.pop() else {
                break;
            };
            process_batch(
                queue.as_ref(),
                &gpu_queue_access_gate,
                &errors,
                &surface_counters,
                *batch,
            );
        }
    }
    // A `DriverMessage::Shutdown` value breaks the loop above; nothing further to do.
}

/// Handles one batch end-to-end: submit, install frame-timing callback, present, signal
/// the oneshot. Each step is instrumented for Tracy.
fn process_batch(
    queue: &wgpu::Queue,
    gpu_queue_access_gate: &GpuQueueAccessGate,
    errors: &DriverErrorState,
    surface_counters: &SurfaceCounters,
    batch: SubmitBatch,
) {
    profiling::scope!("driver::frame");
    let SubmitBatch {
        command_buffers,
        surface_texture,
        on_submitted_work_done,
        frame_timing,
        frame_bracket_readback,
        wait,
        frame_seq,
    } = batch;

    {
        profiling::scope!("driver::submit");
        // Serialise against texture uploads and OpenXR queue-access calls via the shared gate.
        let _gate = gpu_queue_access_gate.lock();
        queue.submit(command_buffers)
    };

    if let Some(track) = frame_timing {
        // Capture the post-submit instant on this thread for the `submit_latency_ms`
        // diagnostic. The HUD's CPU column is published synchronously from the runtime tick
        // epilogue via `record_main_thread_cpu_end` and does not depend on this instant.
        let real_submit_at = record_real_submit(&track);
        register_gpu_completion(queue, track, real_submit_at, frame_bracket_readback);
    }

    for cb in on_submitted_work_done {
        queue.on_submitted_work_done(cb);
    }

    if let Some(tex) = surface_texture {
        {
            profiling::scope!("driver::present");
            // `SurfaceTexture::present` is infallible in the current wgpu API; if that
            // changes, route the error into `errors` with `DriverErrorKind::Present`.
            tex.present();
        };
        // Signal to the main thread that the previous surface texture is no longer
        // outstanding so its next `get_current_texture` call can proceed without a
        // full ring flush.
        surface_counters.note_presented();
    }

    if let Some(wait) = wait {
        wait.signal();
    }

    // `frame_seq` is carried for future error-context enrichment; reference it so the
    // compiler does not warn about unused fields while we grow the error path.
    let _ = frame_seq;
    let _ = errors; // `errors` will fill in once wgpu surfaces fallible submit/present.
}

/// Picks the GPU-completion path for a tracked submit.
///
/// When `bracket_readback` is `Some`, schedules a `map_async` on the timestamp readback buffer
/// and publishes the resulting `gpu_frame_ms` (real GPU time) into the timing accumulator
/// labelled as [`crate::gpu::frame_cpu_gpu_timing::GpuMsSource::FrameBracket`]. Otherwise
/// registers a `Queue::on_submitted_work_done` callback that publishes the wall-clock
/// callback-fire latency labelled as [`crate::gpu::frame_cpu_gpu_timing::GpuMsSource::CallbackLatency`].
fn register_gpu_completion(
    queue: &wgpu::Queue,
    track: FrameTimingTrack,
    real_submit_at: std::time::Instant,
    bracket_readback: Option<crate::gpu::frame_bracket::FrameBracketReadback>,
) {
    if let Some(readback) = bracket_readback {
        let handle = track.handle;
        let generation = track.generation;
        let seq = track.seq;
        readback.schedule_readback(move |gpu_ms| {
            if let Some(ms) = gpu_ms {
                record_frame_bracket_gpu_ms(&handle, generation, seq, ms);
            }
        });
        return;
    }
    let gpu_done =
        make_gpu_done_callback(track.handle, track.generation, track.seq, real_submit_at);
    queue.on_submitted_work_done(Box::new(gpu_done));
}
