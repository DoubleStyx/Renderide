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
use super::submit_counters::SubmitCounters;
use super::surface_counters::SurfaceCounters;
use super::xr_finalize::run_xr_finalize;
use crate::crash_context;
use crate::gpu::GpuQueueAccessGate;
use crate::gpu::flight_recorder::{
    GpuFlightCallResult, GpuFlightDriverStage, GpuFlightEventKind, GpuFlightRecorder,
};
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

/// Shared handles used while processing one driver-thread batch.
#[derive(Clone, Copy)]
struct DriverLoopContext<'a> {
    /// Queue that receives command-buffer submits.
    queue: &'a wgpu::Queue,
    /// Gate used to serialize queue access with OpenXR and uploads.
    gpu_queue_access_gate: &'a GpuQueueAccessGate,
    /// Surface present counters for acquire/present synchronization.
    surface_counters: &'a SurfaceCounters,
    /// Submit counters used for backlog snapshots.
    submit_counters: &'a SubmitCounters,
    /// In-memory crash diagnostic recorder.
    flight_recorder: &'a Arc<GpuFlightRecorder>,
}

/// Copyable summary of the batch being processed.
#[derive(Clone, Copy)]
struct DriverBatchSummary {
    /// Driver ring depth after the batch was popped.
    ring_depth: usize,
    /// Frame sequence assigned by frame timing, or zero when untracked.
    frame_seq: u64,
    /// Command buffers in the batch.
    command_buffers: usize,
    /// Whether this batch carries a surface texture.
    has_surface: bool,
    /// Whether this batch carries OpenXR finalize work.
    has_xr_finalize: bool,
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
    _errors: Arc<DriverErrorState>,
    surface_counters: Arc<SurfaceCounters>,
    submit_counters: Arc<SubmitCounters>,
    flight_recorder: Arc<GpuFlightRecorder>,
) {
    profiling::register_thread!("renderer-driver");
    logger::info!("driver thread started");

    let _liveness = ConsumerLivenessGuard { ring: &ring };
    loop {
        let message = {
            profiling::scope!("driver::wait_for_batch");
            ring.pop()
        };
        let DriverMessage::Submit(batch) = message else {
            break;
        };
        let ring_depth = ring.depth();
        let ctx = DriverLoopContext {
            queue: queue.as_ref(),
            gpu_queue_access_gate: &gpu_queue_access_gate,
            surface_counters: &surface_counters,
            submit_counters: &submit_counters,
            flight_recorder: &flight_recorder,
        };
        process_batch(ctx, ring_depth, *batch);
    }
    // A `DriverMessage::Shutdown` value breaks the loop above; nothing further to do.
    let (pushed, done) = submit_counters.snapshot();
    logger::info!(
        "driver thread exiting: backlog={} pushed={} done={}",
        pushed.saturating_sub(done),
        pushed,
        done,
    );
}

/// Handles one batch end-to-end: submit, install frame-timing callback, present, signal
/// the oneshot. Each step is instrumented for Tracy.
fn process_batch(ctx: DriverLoopContext<'_>, ring_depth: usize, batch: SubmitBatch) {
    profiling::scope!("driver::frame");
    let SubmitBatch {
        submit_kind: _submit_kind,
        command_buffers,
        retained_resources: _retained_resources,
        surface_texture,
        on_submitted_work_done,
        frame_timing,
        frame_bracket_readback,
        wait,
        xr_finalize,
        frame_seq,
    } = batch;
    let summary = DriverBatchSummary {
        ring_depth,
        frame_seq,
        command_buffers: command_buffers.len(),
        has_surface: surface_texture.is_some(),
        has_xr_finalize: xr_finalize.is_some(),
    };

    submit_commands(ctx, summary, command_buffers);

    if let Some(track) = frame_timing {
        // Capture the post-submit instant on this thread for the `submit_latency_ms`
        // diagnostic. The HUD's CPU column is published synchronously from the runtime tick
        // epilogue via `record_main_thread_cpu_end` and does not depend on this instant.
        let real_submit_at = record_real_submit(&track);
        register_gpu_completion(ctx.queue, track, real_submit_at, frame_bracket_readback);
    }

    for cb in on_submitted_work_done {
        ctx.queue.on_submitted_work_done(cb);
    }

    finalize_xr_if_present(ctx, summary, xr_finalize);
    present_surface_if_present(ctx, summary, surface_texture);

    if let Some(wait) = wait {
        wait.signal();
    }
}

/// Submits command buffers through wgpu while holding the queue gate.
fn submit_commands(
    ctx: DriverLoopContext<'_>,
    summary: DriverBatchSummary,
    command_buffers: Vec<wgpu::CommandBuffer>,
) {
    record_driver_event(
        ctx,
        summary,
        GpuFlightDriverStage::SubmitStart,
        GpuFlightCallResult::Ok,
    );
    {
        profiling::scope!("driver::submit");
        // Serialise against texture uploads and OpenXR queue-access calls via the shared gate.
        let _gate = {
            profiling::scope!("driver::submit::queue_gate_lock");
            ctx.gpu_queue_access_gate.lock()
        };
        {
            profiling::scope!("driver::submit::queue_submit");
            ctx.queue.submit(command_buffers)
        }
    };
    // Bumped immediately after the submit returns and the gate is dropped so the backlog
    // plot reflects "in-flight on driver" without waiting on `present` or `xr_finalize`.
    ctx.submit_counters.note_submit_done();
    record_driver_event(
        ctx,
        summary,
        GpuFlightDriverStage::SubmitDone,
        GpuFlightCallResult::Ok,
    );
}

/// Runs deferred OpenXR finalize work when the batch carries it.
fn finalize_xr_if_present(
    ctx: DriverLoopContext<'_>,
    summary: DriverBatchSummary,
    xr_finalize: Option<super::XrFinalizeWork>,
) {
    let Some(finalize) = xr_finalize else {
        return;
    };
    // Deferred OpenXR `xrReleaseSwapchainImage` + `xrEndFrame` for the VR HMD path.
    // Running it here keeps the main thread out of `flush_driver` so frame N+1's CPU
    // work overlaps with frame N's compositor handoff. The next tick's `wait_frame`
    // gates on the matching finalize signal before issuing `xrBeginFrame`.
    record_driver_event(
        ctx,
        summary,
        GpuFlightDriverStage::XrFinalizeStart,
        GpuFlightCallResult::Ok,
    );
    let result = run_xr_finalize(
        ctx.gpu_queue_access_gate,
        finalize,
        Arc::clone(ctx.flight_recorder),
    );
    let (failed, flight_result) = match result {
        Ok(()) => (false, GpuFlightCallResult::Ok),
        Err(err) => (true, GpuFlightCallResult::failed_debug(err)),
    };
    record_driver_event(
        ctx,
        summary,
        GpuFlightDriverStage::XrFinalizeDone,
        flight_result,
    );
    if failed {
        ctx.flight_recorder.dump_once("xr-finalize-failed");
    }
}

/// Presents a surface texture when the batch carries one.
fn present_surface_if_present(
    ctx: DriverLoopContext<'_>,
    summary: DriverBatchSummary,
    surface_texture: Option<wgpu::SurfaceTexture>,
) {
    let Some(tex) = surface_texture else {
        return;
    };
    {
        profiling::scope!("driver::present");
        record_driver_event(
            ctx,
            summary,
            GpuFlightDriverStage::PresentStart,
            GpuFlightCallResult::Ok,
        );
        {
            profiling::scope!("driver::present::queue_gate_lock");
            let _gate = ctx.gpu_queue_access_gate.lock();
            {
                profiling::scope!("driver::present::surface_present");
                // `SurfaceTexture::present` is infallible in the current wgpu API; if that
                // changes, route the error into `errors` with `DriverErrorKind::Present`.
                tex.present();
            }
        }
    };
    // Signal to the main thread that the previous surface texture is no longer
    // outstanding so its next `get_current_texture` call can proceed without a
    // full ring flush.
    ctx.surface_counters.note_presented();
    crate::profiling::plot_surface_in_flight_count(ctx.surface_counters.in_flight_count());
    record_driver_event(
        ctx,
        summary,
        GpuFlightDriverStage::PresentDone,
        GpuFlightCallResult::Ok,
    );
}

/// Records a driver event using the current submit backlog snapshot.
fn record_driver_event(
    ctx: DriverLoopContext<'_>,
    summary: DriverBatchSummary,
    stage: GpuFlightDriverStage,
    result: GpuFlightCallResult,
) {
    let (pushed, done) = ctx.submit_counters.snapshot();
    crash_context::set_driver_stage(stage.crash_context_stage());
    ctx.flight_recorder.record(GpuFlightEventKind::Driver {
        stage,
        frame_seq: summary.frame_seq,
        command_buffers: summary.command_buffers,
        has_surface: summary.has_surface,
        has_xr_finalize: summary.has_xr_finalize,
        ring_depth: summary.ring_depth,
        backlog: pushed.saturating_sub(done),
        result,
    });
}

/// Picks the GPU-completion path for a tracked submit.
///
/// When `bracket_readback` is `Some`, schedules a `map_async` on the timestamp readback buffer
/// and publishes the resulting `gpu_frame_ms` (real GPU time) into the timing accumulator.
/// Otherwise registers a `Queue::on_submitted_work_done` callback that completes the CPU/GPU
/// pairing without publishing GPU busy time.
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
