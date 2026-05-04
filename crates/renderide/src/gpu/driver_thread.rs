//! Dedicated GPU-submission thread, modeled after Filament's `CommandBufferQueue`.
//!
//! The main tick records command buffers, assembles a [`SubmitBatch`], and hands it to
//! [`DriverThread::submit`]. The driver thread drains a bounded FIFO ring
//! ([`ring::BoundedRing`]) and runs `Queue::submit` + `SurfaceTexture::present` off the
//! main thread. The ring's fixed capacity enforces at most one frame of pipelining
//! (backpressure when the driver falls behind).
//!
//! # Ordering
//!
//! The ring is FIFO and processed by a single thread, so GPU submit order matches the
//! order the main thread pushed. For one thread producing, one thread consuming, with
//! sequential in-order processing, nothing else is required.
//!
//! # Shutdown
//!
//! [`DriverThread`]'s `Drop` impl pushes a [`submit_batch::DriverMessage::Shutdown`]
//! sentinel and joins the thread. Any batches queued after shutdown will never run, but
//! in practice no caller pushes during shutdown because the renderer's frame loop has
//! already exited by the time [`crate::gpu::GpuContext`] drops.

mod error;
mod ring;
mod submit_batch;
mod submit_counters;
mod surface_counters;
mod worker;
mod xr_finalize;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::thread;

pub use error::{DriverError, DriverErrorKind};
pub use submit_batch::{SubmitBatch, SubmitWait};
pub use submit_counters::SubmitToken;
pub use xr_finalize::{
    XrFinalizeErrorSlot, XrFinalizeKind, XrFinalizeReceiver, XrFinalizeSignal, XrFinalizeWork,
    XrProjectionFinalize, wait_for_finalize,
};

use error::DriverErrorState;
use ring::BoundedRing;
use submit_batch::DriverMessage;
use submit_counters::SubmitCounters;
use surface_counters::SurfaceCounters;

/// Maximum number of frames queued in the ring at once. Matches Filament's
/// `CommandBufferQueue` latency target: one frame in flight on the driver, one being
/// recorded by the main thread.
pub const RING_CAPACITY: usize = 2;

/// Handle to the driver thread owned by [`crate::gpu::GpuContext`].
///
/// `Drop` pushes a shutdown sentinel and joins the thread, so consumers do not need to
/// call any explicit shutdown API.
pub struct DriverThread {
    ring: Arc<BoundedRing<DriverMessage>>,
    errors: Arc<DriverErrorState>,
    surface_counters: Arc<SurfaceCounters>,
    submit_counters: Arc<SubmitCounters>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DriverThread {
    /// Spawns the driver thread. The thread owns its own clone of the wgpu [`wgpu::Queue`];
    /// the main thread keeps the one inside [`crate::gpu::GpuContext`] for
    /// `queue.write_buffer` / `queue.write_texture` use during encoding.
    ///
    /// `gpu_queue_access_gate` is cloned from [`crate::gpu::GpuContext`]; the driver
    /// loop acquires it around every `Queue::submit` so submits cannot overlap with
    /// texture uploads or OpenXR calls that touch the same Vulkan queue. See
    /// [`crate::gpu::GpuQueueAccessGate`] for the queue-access rules it enforces.
    pub fn new(
        queue: Arc<wgpu::Queue>,
        gpu_queue_access_gate: crate::gpu::GpuQueueAccessGate,
    ) -> std::io::Result<Self> {
        let ring = Arc::new(BoundedRing::<DriverMessage>::new(RING_CAPACITY));
        let errors = Arc::new(DriverErrorState::default());
        let surface_counters = Arc::new(SurfaceCounters::default());
        let submit_counters = Arc::new(SubmitCounters::default());

        let ring_clone = Arc::clone(&ring);
        let errors_clone = Arc::clone(&errors);
        let surface_counters_clone = Arc::clone(&surface_counters);
        let submit_counters_clone = Arc::clone(&submit_counters);
        let handle = thread::Builder::new()
            .name("renderer-driver".to_string())
            .spawn(move || {
                worker::driver_loop(
                    ring_clone,
                    queue,
                    gpu_queue_access_gate,
                    errors_clone,
                    surface_counters_clone,
                    submit_counters_clone,
                );
            })?;

        Ok(Self {
            ring,
            errors,
            surface_counters,
            submit_counters,
            handle: Some(handle),
        })
    }

    /// Enqueues a batch for the driver thread to submit and present, returning its submit token.
    /// Blocks while the ring is full -- that block is the frame-pacing backpressure.
    ///
    /// When the batch carries a [`wgpu::SurfaceTexture`], the submitted counter is bumped
    /// so [`Self::wait_for_previous_present`] can gate the next acquire precisely on the
    /// previous present completing (rather than flushing the whole ring).
    ///
    /// Returns [`None`] if the driver thread has exited (clean shutdown or panic), in which case
    /// the batch is dropped rather than blocking the caller forever; the existing
    /// [`Self::take_pending_error`] path surfaces the underlying failure to the main
    /// render loop on the next tick.
    pub fn submit(&self, batch: SubmitBatch) -> Option<SubmitToken> {
        let has_surface = batch.surface_texture.is_some();
        if has_surface {
            self.surface_counters.note_submitted();
        }
        let token = self.submit_counters.note_pushed();
        if let Err(_dropped) = self.ring.push(DriverMessage::Submit(Box::new(batch))) {
            if has_surface {
                // Roll back the submitted counter so `wait_for_previous_present` does not
                // wait on a present that will never happen.
                self.surface_counters.note_presented();
            }
            // Mirror the rollback for the submit counter so the backlog plot does not
            // show a phantom in-flight batch.
            self.submit_counters.note_submit_done();
            logger::warn!("driver thread exited; dropping submit batch");
            return None;
        }
        // Tracy plot of how full the driver ring is right after the push so saturation
        // (depth equal to RING_CAPACITY = the main thread blocked) and steady-state pipelining
        // (depth typically 0-1) show up alongside the other gpu metrics. Cheap: one Mutex lock,
        // once per submit. Gated on the `tracy` feature so non-tracy builds compile without the
        // `tracy_client` dependency, matching every other plot call in this crate.
        #[cfg(feature = "tracy")]
        tracy_client::plot!("driver/ring_depth", self.ring.depth() as f64);
        Some(token)
    }

    /// Blocks until every previously-submitted surface-carrying batch has reached
    /// [`wgpu::SurfaceTexture::present`] on the driver thread.
    ///
    /// Use this right before [`wgpu::Surface::get_current_texture`] to uphold wgpu's
    /// single-outstanding-surface-texture invariant without draining the full driver ring.
    /// Unlike [`Self::flush`] this does not block on non-surface batches or on the driver's
    /// current non-present work -- only on the specific "previous present completed" event.
    pub fn wait_for_previous_present(&self) {
        self.surface_counters.wait_for_present_catchup(0);
    }

    /// Drains and returns any pending driver-thread error, leaving the slot empty.
    ///
    /// The main thread checks this once per tick and routes the result through the
    /// existing device-recovery path.
    pub fn take_pending_error(&self) -> Option<DriverError> {
        self.errors.take()
    }

    /// Snapshot of the (pushed, done) submit counters, suitable for a Tracy backlog plot.
    /// The gap `pushed - done` is the number of batches the driver still owes the producer.
    pub fn submit_counter_snapshot(&self) -> (u64, u64) {
        self.submit_counters.snapshot()
    }

    /// Returns `true` once the driver has returned from `Queue::submit` for `token`.
    pub fn is_submit_done(&self, token: SubmitToken) -> bool {
        self.submit_counters.is_submit_done(token)
    }

    /// Blocks the caller until the driver thread has processed every batch currently in
    /// the ring.
    ///
    /// Implemented by pushing a zero-work [`SubmitBatch`] that carries a [`SubmitWait`]
    /// oneshot and waiting for the driver to signal it. Because the ring is FIFO and
    /// processed by one thread, observing the trailing batch's signal implies every
    /// earlier batch's `Queue::submit` (and present, if any) has already run. Used by
    /// the headless readback path to establish a happens-before edge with the render
    /// work before issuing the texture-to-buffer copy on the main thread.
    pub fn flush(&self) {
        let (wait, rx) = SubmitWait::new();
        let batch = SubmitBatch {
            command_buffers: Vec::new(),
            surface_texture: None,
            on_submitted_work_done: Vec::new(),
            frame_timing: None,
            frame_bracket_readback: None,
            wait: Some(wait),
            xr_finalize: None,
            frame_seq: 0,
        };
        if self.submit(batch).is_none() {
            // Driver thread is gone; no signal will ever arrive, so skip the wait.
            return;
        }
        // Any recv error (channel disconnected due to panic inside the driver) is treated
        // as "driver no longer running" -- callers handle that via the separate error slot.
        let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
    }
}

impl Drop for DriverThread {
    fn drop(&mut self) {
        // If the driver thread is already gone, the push is a no-op; either way the
        // subsequent join completes once the worker function returns.
        let _ = self.ring.push(DriverMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
