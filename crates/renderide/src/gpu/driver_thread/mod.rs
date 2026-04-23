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
mod surface_counters;
mod worker;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::thread;

pub use error::{DriverError, DriverErrorKind};
pub use submit_batch::{SubmitBatch, SubmitWait};

use error::DriverErrorState;
use ring::BoundedRing;
use submit_batch::DriverMessage;
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
    handle: Option<thread::JoinHandle<()>>,
}

impl DriverThread {
    /// Spawns the driver thread. The thread owns its own clone of the wgpu [`wgpu::Queue`];
    /// the main thread keeps the one inside [`crate::gpu::GpuContext`] for
    /// `queue.write_buffer` / `queue.write_texture` use during encoding.
    pub fn new(queue: Arc<wgpu::Queue>) -> Self {
        let ring = Arc::new(BoundedRing::<DriverMessage>::new(RING_CAPACITY));
        let errors = Arc::new(DriverErrorState::default());
        let surface_counters = Arc::new(SurfaceCounters::default());

        let ring_clone = Arc::clone(&ring);
        let errors_clone = Arc::clone(&errors);
        let counters_clone = Arc::clone(&surface_counters);
        #[expect(
            clippy::expect_used,
            reason = "renderer-driver thread spawn failure at startup is unrecoverable"
        )]
        let handle = thread::Builder::new()
            .name("renderer-driver".to_string())
            .spawn(move || {
                worker::driver_loop(ring_clone, queue, errors_clone, counters_clone);
            })
            .expect("spawn renderer-driver thread");

        Self {
            ring,
            errors,
            surface_counters,
            handle: Some(handle),
        }
    }

    /// Enqueues a batch for the driver thread to submit and present. Blocks while the
    /// ring is full — that block is the frame-pacing backpressure.
    ///
    /// When the batch carries a [`wgpu::SurfaceTexture`], the submitted counter is bumped
    /// so [`Self::wait_for_previous_present`] can gate the next acquire precisely on the
    /// previous present completing (rather than flushing the whole ring).
    pub fn submit(&self, batch: SubmitBatch) {
        let has_surface = batch.surface_texture.is_some();
        if has_surface {
            self.surface_counters.note_submitted();
        }
        self.ring.push(DriverMessage::Submit(batch));
    }

    /// Blocks until every previously-submitted surface-carrying batch has reached
    /// [`wgpu::SurfaceTexture::present`] on the driver thread.
    ///
    /// Use this right before [`wgpu::Surface::get_current_texture`] to uphold wgpu's
    /// single-outstanding-surface-texture invariant without draining the full driver ring.
    /// Unlike [`Self::flush`] this does not block on non-surface batches or on the driver's
    /// current non-present work — only on the specific "previous present completed" event.
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
            wait: Some(wait),
            frame_seq: 0,
        };
        self.ring.push(DriverMessage::Submit(batch));
        // Any recv error (channel disconnected due to panic inside the driver) is treated
        // as "driver no longer running" — callers handle that via the separate error slot.
        let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
    }
}

impl Drop for DriverThread {
    fn drop(&mut self) {
        self.ring.push(DriverMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
