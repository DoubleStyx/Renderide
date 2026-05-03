//! Payload types flowing over the driver-thread ring.
//!
//! [`SubmitBatch`] carries the outputs of one frame's encoding phase: the finished
//! [`wgpu::CommandBuffer`]s, the optional swapchain [`wgpu::SurfaceTexture`] to present,
//! a completion-notification callback, and an optional [`SubmitWait`] oneshot for callers
//! (e.g. headless capture) that must block until the batch is processed.
//!
//! [`DriverMessage`] is the private enum actually pushed into the ring; it carries a
//! shutdown sentinel used by [`super::DriverThread::Drop`].

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

use crate::gpu::frame_bracket::FrameBracketReadback;
use crate::gpu::frame_cpu_gpu_timing::FrameTimingTrack;

/// One frame's worth of GPU work queued for the driver thread.
///
/// Built by the main thread after all command encoders for the frame have been finished.
/// Ownership is moved into the ring; the main thread continues executing while the driver
/// thread processes the batch.
pub struct SubmitBatch {
    /// Ordered list of command buffers. Submitted in one `Queue::submit` call.
    pub command_buffers: Vec<wgpu::CommandBuffer>,
    /// Swapchain texture to present after submit. `None` when the frame targets an
    /// offscreen render target (e.g. headless rendering to a persistent offscreen image).
    pub surface_texture: Option<wgpu::SurfaceTexture>,
    /// Installed via repeated [`wgpu::Queue::on_submitted_work_done`] calls after submit.
    /// Callbacks fire on whichever thread next drains the device via [`wgpu::Device::poll`].
    ///
    /// Hi-Z staging-buffer `map_async` and other auxiliary completion work ride this channel
    /// so the main thread can react to submit completion without a full driver-ring flush.
    /// Frame-timing's GPU completion callback is **not** routed through here; it is registered
    /// on the driver thread immediately after submit using [`Self::frame_timing`] so its
    /// baseline instant is taken on the same thread as `Queue::submit`.
    pub on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    /// Frame-timing track for this batch; the driver thread captures the post-submit instant
    /// and registers a [`wgpu::Queue::on_submitted_work_done`] callback against it.
    ///
    /// `None` for non-tracked submits (e.g. probe-bake one-shots that should not contribute to
    /// the per-frame HUD readouts).
    pub frame_timing: Option<FrameTimingTrack>,
    /// Real-GPU-timestamp readback for this batch. When `Some`, the driver thread schedules
    /// `map_async` on the readback buffer immediately after submit; the resulting
    /// `gpu_frame_ms` value is published into the same [`FrameTimingTrack`] handle as the
    /// callback-latency fallback would have used. `None` on adapters that lack the required
    /// timestamp features (the driver thread falls back to the callback path) and on
    /// non-tracked submits.
    pub frame_bracket_readback: Option<FrameBracketReadback>,
    /// Optional oneshot fired after submit + present complete on the driver thread.
    ///
    /// Use this when the main thread must block until the frame is known to be on the
    /// wire -- e.g. headless tests that read back the presented image synchronously.
    pub wait: Option<SubmitWait>,
    /// Monotonic frame counter, surfaced in [`super::DriverError`] and Tracy zone labels.
    pub frame_seq: u64,
}

/// Oneshot used by the driver thread to signal a specific batch has been processed.
///
/// Pair with [`SubmitWait::new`], which returns both the sender (moved into a
/// [`SubmitBatch`]) and a receiver the caller holds to wait on.
pub struct SubmitWait {
    sender: SyncSender<()>,
}

impl SubmitWait {
    /// Creates a new oneshot pair. The returned [`SubmitWait`] goes into a [`SubmitBatch`];
    /// the [`Receiver`] is held by the caller to block on batch completion.
    pub fn new() -> (Self, Receiver<()>) {
        let (tx, rx) = sync_channel::<()>(1);
        (Self { sender: tx }, rx)
    }

    /// Fires the oneshot. Errors are swallowed: the receiver is allowed to drop before the
    /// driver runs (e.g. when the caller timed out), in which case there is nothing to do.
    pub(super) fn signal(self) {
        let _ = self.sender.send(());
    }
}

/// Internal payload the main thread pushes into the ring.
///
/// The shutdown variant lets [`super::DriverThread::Drop`] terminate the driver loop
/// without forcing consumers of the public API to handle a sentinel value. The
/// [`Submit`](Self::Submit) variant is boxed so the enum stays small in the ring's slot array
/// even though [`SubmitBatch`] itself is many hundreds of bytes.
pub(super) enum DriverMessage {
    /// A frame's command-buffer batch ready for submit + present.
    Submit(Box<SubmitBatch>),
    /// Tells the driver loop to exit. Any batches left in the ring after this message are
    /// dropped (their surface textures are dropped without presenting).
    Shutdown,
}
