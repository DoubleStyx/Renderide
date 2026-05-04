//! OpenXR frame-finalize work piggybacked onto the driver thread.
//!
//! The main thread cannot release the OpenXR swapchain image and call `xrEndFrame` until
//! the previous `wgpu::Queue::submit` for the swapchain-touching work has hit the queue.
//! Doing the wait synchronously on the main thread defeats the driver-thread pipelining.
//! Instead the main thread packages everything `xrReleaseSwapchainImage` and
//! `xrEndFrame` need into [`XrFinalizeWork`] and attaches it to the trailing
//! [`super::SubmitBatch`] of the VR tick. The driver runs the finalize on its own
//! thread, then signals back via [`XrFinalizeSignal`] so the next tick's `wait_frame`
//! can synchronise with `xrBeginFrame` ordering.
//!
//! Errors observed on the driver thread are stored in a shared slot and surfaced to the
//! main thread on the next finalize wait so existing recovery paths react one tick later.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::time::Duration;

use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};
use parking_lot::Mutex;

use crate::gpu::GpuQueueAccessGate;
use crate::xr::session::end_frame_watchdog::EndFrameWatchdog;

/// Deadline for a single deferred `xrEndFrame` call before the watchdog logs a stall.
///
/// Matches the timeout used by the original main-thread `end_frame_projection`
/// (`crates/renderide/src/xr/session/state/frame_loop.rs`). 500 ms is well above normal
/// VR frame budgets while short enough that a true freeze surfaces within one log
/// interval.
const END_FRAME_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);

/// Oneshot used by the driver thread to notify the main thread that the finalize work
/// for one frame has completed (or failed and recorded to [`XrFinalizeErrorSlot`]).
pub struct XrFinalizeSignal {
    sender: SyncSender<()>,
}

/// Receiver half of [`XrFinalizeSignal`], held by the main thread until the next tick's
/// `wait_frame` drains it.
pub type XrFinalizeReceiver = Receiver<()>;

impl XrFinalizeSignal {
    /// Creates a fresh sender/receiver pair. Sender goes into [`XrFinalizeWork`]; the
    /// receiver is stored on [`crate::xr::session::XrSessionState::pending_finalize`].
    pub fn new() -> (Self, XrFinalizeReceiver) {
        let (tx, rx) = sync_channel::<()>(1);
        (Self { sender: tx }, rx)
    }

    /// Fires the oneshot; ignores send errors because the main thread may have already
    /// dropped the receiver during shutdown.
    pub(super) fn signal(self) {
        let _ = self.sender.send(());
    }
}

/// Shared first-error-wins slot for OpenXR errors observed on the driver thread.
///
/// The slot is read by the main thread inside `wait_frame` after waiting on the matching
/// [`XrFinalizeReceiver`]; if a value is present it is consumed and propagated as the
/// `wait_frame` error so the existing recovery path runs.
pub type XrFinalizeErrorSlot = Arc<Mutex<Option<xr::sys::Result>>>;

/// One frame's worth of OpenXR finalize work (release + end_frame) executed on the
/// driver thread after `Queue::submit` returns.
pub struct XrFinalizeWork {
    /// What kind of `xrEndFrame` to issue (projection vs empty).
    pub kind: XrFinalizeKind,
    /// Driver-to-main completion oneshot. Always signaled, success or failure.
    pub signal: XrFinalizeSignal,
    /// Shared error slot; on failure the driver writes here before signaling.
    pub error_slot: XrFinalizeErrorSlot,
}

/// Variant of `xrEndFrame` to issue from the driver thread.
///
/// The projection variant is boxed because its inline payload is much larger than the
/// empty variant (two `xr::View`s plus several `Arc`s) and the enum lives inside
/// `SubmitBatch` which is itself sized to the worst-case batch.
pub enum XrFinalizeKind {
    /// Standard stereo projection layer referencing the just-rendered swapchain image.
    Projection(Box<XrProjectionFinalize>),
    /// Close an open frame with no composition layers (recovery path or the
    /// HMD-projection-skipped tick).
    Empty {
        /// Frame stream to issue the empty `xrEndFrame` against.
        frame_stream: Arc<Mutex<xr::FrameStream<xr::Vulkan>>>,
        /// Blend mode passed through to the compositor.
        env_blend_mode: xr::EnvironmentBlendMode,
        /// Predicted display time of the frame being closed.
        predicted_display_time: xr::Time,
        /// Atomic mirror of `XrSessionState::frame_open`; cleared after `xrEndFrame`.
        frame_open: Arc<AtomicBool>,
        /// Shared shutdown flag used to lower expected compositor-stall log severity.
        shutdown_requested: Arc<AtomicBool>,
    },
}

/// Payload for [`XrFinalizeKind::Projection`]. Stored boxed so the enum stays compact.
pub struct XrProjectionFinalize {
    /// Swapchain whose acquired image we must release before `xrEndFrame`.
    pub swapchain: Arc<Mutex<xr::Swapchain<xr::Vulkan>>>,
    /// Frame stream the projection layer is submitted through.
    pub frame_stream: Arc<Mutex<xr::FrameStream<xr::Vulkan>>>,
    /// Reference space the projection layer is anchored in.
    pub stage: Arc<xr::Space>,
    /// Blend mode passed through to the compositor.
    pub env_blend_mode: xr::EnvironmentBlendMode,
    /// Predicted display time of the frame the projection layer represents.
    pub predicted_display_time: xr::Time,
    /// Stereo views from `locate_views`; index 0 = left eye, 1 = right eye.
    pub views: [xr::View; 2],
    /// Per-eye image rectangle (matches the swapchain extent).
    pub rect: xr::Rect2Di,
    /// Atomic mirror of `XrSessionState::frame_open`; cleared after `xrEndFrame`.
    pub frame_open: Arc<AtomicBool>,
    /// Shared shutdown flag used to lower expected compositor-stall log severity.
    pub shutdown_requested: Arc<AtomicBool>,
}

/// Convenience for callers that need to consume a pending finalize on shutdown or before
/// next-frame begin without owning the matching `wait_frame` path.
///
/// Returns `Ok` on signal, `Err` on timeout (treated as "driver thread is unresponsive";
/// callers log and proceed). The 5 s ceiling is the same one used by [`super::DriverThread::flush`].
pub fn wait_for_finalize(rx: XrFinalizeReceiver) -> Result<(), RecvTimeoutError> {
    rx.recv_timeout(Duration::from_secs(5))
}

/// Runs deferred OpenXR finalize work on the driver thread after the trailing batch's
/// `Queue::submit` returns.
///
/// For [`XrFinalizeKind::Projection`]: takes the queue access gate, releases the
/// swapchain image, drops the gate, takes the gate again, calls `xrEndFrame` with a
/// stereo projection layer, drops the gate, clears the `frame_open` flag, and signals.
/// For [`XrFinalizeKind::Empty`]: takes the gate, calls `xrEndFrame` with no layers,
/// drops the gate, clears the flag, and signals.
///
/// Errors are logged with [`logger::warn!`] and recorded in
/// [`XrFinalizeWork::error_slot`] for the next `wait_frame` to surface.
pub(super) fn run_xr_finalize(gate: &GpuQueueAccessGate, work: XrFinalizeWork) {
    profiling::scope!("driver::xr_finalize");
    let XrFinalizeWork {
        kind,
        signal,
        error_slot,
    } = work;

    let result = match kind {
        XrFinalizeKind::Projection(payload) => {
            let release_res = release_swapchain_image_under_gate(gate, &payload.swapchain);
            if let Err(err) = release_res {
                Err(err)
            } else {
                let frame_open = Arc::clone(&payload.frame_open);
                let res = end_frame_projection(gate, &payload);
                frame_open.store(false, Ordering::Release);
                res
            }
        }
        XrFinalizeKind::Empty {
            frame_stream,
            env_blend_mode,
            predicted_display_time,
            frame_open,
            shutdown_requested,
        } => {
            let res = end_frame_empty(
                gate,
                &frame_stream,
                env_blend_mode,
                predicted_display_time,
                &shutdown_requested,
            );
            frame_open.store(false, Ordering::Release);
            res
        }
    };

    if let Err(err) = result {
        logger::warn!("driver: xr finalize failed: {err:?}");
        let mut slot = error_slot.lock();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    signal.signal();
}

/// Releases the OpenXR swapchain image under the queue access gate.
fn release_swapchain_image_under_gate(
    gate: &GpuQueueAccessGate,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> Result<(), xr::sys::Result> {
    profiling::scope!("driver::xr_release_image");
    let _gate = gate.lock();
    swapchain.lock().release_image()
}

/// Builds a stereo projection layer and calls `xrEndFrame` under the queue access gate.
///
/// The watchdog logs an error if the runtime takes longer than [`END_FRAME_WATCHDOG_TIMEOUT`]
/// to return, matching the main-thread behaviour the deferred path replaces. Both the
/// swapchain mutex and the frame-stream mutex are held for the duration of the
/// `xrEndFrame` call: `SwapchainSubImage` borrows the swapchain handle into the
/// composition layer, and the layer's lifetime must outlive the `frame_stream.end` call
/// that consumes it.
#[expect(
    clippy::significant_drop_tightening,
    reason = "swapchain_guard backs the SwapchainSubImage references that frame_stream.end consumes"
)]
fn end_frame_projection(
    gate: &GpuQueueAccessGate,
    payload: &XrProjectionFinalize,
) -> Result<(), xr::sys::Result> {
    profiling::scope!("driver::xr_end_frame");
    let v0 = &payload.views[0];
    let v1 = &payload.views[1];
    let pose0 = sanitize_pose_for_end_frame(v0.pose);
    let pose1 = sanitize_pose_for_end_frame(v1.pose);
    let wd = EndFrameWatchdog::arm_shutdown_aware(
        END_FRAME_WATCHDOG_TIMEOUT,
        "end_frame_projection",
        Arc::clone(&payload.shutdown_requested),
    );
    let res = {
        let _gate = gate.lock();
        let swapchain_guard = payload.swapchain.lock();
        let projection_views = [
            CompositionLayerProjectionView::new()
                .pose(pose0)
                .fov(v0.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(&swapchain_guard)
                        .image_array_index(0)
                        .image_rect(payload.rect),
                ),
            CompositionLayerProjectionView::new()
                .pose(pose1)
                .fov(v1.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(&swapchain_guard)
                        .image_array_index(1)
                        .image_rect(payload.rect),
                ),
        ];
        let layer = CompositionLayerProjection::new()
            .space(payload.stage.as_ref())
            .views(&projection_views);
        let mut stream = payload.frame_stream.lock();
        stream.end(
            payload.predicted_display_time,
            payload.env_blend_mode,
            &[&layer],
        )
    };
    wd.disarm();
    res
}

/// Calls `xrEndFrame` with no composition layers under the queue access gate.
fn end_frame_empty(
    gate: &GpuQueueAccessGate,
    frame_stream: &Mutex<xr::FrameStream<xr::Vulkan>>,
    env_blend_mode: xr::EnvironmentBlendMode,
    predicted_display_time: xr::Time,
    shutdown_requested: &Arc<AtomicBool>,
) -> Result<(), xr::sys::Result> {
    profiling::scope!("driver::xr_end_frame_empty");
    let wd = EndFrameWatchdog::arm_shutdown_aware(
        END_FRAME_WATCHDOG_TIMEOUT,
        "end_frame_empty",
        Arc::clone(shutdown_requested),
    );
    let res = {
        let _gate = gate.lock();
        frame_stream
            .lock()
            .end(predicted_display_time, env_blend_mode, &[])
    };
    wd.disarm();
    res
}

/// OpenXR requires a unit quaternion; some runtimes briefly report `(0,0,0,0)`, which makes
/// `xrEndFrame` fail with `XR_ERROR_POSE_INVALID`. Falls back to identity orientation when
/// the input is degenerate, matching the original main-thread sanitiser.
fn sanitize_pose_for_end_frame(pose: xr::Posef) -> xr::Posef {
    let o = pose.orientation;
    let len_sq =
        o.w.mul_add(o.w, o.z.mul_add(o.z, o.x.mul_add(o.x, o.y * o.y)));
    if len_sq.is_finite() && len_sq >= 1e-10 {
        pose
    } else {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: pose.position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Instant;

    #[test]
    fn signal_unblocks_receiver() {
        let (signal, rx) = XrFinalizeSignal::new();
        let worker = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            signal.signal();
        });
        let start = Instant::now();
        wait_for_finalize(rx).expect("signal arrived");
        assert!(start.elapsed() < Duration::from_secs(1));
        worker.join().expect("worker joined cleanly");
    }

    #[test]
    fn dropped_signal_returns_disconnected() {
        let (signal, rx) = XrFinalizeSignal::new();
        drop(signal);
        let err = wait_for_finalize(rx).expect_err("dropped signal");
        assert!(matches!(err, RecvTimeoutError::Disconnected));
    }
}
