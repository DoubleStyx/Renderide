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

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::time::Duration;

use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};
use parking_lot::Mutex;

use crate::diagnostics::crash_context;
use crate::diagnostics::gpu_flight_recorder::{
    GpuFlightCallResult, GpuFlightOpenXrCall, GpuFlightRecorder,
};
use crate::gpu::GpuQueueAccessGate;
use crate::gpu::driver_thread::BlockingCallWatchdog;

/// Deadline for a single deferred `xrEndFrame` call before the watchdog logs a stall.
///
/// Matches the timeout used by the original main-thread `end_frame_projection`
/// (`crates/renderide/src/xr/session/state/frame_loop.rs`). 500 ms is well above normal
/// VR frame budgets while short enough that a true freeze surfaces within one log
/// interval.
const END_FRAME_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);

/// Submit-side context attached to a deferred OpenXR finalize batch.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XrFinalizeSubmitContext {
    /// Frame sequence assigned by frame timing, or zero when untracked.
    frame_seq: u64,
    /// Command buffer count in the driver batch that carries this finalize.
    command_buffers: usize,
}

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
    /// receiver is held by the XR session until the next frame wait drains it.
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
    /// Submit-side frame context captured when the batch is assembled.
    pub(crate) submit_context: XrFinalizeSubmitContext,
    /// Driver-to-main completion oneshot. Always signaled, success or failure.
    pub signal: XrFinalizeSignal,
    /// Shared error slot; on failure the driver writes here before signaling.
    pub error_slot: XrFinalizeErrorSlot,
}

impl XrFinalizeWork {
    /// Records the driver-batch context that will carry this finalize work.
    pub(crate) fn set_submit_context(&mut self, frame_seq: u64, command_buffers: usize) {
        self.submit_context = XrFinalizeSubmitContext {
            frame_seq,
            command_buffers,
        };
    }
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

/// Moved empty-finalize payload used by the driver helper.
struct XrEmptyFinalizePayload {
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
}

/// Payload for [`XrFinalizeKind::Projection`]. Stored boxed so the enum stays compact.
pub struct XrProjectionFinalize {
    /// Swapchain whose acquired image we must release before `xrEndFrame`.
    pub swapchain: Arc<Mutex<xr::Swapchain<xr::Vulkan>>>,
    /// Per-frame wgpu wrapper for the acquired OpenXR image.
    ///
    /// Kept in the driver batch until after `Queue::submit` returns so command buffers that
    /// referenced the imported image cannot outlive the wrapper. The driver drops it before
    /// `xrReleaseSwapchainImage` so wgpu does not retain tracking state while the compositor owns
    /// the image.
    pub imported_color_texture: Option<wgpu::Texture>,
    /// Swapchain image index acquired for this frame.
    pub image_index: u32,
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
    /// Optional composition-layer depth payload matching the projection views.
    pub depth: Option<XrProjectionDepthFinalize>,
    /// Atomic mirror of `XrSessionState::frame_open`; cleared after `xrEndFrame`.
    pub frame_open: Arc<AtomicBool>,
    /// Shared shutdown flag used to lower expected compositor-stall log severity.
    pub shutdown_requested: Arc<AtomicBool>,
}

/// Per-frame depth payload attached to a stereo projection layer when supported by OpenXR.
pub struct XrProjectionDepthFinalize {
    /// Depth swapchain whose acquired image must be released before `xrEndFrame`.
    pub swapchain: Arc<Mutex<xr::Swapchain<xr::Vulkan>>>,
    /// Per-frame wgpu wrapper for the acquired OpenXR depth image.
    pub imported_depth_texture: Option<wgpu::Texture>,
    /// Depth swapchain image index acquired for this frame.
    pub image_index: u32,
    /// Per-eye depth rectangle matching the color projection rectangle.
    pub rect: xr::Rect2Di,
    /// Reverse-Z depth metadata chained to each projection view.
    pub depth_info: XrCompositionDepthInfo,
}

/// Depth range metadata passed to `XrCompositionLayerDepthInfoKHR`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XrCompositionDepthInfo {
    /// Minimum normalized depth value written to the submitted depth image.
    pub min_depth: f32,
    /// Maximum normalized depth value written to the submitted depth image.
    pub max_depth: f32,
    /// Positive distance represented by `min_depth`.
    pub near_z: f32,
    /// Positive distance represented by `max_depth`.
    pub far_z: f32,
}

impl XrCompositionDepthInfo {
    /// Builds OpenXR depth metadata for Renderide's reverse-Z depth convention.
    pub fn from_clip_planes(near_clip: f32, far_clip: f32) -> Self {
        Self {
            min_depth: 0.0,
            max_depth: 1.0,
            near_z: far_clip,
            far_z: near_clip,
        }
    }
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
/// For [`XrFinalizeKind::Projection`]: drops the per-frame wgpu image wrapper, takes the queue
/// access gate, releases the swapchain image, drops the gate, takes the gate again, calls
/// `xrEndFrame` with a stereo projection layer, drops the gate, clears the `frame_open` flag, and
/// signals. For [`XrFinalizeKind::Empty`]: takes the gate, calls `xrEndFrame` with no layers,
/// drops the gate, clears the flag, and signals.
///
/// Errors are logged with [`logger::warn!`] and recorded in
/// [`XrFinalizeWork::error_slot`] for the next `wait_frame` to surface.
pub(super) fn run_xr_finalize(
    gate: &GpuQueueAccessGate,
    work: XrFinalizeWork,
    flight_recorder: Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    profiling::scope!("driver::xr_finalize");
    let XrFinalizeWork {
        kind,
        submit_context,
        signal,
        error_slot,
    } = work;

    let result = match kind {
        XrFinalizeKind::Projection(payload) => {
            run_projection_finalize(gate, payload, submit_context, &flight_recorder)
        }
        XrFinalizeKind::Empty {
            frame_stream,
            env_blend_mode,
            predicted_display_time,
            frame_open,
            shutdown_requested,
        } => run_empty_finalize(
            gate,
            XrEmptyFinalizePayload {
                frame_stream,
                env_blend_mode,
                predicted_display_time,
                frame_open,
                shutdown_requested,
            },
            submit_context,
            &flight_recorder,
        ),
    };

    if let Err(err) = result {
        logger::warn!("driver: xr finalize failed: {err:?}");
        let mut slot = error_slot.lock();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    signal.signal();
    crash_context::clear_xr_finalize_state();
    result
}

/// Runs stereo projection finalize work.
fn run_projection_finalize(
    gate: &GpuQueueAccessGate,
    mut payload: Box<XrProjectionFinalize>,
    submit_context: XrFinalizeSubmitContext,
    flight_recorder: &Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    set_projection_crash_context(&payload, submit_context);
    drop(payload.imported_color_texture.take());
    if let Some(depth) = payload.depth.as_mut() {
        drop(depth.imported_depth_texture.take());
    }
    record_release_image_started(&payload, flight_recorder);
    let release_res = release_image_with_watchdog(gate, &payload, submit_context, flight_recorder);
    record_release_image_result(&payload, flight_recorder, release_res);
    release_res?;
    if let Some(depth) = payload.depth.as_ref() {
        record_release_depth_image_started(&payload, depth, flight_recorder);
        let release_depth_res = release_depth_image_with_watchdog(
            gate,
            &payload,
            depth,
            submit_context,
            flight_recorder,
        );
        record_release_depth_image_result(&payload, depth, flight_recorder, release_depth_res);
        release_depth_res?;
    }
    record_end_frame_projection_started(&payload, flight_recorder);
    let res = end_frame_projection(gate, &payload, submit_context, Arc::clone(flight_recorder));
    record_end_frame_projection_result(&payload, flight_recorder, res);
    payload.frame_open.store(false, Ordering::Release);
    res
}

/// Runs empty-frame finalize work.
fn run_empty_finalize(
    gate: &GpuQueueAccessGate,
    payload: XrEmptyFinalizePayload,
    submit_context: XrFinalizeSubmitContext,
    flight_recorder: &Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    crash_context::set_xr_finalize_state(
        crash_context::XrFinalizeKind::Empty,
        None,
        submit_context.frame_seq,
        submit_context.command_buffers,
        None,
        Some(payload.predicted_display_time.as_nanos()),
    );
    flight_recorder.record_openxr_call_started(
        GpuFlightOpenXrCall::EndFrameEmpty,
        None,
        Some(payload.predicted_display_time.as_nanos()),
    );
    let res = end_frame_empty(
        gate,
        &payload.frame_stream,
        payload.env_blend_mode,
        payload.predicted_display_time,
        &payload.shutdown_requested,
        submit_context,
        Arc::clone(flight_recorder),
    );
    flight_recorder.record_openxr_call_result(
        GpuFlightOpenXrCall::EndFrameEmpty,
        flight_result(res),
        None,
        Some(payload.predicted_display_time.as_nanos()),
    );
    payload.frame_open.store(false, Ordering::Release);
    res
}

/// Records the start of a projection swapchain release.
fn record_release_image_started(
    payload: &XrProjectionFinalize,
    flight_recorder: &GpuFlightRecorder,
) {
    flight_recorder.record_openxr_call_started(
        GpuFlightOpenXrCall::ReleaseImage,
        Some(payload.image_index),
        Some(payload.predicted_display_time.as_nanos()),
    );
}

/// Releases a projection swapchain image with a timeout hook.
fn release_image_with_watchdog(
    gate: &GpuQueueAccessGate,
    payload: &XrProjectionFinalize,
    submit_context: XrFinalizeSubmitContext,
    flight_recorder: &Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    let release_timeout_context = XrFinalizeTimeoutContext::projection(payload, submit_context);
    let release_timeout_recorder = Arc::clone(flight_recorder);
    let release_watchdog = BlockingCallWatchdog::arm_shutdown_aware_with_timeout_hook(
        END_FRAME_WATCHDOG_TIMEOUT,
        "xr::release_image",
        Arc::clone(&payload.shutdown_requested),
        move || {
            record_xr_finalize_timeout(
                release_timeout_recorder,
                GpuFlightOpenXrCall::ReleaseImage,
                "openxr-release-image-stall",
                release_timeout_context,
            );
        },
    );
    let release_res = release_swapchain_image_under_gate(gate, &payload.swapchain);
    release_watchdog.disarm();
    release_res
}

/// Records the start of a projection depth swapchain release.
fn record_release_depth_image_started(
    payload: &XrProjectionFinalize,
    depth: &XrProjectionDepthFinalize,
    flight_recorder: &GpuFlightRecorder,
) {
    flight_recorder.record_openxr_call_started(
        GpuFlightOpenXrCall::ReleaseImage,
        Some(depth.image_index),
        Some(payload.predicted_display_time.as_nanos()),
    );
}

/// Releases a projection depth swapchain image with a timeout hook.
fn release_depth_image_with_watchdog(
    gate: &GpuQueueAccessGate,
    payload: &XrProjectionFinalize,
    depth: &XrProjectionDepthFinalize,
    submit_context: XrFinalizeSubmitContext,
    flight_recorder: &Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    let release_timeout_context =
        XrFinalizeTimeoutContext::projection_depth(payload, depth, submit_context);
    let release_timeout_recorder = Arc::clone(flight_recorder);
    let release_watchdog = BlockingCallWatchdog::arm_shutdown_aware_with_timeout_hook(
        END_FRAME_WATCHDOG_TIMEOUT,
        "xr::release_depth_image",
        Arc::clone(&payload.shutdown_requested),
        move || {
            record_xr_finalize_timeout(
                release_timeout_recorder,
                GpuFlightOpenXrCall::ReleaseImage,
                "openxr-release-depth-image-stall",
                release_timeout_context,
            );
        },
    );
    let release_res = release_swapchain_image_under_gate(gate, &depth.swapchain);
    release_watchdog.disarm();
    release_res
}

/// Records the result of a projection depth swapchain release.
fn record_release_depth_image_result(
    payload: &XrProjectionFinalize,
    depth: &XrProjectionDepthFinalize,
    flight_recorder: &GpuFlightRecorder,
    release_res: Result<(), xr::sys::Result>,
) {
    flight_recorder.record_openxr_call_result(
        GpuFlightOpenXrCall::ReleaseImage,
        flight_result(release_res),
        Some(depth.image_index),
        Some(payload.predicted_display_time.as_nanos()),
    );
}

/// Records the result of a projection swapchain release.
fn record_release_image_result(
    payload: &XrProjectionFinalize,
    flight_recorder: &GpuFlightRecorder,
    release_res: Result<(), xr::sys::Result>,
) {
    flight_recorder.record_openxr_call_result(
        GpuFlightOpenXrCall::ReleaseImage,
        flight_result(release_res),
        Some(payload.image_index),
        Some(payload.predicted_display_time.as_nanos()),
    );
}

/// Records the start of projection end-frame submission.
fn record_end_frame_projection_started(
    payload: &XrProjectionFinalize,
    flight_recorder: &GpuFlightRecorder,
) {
    flight_recorder.record_openxr_call_started(
        GpuFlightOpenXrCall::EndFrameProjection,
        Some(payload.image_index),
        Some(payload.predicted_display_time.as_nanos()),
    );
}

/// Records the result of projection end-frame submission.
fn record_end_frame_projection_result(
    payload: &XrProjectionFinalize,
    flight_recorder: &GpuFlightRecorder,
    res: Result<(), xr::sys::Result>,
) {
    flight_recorder.record_openxr_call_result(
        GpuFlightOpenXrCall::EndFrameProjection,
        flight_result(res),
        Some(payload.image_index),
        Some(payload.predicted_display_time.as_nanos()),
    );
}

/// Converts a fallible OpenXR result to a compact flight-recorder result.
fn flight_result(result: Result<(), xr::sys::Result>) -> GpuFlightCallResult {
    result.map_or_else(GpuFlightCallResult::failed_debug, |()| {
        GpuFlightCallResult::Ok
    })
}

/// Records projection finalize state for native crash handlers.
fn set_projection_crash_context(
    payload: &XrProjectionFinalize,
    submit_context: XrFinalizeSubmitContext,
) {
    let extent = payload.rect.extent;
    let width = u32::try_from(extent.width).ok();
    let height = u32::try_from(extent.height).ok();
    crash_context::set_xr_finalize_state(
        crash_context::XrFinalizeKind::Projection,
        Some(payload.image_index),
        submit_context.frame_seq,
        submit_context.command_buffers,
        width.zip(height),
        Some(payload.predicted_display_time.as_nanos()),
    );
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
    submit_context: XrFinalizeSubmitContext,
    flight_recorder: Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    profiling::scope!("driver::xr_end_frame");
    let v0 = &payload.views[0];
    let v1 = &payload.views[1];
    let pose0 = sanitize_pose_for_end_frame(v0.pose);
    let pose1 = sanitize_pose_for_end_frame(v1.pose);
    let timeout_context = XrFinalizeTimeoutContext::projection(payload, submit_context);
    let wd = BlockingCallWatchdog::arm_shutdown_aware_with_timeout_hook(
        END_FRAME_WATCHDOG_TIMEOUT,
        "xr::end_frame_projection",
        Arc::clone(&payload.shutdown_requested),
        move || {
            record_xr_finalize_timeout(
                flight_recorder,
                GpuFlightOpenXrCall::EndFrameProjection,
                "openxr-end-frame-projection-stall",
                timeout_context,
            );
        },
    );
    let res = {
        let _gate = gate.lock();
        let swapchain_guard = payload.swapchain.lock();
        let depth_guard = payload.depth.as_ref().map(|depth| depth.swapchain.lock());
        let depth_infos = depth_guard.as_ref().and_then(|guard| {
            payload
                .depth
                .as_ref()
                .map(|depth| build_depth_infos(depth, guard))
        });
        let projection_views = build_projection_views(
            payload,
            &swapchain_guard,
            [pose0, pose1],
            depth_infos.as_ref(),
        );
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

fn build_projection_views<'a>(
    payload: &XrProjectionFinalize,
    color_swapchain: &'a xr::Swapchain<xr::Vulkan>,
    poses: [xr::Posef; 2],
    depth_infos: Option<&'a [xr::sys::CompositionLayerDepthInfoKHR; 2]>,
) -> [CompositionLayerProjectionView<'a, xr::Vulkan>; 2] {
    [
        build_projection_view(
            color_swapchain,
            payload.rect,
            0,
            poses[0],
            payload.views[0].fov,
            depth_infos.map(|infos| &infos[0]),
        ),
        build_projection_view(
            color_swapchain,
            payload.rect,
            1,
            poses[1],
            payload.views[1].fov,
            depth_infos.map(|infos| &infos[1]),
        ),
    ]
}

fn build_projection_view<'a>(
    color_swapchain: &'a xr::Swapchain<xr::Vulkan>,
    rect: xr::Rect2Di,
    layer: u32,
    pose: xr::Posef,
    fov: xr::Fovf,
    depth_info: Option<&'a xr::sys::CompositionLayerDepthInfoKHR>,
) -> CompositionLayerProjectionView<'a, xr::Vulkan> {
    let next = depth_info.map_or(ptr::null(), |info| ptr::from_ref(info).cast::<c_void>());
    let sub_image = SwapchainSubImage::new()
        .swapchain(color_swapchain)
        .image_array_index(layer)
        .image_rect(rect)
        .into_raw();
    let raw = xr::sys::CompositionLayerProjectionView {
        ty: xr::sys::CompositionLayerProjectionView::TYPE,
        next,
        pose,
        fov,
        sub_image,
    };
    // SAFETY: `raw.sub_image.swapchain` references `color_swapchain`, which is locked until
    // `xrEndFrame` returns, and `raw.next` either is null or points at an element of
    // `depth_infos`, which also lives until `xrEndFrame` consumes the projection layer.
    unsafe { CompositionLayerProjectionView::from_raw(raw) }
}

fn build_depth_infos(
    depth: &XrProjectionDepthFinalize,
    depth_swapchain: &xr::Swapchain<xr::Vulkan>,
) -> [xr::sys::CompositionLayerDepthInfoKHR; 2] {
    [
        build_depth_info(depth, depth_swapchain, 0),
        build_depth_info(depth, depth_swapchain, 1),
    ]
}

fn build_depth_info(
    depth: &XrProjectionDepthFinalize,
    depth_swapchain: &xr::Swapchain<xr::Vulkan>,
    layer: u32,
) -> xr::sys::CompositionLayerDepthInfoKHR {
    let sub_image = SwapchainSubImage::new()
        .swapchain(depth_swapchain)
        .image_array_index(layer)
        .image_rect(depth.rect)
        .into_raw();
    xr::sys::CompositionLayerDepthInfoKHR {
        ty: xr::sys::CompositionLayerDepthInfoKHR::TYPE,
        next: ptr::null(),
        sub_image,
        min_depth: depth.depth_info.min_depth,
        max_depth: depth.depth_info.max_depth,
        near_z: depth.depth_info.near_z,
        far_z: depth.depth_info.far_z,
    }
}

/// Calls `xrEndFrame` with no composition layers under the queue access gate.
fn end_frame_empty(
    gate: &GpuQueueAccessGate,
    frame_stream: &Mutex<xr::FrameStream<xr::Vulkan>>,
    env_blend_mode: xr::EnvironmentBlendMode,
    predicted_display_time: xr::Time,
    shutdown_requested: &Arc<AtomicBool>,
    submit_context: XrFinalizeSubmitContext,
    flight_recorder: Arc<GpuFlightRecorder>,
) -> Result<(), xr::sys::Result> {
    profiling::scope!("driver::xr_end_frame_empty");
    let timeout_context = XrFinalizeTimeoutContext::empty(predicted_display_time, submit_context);
    let wd = BlockingCallWatchdog::arm_shutdown_aware_with_timeout_hook(
        END_FRAME_WATCHDOG_TIMEOUT,
        "xr::end_frame_empty",
        Arc::clone(shutdown_requested),
        move || {
            record_xr_finalize_timeout(
                flight_recorder,
                GpuFlightOpenXrCall::EndFrameEmpty,
                "openxr-end-frame-empty-stall",
                timeout_context,
            );
        },
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

/// Copyable context emitted when an OpenXR finalize call exceeds its watchdog timeout.
#[derive(Clone, Copy, Debug)]
struct XrFinalizeTimeoutContext {
    /// Human-readable finalize kind.
    kind: &'static str,
    /// Swapchain image index for projection finalizes.
    image_index: Option<u32>,
    /// Frame sequence assigned to the driver batch.
    frame_seq: u64,
    /// Command buffers in the driver batch.
    command_buffers: usize,
    /// OpenXR swapchain extent for projection finalizes.
    extent: Option<(u32, u32)>,
    /// Predicted display time in OpenXR nanoseconds.
    predicted_display_time_nanos: Option<i64>,
}

impl XrFinalizeTimeoutContext {
    /// Builds timeout context for stereo projection end-frame.
    fn projection(payload: &XrProjectionFinalize, submit_context: XrFinalizeSubmitContext) -> Self {
        let extent = payload.rect.extent;
        let width = u32::try_from(extent.width).ok();
        let height = u32::try_from(extent.height).ok();
        Self {
            kind: "projection",
            image_index: Some(payload.image_index),
            frame_seq: submit_context.frame_seq,
            command_buffers: submit_context.command_buffers,
            extent: width.zip(height),
            predicted_display_time_nanos: Some(payload.predicted_display_time.as_nanos()),
        }
    }

    /// Builds timeout context for stereo projection depth release.
    fn projection_depth(
        payload: &XrProjectionFinalize,
        depth: &XrProjectionDepthFinalize,
        submit_context: XrFinalizeSubmitContext,
    ) -> Self {
        let extent = depth.rect.extent;
        let width = u32::try_from(extent.width).ok();
        let height = u32::try_from(extent.height).ok();
        Self {
            kind: "projection-depth",
            image_index: Some(depth.image_index),
            frame_seq: submit_context.frame_seq,
            command_buffers: submit_context.command_buffers,
            extent: width.zip(height),
            predicted_display_time_nanos: Some(payload.predicted_display_time.as_nanos()),
        }
    }

    /// Builds timeout context for empty end-frame.
    fn empty(predicted_display_time: xr::Time, submit_context: XrFinalizeSubmitContext) -> Self {
        Self {
            kind: "empty",
            image_index: None,
            frame_seq: submit_context.frame_seq,
            command_buffers: submit_context.command_buffers,
            extent: None,
            predicted_display_time_nanos: Some(predicted_display_time.as_nanos()),
        }
    }
}

/// Records a detailed timeout event and flushes logs before the process can be aborted.
fn record_xr_finalize_timeout(
    flight_recorder: Arc<GpuFlightRecorder>,
    call: GpuFlightOpenXrCall,
    reason: &'static str,
    context: XrFinalizeTimeoutContext,
) {
    logger::error!(
        "driver: OpenXR finalize timed out: reason={} call={} kind={} frame_seq={} command_buffers={} image_index={} extent={} predicted_time_ns={}",
        reason,
        call,
        context.kind,
        context.frame_seq,
        context.command_buffers,
        optional_u32(context.image_index),
        optional_extent(context.extent),
        optional_i64(context.predicted_display_time_nanos),
    );
    flight_recorder.record_openxr_call_timeout(
        call,
        reason,
        context.image_index,
        context.predicted_display_time_nanos,
    );
    flight_recorder.dump_once(reason);
    logger::flush();
}

/// Formats an optional unsigned integer for log output.
fn optional_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

/// Formats an optional signed integer for log output.
fn optional_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

/// Formats an optional extent for log output.
fn optional_extent(value: Option<(u32, u32)>) -> String {
    value.map_or_else(
        || "none".to_owned(),
        |(width, height)| format!("{width}x{height}"),
    )
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

    #[test]
    fn finalize_work_can_cross_driver_thread_boundary() {
        fn assert_send<T: Send>() {}

        assert_send::<XrFinalizeWork>();
        assert_send::<XrProjectionFinalize>();
    }

    #[test]
    fn composition_depth_info_matches_reverse_z_clip_planes() {
        let info = XrCompositionDepthInfo::from_clip_planes(0.05, 500.0);

        assert_eq!(info.min_depth, 0.0);
        assert_eq!(info.max_depth, 1.0);
        assert_eq!(info.near_z, 500.0);
        assert_eq!(info.far_z, 0.05);
    }
}
