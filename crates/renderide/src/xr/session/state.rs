//! OpenXR session frame loop: wait, begin, locate views, end.
//!
//! Tracks the latest [`xr::SessionState`] from the runtime so submission gates (compositor
//! visibility, exit propagation) can react to lifecycle transitions, and maintains a `frame_open`
//! flag so every successful `xrBeginFrame` is matched by exactly one `xrEndFrame`. Entry points
//! that call the compositor (`xrEndFrame`) are wrapped with an
//! [`super::end_frame_watchdog::EndFrameWatchdog`] so runtime stalls surface as `logger::error!`
//! lines instead of silent freezes.

mod frame_loop;
mod lifecycle;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use openxr as xr;
use parking_lot::Mutex;

pub use lifecycle::TrackedSessionState;
use lifecycle::is_visible_tracked;

use crate::gpu::driver_thread::{XrFinalizeErrorSlot, XrFinalizeReceiver};

/// Owns OpenXR session objects (constructed in [`super::super::bootstrap::init_wgpu_openxr`]).
pub struct XrSessionState {
    /// OpenXR instance (retained for the session lifetime).
    pub(super) xr_instance: xr::Instance,
    /// Dropped before [`Self::xr_instance`] so the messenger handle is destroyed first; held only
    /// for this Drop ordering, hence never read after construction.
    #[expect(dead_code, reason = "drop-order-only field; see doc comment above")]
    pub(super) openxr_debug_messenger: Option<super::super::debug_utils::OpenxrDebugUtilsMessenger>,
    /// Blend mode used for `xrEndFrame`.
    pub(super) environment_blend_mode: xr::EnvironmentBlendMode,
    /// Vulkan-backed session.
    pub(super) session: xr::Session<xr::Vulkan>,
    /// Whether `xrBeginSession` has been called and `xrEndSession` has not.
    pub(super) session_running: bool,
    /// Latest [`xr::SessionState`] observed via `SessionStateChanged`.
    pub(super) last_session_state: TrackedSessionState,
    /// `true` between a successful `frame_stream.begin()` and the matching `frame_stream.end()`;
    /// prevents orphaned frames on error paths. Stored as an atomic so the driver thread
    /// can clear it inside the deferred `xrEndFrame` finalize.
    pub(super) frame_open: Arc<AtomicBool>,
    /// Set when the runtime requests teardown (`EXITING` / `LOSS_PENDING` / instance loss);
    /// read by the app loop to trigger `event_loop.exit()`.
    pub(super) exit_requested: bool,
    /// Blocks until the compositor signals frame timing.
    pub(super) frame_wait: xr::FrameWaiter,
    /// Submits composition layers to the compositor. Behind a [`Mutex`] so the driver
    /// thread can call `frame_stream.end()` for the deferred finalize while the main
    /// thread keeps `frame_stream.begin()` on the next tick.
    pub(super) frame_stream: Arc<Mutex<xr::FrameStream<xr::Vulkan>>>,
    /// Stage reference space for view and controller pose location. Wrapped in [`Arc`]
    /// so the projection-layer finalize on the driver thread can hold a reference
    /// independent of the main-thread session borrow.
    pub(super) stage: Arc<xr::Space>,
    /// Scratch buffer for `xrPollEvent`.
    pub(super) event_storage: xr::EventDataBuffer,
    /// Receiver for the in-flight finalize signal queued by the previous tick. The next
    /// `wait_frame` consumes this before calling `xrBeginFrame` so begin/end ordering is
    /// preserved across the driver-thread handoff.
    pub(super) pending_finalize: Option<XrFinalizeReceiver>,
    /// First-error-wins slot the driver thread writes to on a finalize failure; the next
    /// `wait_frame` drains this and returns the recorded error so the existing recovery
    /// path runs one tick later than it would have on the main thread.
    pub(super) finalize_error_slot: XrFinalizeErrorSlot,
}

/// Bundle of values needed to construct [`XrSessionState`] - `new` takes this instead of seven
/// separate parameters to keep the bootstrap signature readable.
pub(in crate::xr) struct XrSessionStateDescriptor {
    /// OpenXR instance (retained for the session lifetime).
    pub(in crate::xr) xr_instance: xr::Instance,
    /// Debug-utils messenger; must drop before the instance. See [`XrSessionState`].
    pub(in crate::xr) openxr_debug_messenger:
        Option<super::super::debug_utils::OpenxrDebugUtilsMessenger>,
    /// Blend mode used for `xrEndFrame`.
    pub(in crate::xr) environment_blend_mode: xr::EnvironmentBlendMode,
    /// Vulkan-backed session.
    pub(in crate::xr) session: xr::Session<xr::Vulkan>,
    /// Frame waiter from the session tuple.
    pub(in crate::xr) frame_wait: xr::FrameWaiter,
    /// Frame stream from the session tuple.
    pub(in crate::xr) frame_stream: xr::FrameStream<xr::Vulkan>,
    /// Stage reference space used for view + controller pose location.
    pub(in crate::xr) stage: xr::Space,
}

impl XrSessionState {
    /// Constructed only from [`crate::xr::bootstrap::init_wgpu_openxr`].
    pub(in crate::xr) fn new(desc: XrSessionStateDescriptor) -> Self {
        Self {
            xr_instance: desc.xr_instance,
            openxr_debug_messenger: desc.openxr_debug_messenger,
            environment_blend_mode: desc.environment_blend_mode,
            session: desc.session,
            session_running: false,
            last_session_state: TrackedSessionState::Unknown,
            frame_open: Arc::new(AtomicBool::new(false)),
            exit_requested: false,
            frame_wait: desc.frame_wait,
            frame_stream: Arc::new(Mutex::new(desc.frame_stream)),
            stage: Arc::new(desc.stage),
            event_storage: xr::EventDataBuffer::new(),
            pending_finalize: None,
            finalize_error_slot: Arc::new(Mutex::new(None)),
        }
    }

    /// Whether the OpenXR session is running (`xrBeginSession` called, `xrEndSession` not yet).
    pub fn session_running(&self) -> bool {
        self.session_running
    }

    /// Latest [`TrackedSessionState`] observed from the runtime; [`TrackedSessionState::Unknown`]
    /// until the first `SessionStateChanged` event.
    pub fn last_session_state(&self) -> TrackedSessionState {
        self.last_session_state
    }

    /// Whether the compositor is currently displaying this app's frames
    /// ([`TrackedSessionState::Visible`] or [`TrackedSessionState::Focused`]). Used to gate real
    /// projection-layer submission; the empty-frame path still runs to satisfy the OpenXR
    /// begin/end frame contract.
    pub fn is_visible(&self) -> bool {
        is_visible_tracked(self.last_session_state)
    }

    /// Whether the runtime has asked the renderer to exit (EXITING / LOSS_PENDING / instance
    /// loss). Checked by the app loop after each `poll_events`.
    pub fn exit_requested(&self) -> bool {
        self.exit_requested
    }

    /// Whether a frame scope is currently open (`xrBeginFrame` called without matching
    /// `xrEndFrame`). Reads through the shared atomic so the value reflects the deferred
    /// finalize on the driver thread as well as `wait_frame` on the main thread.
    pub fn frame_open(&self) -> bool {
        self.frame_open.load(Ordering::Acquire)
    }

    /// OpenXR instance handle (swapchain creation, view enumeration).
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    /// Underlying Vulkan session (swapchain lifetime).
    pub fn xr_vulkan_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.session
    }

    /// Stage reference space used for [`Self::locate_views`] and controller [`xr::Space`] location.
    pub fn stage_space(&self) -> &xr::Space {
        self.stage.as_ref()
    }

    /// Drains and returns any pending finalize error captured by the driver thread.
    ///
    /// Called from `wait_frame` after waiting on [`Self::pending_finalize`]; if a value
    /// is returned the existing recovery path runs one tick later than it would if the
    /// finalize had stayed on the main thread.
    pub(super) fn take_finalize_error(&self) -> Option<xr::sys::Result> {
        self.finalize_error_slot.lock().take()
    }

    /// Builds a stereo-projection finalize payload referencing the just-rendered swapchain
    /// image. The returned receiver should be stored on [`Self::set_pending_finalize`] so
    /// the next `wait_frame` waits on it.
    pub(crate) fn build_projection_finalize(
        &self,
        swapchain: Arc<Mutex<xr::Swapchain<xr::Vulkan>>>,
        predicted_display_time: xr::Time,
        views: [xr::View; 2],
        rect: xr::Rect2Di,
    ) -> (
        crate::gpu::driver_thread::XrFinalizeWork,
        XrFinalizeReceiver,
    ) {
        let (signal, rx) = crate::gpu::driver_thread::XrFinalizeSignal::new();
        let payload = crate::gpu::driver_thread::XrProjectionFinalize {
            swapchain,
            frame_stream: Arc::clone(&self.frame_stream),
            stage: Arc::clone(&self.stage),
            env_blend_mode: self.environment_blend_mode,
            predicted_display_time,
            views,
            rect,
            frame_open: Arc::clone(&self.frame_open),
        };
        let work = crate::gpu::driver_thread::XrFinalizeWork {
            kind: crate::gpu::driver_thread::XrFinalizeKind::Projection(Box::new(payload)),
            signal,
            error_slot: Arc::clone(&self.finalize_error_slot),
        };
        (work, rx)
    }

    /// Builds an empty-end-frame finalize payload, used for recovery paths and ticks where
    /// the HMD render was skipped after `xrBeginFrame` had already opened the frame.
    pub(crate) fn build_empty_finalize(
        &self,
        predicted_display_time: xr::Time,
    ) -> (
        crate::gpu::driver_thread::XrFinalizeWork,
        XrFinalizeReceiver,
    ) {
        let (signal, rx) = crate::gpu::driver_thread::XrFinalizeSignal::new();
        let work = crate::gpu::driver_thread::XrFinalizeWork {
            kind: crate::gpu::driver_thread::XrFinalizeKind::Empty {
                frame_stream: Arc::clone(&self.frame_stream),
                env_blend_mode: self.environment_blend_mode,
                predicted_display_time,
                frame_open: Arc::clone(&self.frame_open),
            },
            signal,
            error_slot: Arc::clone(&self.finalize_error_slot),
        };
        (work, rx)
    }

    /// Stores the receiver matching a finalize payload pushed to the driver thread.
    /// The next `wait_frame` consumes this before issuing `xrBeginFrame`.
    pub(crate) fn set_pending_finalize(&mut self, rx: XrFinalizeReceiver) {
        self.pending_finalize = Some(rx);
    }
}
