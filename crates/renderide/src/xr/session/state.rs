//! OpenXR session frame loop: wait, begin, locate views, end.
//!
//! Tracks the latest [`xr::SessionState`] from the runtime so submission gates (compositor
//! visibility, exit propagation) can react to lifecycle transitions, and maintains a `frame_open`
//! flag so every successful `xrBeginFrame` is matched by exactly one `xrEndFrame`. Entry points
//! that call the compositor (`xrEndFrame`) are wrapped with an
//! [`end_frame_watchdog::EndFrameWatchdog`] so runtime stalls surface as `logger::error!` lines
//! instead of silent freezes.

use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};
use std::time::Duration;

use super::end_frame_watchdog::EndFrameWatchdog;

/// Deadline for a single `xrEndFrame` call before the watchdog logs a compositor stall.
///
/// 500 ms is an order of magnitude above normal VR frame budgets (≤ ~16 ms at 60 Hz, ~11 ms at
/// 90 Hz) while short enough that a true freeze surfaces within one log-visible interval.
const END_FRAME_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);

/// Renderer-local projection of [`xr::SessionState`] used for exhaustive matching on
/// compositor visibility without depending on the raw OpenXR newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackedSessionState {
    /// No `SessionStateChanged` event has been observed yet.
    Unknown,
    /// Runtime-side session exists but no rendering is expected.
    Idle,
    /// Runtime signalled the app should call `xrBeginSession`.
    Ready,
    /// Frame loop is running but the compositor is not displaying the app.
    Synchronized,
    /// Compositor is displaying frames but the app does not have input focus.
    Visible,
    /// Compositor is displaying frames and the app has input focus.
    Focused,
    /// Runtime signalled the app should call `xrEndSession`.
    Stopping,
    /// Runtime plans to lose the session; the app should exit.
    LossPending,
    /// Runtime signalled the app to exit.
    Exiting,
}

/// Projects a raw [`xr::SessionState`] onto [`TrackedSessionState`]; unknown numeric values fall
/// back to [`TrackedSessionState::Unknown`] so future OpenXR additions don't panic.
pub(super) fn tracked_from_xr(state: xr::SessionState) -> TrackedSessionState {
    match state {
        xr::SessionState::IDLE => TrackedSessionState::Idle,
        xr::SessionState::READY => TrackedSessionState::Ready,
        xr::SessionState::SYNCHRONIZED => TrackedSessionState::Synchronized,
        xr::SessionState::VISIBLE => TrackedSessionState::Visible,
        xr::SessionState::FOCUSED => TrackedSessionState::Focused,
        xr::SessionState::STOPPING => TrackedSessionState::Stopping,
        xr::SessionState::LOSS_PENDING => TrackedSessionState::LossPending,
        xr::SessionState::EXITING => TrackedSessionState::Exiting,
        _ => TrackedSessionState::Unknown,
    }
}

/// Whether the compositor is currently displaying the app and will accept projection layers.
pub(super) fn is_visible_tracked(state: TrackedSessionState) -> bool {
    matches!(
        state,
        TrackedSessionState::Visible | TrackedSessionState::Focused
    )
}

/// Outcome of inspecting one `xrPollEvent` result, decoded into an owned enum so the caller can
/// release the event borrow before invoking `&mut self` side-effects.
enum PollEventAction {
    /// Session transitioned to a new [`xr::SessionState`]; apply via
    /// [`XrSessionState::handle_session_state_change`].
    SessionStateChanged(xr::SessionState),
    /// Instance is being destroyed; renderer must exit.
    InstanceLoss,
    /// Controller interaction profile was re-bound; informational only.
    InteractionProfileChanged,
    /// Event variant not reacted to by this renderer.
    Ignore,
}

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
    /// prevents orphaned frames on error paths.
    pub(super) frame_open: bool,
    /// Set when the runtime requests teardown (`EXITING` / `LOSS_PENDING` / instance loss);
    /// read by the app loop to trigger `event_loop.exit()`.
    pub(super) exit_requested: bool,
    /// Blocks until the compositor signals frame timing.
    pub(super) frame_wait: xr::FrameWaiter,
    /// Submits composition layers to the compositor.
    pub(super) frame_stream: xr::FrameStream<xr::Vulkan>,
    /// Stage reference space for view and controller pose location.
    pub(super) stage: xr::Space,
    /// Scratch buffer for `xrPollEvent`.
    pub(super) event_storage: xr::EventDataBuffer,
}

/// Bundle of values needed to construct [`XrSessionState`] — `new` takes this instead of seven
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
            frame_open: false,
            exit_requested: false,
            frame_wait: desc.frame_wait,
            frame_stream: desc.frame_stream,
            stage: desc.stage,
            event_storage: xr::EventDataBuffer::new(),
        }
    }

    /// Poll events and return `false` if the session should exit.
    ///
    /// Callers may also read [`Self::exit_requested`] directly; both signals are kept in sync so a
    /// dropped return value (as at [`crate::xr::app_integration::openxr_begin_frame_tick`]) no
    /// longer silently strands the app in a terminating session.
    pub fn poll_events(&mut self) -> Result<bool, xr::sys::Result> {
        loop {
            // Bind the next event in an inner scope so its borrow on `self.event_storage` ends
            // before we invoke any `&mut self` state-change side-effects below.
            let action = {
                let Some(event) = self.xr_instance.poll_event(&mut self.event_storage)? else {
                    break;
                };
                use xr::Event::*;
                match event {
                    SessionStateChanged(e) => PollEventAction::SessionStateChanged(e.state()),
                    InstanceLossPending(_) => PollEventAction::InstanceLoss,
                    InteractionProfileChanged(_) => PollEventAction::InteractionProfileChanged,
                    _ => PollEventAction::Ignore,
                }
            };
            match action {
                PollEventAction::SessionStateChanged(state) => {
                    if !self.handle_session_state_change(state)? {
                        return Ok(false);
                    }
                }
                PollEventAction::InstanceLoss => {
                    self.exit_requested = true;
                    return Ok(false);
                }
                PollEventAction::InteractionProfileChanged => {
                    logger::info!("OpenXR interaction profile changed");
                }
                PollEventAction::Ignore => {}
            }
        }
        Ok(!self.exit_requested)
    }

    /// Applies a `SessionStateChanged` event, running any required runtime side-effects
    /// (`xrBeginSession` / `xrEndSession`). Returns `Ok(false)` on terminal transitions so the
    /// caller can break out of its event loop.
    fn handle_session_state_change(
        &mut self,
        new_state: xr::SessionState,
    ) -> Result<bool, xr::sys::Result> {
        let new_tracked = tracked_from_xr(new_state);
        if new_tracked != self.last_session_state {
            logger::info!(
                "OpenXR session state: {:?} -> {:?}",
                self.last_session_state,
                new_tracked
            );
        }
        self.last_session_state = new_tracked;
        match new_state {
            xr::SessionState::READY => {
                self.session
                    .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                self.session_running = true;
                Ok(true)
            }
            xr::SessionState::STOPPING => {
                self.session.end()?;
                self.session_running = false;
                Ok(true)
            }
            xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                self.exit_requested = true;
                Ok(false)
            }
            _ => Ok(true),
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
    /// `xrEndFrame`).
    pub fn frame_open(&self) -> bool {
        self.frame_open
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
        &self.stage
    }

    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    ///
    /// On a successful `frame_stream.begin()` sets [`Self::frame_open`] so the outer loop knows a
    /// matching `end_frame_*` must be called.
    pub fn wait_frame(&mut self) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if !self.session_running {
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        self.frame_stream.begin()?;
        self.frame_open = true;
        Ok(Some(state))
    }

    /// Ends the frame with no composition layers (mirror path, or visibility fallback).
    pub fn end_frame_empty(
        &mut self,
        predicted_display_time: xr::Time,
    ) -> Result<(), xr::sys::Result> {
        let wd = EndFrameWatchdog::arm(END_FRAME_WATCHDOG_TIMEOUT, "end_frame_empty");
        let res = self
            .frame_stream
            .end(predicted_display_time, self.environment_blend_mode, &[]);
        self.frame_open = false;
        wd.disarm();
        res
    }

    /// Ends the frame via [`Self::end_frame_empty`] only if a frame scope is currently open; a
    /// no-op otherwise. Error paths in `xr::app_integration` call this after bailing out of HMD
    /// submit so the begin/end frame contract is honoured regardless of where submission failed.
    pub fn end_frame_if_open(
        &mut self,
        predicted_display_time: xr::Time,
    ) -> Result<(), xr::sys::Result> {
        if !self.frame_open {
            return Ok(());
        }
        self.end_frame_empty(predicted_display_time)
    }

    /// Submits a stereo projection layer referencing the acquired swapchain image (`image_rect` in pixels).
    ///
    /// For the primary stereo view configuration (`PRIMARY_STEREO`), `views[0]` is the left eye and
    /// `views[1]` the right eye. Composition layer 0 / `image_array_index` 0 is the left eye, layer 1
    /// / index 1 the right eye, matching multiview `view_index` in the stereo path.
    pub fn end_frame_projection(
        &mut self,
        predicted_display_time: xr::Time,
        swapchain: &xr::Swapchain<xr::Vulkan>,
        views: &[xr::View],
        image_rect: xr::Rect2Di,
    ) -> Result<(), xr::sys::Result> {
        profiling::scope!("xr::end_frame");
        if views.len() < 2 {
            return self.end_frame_empty(predicted_display_time);
        }
        let v0 = &views[0]; // left eye
        let v1 = &views[1]; // right eye
        let pose0 = sanitize_pose_for_end_frame(v0.pose);
        let pose1 = sanitize_pose_for_end_frame(v1.pose);
        let projection_views = [
            CompositionLayerProjectionView::new()
                .pose(pose0)
                .fov(v0.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(0)
                        .image_rect(image_rect),
                ),
            CompositionLayerProjectionView::new()
                .pose(pose1)
                .fov(v1.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(1)
                        .image_rect(image_rect),
                ),
        ];
        let layer = CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&projection_views);
        let wd = EndFrameWatchdog::arm(END_FRAME_WATCHDOG_TIMEOUT, "end_frame_projection");
        let res = self.frame_stream.end(
            predicted_display_time,
            self.environment_blend_mode,
            &[&layer],
        );
        self.frame_open = false;
        wd.disarm();
        res
    }

    /// Locates stereo views for the predicted display time.
    pub fn locate_views(
        &self,
        predicted_display_time: xr::Time,
    ) -> Result<Vec<xr::View>, xr::sys::Result> {
        let (_, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            predicted_display_time,
            &self.stage,
        )?;
        Ok(views)
    }
}

/// OpenXR requires a unit quaternion; some runtimes briefly report `(0,0,0,0)`, which makes
/// `xrEndFrame` fail with `XR_ERROR_POSE_INVALID`.
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
mod state_machine_tests {
    //! State-machine logic testable without a live OpenXR runtime. Frame-stream integration
    //! (`wait_frame` / `end_frame_*`) requires a real `xr::FrameWaiter` + `xr::FrameStream` and is
    //! exercised by the VR integration path instead of unit tests.
    use super::*;

    #[test]
    fn tracked_from_xr_covers_every_documented_variant() {
        assert_eq!(
            tracked_from_xr(xr::SessionState::IDLE),
            TrackedSessionState::Idle
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::READY),
            TrackedSessionState::Ready
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::SYNCHRONIZED),
            TrackedSessionState::Synchronized
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::VISIBLE),
            TrackedSessionState::Visible
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::FOCUSED),
            TrackedSessionState::Focused
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::STOPPING),
            TrackedSessionState::Stopping
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::LOSS_PENDING),
            TrackedSessionState::LossPending
        );
        assert_eq!(
            tracked_from_xr(xr::SessionState::EXITING),
            TrackedSessionState::Exiting
        );
    }

    #[test]
    fn is_visible_tracked_only_true_for_visible_and_focused() {
        for (s, expected) in [
            (TrackedSessionState::Unknown, false),
            (TrackedSessionState::Idle, false),
            (TrackedSessionState::Ready, false),
            (TrackedSessionState::Synchronized, false),
            (TrackedSessionState::Visible, true),
            (TrackedSessionState::Focused, true),
            (TrackedSessionState::Stopping, false),
            (TrackedSessionState::LossPending, false),
            (TrackedSessionState::Exiting, false),
        ] {
            assert_eq!(is_visible_tracked(s), expected, "state {s:?}");
        }
    }
}
