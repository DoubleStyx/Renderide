//! OpenXR session frame loop: wait, begin, locate views, end.

use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};

/// Owns OpenXR session objects (constructed in [`super::super::bootstrap::init_wgpu_openxr`]).
pub struct XrSessionState {
    pub(super) xr_instance: xr::Instance,
    /// Dropped before [`Self::xr_instance`] so the messenger handle is destroyed first; held only
    /// for this Drop ordering, hence never read after construction.
    #[expect(dead_code, reason = "drop-order-only field; see doc comment above")]
    pub(super) openxr_debug_messenger: Option<super::super::debug_utils::OpenxrDebugUtilsMessenger>,
    pub(super) environment_blend_mode: xr::EnvironmentBlendMode,
    pub(super) session: xr::Session<xr::Vulkan>,
    pub(super) session_running: bool,
    pub(super) frame_wait: xr::FrameWaiter,
    pub(super) frame_stream: xr::FrameStream<xr::Vulkan>,
    pub(super) stage: xr::Space,
    pub(super) event_storage: xr::EventDataBuffer,
}

impl XrSessionState {
    /// Constructed only from [`crate::xr::bootstrap::init_wgpu_openxr`].
    pub(in crate::xr) fn new(
        xr_instance: xr::Instance,
        openxr_debug_messenger: Option<super::super::debug_utils::OpenxrDebugUtilsMessenger>,
        environment_blend_mode: xr::EnvironmentBlendMode,
        session: xr::Session<xr::Vulkan>,
        frame_wait: xr::FrameWaiter,
        frame_stream: xr::FrameStream<xr::Vulkan>,
        stage: xr::Space,
    ) -> Self {
        Self {
            xr_instance,
            openxr_debug_messenger,
            environment_blend_mode,
            session,
            session_running: false,
            frame_wait,
            frame_stream,
            stage,
            event_storage: xr::EventDataBuffer::new(),
        }
    }

    /// Poll events and return `false` if the session should exit.
    pub fn poll_events(&mut self) -> Result<bool, xr::sys::Result> {
        while let Some(event) = self.xr_instance.poll_event(&mut self.event_storage)? {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        self.session
                            .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                        self.session_running = true;
                    }
                    xr::SessionState::STOPPING => {
                        self.session.end()?;
                        self.session_running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                        return Ok(false);
                    }
                    _ => {}
                },
                InstanceLossPending(_) => return Ok(false),
                _ => {}
            }
        }
        Ok(true)
    }

    /// Whether the OpenXR session is running.
    pub fn session_running(&self) -> bool {
        self.session_running
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
    pub fn wait_frame(&mut self) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if !self.session_running {
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        self.frame_stream.begin()?;
        Ok(Some(state))
    }

    /// Ends the frame with no composition layers (mirror path until swapchain submission is wired).
    pub fn end_frame_empty(
        &mut self,
        predicted_display_time: xr::Time,
    ) -> Result<(), xr::sys::Result> {
        self.frame_stream
            .end(predicted_display_time, self.environment_blend_mode, &[])
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
        self.frame_stream.end(
            predicted_display_time,
            self.environment_blend_mode,
            &[&layer],
        )
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
