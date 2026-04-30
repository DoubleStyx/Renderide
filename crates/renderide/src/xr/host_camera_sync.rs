//! Narrow traits so OpenXR integration does not depend on the full [`crate::runtime::RendererRuntime`] surface.
//!
//! Implementations live on [`crate::runtime::RendererRuntime`] in [`crate::runtime`].

use glam::{Mat4, Quat, Vec3};

use crate::gpu::GpuContext;
use crate::render_graph::{ExternalFrameTargets, GraphExecuteError, StereoViewMatrices};
use crate::shared::HeadOutputDevice;

/// Read/write hooks for per-eye matrices and head-output positioning used by OpenXR frame ticks.
pub trait XrHostCameraSync {
    /// Effective near clip plane distance for the current frame (world units).
    fn near_clip(&self) -> f32;
    /// Effective far clip plane distance for the current frame (world units).
    fn far_clip(&self) -> f32;
    /// Host-selected head output device (desktop vs HMD class).
    fn output_device(&self) -> HeadOutputDevice;
    /// Whether VR submission is active this frame (OpenXR session running).
    fn vr_active(&self) -> bool;
    /// Active main space root scale for [`crate::camera::effective_head_output_clip_planes`].
    fn scene_root_scale_for_clip(&self) -> Option<Vec3>;
    /// Same rig alignment as [`crate::xr::tracking_space_to_world_matrix`].
    fn world_from_tracking(&self, center_pose_tracking: Option<(Vec3, Quat)>) -> Mat4;
    /// Updates the head-output rig transform used for overlay alignment and host IPC replies.
    fn set_head_output_transform(&mut self, transform: Mat4);
    /// Stores the center-eye world position used by mono fallback paths and CPU view sorting.
    fn set_eye_world_position(&mut self, position: Vec3);
    /// Stores per-eye stereo matrices used by the HMD multiview view this tick.
    fn set_stereo(&mut self, stereo: Option<StereoViewMatrices>);
    /// Hook when OpenXR `wait_frame` returns an error (recoverable; tick may skip XR work).
    fn note_openxr_wait_frame_failed(&mut self) {}
    /// Hook when OpenXR `locate_views` fails while the runtime expected rendering views.
    fn note_openxr_locate_views_failed(&mut self) {}
}

/// Per-tick render entry points used by the OpenXR frame submit helper.
///
/// Split from the desktop inherent entry so the VR path does not need to encode mode selection
/// with boolean flags. [`Self::submit_hmd_view`] renders the HMD stereo view plus any active
/// secondary render-texture cameras in a single submit; [`Self::submit_secondary_only`] is used
/// when HMD swapchain acquire failed but secondary RTs should still render.
pub trait XrFrameRenderer: XrHostCameraSync {
    /// Records and submits the compiled render graph for the HMD stereo view plus all active
    /// secondary render-texture cameras. The HMD view replaces the main camera this tick.
    fn submit_hmd_view(
        &mut self,
        gpu: &mut GpuContext,
        hmd: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError>;

    /// Records and submits only the active secondary render-texture cameras. Used when the HMD
    /// swapchain acquire failed; the desktop mirror stays on its last frame.
    fn submit_secondary_only(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError>;
}
