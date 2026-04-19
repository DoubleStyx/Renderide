//! Narrow traits so OpenXR integration does not depend on the full [`crate::runtime::RendererRuntime`] surface.
//!
//! Implementations live on [`crate::runtime::RendererRuntime`] in [`crate::runtime`].

use glam::{Mat4, Quat, Vec3};

use crate::gpu::GpuContext;
use crate::render_graph::{ExternalFrameTargets, GraphExecuteError};
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
    /// Active main space root scale for [`crate::render_graph::camera::effective_head_output_clip_planes`].
    fn scene_root_scale_for_clip(&self) -> Option<Vec3>;
    /// Same rig alignment as [`crate::xr::tracking_space_to_world_matrix`].
    fn world_from_tracking(&self, center_pose_tracking: Option<(Vec3, Quat)>) -> Mat4;
    /// Updates legacy head-output rig transform for overlay alignment and host IPC.
    fn set_head_output_transform(&mut self, transform: Mat4);
    /// Stores per-eye view–projection for stereo world mesh draws and clustering.
    fn set_stereo_view_proj(&mut self, vp: Option<(Mat4, Mat4)>);
    /// Per-eye **view-only** matrices (world-to-view, handedness-fixed) for stereo clustering.
    fn set_stereo_views(&mut self, views: Option<(Mat4, Mat4)>);
    /// Hook when OpenXR `wait_frame` returns an error (recoverable; tick may skip XR work).
    fn note_openxr_wait_frame_failed(&mut self) {}
    /// Hook when OpenXR `locate_views` fails while the runtime expected rendering views.
    fn note_openxr_locate_views_failed(&mut self) {}
}

/// Multiview submission path that reuses the render graph with external stereo targets.
pub trait XrMultiviewFrameRenderer: XrHostCameraSync {
    /// Renders to OpenXR array color / depth ([`RenderBackend::execute_frame_graph_external_multiview`](crate::backend::RenderBackend::execute_frame_graph_external_multiview)).
    ///
    /// When `skip_hi_z_begin_readback` is `true`, the caller has already drained Hi-Z readbacks
    /// this tick (see [`crate::runtime::RendererRuntime::drain_hi_z_readback`]).
    fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        external: ExternalFrameTargets<'_>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError>;
}
