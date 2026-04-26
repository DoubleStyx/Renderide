//! [`crate::xr::XrHostCameraSync`] and [`crate::xr::XrFrameRenderer`] for [`super::RendererRuntime`].

use glam::{Mat4, Quat, Vec3};

use crate::gpu::GpuContext;
use crate::render_graph::ExternalFrameTargets;
use crate::render_graph::{GraphExecuteError, StereoViewMatrices};
use crate::shared::HeadOutputDevice;

use super::RendererRuntime;

impl crate::xr::XrHostCameraSync for RendererRuntime {
    fn near_clip(&self) -> f32 {
        self.host_camera.near_clip
    }

    fn far_clip(&self) -> f32 {
        self.host_camera.far_clip
    }

    fn output_device(&self) -> HeadOutputDevice {
        self.host_camera.output_device
    }

    fn vr_active(&self) -> bool {
        RendererRuntime::vr_active(self)
    }

    fn scene_root_scale_for_clip(&self) -> Option<Vec3> {
        self.scene
            .active_main_space()
            .map(|space| space.root_transform.scale)
    }

    fn world_from_tracking(&self, center_pose_tracking: Option<(Vec3, Quat)>) -> Mat4 {
        self.scene
            .active_main_space()
            .map(|space| {
                crate::xr::tracking_space_to_world_matrix(
                    &space.root_transform,
                    &space.view_transform,
                    space.override_view_position,
                    center_pose_tracking,
                )
            })
            .unwrap_or(Mat4::IDENTITY)
    }

    fn set_head_output_transform(&mut self, transform: Mat4) {
        self.host_camera.head_output_transform = transform;
        // In VR the head pose is also the eye pose (per-eye offsets are applied via stereo
        // view matrices, not the camera_world_pos uniform). Mirror the translation into
        // `eye_world_position` so PBS shaders see the actual eye for view-direction math
        // instead of the desktop-only render-space root.
        self.host_camera.eye_world_position = Some(transform.col(3).truncate());
    }

    fn set_stereo(&mut self, stereo: Option<StereoViewMatrices>) {
        self.host_camera.stereo = stereo;
    }

    fn note_openxr_wait_frame_failed(&mut self) {
        self.xr_wait_frame_failures = self.xr_wait_frame_failures.saturating_add(1);
    }

    fn note_openxr_locate_views_failed(&mut self) {
        self.xr_locate_views_failures = self.xr_locate_views_failures.saturating_add(1);
    }
}

impl crate::xr::XrFrameRenderer for RendererRuntime {
    fn submit_hmd_view(
        &mut self,
        gpu: &mut GpuContext,
        hmd: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        RendererRuntime::render_frame(
            self,
            gpu,
            super::frame_render::FrameRenderMode::VrWithHmd(hmd),
        )
    }

    fn submit_secondary_only(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        RendererRuntime::render_frame(
            self,
            gpu,
            super::frame_render::FrameRenderMode::VrSecondariesOnly,
        )
    }
}
