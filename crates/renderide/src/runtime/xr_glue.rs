//! [`crate::xr::XrHostCameraSync`] and [`crate::xr::XrFrameRenderer`] for [`RendererRuntime`].
//!
//! Lives next to the type it impls so `xr/` never reaches into runtime internals.

use glam::{Mat4, Quat, Vec3};

use crate::camera::StereoViewMatrices;
use crate::gpu::GpuContext;
use crate::render_graph::ExternalFrameTargets;
use crate::render_graph::GraphExecuteError;
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
            .map_or(Mat4::IDENTITY, |space| {
                crate::xr::tracking_space_to_world_matrix(
                    &space.root_transform,
                    &space.view_transform,
                    space.override_view_position,
                    center_pose_tracking,
                )
            })
    }

    fn set_head_output_transform(&mut self, transform: Mat4) {
        self.host_camera.head_output_transform = transform;
    }

    fn set_eye_world_position(&mut self, position: Vec3) {
        self.host_camera.eye_world_position = Some(position);
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
            crate::runtime::frame_render::FrameRenderMode::VrWithHmd(hmd),
        )
    }

    fn submit_secondary_only(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        RendererRuntime::render_frame(
            self,
            gpu,
            crate::runtime::frame_render::FrameRenderMode::VrSecondariesOnly,
        )
    }
}
