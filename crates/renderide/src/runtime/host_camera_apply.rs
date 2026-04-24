//! Centralized updates to [`crate::render_graph::HostCameraFrame`] from host [`FrameSubmitData`] and scene.
//!
//! OpenXR integration must use [`crate::xr::XrHostCameraSync`] for stereo / head-output writes so
//! stereo clearing when `!vr_active` stays consistent with [`apply_frame_submit_fields`].

use glam::Mat4;

use crate::render_graph::HostCameraFrame;
use crate::scene::SceneCoordinator;
use crate::shared::{CameraProjection, FrameSubmitData};

/// Applies host clip, FOV, VR flag, ortho hint, and clears stereo when desktop mode.
pub(crate) fn apply_frame_submit_fields(host_camera: &mut HostCameraFrame, data: &FrameSubmitData) {
    host_camera.frame_index = data.frame_index;
    host_camera.near_clip = data.near_clip;
    host_camera.far_clip = data.far_clip;
    host_camera.desktop_fov_degrees = data.desktop_fov;
    host_camera.vr_active = data.vr_active;
    if !data.vr_active {
        host_camera.stereo = None;
    }
    host_camera.primary_ortho_task = data.render_tasks.iter().find_map(|t| {
        t.parameters.as_ref().and_then(|p| {
            if p.projection == CameraProjection::Orthographic {
                Some((p.orthographic_size, p.near_clip.max(0.01), p.far_clip))
            } else {
                None
            }
        })
    });
}

/// Head-output matrix derived from the active main render space root (the host `HeadOutput` pose).
pub(crate) fn head_output_from_active_main_space(scene: &SceneCoordinator) -> Mat4 {
    scene
        .active_main_space()
        .map(|space| crate::scene::render_transform_to_matrix(&space.root_transform))
        .unwrap_or(Mat4::IDENTITY)
}
