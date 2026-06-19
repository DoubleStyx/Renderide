//! Host frame-submit camera reduction and scene-derived camera poses.

use glam::{Mat4, Vec3};

use crate::scene::{RenderSpaceRead, SceneSpaceRead};
use crate::shared::{CameraProjection, FrameSubmitData};

use super::{CameraClipPlanes, CameraProjectionKind, HostCameraFrame, OrthographicProjectionSpec};

/// Applies host clip, FOV, VR flag, ortho hint, and clears stereo when desktop mode.
pub(crate) fn apply_frame_submit_fields(host_camera: &mut HostCameraFrame, data: &FrameSubmitData) {
    host_camera.frame_index = data.frame_index;
    host_camera.clip = CameraClipPlanes::new(data.near_clip, data.far_clip);
    host_camera.desktop_fov_degrees = data.desktop_fov;
    host_camera.vr_active = data.vr_active;
    host_camera.projection_kind = CameraProjectionKind::Perspective;
    if !data.vr_active {
        host_camera.stereo = None;
    }
    host_camera.primary_ortho_task = data.render_tasks.iter().find_map(|task| {
        task.parameters.as_ref().and_then(|parameters| {
            (parameters.projection == CameraProjection::Orthographic).then(|| {
                OrthographicProjectionSpec::new(
                    parameters.orthographic_size,
                    CameraClipPlanes::new(parameters.near_clip.max(0.01), parameters.far_clip),
                )
            })
        })
    });
}

/// Head-output matrix derived from the active main render space root.
#[inline]
pub(crate) fn head_output_from_active_main_space(scene: &(impl SceneSpaceRead + ?Sized)) -> Mat4 {
    scene.active_main_space().map_or(Mat4::IDENTITY, |space| {
        crate::scene::render_transform_to_matrix(space.root_transform())
    })
}

/// Eye/camera world position derived from the active main render space's resolved view transform.
#[inline]
pub(crate) fn eye_world_position_from_active_main_space(
    scene: &(impl SceneSpaceRead + ?Sized),
) -> Option<Vec3> {
    scene
        .active_main_space()
        .map(|space| space.view_transform().position)
}
