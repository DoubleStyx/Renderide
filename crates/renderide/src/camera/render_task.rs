//! Host camera render-task projection and pose conversion.

use glam::{Mat4, Vec3};

use crate::shared::CameraRenderParameters;

use super::host_camera_frame::{SingleCameraInputs, build_single_camera_frame};
use super::{CameraClipPlanes, CameraPose, HostCameraFrame, Viewport, clamp_desktop_fov_degrees};

/// Builds a [`HostCameraFrame`] for a host camera readback task.
pub fn host_camera_frame_for_render_task(
    base: &HostCameraFrame,
    parameters: &CameraRenderParameters,
    viewport_px: (u32, u32),
    camera_world_matrix: Mat4,
) -> HostCameraFrame {
    build_single_camera_frame(
        base,
        SingleCameraInputs {
            viewport: Viewport::from_tuple(viewport_px),
            pose: CameraPose::from_world_matrix(camera_world_matrix),
            clip: camera_render_task_clip(parameters),
            fov_degrees: clamp_desktop_fov_degrees(parameters.fov),
            orthographic_size: parameters.orthographic_size,
            projection: parameters.projection,
            suppress_occlusion_temporal: true,
        },
    )
}

/// Returns sanitized clip planes for a host camera readback task.
#[inline]
pub(crate) fn camera_render_task_clip(parameters: &CameraRenderParameters) -> CameraClipPlanes {
    let near = finite_positive_or(parameters.near_clip, CameraClipPlanes::default().near).max(0.01);
    let far_default = CameraClipPlanes::default().far;
    let far = finite_positive_or(parameters.far_clip, far_default).max(near + 0.01);
    CameraClipPlanes::new(near, far)
}

#[inline]
fn finite_positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

/// Builds a camera world matrix from a host task position and rotation.
#[inline]
pub fn camera_render_task_world_matrix(position: Vec3, rotation: glam::Quat) -> Mat4 {
    Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, position)
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::shared::{CameraProjection, CameraRenderParameters};

    use super::*;
    use crate::camera::CameraProjectionKind;

    #[test]
    fn camera_render_task_perspective_sets_explicit_view_and_clamps_fov() {
        let base = HostCameraFrame {
            frame_index: 42,
            ..Default::default()
        };
        let parameters = CameraRenderParameters {
            projection: CameraProjection::Perspective,
            fov: 200.0,
            near_clip: 0.05,
            far_clip: 500.0,
            ..Default::default()
        };
        let world = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));

        let out = host_camera_frame_for_render_task(&base, &parameters, (1280, 720), world);

        assert_eq!(out.frame_index, 42);
        assert_eq!(out.projection_kind, CameraProjectionKind::Perspective);
        assert_eq!(out.primary_ortho_task, None);
        assert_eq!(out.clip, CameraClipPlanes::new(0.05, 500.0));
        assert!(out.desktop_fov_degrees < 180.0);
        assert!(
            out.explicit_view
                .expect("task explicit view")
                .proj
                .is_finite()
        );
        assert!(out.suppress_occlusion_temporal);
    }

    #[test]
    fn camera_render_task_orthographic_sets_projection_override() {
        let parameters = CameraRenderParameters {
            projection: CameraProjection::Orthographic,
            orthographic_size: 8.0,
            near_clip: 0.1,
            far_clip: 900.0,
            ..Default::default()
        };

        let out = host_camera_frame_for_render_task(
            &HostCameraFrame::default(),
            &parameters,
            (640, 480),
            Mat4::IDENTITY,
        );

        let ortho = out.primary_ortho_task.expect("orthographic task");
        assert_eq!(ortho.half_height, 8.0);
        assert_eq!(ortho.clip, out.clip);
        assert_eq!(out.projection_kind, CameraProjectionKind::Orthographic);
    }

    #[test]
    fn camera_render_task_world_matrix_uses_position_and_rotation() {
        let rotation = Quat::from_rotation_y(1.0);
        let position = Vec3::new(4.0, 5.0, 6.0);

        let matrix = camera_render_task_world_matrix(position, rotation);
        let (_, actual_rotation, actual_position) = matrix.to_scale_rotation_translation();

        assert_eq!(actual_position, position);
        assert!((actual_rotation.dot(rotation).abs() - 1.0).abs() < 1e-5);
    }
}
