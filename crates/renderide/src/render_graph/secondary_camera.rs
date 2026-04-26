//! Secondary (render-texture) camera parameters from host [`crate::shared::CameraState`].

use glam::Mat4;

use crate::scene::SceneCoordinator;
use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

use super::camera::{
    apply_view_handedness_fix, clamp_desktop_fov_degrees, effective_head_output_clip_planes,
    reverse_z_orthographic, reverse_z_perspective,
};
use super::frame_params::HostCameraFrame;

/// Returns `true` when [`CameraState::flags`] bit 0 is set (FrooxEngine `Camera.enabled`).
#[inline]
pub fn camera_state_enabled(flags: u16) -> bool {
    flags & 1 != 0
}

/// Builds a [`HostCameraFrame`] for rendering through a secondary camera to a render texture.
pub fn host_camera_frame_for_render_texture(
    base: &HostCameraFrame,
    state: &CameraState,
    viewport_px: (u32, u32),
    camera_world_matrix: Mat4,
    scene: &SceneCoordinator,
) -> HostCameraFrame {
    let (vw, vh) = viewport_px;
    let aspect = vw as f32 / vh.max(1) as f32;
    let root_scale = scene
        .active_main_space()
        .map(|space| space.root_transform.scale);
    let (near_clip, far_clip) = effective_head_output_clip_planes(
        state.near_clip,
        state.far_clip,
        HeadOutputDevice::Screen,
        root_scale,
    );
    let world_to_view = apply_view_handedness_fix(camera_world_matrix.inverse());
    let camera_world = camera_world_matrix.col(3).truncate();
    let world_proj = match state.projection {
        CameraProjection::Orthographic => {
            let half_h = state.orthographic_size.max(1e-6);
            let half_w = half_h * aspect;
            reverse_z_orthographic(half_w, half_h, near_clip, far_clip)
        }
        CameraProjection::Perspective | CameraProjection::Panoramic => {
            let fov_deg = clamp_desktop_fov_degrees(state.field_of_view);
            let fov_rad = fov_deg.to_radians();
            reverse_z_perspective(aspect, fov_rad, near_clip, far_clip)
        }
    };

    let primary_ortho_task = match state.projection {
        CameraProjection::Orthographic => {
            Some((state.orthographic_size.max(1e-6), near_clip, far_clip))
        }
        CameraProjection::Perspective | CameraProjection::Panoramic => None,
    };

    let desktop_fov = clamp_desktop_fov_degrees(state.field_of_view);

    HostCameraFrame {
        frame_index: base.frame_index,
        near_clip,
        far_clip,
        desktop_fov_degrees: desktop_fov,
        vr_active: false,
        output_device: base.output_device,
        primary_ortho_task,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_world_to_view: Some(world_to_view),
        cluster_view_override: Some(world_to_view),
        cluster_proj_override: Some(world_proj),
        explicit_camera_world_position: Some(camera_world),
        eye_world_position: base.eye_world_position,
        suppress_occlusion_temporal: false,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`camera_state_enabled`] and [`host_camera_frame_for_render_texture`].

    use glam::{Mat4, Vec3};

    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

    use super::super::camera::apply_view_handedness_fix;
    use super::super::frame_params::HostCameraFrame;
    use super::{camera_state_enabled, host_camera_frame_for_render_texture};

    #[test]
    fn camera_state_enabled_reads_bit_zero() {
        assert!(!camera_state_enabled(0));
        assert!(camera_state_enabled(1));
        assert!(camera_state_enabled(0xffff));
        assert!(!camera_state_enabled(2));
    }

    #[test]
    fn host_camera_frame_secondary_sets_world_to_view_and_cluster_overrides() {
        let scene = SceneCoordinator::new();
        let base = HostCameraFrame {
            output_device: HeadOutputDevice::Screen,
            ..Default::default()
        };
        let cam_world = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        let state = CameraState {
            projection: CameraProjection::Perspective,
            field_of_view: 55.0,
            near_clip: 0.05,
            far_clip: 2000.0,
            ..Default::default()
        };

        let out =
            host_camera_frame_for_render_texture(&base, &state, (1280, 720), cam_world, &scene);

        let expected_w2v = apply_view_handedness_fix(cam_world.inverse());
        assert_eq!(out.explicit_world_to_view, Some(expected_w2v));
        assert_eq!(out.cluster_view_override, Some(expected_w2v));
        assert!(out.cluster_proj_override.is_some());
        assert_eq!(out.primary_ortho_task, None);
        assert_eq!(out.desktop_fov_degrees, state.field_of_view);
        assert!(!out.vr_active);
    }

    #[test]
    fn host_camera_frame_secondary_orthographic_sets_primary_ortho_task() {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![crate::shared::RenderTransform::default()],
            vec![-1],
        );
        let base = HostCameraFrame::default();
        let cam_world = Mat4::IDENTITY;
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 8.0,
            near_clip: 0.1,
            far_clip: 500.0,
            ..Default::default()
        };

        let out =
            host_camera_frame_for_render_texture(&base, &state, (640, 480), cam_world, &scene);

        assert_eq!(
            out.primary_ortho_task,
            Some((8.0, out.near_clip, out.far_clip))
        );
        assert!(out.cluster_proj_override.is_some());
    }
}
