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
        stereo_view_proj: None,
        stereo_views: None,
        head_output_transform: base.head_output_transform,
        secondary_camera_world_to_view: Some(world_to_view),
        cluster_view_override: Some(world_to_view),
        cluster_proj_override: Some(world_proj),
        secondary_camera_world_position: Some(camera_world),
        suppress_occlusion_temporal: false,
    }
}
