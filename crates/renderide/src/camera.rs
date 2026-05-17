//! Host camera state, view identity, and reverse-Z projection / view-matrix math.
//!
//! Holds the types and pure-CPU helpers that any module talking about cameras / views uses, so
//! non-graph modules don't have to import `crate::render_graph` just to reference a camera.
//! World-to-view applies a Z flip for Vulkan/WebGPU clip, and perspective uses vertical FOV in
//! **radians** with clip planes from [`HostCameraFrame`].
//!
//! OpenXR HMD views use [`reverse_z_perspective_openxr_fov`] (asymmetric frustum from tangents).

mod frame;
mod geometry;
mod host_camera_frame;
mod projection;
mod projection_plan;
mod render_task;
mod secondary;
mod stereo;
mod view;
mod view_id;

pub(crate) use frame::{
    apply_frame_submit_fields, eye_world_position_from_active_main_space,
    head_output_from_active_main_space,
};
pub use geometry::{CameraClipPlanes, CameraPose, EyeView, OrthographicProjectionSpec, Viewport};
pub use host_camera_frame::{CameraProjectionKind, HostCameraFrame};
pub use projection::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_perspective,
    reverse_z_perspective_openxr_fov,
};
pub use projection_plan::WorldProjectionSet;
pub use render_task::{camera_render_task_world_matrix, host_camera_frame_for_render_task};
pub use secondary::{
    camera_state_enabled, camera_state_motion_blur, camera_state_post_processing,
    camera_state_screen_space_reflections, host_camera_frame_for_render_texture,
};
pub use stereo::StereoViewMatrices;
pub use view::{
    apply_view_handedness_fix, view_matrix_for_host_world_mesh_space,
    view_matrix_for_world_mesh_render_space, view_matrix_from_render_transform,
    world_to_view_pair_for_skybox,
};
#[cfg(test)]
pub(crate) use view_id::SecondaryCameraId;
pub use view_id::ViewId;

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};
    use openxr::Fovf;

    use crate::scene::render_transform_to_matrix;
    use crate::shared::{HeadOutputDevice, RenderTransform};

    use super::projection::{
        DEFAULT_DESKTOP_FOV_DEGREES, DESKTOP_FOV_DEGREES_MAX, DESKTOP_FOV_DEGREES_MIN,
        clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_orthographic,
        reverse_z_perspective, reverse_z_perspective_openxr_fov,
    };
    use super::view::view_matrix_from_render_transform;

    fn expected_reverse_z_perspective_cols(
        aspect: f32,
        vertical_fov: f32,
        near: f32,
        far: f32,
    ) -> [f32; 16] {
        let tan_vertical_half = (vertical_fov * 0.5).tan();
        let f_y = 1.0 / tan_vertical_half;
        let f_x = f_y / aspect.max(f32::MIN_POSITIVE);
        let z2 = near / (far - near);
        let z3 = (far * near) / (far - near);
        [
            f_x, 0.0, 0.0, 0.0, 0.0, f_y, 0.0, 0.0, 0.0, 0.0, z2, -1.0, 0.0, 0.0, z3, 0.0,
        ]
    }

    #[test]
    fn reverse_z_perspective_matches_expected_coeffs() {
        let aspect = 16.0 / 9.0;
        let vertical_fov = 55f32.to_radians();
        let near = 0.1_f32;
        let far = 2000.0_f32;
        let glam_m = reverse_z_perspective(aspect, vertical_fov, near, far);
        let expected_cols = expected_reverse_z_perspective_cols(aspect, vertical_fov, near, far);
        let glam_cols = glam_m.to_cols_array();
        assert_eq!(glam_cols.len(), expected_cols.len());
        for (i, (&g, &expected)) in glam_cols.iter().zip(expected_cols.iter()).enumerate() {
            assert!(
                (g - expected).abs() < 1e-5,
                "coeff mismatch at {i}: glam={g} expected={expected}"
            );
        }
    }

    #[test]
    fn clamp_desktop_fov_degrees_nan_default_and_range_clamps() {
        assert!((clamp_desktop_fov_degrees(0.0) - DESKTOP_FOV_DEGREES_MIN).abs() < 1e-6);
        assert!((clamp_desktop_fov_degrees(200.0) - DESKTOP_FOV_DEGREES_MAX).abs() < 1e-6);
        assert_eq!(
            clamp_desktop_fov_degrees(f32::NAN),
            DEFAULT_DESKTOP_FOV_DEGREES
        );
        assert_eq!(
            clamp_desktop_fov_degrees(f32::INFINITY),
            DESKTOP_FOV_DEGREES_MAX
        );
        assert_eq!(
            clamp_desktop_fov_degrees(f32::NEG_INFINITY),
            DESKTOP_FOV_DEGREES_MIN
        );
    }

    #[test]
    fn reverse_z_perspective_finite_diagonal() {
        let m = reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 500.0);
        assert!(m.w_axis.w.is_finite());
        assert!(m.x_axis.x > 0.0 && m.y_axis.y > 0.0);
        assert!(m.z_axis.w == -1.0);
    }

    #[test]
    fn view_handedness_applies_z_flip() {
        let tr = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        };
        let v = view_matrix_from_render_transform(&tr);
        let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
        let unflipped = render_transform_to_matrix(&tr).inverse();
        let expected = z_flip * unflipped;
        assert!(
            (v - expected)
                .to_cols_array()
                .iter()
                .map(|x| x.abs())
                .sum::<f32>()
                < 1e-5
        );
    }

    #[test]
    fn orthographic_reverse_z_depth_maps_near_to_one_far_to_zero() {
        let m = reverse_z_orthographic(2.0, 1.0, 0.05, 100.0);
        let near_clip = m * Vec3::new(0.0, 0.0, -0.05).extend(1.0);
        let far_clip = m * Vec3::new(0.0, 0.0, -100.0).extend(1.0);

        assert!((near_clip.z / near_clip.w - 1.0).abs() < 1e-5);
        assert!((far_clip.z / far_clip.w).abs() < 1e-5);
    }

    #[test]
    fn effective_head_output_clip_planes_match_unity_rules() {
        let (near, far) = effective_head_output_clip_planes(
            0.0001,
            0.25,
            HeadOutputDevice::Screen360,
            Some(Vec3::splat(2.0)),
        );
        assert!((near - 0.5).abs() < 1e-6);
        assert!((far - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reverse_z_openxr_fov_symmetric_near_symmetric_perspective() {
        let a = 0.45_f32;
        let b = 0.45_f32;
        let fov = Fovf {
            angle_left: -a,
            angle_right: a,
            angle_down: -b,
            angle_up: b,
        };
        let near = 0.01_f32;
        let far = 500.0_f32;
        let m_oxr = reverse_z_perspective_openxr_fov(&fov, near, far);
        let aspect = (a.tan() - (-a).tan()) / (b.tan() - (-b).tan());
        let m_sym = reverse_z_perspective(aspect, 2.0 * b, near, far);
        for i in 0..16 {
            assert!(
                (m_oxr.to_cols_array()[i] - m_sym.to_cols_array()[i]).abs() < 2e-3,
                "coeff {i} mismatch"
            );
        }
    }
}
