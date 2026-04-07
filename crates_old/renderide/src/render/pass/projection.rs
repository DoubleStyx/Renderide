//! Projection matrix helpers for perspective and orthographic views.
//!
//! Uses reverse-Z depth (near → 1, far → -1 in NDC).

use nalgebra::Matrix4;

use crate::shared::{CameraProjection, CameraRenderParameters};

/// Reverse-Z projection matrix for the given aspect and frustum.
pub fn reverse_z_projection(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Matrix4<f32> {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;

    Matrix4::new(
        f_x,
        0.0,
        0.0,
        0.0,
        0.0,
        f_y,
        0.0,
        0.0,
        0.0,
        0.0,
        near / (far - near),
        (far * near) / (far - near),
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}

/// Reverse-Z orthographic projection matrix.
///
/// Used for UI and render tasks (e.g. CameraRenderTask with orthographic projection).
/// Maps z from [near, far] to NDC such that near → 1, far → -1 (reverse-Z depth).
///
/// * `half_width` - Half-width of the orthographic view volume.
/// * `half_height` - Half-height of the orthographic view volume.
/// * `near` - Near clip plane (closer to camera).
/// * `far` - Far clip plane (farther from camera).
pub fn orthographic_projection_reverse_z(
    half_width: f32,
    half_height: f32,
    near: f32,
    far: f32,
) -> Matrix4<f32> {
    let range = far - near;
    let z_scale = -2.0 / range;
    let z_offset = (far + near) / range;

    Matrix4::new(
        1.0 / half_width,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / half_height,
        0.0,
        0.0,
        0.0,
        0.0,
        z_scale,
        0.0,
        0.0,
        0.0,
        z_offset,
        1.0,
    )
}

/// Returns a projection matrix for the given camera parameters.
///
/// Chooses perspective or orthographic based on `params.projection`. Orthographic
/// is used for UI and render tasks (e.g. offscreen CameraRenderTask); the main
/// view continues to use the existing perspective path via [`reverse_z_projection`].
///
/// * `params` - Camera parameters from `CameraRenderTask` or similar.
/// * `aspect` - Aspect ratio (width / height).
pub fn projection_for_params(params: &CameraRenderParameters, aspect: f32) -> Matrix4<f32> {
    let near = params.near_clip.max(0.01);
    let far = params.far_clip;

    match params.projection {
        CameraProjection::orthographic => {
            let half_height = params.orthographic_size;
            let half_width = half_height * aspect;
            orthographic_projection_reverse_z(half_width, half_height, near, far)
        }
        CameraProjection::perspective | CameraProjection::panoramic => {
            reverse_z_projection(aspect, params.fov.to_radians(), near, far)
        }
    }
}
