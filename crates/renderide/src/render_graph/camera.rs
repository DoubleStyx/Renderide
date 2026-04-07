//! Host-facing view and **reverse-Z** projection (`crates_old` `reverse_z_projection` / `ViewParams`).
//!
//! Matches the legacy mesh pass: world-to-view applies a Z flip for Vulkan/WebGPU clip, and
//! perspective uses vertical FOV in **radians** with clip planes from [`super::frame_params::HostCameraFrame`].
//!
//! OpenXR HMD views use [`reverse_z_perspective_openxr_fov`] (asymmetric frustum from tangents).

use glam::{Mat4, Vec3, Vec4};
use openxr::Fovf;

use crate::scene::render_transform_to_matrix;
use crate::shared::RenderTransform;

/// Minimum desktop vertical FOV in **degrees** after clamping.
///
/// Mirrors a small positive host lower bound so `tan(fov/2)` stays finite and non-zero.
pub const DESKTOP_FOV_DEGREES_MIN: f32 = 1e-4;

/// Maximum desktop vertical FOV in **degrees** after clamping (non-inclusive of 180° degeneracy).
pub const DESKTOP_FOV_DEGREES_MAX: f32 = 179.0;

/// Default fallback when the host sends non-finite FOV (matches [`super::frame_params::HostCameraFrame::default`]).
const DEFAULT_DESKTOP_FOV_DEGREES: f32 = 60.0;

/// Clamps host `desktopFOV` to a sane range before perspective projection.
///
/// [`f32::NAN`] falls back to [`DEFAULT_DESKTOP_FOV_DEGREES`]. Infinities clamp to the min/max
/// bounds like any other out-of-range value.
pub fn clamp_desktop_fov_degrees(degrees: f32) -> f32 {
    if degrees.is_nan() {
        DEFAULT_DESKTOP_FOV_DEGREES
    } else {
        degrees.clamp(DESKTOP_FOV_DEGREES_MIN, DESKTOP_FOV_DEGREES_MAX)
    }
}

/// Clamps scale for view matrix construction: if any axis is nearly zero, use unit scale (legacy `filter_scale`).
fn filter_scale_for_view(scale: Vec3) -> Vec3 {
    const MIN: f32 = 1e-8;
    if scale.x.abs() < MIN || scale.y.abs() < MIN || scale.z.abs() < MIN {
        Vec3::splat(1.0)
    } else {
        scale
    }
}

/// Z-flip for RH engine space → Vulkan/WebGPU-style clip (legacy `apply_view_handedness_fix`).
/// Z-flip for RH engine space → Vulkan/WebGPU-style clip (legacy `apply_view_handedness_fix`).
#[inline]
pub fn apply_view_handedness_fix(view: Mat4) -> Mat4 {
    let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// World-to-view matrix from a host [`RenderTransform`] (camera / eye TRS).
///
/// Applies legacy scale filtering and handedness fix so `view_proj * world_pos` matches the old mesh path.
pub fn view_matrix_from_render_transform(tr: &RenderTransform) -> Mat4 {
    let mut t = *tr;
    let fs = filter_scale_for_view(Vec3::new(tr.scale.x, tr.scale.y, tr.scale.z));
    t.scale.x = fs.x;
    t.scale.y = fs.y;
    t.scale.z = fs.z;
    let cam = render_transform_to_matrix(&t);
    apply_view_handedness_fix(cam.inverse())
}

/// Reverse-Z perspective projection (column-major [`Mat4`], same coefficients as legacy nalgebra path).
///
/// * `vertical_fov` — vertical field of view in **radians**
/// * `near` / `far` — positive distances (`far > near`)
pub fn reverse_z_perspective(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Mat4 {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1_f32, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;
    reverse_z_perspective_from_scales(f_x, f_y, 0.0, 0.0, near, far)
}

/// Reverse-Z perspective with optional **off-center** (asymmetric) X/Y skew from OpenXR tangents.
///
/// `skew_x` / `skew_y` are `(tan_right + tan_left) / (tan_right - tan_left)` and
/// `(tan_up + tan_down) / (tan_up - tan_down)` on the **Z basis column** so clip X/Y depend on view-space Z.
fn reverse_z_perspective_from_scales(
    x_scale: f32,
    y_scale: f32,
    skew_x: f32,
    skew_y: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let z2 = near / (far - near);
    let z3 = (far * near) / (far - near);
    Mat4::from_cols(
        Vec4::new(x_scale, 0.0, 0.0, 0.0),
        Vec4::new(0.0, y_scale, 0.0, 0.0),
        Vec4::new(skew_x, skew_y, z2, -1.0),
        Vec4::new(0.0, 0.0, z3, 0.0),
    )
}

/// Asymmetric reverse-Z projection from OpenXR [`Fovf`] tangents (Khronos `XrMatrix4x4f_CreateProjectionFov` X/Y,
/// with the same reverse-Z depth row as [`reverse_z_perspective`]).
///
/// View space matches the renderer: **right-handed**, **−Z** forward, **+Y** up.
pub fn reverse_z_perspective_openxr_fov(fov: &Fovf, near: f32, far: f32) -> Mat4 {
    let tl = fov.angle_left.tan();
    let tr = fov.angle_right.tan();
    let td = fov.angle_down.tan();
    let tu = fov.angle_up.tan();
    let w = tr - tl;
    let h = tu - td;
    if !(w.is_finite() && h.is_finite()) || w.abs() < 1e-6 || h.abs() < 1e-6 {
        logger::trace!(
            "OpenXR FOV degenerate; using symmetric fallback (16:9, 45° vertical). raw angles rad: left={:.4} right={:.4} down={:.4} up={:.4} w={w} h={h}",
            fov.angle_left,
            fov.angle_right,
            fov.angle_down,
            fov.angle_up
        );
        let aspect = 16.0 / 9.0;
        let vertical_fov = std::f32::consts::FRAC_PI_2 * 0.5;
        return reverse_z_perspective(aspect, vertical_fov, near, far);
    }
    let x_scale = 2.0 / w;
    let y_scale = 2.0 / h;
    let skew_x = (tr + tl) / w;
    let skew_y = (tu + td) / h;
    reverse_z_perspective_from_scales(x_scale, y_scale, skew_x, skew_y, near, far)
}

/// Reverse-Z orthographic projection (`half_width`, `half_height` in view space).
pub fn reverse_z_orthographic(half_width: f32, half_height: f32, near: f32, far: f32) -> Mat4 {
    let range = far - near;
    let z_scale = -2.0 / range;
    let z_offset = (far + near) / range;
    Mat4::from_cols(
        Vec4::new(1.0 / half_width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / half_height, 0.0, 0.0),
        Vec4::new(0.0, 0.0, z_scale, 0.0),
        Vec4::new(0.0, 0.0, z_offset, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    fn legacy_nalgebra_reverse_z_projection(
        aspect: f32,
        vertical_fov: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        let vertical_half = vertical_fov / 2.0;
        let tan_vertical_half = vertical_half.tan();
        let horizontal_fov = (tan_vertical_half * aspect)
            .atan()
            .clamp(0.1_f32, std::f32::consts::FRAC_PI_2 - 0.1_f32)
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

    #[test]
    fn reverse_z_perspective_matches_legacy_nalgebra_coeffs() {
        let aspect = 16.0 / 9.0;
        let vertical_fov = 55f32.to_radians();
        let near = 0.1_f32;
        let far = 2000.0_f32;
        let glam_m = reverse_z_perspective(aspect, vertical_fov, near, far);
        let na_m = legacy_nalgebra_reverse_z_projection(aspect, vertical_fov, near, far);
        let glam_cols = glam_m.to_cols_array();
        let na_slice = na_m.as_slice();
        assert_eq!(glam_cols.len(), na_slice.len());
        for (i, (&g, &n)) in glam_cols.iter().zip(na_slice.iter()).enumerate() {
            assert!(
                (g - n).abs() < 1e-5,
                "coeff mismatch at {i}: glam={g} nalgebra={n}"
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
        let tr = RenderTransform::default();
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
    fn orthographic_matches_legacy_z_coeffs() {
        let m = reverse_z_orthographic(2.0, 1.0, 0.05, 100.0);
        let range = 100.0 - 0.05;
        let z_scale = -2.0 / range;
        let z_off = (100.0 + 0.05) / range;
        assert!((m.z_axis.z - z_scale).abs() < 1e-5);
        assert!((m.w_axis.z - z_off).abs() < 1e-5);
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
