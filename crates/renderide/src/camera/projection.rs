//! Reverse-Z projections and host clip-plane helpers for desktop and OpenXR.

use glam::{Mat4, Vec3, Vec4};
use openxr::Fovf;

use crate::shared::HeadOutputDevice;

use super::view::filter_scale_legacy;

/// Minimum desktop vertical FOV in **degrees** after clamping.
///
/// Mirrors a small positive host lower bound so `tan(fov/2)` stays finite and non-zero.
pub const DESKTOP_FOV_DEGREES_MIN: f32 = 1e-4;

/// Maximum desktop vertical FOV in **degrees** after clamping (non-inclusive of 180 deg degeneracy).
pub const DESKTOP_FOV_DEGREES_MAX: f32 = 179.0;

/// Default fallback when the host sends non-finite FOV (matches [`crate::camera::HostCameraFrame::default`]).
pub(crate) const DEFAULT_DESKTOP_FOV_DEGREES: f32 = 60.0;

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

/// Clip-plane adjustment derived from head output device and root scale (Unity-style parity).
pub fn effective_head_output_clip_planes(
    near_clip: f32,
    far_clip: f32,
    output_device: HeadOutputDevice,
    root_scale: Option<Vec3>,
) -> (f32, f32) {
    let near_min = if output_device == HeadOutputDevice::Screen360 {
        0.25
    } else {
        0.001
    };
    let filtered_root_scale = filter_scale_legacy(root_scale.unwrap_or(Vec3::ONE));
    (
        near_clip.max(near_min) * filtered_root_scale.x,
        far_clip.max(0.5) * filtered_root_scale.x,
    )
}

/// Reverse-Z perspective projection (column-major [`Mat4`], same coefficients as the historical nalgebra path).
///
/// Horizontal FOV is *derived* from vertical FOV and aspect; the X scale is
/// `f_x = f_y / aspect`, which preserves the canonical
/// `f_x / f_y = 1 / aspect` relationship at every FOV/aspect pair. Input safety relies on
/// the upstream [`clamp_desktop_fov_degrees`] sanitisation, so no further per-axis clamping
/// is applied here (an asymmetric clamp would decouple X and Y and visibly squish the scene
/// at low FOV / non-`16:9` aspects).
///
/// * `vertical_fov` -- vertical field of view in **radians**
/// * `near` / `far` -- positive distances (`far > near`)
pub fn reverse_z_perspective(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Mat4 {
    let tan_vertical_half = (vertical_fov / 2.0).tan();
    let f_y = 1.0 / tan_vertical_half;
    let f_x = f_y / aspect.max(f32::MIN_POSITIVE);
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
/// View space matches the renderer: **right-handed**, **-Z** forward, **+Y** up.
pub fn reverse_z_perspective_openxr_fov(fov: &Fovf, near: f32, far: f32) -> Mat4 {
    let tl = fov.angle_left.tan();
    let tr = fov.angle_right.tan();
    let td = fov.angle_down.tan();
    let tu = fov.angle_up.tan();
    let w = tr - tl;
    let h = tu - td;
    if !(w.is_finite() && h.is_finite()) || w.abs() < 1e-6 || h.abs() < 1e-6 {
        logger::trace!(
            "OpenXR FOV degenerate; using symmetric fallback (16:9, 45 deg vertical). raw angles rad: left={:.4} right={:.4} down={:.4} up={:.4} w={w} h={h}",
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
mod effective_clip_plane_tests {
    use glam::Vec3;

    use crate::shared::HeadOutputDevice;

    use super::effective_head_output_clip_planes;

    #[test]
    fn screen360_uses_higher_near_floor_than_screen() {
        let (n360, f360) =
            effective_head_output_clip_planes(0.01, 100.0, HeadOutputDevice::Screen360, None);
        let (n_screen, f_screen) =
            effective_head_output_clip_planes(0.01, 100.0, HeadOutputDevice::Screen, None);
        assert!((n360 - 0.25).abs() < 1e-5);
        assert!((n_screen - 0.01).abs() < 1e-5);
        assert!((f360 - 100.0).abs() < 1e-4);
        assert!((f_screen - 100.0).abs() < 1e-4);
    }

    #[test]
    fn root_scale_multiplies_adjusted_planes_when_non_degenerate() {
        let scale = Vec3::new(2.0, 1.0, 1.0);
        let (n, f) =
            effective_head_output_clip_planes(0.1, 50.0, HeadOutputDevice::Screen, Some(scale));
        assert!((n - 0.2).abs() < 1e-5);
        assert!((f - 100.0).abs() < 1e-4);
    }

    #[test]
    fn near_zero_root_scale_axis_falls_back_to_unit_scale() {
        let scale = Vec3::new(1e-9, 1.0, 1.0);
        let (n, f) =
            effective_head_output_clip_planes(0.1, 50.0, HeadOutputDevice::Screen, Some(scale));
        assert!((n - 0.1).abs() < 1e-5);
        assert!((f - 50.0).abs() < 1e-4);
    }
}

#[cfg(test)]
mod projection_math_tests {
    use glam::{Vec2, Vec3, Vec4};
    use openxr::Fovf;
    use std::f32::consts::FRAC_PI_2;

    use super::{
        DEFAULT_DESKTOP_FOV_DEGREES, DESKTOP_FOV_DEGREES_MAX, DESKTOP_FOV_DEGREES_MIN,
        clamp_desktop_fov_degrees, reverse_z_perspective, reverse_z_perspective_openxr_fov,
    };

    /// Projects a view-space point through `m` and returns its `z / w` clip depth.
    fn project_depth(m: &glam::Mat4, view_z: f32) -> f32 {
        let clip = m.mul_vec4(Vec4::new(0.0, 0.0, view_z, 1.0));
        clip.z / clip.w
    }

    /// Projects a point on the skybox's `view_z = -1` plane and returns NDC XY.
    fn project_unit_depth_view_xy(m: &glam::Mat4, view_xy: Vec2) -> Vec2 {
        let clip = m.mul_vec4(Vec4::new(view_xy.x, view_xy.y, -1.0, 1.0));
        Vec2::new(clip.x / clip.w, clip.y / clip.w)
    }

    /// Extracts the same projection coefficients uploaded to `FrameGlobals::proj_params_*`.
    fn skybox_proj_params_for_test(m: glam::Mat4) -> [f32; 4] {
        [m.x_axis.x, m.y_axis.y, m.z_axis.x, m.z_axis.y]
    }

    /// Mirrors `skybox_common.wgsl::view_ray_from_ndc`.
    fn skybox_view_ray_from_ndc_for_test(ndc: Vec2, proj_params: [f32; 4]) -> Vec3 {
        let view_x = (ndc.x + proj_params[2]) / proj_params[0].abs().max(1e-6);
        let view_y = (ndc.y + proj_params[3]) / proj_params[1].abs().max(1e-6);
        Vec3::new(view_x, view_y, -1.0).normalize()
    }

    /// [`clamp_desktop_fov_degrees`] maps NaN to the default and clamps out-of-range inputs to the
    /// declared bounds without changing in-range values.
    #[test]
    fn clamp_desktop_fov_handles_special_values() {
        assert_eq!(
            clamp_desktop_fov_degrees(f32::NAN),
            DEFAULT_DESKTOP_FOV_DEGREES
        );
        assert_eq!(clamp_desktop_fov_degrees(-1.0), DESKTOP_FOV_DEGREES_MIN);
        assert_eq!(clamp_desktop_fov_degrees(500.0), DESKTOP_FOV_DEGREES_MAX);
        assert_eq!(
            clamp_desktop_fov_degrees(f32::INFINITY),
            DESKTOP_FOV_DEGREES_MAX
        );
        assert_eq!(
            clamp_desktop_fov_degrees(f32::NEG_INFINITY),
            DESKTOP_FOV_DEGREES_MIN
        );
        assert_eq!(clamp_desktop_fov_degrees(45.0), 45.0);
    }

    /// The reverse-Z perspective matrix must have positive scale on the diagonal, a `-1` on the
    /// view-Z -> clip-W row, and map the near/far view-space planes to clip depths `1` and `0`.
    #[test]
    fn reverse_z_perspective_near_and_far_depth_values() {
        let near = 0.1_f32;
        let far = 100.0_f32;
        let m = reverse_z_perspective(1.0, FRAC_PI_2, near, far);
        let cols = m.to_cols_array_2d();

        assert!(cols[0][0].is_finite() && cols[0][0] > 0.0);
        assert!(cols[1][1].is_finite() && cols[1][1] > 0.0);
        assert!((cols[2][3] - -1.0).abs() < 1e-6);

        assert!((project_depth(&m, -near) - 1.0).abs() < 1e-4);
        assert!(project_depth(&m, -far).abs() < 1e-4);
    }

    /// Fullscreen skybox ray reconstruction must invert asymmetric OpenXR skew with a plus sign.
    #[test]
    fn skybox_view_ray_roundtrips_asymmetric_openxr_projection() {
        let fov = Fovf {
            angle_left: -0.82,
            angle_right: 0.68,
            angle_down: -0.53,
            angle_up: 0.74,
        };
        let m = reverse_z_perspective_openxr_fov(&fov, 0.1, 100.0);
        let proj_params = skybox_proj_params_for_test(m);

        for view_xy in [Vec2::new(-0.35, 0.22), Vec2::ZERO, Vec2::new(0.28, -0.31)] {
            let ndc = project_unit_depth_view_xy(&m, view_xy);
            let actual = skybox_view_ray_from_ndc_for_test(ndc, proj_params);
            let expected = Vec3::new(view_xy.x, view_xy.y, -1.0).normalize();
            assert!(
                actual.dot(expected) > 0.999_99,
                "skybox ray mismatch for {view_xy:?}: expected {expected:?}, got {actual:?}"
            );
        }
    }

    /// Shader source should keep the same asymmetric-skew sign as the CPU roundtrip above.
    #[test]
    fn skybox_shader_uses_positive_asymmetric_skew_sign() {
        let source = include_str!("../../shaders/modules/skybox_common.wgsl");
        assert!(source.contains("ndc.x + proj_params.z"));
        assert!(source.contains("ndc.y + proj_params.w"));
    }

    /// A degenerate all-zero OpenXR FOV takes the symmetric 16:9 fallback and yields a finite
    /// matrix.
    #[test]
    fn openxr_degenerate_fov_falls_back_to_finite_matrix() {
        let fov = Fovf {
            angle_left: 0.0,
            angle_right: 0.0,
            angle_down: 0.0,
            angle_up: 0.0,
        };
        let m = reverse_z_perspective_openxr_fov(&fov, 0.1, 100.0);
        for v in m.to_cols_array() {
            assert!(v.is_finite(), "matrix element must be finite: {v}");
        }
    }

    /// Regression: at low vertical FOV with non-`16:9` aspect ratios the projection's X and Y
    /// scales must stay tied through `f_x = f_y / aspect`. The previous implementation clamped
    /// the *derived* horizontal FOV to `[0.1, PI/2 - 0.1]` rad independently of the vertical
    /// scale, which decoupled the axes for `vfov <~ 3.2 deg` at 16:9 (and at much larger vfov for
    /// portrait/square aspects) and visibly squished the rendered scene.
    #[test]
    fn perspective_preserves_aspect_at_low_vertical_fov() {
        const VFOVS_DEG: &[f32] = &[10.0, 5.0, 1.0, 0.1];
        const ASPECTS: &[f32] = &[16.0 / 9.0, 4.0 / 3.0, 1.0, 0.5, 0.25];
        for &vfov_deg in VFOVS_DEG {
            for &aspect in ASPECTS {
                let m = reverse_z_perspective(aspect, vfov_deg.to_radians(), 0.1, 100.0);
                let f_x = m.x_axis.x;
                let f_y = m.y_axis.y;
                assert!(f_x.is_finite() && f_x > 0.0, "f_x must be positive finite");
                assert!(f_y.is_finite() && f_y > 0.0, "f_y must be positive finite");
                let observed = f_x / f_y;
                let expected = 1.0 / aspect;
                assert!(
                    (observed - expected).abs() < 1e-4 * expected.abs().max(1.0),
                    "f_x / f_y mismatch for vfov={vfov_deg} deg aspect={aspect}: \
                     expected {expected}, got {observed} (f_x={f_x}, f_y={f_y})"
                );
            }
        }
    }

    /// At wide vertical FOV the matrix must remain finite and keep the same canonical
    /// `f_x / f_y = 1 / aspect` relationship; previously the upper clamp on the derived
    /// horizontal FOV could decouple the axes near the host upper bound (179 deg).
    #[test]
    fn perspective_preserves_aspect_at_wide_vertical_fov() {
        for &aspect in &[16.0 / 9.0, 1.0 / 4.0_f32] {
            let m = reverse_z_perspective(aspect, 170f32.to_radians(), 0.1, 100.0);
            let f_x = m.x_axis.x;
            let f_y = m.y_axis.y;
            assert!(f_x.is_finite() && f_x > 0.0);
            assert!(f_y.is_finite() && f_y > 0.0);
            let observed = f_x / f_y;
            let expected = 1.0 / aspect;
            assert!(
                (observed - expected).abs() < 1e-4 * expected.abs().max(1.0),
                "f_x / f_y mismatch for vfov=170 deg aspect={aspect}: \
                 expected {expected}, got {observed}"
            );
        }
    }

    /// Reverse-Z near/far depth invariants must hold independently of the input FOV -- pin them
    /// at a small FOV so a future change to the X/Y scales can't silently regress the depth row.
    #[test]
    fn perspective_depth_row_holds_at_low_vertical_fov() {
        let near = 0.1_f32;
        let far = 100.0_f32;
        let m = reverse_z_perspective(16.0 / 9.0, 1f32.to_radians(), near, far);
        assert!((project_depth(&m, -near) - 1.0).abs() < 1e-4);
        assert!(project_depth(&m, -far).abs() < 1e-4);
    }

    /// Defensive: a degenerate viewport with zero aspect must not produce NaN/inf -- the
    /// `aspect.max(f32::MIN_POSITIVE)` guard keeps the matrix finite (X scale becomes huge but
    /// stays representable in `f32`).
    #[test]
    fn perspective_zero_aspect_stays_finite() {
        let m = reverse_z_perspective(0.0, 60f32.to_radians(), 0.1, 100.0);
        for v in m.to_cols_array() {
            assert!(v.is_finite(), "matrix element must be finite: {v}");
        }
    }
}
