//! TRS -> [`glam::Mat4`] for hierarchy and future GPU uploads.

use glam::{Mat4, Quat, Vec3};

use crate::shared::RenderTransform;

/// Maximum absolute raw object scale axis treated as collapsed for draw culling.
pub(crate) const DEGENERATE_RENDER_SCALE_EPS: f32 = 1e-8;

/// Returns `true` when one raw scale axis is unusable for draw culling.
#[inline]
fn scale_axis_collapses(axis: f32) -> bool {
    !axis.is_finite() || axis.abs() <= DEGENERATE_RENDER_SCALE_EPS
}

/// Returns `true` when all raw scale axes collapse an object draw.
#[inline]
pub(crate) fn render_transform_has_degenerate_scale(t: &RenderTransform) -> bool {
    let scale = t.scale;
    scale_axis_collapses(scale.x) && scale_axis_collapses(scale.y) && scale_axis_collapses(scale.z)
}

/// Repairs one raw scale axis for matrix construction while preserving finite authored scale.
#[inline]
fn matrix_scale_axis(axis: f32) -> f32 {
    if !axis.is_finite() {
        return 1.0;
    }
    axis
}

/// Builds column-major TRS = `T * R * S`, matching the host `RenderTransform` convention.
#[inline]
pub fn render_transform_to_matrix(t: &RenderTransform) -> Mat4 {
    let scale = Vec3::new(
        matrix_scale_axis(t.scale.x),
        matrix_scale_axis(t.scale.y),
        matrix_scale_axis(t.scale.z),
    );
    let rot = if t.rotation.w.abs() >= 1e-8
        || t.rotation.x.abs() >= 1e-8
        || t.rotation.y.abs() >= 1e-8
        || t.rotation.z.abs() >= 1e-8
    {
        t.rotation
    } else {
        Quat::IDENTITY
    };
    let pos = if t.position.x.is_finite() && t.position.y.is_finite() && t.position.z.is_finite() {
        Vec3::new(t.position.x, t.position.y, t.position.z)
    } else {
        Vec3::ZERO
    };
    Mat4::from_scale_rotation_translation(scale, rot, pos)
}

/// Left-multiplies a hierarchy world matrix by the render-space root TRS.
///
/// [`super::coordinator::SceneCoordinator::world_matrix`] already encodes the full parent chain
/// for objects. The host uses [`RenderSpaceState::root_transform`](super::render_space::RenderSpaceState)
/// for the **view / rig** basis; combining it with object matrices is only for exceptional host
/// contracts--not default mesh, light, or skinning paths.
#[inline]
#[cfg(test)]
pub fn multiply_root(world_local: Mat4, root: &RenderTransform) -> Mat4 {
    render_transform_to_matrix(root) * world_local
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a transform with the requested scale and otherwise identity components.
    fn scaled_transform(scale: Vec3) -> RenderTransform {
        RenderTransform {
            position: Vec3::ZERO,
            scale,
            rotation: Quat::IDENTITY,
        }
    }

    /// Unit scale is a renderable object scale.
    #[test]
    fn unit_scale_is_not_degenerate_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::ONE
        )));
    }

    /// Exact zero on one object scale axis remains renderable.
    #[test]
    fn single_zero_scale_axis_is_not_degenerate_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(1.0, 0.0, 1.0)
        )));
    }

    /// Two exact-zero object scale axes remain renderable for flat edge geometry.
    #[test]
    fn two_zero_scale_axes_are_not_degenerate_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(0.0, 0.0, 1.0)
        )));
    }

    /// All exact-zero object scale axes collapse mesh draws.
    #[test]
    fn all_zero_scale_axes_are_degenerate_for_draws() {
        assert!(render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::ZERO
        )));
    }

    /// A single near-zero transform axis remains renderable.
    #[test]
    fn single_near_zero_scale_axis_is_not_degenerate_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(1.0, DEGENERATE_RENDER_SCALE_EPS, 1.0)
        )));
    }

    /// All near-zero transform axes collapse mesh draws.
    #[test]
    fn all_near_zero_scale_axes_are_degenerate_for_draws() {
        assert!(render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::splat(DEGENERATE_RENDER_SCALE_EPS)
        )));
    }

    /// Negative nonzero scale preserves mirrored draw semantics instead of being skipped.
    #[test]
    fn negative_nonzero_scale_is_not_degenerate_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(-1.0, 1.0, 1.0)
        )));
    }

    #[test]
    fn render_transform_to_matrix_trs() {
        let t = RenderTransform {
            position: Vec3::new(1.0, 2.0, 3.0),
            scale: Vec3::new(2.0, 2.0, 2.0),
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);
        let col3 = m.col(3);
        assert!((col3.x - 1.0).abs() < 1e-5);
        assert!((col3.y - 2.0).abs() < 1e-5);
        assert!((col3.z - 3.0).abs() < 1e-5);
        assert!((m.col(0).x - 2.0).abs() < 1e-5);
        assert!((m.col(1).y - 2.0).abs() < 1e-5);
        assert!((m.col(2).z - 2.0).abs() < 1e-5);
    }

    /// Finite scale axes, including exact zero, are preserved while non-finite axes fall back to 1.
    #[test]
    fn matrix_scale_axes_preserve_finite_values_and_repair_non_finite_axes() {
        let t = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::new(0.0, f32::NAN, f32::INFINITY),
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);
        assert_eq!(m.col(0).x, 0.0);
        assert!((m.col(1).y - 1.0).abs() < 1e-6);
        assert!((m.col(2).z - 1.0).abs() < 1e-6);
    }

    /// Negative zero scale axes preserve their sign when uploaded to the model matrix.
    #[test]
    fn matrix_scale_axes_preserve_negative_zero_sign() {
        let t = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::new(-0.0, 1.0, 1.0),
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);

        assert!(m.col(0).x.is_sign_negative());
        assert_eq!(m.col(0).x, -0.0);
    }

    /// An all-zero scale vector preserves a collapsed linear part and finite translation.
    #[test]
    fn zero_scale_vector_preserves_zero_scale_matrix() {
        let t = RenderTransform {
            position: Vec3::new(3.0, 4.0, 5.0),
            scale: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);

        assert_eq!(m.col(0).x, 0.0);
        assert_eq!(m.col(1).y, 0.0);
        assert_eq!(m.col(2).z, 0.0);
        assert_eq!(m.col(3).truncate(), Vec3::new(3.0, 4.0, 5.0));
    }

    /// Single-axis zero transforms remain exact singular matrices instead of growing thickness.
    #[test]
    fn single_zero_axis_matrices_preserve_exact_zero_scale_axes() {
        for scale in [
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0),
        ] {
            let m = render_transform_to_matrix(&scaled_transform(scale));

            assert_eq!(m.col(0).truncate().length(), scale.x.abs());
            assert_eq!(m.col(1).truncate().length(), scale.y.abs());
            assert_eq!(m.col(2).truncate().length(), scale.z.abs());
        }
    }

    /// A zero-length rotation quaternion falls back to identity; a finite non-unit quaternion is
    /// passed through so glam can normalize it inside [`glam::Mat4::from_scale_rotation_translation`].
    #[test]
    fn zero_quaternion_falls_back_to_identity_rotation() {
        let t = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        };
        let m = render_transform_to_matrix(&t);
        assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    /// A near-zero quaternion below the guard threshold follows the same identity fallback.
    #[test]
    fn near_zero_quaternion_falls_back_to_identity_rotation() {
        let t = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::from_xyzw(1.0e-10, -1.0e-10, 1.0e-10, -1.0e-10),
        };
        let m = render_transform_to_matrix(&t);

        assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    /// Non-finite position components collapse the translation column to the origin so the matrix
    /// does not leak NaN/inf downstream.
    #[test]
    fn non_finite_position_collapses_to_origin() {
        let t = RenderTransform {
            position: Vec3::new(f32::NAN, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);
        let col3 = m.col(3);
        assert_eq!(col3.x, 0.0);
        assert_eq!(col3.y, 0.0);
        assert_eq!(col3.z, 0.0);
    }

    /// Large finite translations and tiny finite scales survive without producing non-finite output.
    #[test]
    fn large_translation_and_tiny_scale_stay_finite() {
        let position = Vec3::new(1.0e20, -1.0e20, 3.5e19);
        let scale_axis = DEGENERATE_RENDER_SCALE_EPS * 10.0;
        let t = RenderTransform {
            position,
            scale: Vec3::splat(scale_axis),
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);

        for value in m.to_cols_array() {
            assert!(value.is_finite());
        }
        assert_eq!(m.col(3).truncate(), position);
        assert_eq!(m.col(0).x, scale_axis);
        assert_eq!(m.col(1).y, scale_axis);
        assert_eq!(m.col(2).z, scale_axis);
    }

    /// [`multiply_root`] composes the root TRS on the **left**: applying it to an object-local
    /// identity world reproduces the root translation in column 3.
    #[test]
    fn multiply_root_applies_root_transform_on_left() {
        let root = RenderTransform {
            position: Vec3::new(10.0, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        };
        let world_local = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let composed = multiply_root(world_local, &root);
        let col3 = composed.col(3);
        assert!((col3.x - 11.0).abs() < 1e-5);
        assert!(col3.y.abs() < 1e-6);
        assert!(col3.z.abs() < 1e-6);
    }
}
