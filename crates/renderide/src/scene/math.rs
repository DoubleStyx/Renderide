//! TRS -> [`glam::Mat4`] for hierarchy and future GPU uploads.

use glam::{Mat4, Quat, Vec3};

use crate::shared::RenderTransform;

/// Minimum absolute object scale axis that contributes one transform dimension.
pub(crate) const MIN_RENDER_SCALE: f32 = 1e-8;

/// Minimum squared cross-product area that counts as two independent transform dimensions.
const MIN_RENDER_AREA_SQ: f32 =
    MIN_RENDER_SCALE * MIN_RENDER_SCALE * MIN_RENDER_SCALE * MIN_RENDER_SCALE;

/// Returns the sanitized scale axis used for matrix construction.
#[inline]
fn sanitized_scale_axis(axis: f32) -> f32 {
    if !axis.is_finite() {
        1.0
    } else if axis.abs() <= MIN_RENDER_SCALE {
        0.0
    } else {
        axis
    }
}

/// Returns whether an input scale axis contributes one dimension to rasterized triangle geometry.
#[inline]
fn scale_axis_contributes_dimension(axis: f32) -> bool {
    axis.is_finite() && axis.abs() > MIN_RENDER_SCALE
}

/// Returns the sanitized scale vector used for matrix construction.
///
/// When every axis is at or below [`MIN_RENDER_SCALE`] (including exact zeros), matrix construction
/// must use **`Vec3::ZERO`**, not unit scale—otherwise rigs and objects with `(0,0,0)` local scale
/// incorrectly behave like `(1,1,1)` in world matrices and skinning palettes.
#[inline]
fn sanitized_scale(scale: Vec3) -> Vec3 {
    let contributes = scale_axis_contributes_dimension(scale.x)
        || scale_axis_contributes_dimension(scale.y)
        || scale_axis_contributes_dimension(scale.z);
    if scale.is_finite() && !contributes {
        return Vec3::ZERO;
    }
    Vec3::new(
        sanitized_scale_axis(scale.x),
        sanitized_scale_axis(scale.y),
        sanitized_scale_axis(scale.z),
    )
}

/// Returns whether a transform carries non-finite scale data.
#[inline]
pub(crate) fn render_transform_has_invalid_scale(t: &RenderTransform) -> bool {
    let scale = t.scale;
    !(scale.x.is_finite() && scale.y.is_finite() && scale.z.is_finite())
}

/// Returns `true` when the local scale has fewer than two renderable dimensions.
#[inline]
pub(crate) fn render_transform_has_degenerate_scale(t: &RenderTransform) -> bool {
    let scale = t.scale;
    if render_transform_has_invalid_scale(t) {
        return true;
    }
    let axis_count = u8::from(scale_axis_contributes_dimension(scale.x))
        + u8::from(scale_axis_contributes_dimension(scale.y))
        + u8::from(scale_axis_contributes_dimension(scale.z));
    axis_count < 2
}

/// Returns `true` when an effective model matrix has fewer than two independent dimensions.
#[inline]
pub(crate) fn render_matrix_has_degenerate_scale(model: Mat4) -> bool {
    let x = model.x_axis.truncate();
    let y = model.y_axis.truncate();
    let z = model.z_axis.truncate();
    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
        return true;
    }
    let xy = x.cross(y).length_squared();
    let yz = y.cross(z).length_squared();
    let zx = z.cross(x).length_squared();
    if !(xy.is_finite() && yz.is_finite() && zx.is_finite()) {
        return true;
    }
    xy.max(yz).max(zx) <= MIN_RENDER_AREA_SQ
}

/// Builds column-major TRS = `T * R * S`, matching the host `RenderTransform` convention.
#[inline]
pub fn render_transform_to_matrix(t: &RenderTransform) -> Mat4 {
    let scale = sanitized_scale(t.scale);
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

    /// Exact zero on one object scale axis still leaves a rasterizable plane.
    #[test]
    fn one_zero_scale_axis_is_not_degenerate_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(1.0, 0.0, 1.0)
        )));
    }

    /// Two exact zero object scale axes collapse triangle geometry to a line.
    #[test]
    fn two_zero_scale_axes_are_degenerate_for_draws() {
        assert!(render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(1.0, 0.0, 0.0)
        )));
    }

    /// The existing near-zero transform threshold classifies axes as collapsed.
    #[test]
    fn near_zero_scale_axes_are_collapsed_for_draws() {
        assert!(!render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(1.0, MIN_RENDER_SCALE, 1.0)
        )));
        assert!(render_transform_has_degenerate_scale(&scaled_transform(
            Vec3::new(1.0, MIN_RENDER_SCALE, MIN_RENDER_SCALE)
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

    /// Finite zero scale axes are preserved, while non-finite axes fall back to unit scale.
    #[test]
    fn zero_scale_axes_are_preserved_and_non_finite_axes_fall_back() {
        let t = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::new(0.0, f32::NAN, f32::INFINITY),
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);
        assert!(m.col(0).x.abs() < 1e-6);
        assert!((m.col(1).y - 1.0).abs() < 1e-6);
        assert!((m.col(2).z - 1.0).abs() < 1e-6);
    }

    /// An all-zero scale vector maps to collapsed matrix axes while translation stays unchanged.
    #[test]
    fn zero_scale_vector_preserves_collapsed_scale_in_matrix() {
        let t = RenderTransform {
            position: Vec3::new(3.0, 4.0, 5.0),
            scale: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);

        assert!(m.col(0).truncate().length_squared() < 1e-12);
        assert!(m.col(1).truncate().length_squared() < 1e-12);
        assert!(m.col(2).truncate().length_squared() < 1e-12);
        assert_eq!(m.col(3).truncate(), Vec3::new(3.0, 4.0, 5.0));
        assert!(render_transform_has_degenerate_scale(&t));
    }

    /// Matrix rank, not raw local axes, determines the effective drawability of a transform chain.
    #[test]
    fn render_matrix_degenerate_scale_requires_two_independent_dimensions() {
        assert!(!render_matrix_has_degenerate_scale(Mat4::from_scale(
            Vec3::new(1.0, 0.0, 1.0)
        )));
        assert!(render_matrix_has_degenerate_scale(Mat4::from_scale(
            Vec3::new(1.0, 0.0, 0.0)
        )));
        assert!(render_matrix_has_degenerate_scale(Mat4::from_scale(
            Vec3::ZERO
        )));
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
        let scale_axis = MIN_RENDER_SCALE * 10.0;
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
