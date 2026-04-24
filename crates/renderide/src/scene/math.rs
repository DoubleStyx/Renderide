//! TRS → [`glam::Mat4`] for hierarchy and future GPU uploads.

use glam::{Mat4, Quat, Vec3};

use crate::shared::RenderTransform;

const MIN_SCALE: f32 = 1e-8;

/// Builds column-major TRS = `T * R * S`, matching Unity / host `RenderTransform` convention.
#[inline]
pub fn render_transform_to_matrix(t: &RenderTransform) -> Mat4 {
    let sx = if t.scale.x.is_finite() && t.scale.x.abs() >= MIN_SCALE {
        t.scale.x
    } else {
        1.0
    };
    let sy = if t.scale.y.is_finite() && t.scale.y.abs() >= MIN_SCALE {
        t.scale.y
    } else {
        1.0
    };
    let sz = if t.scale.z.is_finite() && t.scale.z.abs() >= MIN_SCALE {
        t.scale.z
    } else {
        1.0
    };
    let scale = Vec3::new(sx, sy, sz);
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
/// contracts—not default mesh, light, or skinning paths.
#[inline]
pub fn multiply_root(world_local: Mat4, root: &RenderTransform) -> Mat4 {
    render_transform_to_matrix(root) * world_local
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Near-zero, NaN, or infinite scale axes fall back to a scale of 1 so the resulting matrix
    /// remains invertible. Each axis is independent.
    #[test]
    fn degenerate_scale_axes_fall_back_to_unit_scale() {
        let t = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::new(0.0, f32::NAN, f32::INFINITY),
            rotation: Quat::IDENTITY,
        };
        let m = render_transform_to_matrix(&t);
        assert!((m.col(0).x - 1.0).abs() < 1e-6);
        assert!((m.col(1).y - 1.0).abs() < 1e-6);
        assert!((m.col(2).z - 1.0).abs() < 1e-6);
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
        assert!(m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-6));
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

    /// [`multiply_root`] composes the root TRS on the **left**: applying it to an object-local
    /// identity world reproduces the root translation in column 3.
    #[test]
    fn multiply_root_applies_root_transform_on_left() {
        let root = RenderTransform {
            position: Vec3::new(10.0, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        };
        let world_local = glam::Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let composed = super::multiply_root(world_local, &root);
        let col3 = composed.col(3);
        assert!((col3.x - 11.0).abs() < 1e-5);
        assert!(col3.y.abs() < 1e-6);
        assert!(col3.z.abs() < 1e-6);
    }
}
