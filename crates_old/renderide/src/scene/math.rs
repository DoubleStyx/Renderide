//! Math utilities for transform and matrix operations.
//!
//! Uses glam for SIMD-optimized matrix operations in the hot path.
//! Provides conversions between nalgebra and glam for pipeline uniform upload.

use glam::{Mat4, Quat, Vec3};
use nalgebra::Matrix4;

use crate::shared::RenderTransform;

/// Converts nalgebra `Matrix4` to glam `Mat4`.
///
/// Both use column-major layout; the conversion preserves the same memory layout.
#[inline(always)]
pub fn matrix_na_to_glam(m: &Matrix4<f32>) -> Mat4 {
    Mat4::from_cols_array(&[
        m[(0, 0)],
        m[(1, 0)],
        m[(2, 0)],
        m[(3, 0)],
        m[(0, 1)],
        m[(1, 1)],
        m[(2, 1)],
        m[(3, 1)],
        m[(0, 2)],
        m[(1, 2)],
        m[(2, 2)],
        m[(3, 2)],
        m[(0, 3)],
        m[(1, 3)],
        m[(2, 3)],
        m[(3, 3)],
    ])
}

/// Converts glam `Mat4` to nalgebra `Matrix4` for pipeline uniform upload.
///
/// Both use column-major layout; the conversion preserves the same memory layout.
#[inline(always)]
pub fn matrix_glam_to_na(m: Mat4) -> Matrix4<f32> {
    let a = m.to_cols_array();
    Matrix4::from_fn(|r, c| a[c * 4 + r])
}

const MIN_SCALE: f32 = 1e-8;

/// Converts a RenderTransform to a 4x4 model matrix (translation * rotation * scale).
/// Uses glam for SIMD-optimized TRS construction.
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
        || t.rotation.i.abs() >= 1e-8
        || t.rotation.j.abs() >= 1e-8
        || t.rotation.k.abs() >= 1e-8
    {
        Quat::from_xyzw(t.rotation.i, t.rotation.j, t.rotation.k, t.rotation.w)
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    /// Round-trip test: position (1,2,3), identity rotation, scale (2,2,2).
    /// Verifies TRS order (trans * rot * scale) matches Unity Transform convention.
    #[test]
    fn test_render_transform_to_matrix_trs() {
        let t = RenderTransform {
            position: nalgebra::Vector3::new(1.0, 2.0, 3.0),
            scale: nalgebra::Vector3::new(2.0, 2.0, 2.0),
            rotation: Quaternion::identity(),
        };
        let m = render_transform_to_matrix(&t);
        let col3 = m.col(3);
        assert!((col3.x - 1.0).abs() < 1e-5, "translation x");
        assert!((col3.y - 2.0).abs() < 1e-5, "translation y");
        assert!((col3.z - 3.0).abs() < 1e-5, "translation z");
        assert!((m.col(0).x - 2.0).abs() < 1e-5, "scale xx");
        assert!((m.col(1).y - 2.0).abs() < 1e-5, "scale yy");
        assert!((m.col(2).z - 2.0).abs() < 1e-5, "scale zz");
    }
}
