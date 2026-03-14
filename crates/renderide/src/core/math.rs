//! Math utilities for transform and matrix operations.

use nalgebra::{Matrix4, UnitQuaternion, Vector3};

use crate::shared::RenderTransform;

const MIN_SCALE: f32 = 1e-8;

/// Converts a RenderTransform to a 4x4 model matrix (translation * rotation * scale).
pub fn render_transform_to_matrix(t: &RenderTransform) -> Matrix4<f32> {
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
    let scale = Matrix4::new_nonuniform_scaling(&Vector3::new(sx, sy, sz));
    let rot: Matrix4<f32> = UnitQuaternion::try_new(t.rotation, 1e-8)
        .map(|u| u.to_homogeneous())
        .unwrap_or_else(Matrix4::identity);
    let pos = if t.position.x.is_finite() && t.position.y.is_finite() && t.position.z.is_finite() {
        t.position
    } else {
        Vector3::zeros()
    };
    let trans = Matrix4::new_translation(&pos);
    trans * rot * scale
}
