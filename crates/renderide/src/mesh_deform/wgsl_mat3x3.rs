//! Column-major `mat3x3<f32>` packed for WGSL storage alignment (each column padded to 16 bytes).

use glam::{Mat3, Mat4};

/// Column-major `mat3x3` with WGSL storage layout: each column is `vec3` padded to 16 bytes.
///
/// Matches [`mat3x3<f32>`](https://www.w3.org/TR/WGSL/#alignment-and-size) in storage (`vec3` stride 16).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WgslMat3x3 {
    /// First column (x, y, z, _pad).
    pub col0: [f32; 4],
    /// Second column (x, y, z, _pad).
    pub col1: [f32; 4],
    /// Third column (x, y, z, _pad).
    pub col2: [f32; 4],
}

impl WgslMat3x3 {
    /// Identity `mat3x3` (flat normals unchanged when `model` is identity).
    pub(super) const IDENTITY: Self = Self {
        col0: [1.0, 0.0, 0.0, 0.0],
        col1: [0.0, 1.0, 0.0, 0.0],
        col2: [0.0, 0.0, 1.0, 0.0],
    };

    /// Packs a glam [`Mat3`] into WGSL column-major storage layout.
    #[must_use]
    pub(super) fn from_mat3(matrix: Mat3) -> Self {
        let c0 = matrix.x_axis;
        let c1 = matrix.y_axis;
        let c2 = matrix.z_axis;
        Self {
            col0: [c0.x, c0.y, c0.z, 0.0],
            col1: [c1.x, c1.y, c1.z, 0.0],
            col2: [c2.x, c2.y, c2.z, 0.0],
        }
    }

    /// `transpose(inverse(M))` for the upper 3x3 of `model`, packed for WGSL `normal_matrix`.
    ///
    /// For singular or near-singular linear parts, returns identity to avoid NaNs in the shader.
    #[must_use]
    pub(super) fn from_model_upper_3x3(model: Mat4) -> Self {
        let m3 = Mat3::from_mat4(model);
        let det = m3.determinant();
        if !det.is_finite() || det.abs() < 1e-20 {
            return Self::IDENTITY;
        }
        let nm = m3.inverse().transpose();
        Self::from_mat3(nm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn normal_matrix_uniform_scale_matches_model_linear() {
        let m = Mat4::from_scale(Vec3::splat(2.0));
        let nm = WgslMat3x3::from_model_upper_3x3(m);
        let m3 = Mat3::from_mat4(m);
        let expected = m3.inverse().transpose();
        let c0 = Vec3::new(nm.col0[0], nm.col0[1], nm.col0[2]);
        assert!((c0 - expected.x_axis).length() < 1e-4);
    }
}
