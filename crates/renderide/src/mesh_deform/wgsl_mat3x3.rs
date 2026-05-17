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

    /// Cofactor normal matrix for the upper 3x3 of `model`, packed for WGSL `normal_matrix`.
    ///
    /// The shader normalizes the transformed normal after multiplication, so the determinant scale
    /// from a full inverse-transpose normal transform is unnecessary. For non-singular mirrored
    /// transforms, the determinant sign is still applied so normalized directions match the
    /// inverse-transpose result. Keeping the cofactor form also preserves finite singular
    /// transforms instead of replacing zero-scale axes with a fake inverse.
    #[must_use]
    pub(super) fn from_model_upper_3x3(model: Mat4) -> Self {
        let m3 = Mat3::from_mat4(model);
        let cofactor = Mat3::from_cols(
            m3.y_axis.cross(m3.z_axis),
            m3.z_axis.cross(m3.x_axis),
            m3.x_axis.cross(m3.y_axis),
        );
        let determinant = m3.determinant();
        let normal_matrix = if determinant.is_finite() && determinant < 0.0 {
            -cofactor
        } else {
            cofactor
        };
        if !normal_matrix.is_finite() {
            return Self::IDENTITY;
        }
        Self::from_mat3(normal_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Returns a packed matrix column as a `Vec3`.
    fn packed_column(matrix: WgslMat3x3, index: usize) -> Vec3 {
        let column = match index {
            0 => matrix.col0,
            1 => matrix.col1,
            2 => matrix.col2,
            _ => unreachable!("test only requests existing columns"),
        };
        Vec3::new(column[0], column[1], column[2])
    }

    /// Verifies cofactor packing for a simple uniform scale.
    #[test]
    fn normal_matrix_uniform_scale_uses_cofactor_matrix() {
        let m = Mat4::from_scale(Vec3::splat(2.0));
        let nm = WgslMat3x3::from_model_upper_3x3(m);

        assert_eq!(packed_column(nm, 0), Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(packed_column(nm, 1), Vec3::new(0.0, 4.0, 0.0));
        assert_eq!(packed_column(nm, 2), Vec3::new(0.0, 0.0, 4.0));
    }

    /// Mirrored non-singular transforms keep inverse-transpose direction parity after shader normalization.
    #[test]
    fn normal_matrix_negative_scale_matches_inverse_transpose_direction() {
        let m = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        let nm = WgslMat3x3::from_model_upper_3x3(m);

        assert_eq!(packed_column(nm, 0), Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(packed_column(nm, 1), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(packed_column(nm, 2), Vec3::new(0.0, 0.0, 1.0));
    }

    /// Keeps one usable normal axis for a transform flattened onto the YZ plane.
    #[test]
    fn normal_matrix_single_zero_scale_axis_remains_finite() {
        let m = Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0));
        let nm = WgslMat3x3::from_model_upper_3x3(m);

        assert_eq!(packed_column(nm, 0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(packed_column(nm, 1), Vec3::ZERO);
        assert_eq!(packed_column(nm, 2), Vec3::ZERO);
    }

    /// Fully collapsed normal planes stay finite and rely on shader fallback normalization.
    #[test]
    fn normal_matrix_two_zero_scale_axes_remains_finite_zero() {
        let m = Mat4::from_scale(Vec3::new(0.0, 0.0, 1.0));
        let nm = WgslMat3x3::from_model_upper_3x3(m);

        assert_eq!(packed_column(nm, 0), Vec3::ZERO);
        assert_eq!(packed_column(nm, 1), Vec3::ZERO);
        assert_eq!(packed_column(nm, 2), Vec3::ZERO);
    }

    /// Falls back only when the incoming matrix cannot produce finite cofactors.
    #[test]
    fn normal_matrix_non_finite_model_uses_identity() {
        let m = Mat4::from_cols_array(&[
            f32::NAN,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]);
        let nm = WgslMat3x3::from_model_upper_3x3(m);

        assert_eq!(nm, WgslMat3x3::IDENTITY);
    }
}
