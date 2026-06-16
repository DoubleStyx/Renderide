//! Point-light shadow cube-face topology for shadow projection.

use glam::Vec3;

/// Point-light shadow cube face count.
pub(super) const POINT_FACE_COUNT: u32 = 6;

/// Returns the camera direction and up axis for a point-light shadow face.
pub(super) fn basis(face: u32) -> (Vec3, Vec3) {
    match face % POINT_FACE_COUNT {
        0 => (Vec3::X, Vec3::Y),
        1 => (Vec3::NEG_X, Vec3::Y),
        2 => (Vec3::Y, Vec3::NEG_Z),
        3 => (Vec3::NEG_Y, Vec3::Z),
        4 => (Vec3::Z, Vec3::Y),
        _ => (Vec3::NEG_Z, Vec3::Y),
    }
}

#[cfg(test)]
fn face_index(direction: Vec3) -> u32 {
    let a = direction.abs();
    if a.x >= a.y && a.x >= a.z {
        return u32::from(direction.x < 0.0);
    }
    if a.y >= a.z {
        return 2 + u32::from(direction.y < 0.0);
    }
    4 + u32::from(direction.z < 0.0)
}

#[cfg(test)]
fn right_axis(face: u32) -> Vec3 {
    let (direction, up) = basis(face);
    direction.cross(up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_face_order_matches_shader_face_indices() {
        let expected = [
            (Vec3::X, Vec3::Y),
            (Vec3::NEG_X, Vec3::Y),
            (Vec3::Y, Vec3::NEG_Z),
            (Vec3::NEG_Y, Vec3::Z),
            (Vec3::Z, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y),
        ];
        for (face, expected_basis) in expected.into_iter().enumerate() {
            assert_eq!(basis(face as u32), expected_basis);
        }
    }

    #[test]
    fn point_face_indices_follow_largest_axis_tie_order() {
        for (face, (direction, _)) in (0..POINT_FACE_COUNT).map(|face| (face, basis(face))) {
            assert_eq!(face_index(direction), face);
        }
        assert_eq!(face_index(Vec3::new(1.0, 1.0, 0.0)), 0);
        assert_eq!(face_index(Vec3::new(0.0, 1.0, 1.0)), 2);
    }

    #[test]
    fn point_face_right_axes_match_look_at_convention() {
        let expected = [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::NEG_X,
            Vec3::NEG_X,
            Vec3::NEG_X,
            Vec3::X,
        ];
        for (face, expected_right) in expected.into_iter().enumerate() {
            assert_eq!(right_axis(face as u32), expected_right);
        }
    }
}
