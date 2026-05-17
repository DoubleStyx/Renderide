//! CPU-side cubemap topology checks matching the WGSL IBL filtering helpers.

use glam::{Vec2, Vec3};

/// Cubemap face in the renderer's canonical layer order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CubeFace {
    /// Positive X face.
    PosX,
    /// Negative X face.
    NegX,
    /// Positive Y face.
    PosY,
    /// Negative Y face.
    NegY,
    /// Positive Z face.
    PosZ,
    /// Negative Z face.
    NegZ,
}

impl CubeFace {
    /// Returns the numeric cubemap array layer offset for this face.
    const fn index(self) -> u32 {
        match self {
            Self::PosX => 0,
            Self::NegX => 1,
            Self::PosY => 2,
            Self::NegY => 3,
            Self::PosZ => 4,
            Self::NegZ => 5,
        }
    }

    /// Returns the face for a numeric cubemap array layer offset.
    fn from_index(index: u32) -> Self {
        match index {
            0 => Self::PosX,
            1 => Self::NegX,
            2 => Self::PosY,
            3 => Self::NegY,
            4 => Self::PosZ,
            _ => Self::NegZ,
        }
    }
}

/// Canonical cubemap address.
#[derive(Clone, Copy, Debug)]
struct CubeAddress {
    /// Canonical cubemap face.
    face: CubeFace,
    /// Normalized face UV in `[0, 1]`.
    uv: Vec2,
}

/// Converts a face and normalized UV to a canonical world direction.
fn face_uv_to_dir(face: CubeFace, uv: Vec2) -> Vec3 {
    let st = uv * 2.0 - Vec2::ONE;
    let s = st.x;
    let t = st.y;
    match face {
        CubeFace::PosX => Vec3::new(1.0, -t, -s).normalize(),
        CubeFace::NegX => Vec3::new(-1.0, -t, s).normalize(),
        CubeFace::PosY => Vec3::new(s, 1.0, t).normalize(),
        CubeFace::NegY => Vec3::new(s, -1.0, -t).normalize(),
        CubeFace::PosZ => Vec3::new(s, -t, 1.0).normalize(),
        CubeFace::NegZ => Vec3::new(-s, -t, -1.0).normalize(),
    }
}

/// Converts a face and texel-space coordinate to a canonical world direction.
fn face_coord_to_dir(face: CubeFace, coord: Vec2, face_size: u32) -> Vec3 {
    let size = face_size.max(1) as f32;
    face_uv_to_dir(face, (coord + Vec2::splat(0.5)) / size)
}

/// Converts a direction to its canonical cubemap face and normalized UV.
fn dir_to_face_uv(dir: Vec3) -> CubeAddress {
    let d = dir.normalize();
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        if d.x >= 0.0 {
            return CubeAddress {
                face: CubeFace::PosX,
                uv: Vec2::new(-d.z / a.x, -d.y / a.x) * 0.5 + Vec2::splat(0.5),
            };
        }
        return CubeAddress {
            face: CubeFace::NegX,
            uv: Vec2::new(d.z / a.x, -d.y / a.x) * 0.5 + Vec2::splat(0.5),
        };
    }
    if a.y >= a.z {
        if d.y >= 0.0 {
            return CubeAddress {
                face: CubeFace::PosY,
                uv: Vec2::new(d.x / a.y, d.z / a.y) * 0.5 + Vec2::splat(0.5),
            };
        }
        return CubeAddress {
            face: CubeFace::NegY,
            uv: Vec2::new(d.x / a.y, -d.z / a.y) * 0.5 + Vec2::splat(0.5),
        };
    }
    if d.z >= 0.0 {
        return CubeAddress {
            face: CubeFace::PosZ,
            uv: Vec2::new(d.x / a.z, -d.y / a.z) * 0.5 + Vec2::splat(0.5),
        };
    }
    CubeAddress {
        face: CubeFace::NegZ,
        uv: Vec2::new(-d.x / a.z, -d.y / a.z) * 0.5 + Vec2::splat(0.5),
    }
}

/// Returns the canonical face reached by a virtual texel-space coordinate.
fn virtual_neighbor_face(face: CubeFace, coord: Vec2, face_size: u32) -> CubeFace {
    dir_to_face_uv(face_coord_to_dir(face, coord, face_size)).face
}

/// Exact area-element primitive for cubemap texel solid angles.
fn area_element(x: f32, y: f32) -> f32 {
    (x * y).atan2((x * x + y * y + 1.0).sqrt())
}

/// Exact solid angle of one texel in a cubemap face.
fn texel_solid_angle(x: u32, y: u32, face_size: u32) -> f32 {
    let size = face_size.max(1) as f32;
    let x0 = 2.0 * x as f32 / size - 1.0;
    let y0 = 2.0 * y as f32 / size - 1.0;
    let x1 = 2.0 * (x + 1) as f32 / size - 1.0;
    let y1 = 2.0 * (y + 1) as f32 / size - 1.0;
    (area_element(x0, y0) - area_element(x0, y1) - area_element(x1, y0) + area_element(x1, y1))
        .abs()
}

/// Sum of all cubemap texel solid angles for one face size.
fn cube_solid_angle_sum(face_size: u32) -> f32 {
    let mut sum = 0.0;
    for _face in 0..6 {
        for y in 0..face_size {
            for x in 0..face_size {
                sum += texel_solid_angle(x, y, face_size);
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    /// Face center directions match the canonical layer order.
    #[test]
    fn face_centers_match_canonical_axes() {
        let centers = [
            (CubeFace::PosX, Vec3::X),
            (CubeFace::NegX, Vec3::NEG_X),
            (CubeFace::PosY, Vec3::Y),
            (CubeFace::NegY, Vec3::NEG_Y),
            (CubeFace::PosZ, Vec3::Z),
            (CubeFace::NegZ, Vec3::NEG_Z),
        ];

        for (face, expected) in centers {
            let dir = face_uv_to_dir(face, Vec2::splat(0.5));
            assert!(dir.abs_diff_eq(expected, 1e-6), "{face:?} -> {dir:?}");
            assert_eq!(CubeFace::from_index(face.index()), face);
        }
    }

    /// Direction addressing round-trips representative interior UVs on every face.
    #[test]
    fn direction_address_round_trips_face_uv() {
        for face_index in 0..6 {
            let face = CubeFace::from_index(face_index);
            for uv in [
                Vec2::new(0.25, 0.25),
                Vec2::new(0.5, 0.75),
                Vec2::new(0.8, 0.4),
            ] {
                let addr = dir_to_face_uv(face_uv_to_dir(face, uv));
                assert_eq!(addr.face, face);
                assert!(
                    addr.uv.abs_diff_eq(uv, 1e-6),
                    "{face:?} {uv:?} -> {:?}",
                    addr.uv
                );
            }
        }
    }

    /// Top and bottom faces remap across their edges into the lateral faces with correct polarity.
    #[test]
    fn top_and_bottom_edges_have_expected_neighbors() {
        let n = 8;
        let mid = Vec2::splat(3.5);

        assert_eq!(
            virtual_neighbor_face(CubeFace::PosY, Vec2::new(mid.x, -1.0), n),
            CubeFace::NegZ
        );
        assert_eq!(
            virtual_neighbor_face(CubeFace::PosY, Vec2::new(mid.x, n as f32), n),
            CubeFace::PosZ
        );
        assert_eq!(
            virtual_neighbor_face(CubeFace::PosY, Vec2::new(-1.0, mid.y), n),
            CubeFace::NegX
        );
        assert_eq!(
            virtual_neighbor_face(CubeFace::PosY, Vec2::new(n as f32, mid.y), n),
            CubeFace::PosX
        );

        assert_eq!(
            virtual_neighbor_face(CubeFace::NegY, Vec2::new(mid.x, -1.0), n),
            CubeFace::PosZ
        );
        assert_eq!(
            virtual_neighbor_face(CubeFace::NegY, Vec2::new(mid.x, n as f32), n),
            CubeFace::NegZ
        );
        assert_eq!(
            virtual_neighbor_face(CubeFace::NegY, Vec2::new(-1.0, mid.y), n),
            CubeFace::NegX
        );
        assert_eq!(
            virtual_neighbor_face(CubeFace::NegY, Vec2::new(n as f32, mid.y), n),
            CubeFace::PosX
        );
    }

    /// Cubemap solid-angle weights integrate to the unit sphere area.
    #[test]
    fn solid_angle_weights_sum_to_four_pi() {
        let sum = cube_solid_angle_sum(32);
        assert!((sum - 4.0 * PI).abs() < 1e-4, "sum={sum}");
    }
}
