//! CPU-side cubemap topology checks matching the WGSL IBL filtering helpers.

use glam::{Vec2, Vec3};

/// Cubemap face in the renderer's canonical layer order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
    /// Every face in canonical cubemap layer order.
    const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

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

/// Cubemap face edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CubeEdge {
    /// Minimum face-local X edge.
    Left,
    /// Maximum face-local X edge.
    Right,
    /// Minimum face-local Y edge.
    Top,
    /// Maximum face-local Y edge.
    Bottom,
}

impl CubeEdge {
    /// Every face edge.
    const ALL: [Self; 4] = [Self::Left, Self::Right, Self::Top, Self::Bottom];

    /// Returns the texel distance from this edge.
    const fn distance(self, x: u32, y: u32, face_size: u32) -> u32 {
        match self {
            Self::Left => x,
            Self::Right => face_size - 1 - x,
            Self::Top => y,
            Self::Bottom => face_size - 1 - y,
        }
    }

    /// Returns the texel paired across this edge for seam-band fix-up.
    fn paired_texel(
        self,
        face: CubeFace,
        x: u32,
        y: u32,
        face_size: u32,
        distance: u32,
    ) -> CubeTexel {
        let last = face_size - 1;
        match self {
            Self::Left => {
                let i = y;
                match face {
                    CubeFace::PosX => CubeTexel::new(CubeFace::PosZ, last - distance, i),
                    CubeFace::NegX => CubeTexel::new(CubeFace::NegZ, last - distance, i),
                    CubeFace::PosY => CubeTexel::new(CubeFace::NegX, i, distance),
                    CubeFace::NegY => CubeTexel::new(CubeFace::NegX, last - i, last - distance),
                    CubeFace::PosZ => CubeTexel::new(CubeFace::NegX, last - distance, i),
                    CubeFace::NegZ => CubeTexel::new(CubeFace::PosX, last - distance, i),
                }
            }
            Self::Right => {
                let i = y;
                match face {
                    CubeFace::PosX => CubeTexel::new(CubeFace::NegZ, distance, i),
                    CubeFace::NegX => CubeTexel::new(CubeFace::PosZ, distance, i),
                    CubeFace::PosY => CubeTexel::new(CubeFace::PosX, last - i, distance),
                    CubeFace::NegY => CubeTexel::new(CubeFace::PosX, i, last - distance),
                    CubeFace::PosZ => CubeTexel::new(CubeFace::PosX, distance, i),
                    CubeFace::NegZ => CubeTexel::new(CubeFace::NegX, distance, i),
                }
            }
            Self::Top => {
                let i = x;
                match face {
                    CubeFace::PosX => CubeTexel::new(CubeFace::PosY, last - distance, last - i),
                    CubeFace::NegX => CubeTexel::new(CubeFace::PosY, distance, i),
                    CubeFace::PosY => CubeTexel::new(CubeFace::NegZ, last - i, distance),
                    CubeFace::NegY => CubeTexel::new(CubeFace::PosZ, i, last - distance),
                    CubeFace::PosZ => CubeTexel::new(CubeFace::PosY, i, last - distance),
                    CubeFace::NegZ => CubeTexel::new(CubeFace::PosY, last - i, distance),
                }
            }
            Self::Bottom => {
                let i = x;
                match face {
                    CubeFace::PosX => CubeTexel::new(CubeFace::NegY, last - distance, i),
                    CubeFace::NegX => CubeTexel::new(CubeFace::NegY, distance, last - i),
                    CubeFace::PosY => CubeTexel::new(CubeFace::PosZ, i, distance),
                    CubeFace::NegY => CubeTexel::new(CubeFace::NegZ, last - i, last - distance),
                    CubeFace::PosZ => CubeTexel::new(CubeFace::NegY, i, distance),
                    CubeFace::NegZ => CubeTexel::new(CubeFace::NegY, last - i, last - distance),
                }
            }
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

/// Canonical cubemap texel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CubeTexel {
    /// Canonical cubemap face.
    face: CubeFace,
    /// Texel X coordinate.
    x: u32,
    /// Texel Y coordinate.
    y: u32,
}

impl CubeTexel {
    /// Builds a canonical cubemap texel address.
    const fn new(face: CubeFace, x: u32, y: u32) -> Self {
        Self { face, x, y }
    }
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

/// Converts a direction to a canonical cubemap texel address.
fn canonical_texel_from_dir(dir: Vec3, face_size: u32) -> CubeTexel {
    let size = face_size.max(1);
    let addr = dir_to_face_uv(dir);
    let xy = (addr.uv * size as f32).floor();
    let max_coord = (size - 1) as f32;
    CubeTexel::new(
        addr.face,
        xy.x.clamp(0.0, max_coord) as u32,
        xy.y.clamp(0.0, max_coord) as u32,
    )
}

/// Converts a face-local virtual texel coordinate to a canonical cubemap texel.
fn canonical_texel_from_face_coord(face: CubeFace, coord: Vec2, face_size: u32) -> CubeTexel {
    canonical_texel_from_dir(face_coord_to_dir(face, coord, face_size), face_size)
}

/// Returns the canonical face reached by a virtual texel-space coordinate.
fn virtual_neighbor_face(face: CubeFace, coord: Vec2, face_size: u32) -> CubeFace {
    dir_to_face_uv(face_coord_to_dir(face, coord, face_size)).face
}

/// Returns the stitch fix-up band width matching `skybox_ibl_stitch.wgsl`.
fn seam_fixup_width(face_size: u32) -> u32 {
    const EDGE_FIXUP_DIVISOR: u32 = 64;
    const MAX_EDGE_FIXUP_WIDTH: u32 = 4;

    if face_size <= 1 {
        return 0;
    }
    let scaled = (face_size / EDGE_FIXUP_DIVISOR).max(1);
    let half_size = (face_size / 2).max(1);
    MAX_EDGE_FIXUP_WIDTH.min(scaled.min(half_size))
}

/// Returns the exact corner stitch group for one face corner.
fn corner_stitch_group(face: CubeFace, x: u32, y: u32, face_size: u32) -> Vec<CubeTexel> {
    let mut group = Vec::with_capacity(3);
    group.push(CubeTexel::new(face, x, y));
    if x == 0 {
        group.push(canonical_texel_from_face_coord(
            face,
            Vec2::new(-1.0, y as f32),
            face_size,
        ));
    }
    if x + 1 >= face_size {
        group.push(canonical_texel_from_face_coord(
            face,
            Vec2::new(face_size as f32, y as f32),
            face_size,
        ));
    }
    if y == 0 {
        group.push(canonical_texel_from_face_coord(
            face,
            Vec2::new(x as f32, -1.0),
            face_size,
        ));
    }
    if y + 1 >= face_size {
        group.push(canonical_texel_from_face_coord(
            face,
            Vec2::new(x as f32, face_size as f32),
            face_size,
        ));
    }
    group
}

/// Returns the stable sort key for a cubemap texel.
fn texel_key(texel: CubeTexel) -> (u32, u32, u32) {
    (texel.face.index(), texel.x, texel.y)
}

/// Returns whether a texel lies on an exact face corner.
const fn is_corner(x: u32, y: u32, face_size: u32) -> bool {
    (x == 0 || x + 1 >= face_size) && (y == 0 || y + 1 >= face_size)
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

    use hashbrown::HashMap;

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
        for face in CubeFace::ALL {
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

    /// Edge fix-up pairs are reciprocal through the whole pull-linear seam band.
    #[test]
    fn seam_fixup_band_pairs_are_reciprocal() {
        for face_size in [4, 8, 128, 256] {
            let width = seam_fixup_width(face_size);
            let mut pairs = HashMap::new();

            for face in CubeFace::ALL {
                for y in 0..face_size {
                    for x in 0..face_size {
                        if is_corner(x, y, face_size) {
                            continue;
                        }
                        let texel = CubeTexel::new(face, x, y);
                        for edge in CubeEdge::ALL {
                            let distance = edge.distance(x, y, face_size);
                            if distance >= width {
                                continue;
                            }
                            let neighbor = edge.paired_texel(face, x, y, face_size, distance);
                            let mut key = [texel, neighbor];
                            key.sort_by_key(|texel| texel_key(*texel));
                            *pairs.entry(key).or_insert(0u32) += 1;
                        }
                    }
                }
            }

            for (pair, count) in pairs {
                assert_eq!(
                    count, 2,
                    "face_size={face_size} reciprocal pair mismatch: {pair:?}"
                );
            }
        }
    }

    /// Exact face corners collapse to the same three cubemap texels from all adjacent faces.
    #[test]
    fn seam_corner_groups_share_three_texels() {
        let face_size = 8;
        let mut groups: HashMap<Vec<CubeTexel>, Vec<CubeTexel>> = HashMap::new();

        for face in CubeFace::ALL {
            for (x, y) in [
                (0, 0),
                (face_size - 1, 0),
                (0, face_size - 1),
                (face_size - 1, face_size - 1),
            ] {
                let mut group = corner_stitch_group(face, x, y, face_size);
                group.sort_by_key(|texel| texel_key(*texel));
                groups
                    .entry(group)
                    .or_default()
                    .push(CubeTexel::new(face, x, y));
            }
        }

        assert_eq!(groups.len(), 8);
        for (group, mut corners) in groups {
            corners.sort_by_key(|texel| texel_key(*texel));
            assert_eq!(group.len(), 3, "group={group:?}");
            assert_eq!(corners.len(), 3, "corners={corners:?}");
            assert_eq!(group, corners);
        }
    }

    /// Cubemap solid-angle weights integrate to the unit sphere area.
    #[test]
    fn solid_angle_weights_sum_to_four_pi() {
        let sum = cube_solid_angle_sum(32);
        assert!((sum - 4.0 * PI).abs() < 1e-4, "sum={sum}");
    }
}
