//! Procedural torus mesh generator.
//!
//! Produces a [`Mesh`] in object space with smooth normals and UVs spanning `[0,1]x[0,1]`,
//! where U wraps around the major (large) circle and V wraps around the minor (tube) circle.
//! Clockwise winding matches the renderer's `FrontFace::Cw` convention.

use glam::{Vec2, Vec3};

use super::mesh::{Mesh, Vertex};

/// Generates a torus with `major_segments` divisions around the large circle and
/// `minor_segments` divisions around the tube cross-section. `major_radius` is the distance
/// from the torus center to the tube center; `minor_radius` is the tube cross-section radius.
///
/// The defaults `(48, 24, 0.65, 0.25)` produce a torus that fits comfortably inside a unit
/// cube centered at the origin (`r_major + r_minor = 0.9`).
pub fn generate_torus(
    major_segments: u32,
    minor_segments: u32,
    major_radius: f32,
    minor_radius: f32,
) -> Mesh {
    assert!(
        major_segments >= 3 && minor_segments >= 3,
        "torus needs at least 3 major and 3 minor segments"
    );
    assert!(
        major_radius > 0.0 && minor_radius > 0.0 && minor_radius < major_radius,
        "torus requires 0 < minor_radius < major_radius (got minor={minor_radius}, major={major_radius})"
    );

    let maj = major_segments;
    let min = minor_segments;
    let mut vertices = Vec::with_capacity(((maj + 1) * (min + 1)) as usize);
    for i in 0..=maj {
        let u = i as f32 / maj as f32;
        let theta = u * std::f32::consts::TAU; // around major circle
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        for j in 0..=min {
            let v = j as f32 / min as f32;
            let phi = v * std::f32::consts::TAU; // around tube cross section
            let cos_p = phi.cos();
            let sin_p = phi.sin();

            let r = major_radius + minor_radius * cos_p;
            let pos = Vec3::new(r * cos_t, minor_radius * sin_p, r * sin_t);
            // Tube cross-section center for this ring; the surface normal points from there
            // outward to the surface position.
            let tube_center = Vec3::new(major_radius * cos_t, 0.0, major_radius * sin_t);
            let normal = (pos - tube_center).normalize_or_zero();
            let uv = Vec2::new(u, v);
            vertices.push(Vertex {
                position: pos.to_array(),
                normal: normal.to_array(),
                uv: uv.to_array(),
            });
        }
    }

    let mut indices = Vec::with_capacity((maj * min * 6) as usize);
    let row = min + 1;
    for i in 0..maj {
        for j in 0..min {
            let v0 = i * row + j;
            let v1 = (i + 1) * row + j;
            let v2 = (i + 1) * row + (j + 1);
            let v3 = i * row + (j + 1);
            indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
        }
    }

    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::generate_torus;

    #[test]
    fn generates_expected_counts() {
        let m = generate_torus(48, 24, 0.65, 0.25);
        assert_eq!(m.vertices.len(), (48 + 1) * (24 + 1));
        assert_eq!(m.indices.len(), 48 * 24 * 6);
    }

    #[test]
    fn vertices_have_unit_length_normals() {
        let m = generate_torus(16, 8, 0.6, 0.2);
        for v in &m.vertices {
            let n = glam::Vec3::from_array(v.normal);
            let len = n.length();
            assert!(
                len == 0.0 || (0.999..1.001).contains(&len),
                "expected unit normal, got {:?}",
                v.normal
            );
        }
    }

    #[test]
    fn surface_distance_matches_minor_radius() {
        let m = generate_torus(16, 8, 0.6, 0.2);
        for v in &m.vertices {
            let pos = glam::Vec3::from_array(v.position);
            let r_major_from_pos = (pos.x * pos.x + pos.z * pos.z).sqrt();
            let dx = r_major_from_pos - 0.6;
            let dy = pos.y;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!(
                (0.199..0.201).contains(&dist),
                "surface point {pos:?} not on tube of minor radius 0.2 (got {dist})"
            );
        }
    }

    #[test]
    fn rejects_invalid_radii() {
        let result = std::panic::catch_unwind(|| generate_torus(8, 6, 0.2, 0.6));
        assert!(result.is_err(), "expected panic when minor >= major");
    }
}
