//! UV sphere mesh generator.
//!
//! Produces a [`Mesh`] with smooth normals, UV coords in `[0,1]x[0,1]`, and clockwise winding.

use glam::{Vec2, Vec3};

use super::mesh::{Mesh, Vertex};

/// Generates a unit sphere with `latitude` rings (excluding the poles, which are added)
/// and `longitude` segments around the equator.
///
/// Choosing `latitude = 16, longitude = 24` yields ~624 verts / ~1152 tris which is small
/// enough to fit comfortably in any IPC ring while still showing recognizable shading.
pub fn generate_sphere(latitude: u32, longitude: u32) -> Mesh {
    assert!(
        latitude >= 2 && longitude >= 3,
        "sphere needs at least 2 latitude rings and 3 longitude segments"
    );
    let lat = latitude;
    let lon = longitude;
    let mut vertices = Vec::with_capacity(((lat + 1) * (lon + 1)) as usize);
    for i in 0..=lat {
        let v = i as f32 / lat as f32;
        let phi = v * std::f32::consts::PI;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=lon {
            let u = j as f32 / lon as f32;
            let theta = u * std::f32::consts::TAU;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let pos = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            let normal = pos.normalize_or_zero();
            let uv = Vec2::new(u, 1.0 - v);
            vertices.push(Vertex {
                position: pos.to_array(),
                normal: normal.to_array(),
                uv: uv.to_array(),
            });
        }
    }
    let mut indices = Vec::with_capacity((lat * lon * 6) as usize);
    for i in 0..lat {
        for j in 0..lon {
            let row = lon + 1;
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
    use super::generate_sphere;

    #[test]
    fn generates_expected_counts() {
        let m = generate_sphere(16, 24);
        assert_eq!(m.vertices.len(), (16 + 1) * (24 + 1));
        assert_eq!(m.indices.len(), 16 * 24 * 6);
    }

    #[test]
    fn vertices_have_unit_length_normals() {
        let m = generate_sphere(8, 12);
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
}
