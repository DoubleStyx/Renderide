//! UV sphere mesh generator used as the test object.
//!
//! Produces interleaved-friendly position/normal/UV vertex arrays plus a triangle-list index buffer.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// One vertex of the test sphere (deterministic packing for `bytemuck` shared-memory transfer).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SphereVertex {
    /// Object-space position (radius `1.0`).
    pub position: [f32; 3],
    /// Smooth shading normal (unit length).
    pub normal: [f32; 3],
    /// UV in `[0, 1] x [0, 1]`.
    pub uv: [f32; 2],
}

/// Procedural UV sphere geometry.
#[derive(Clone, Debug)]
pub struct SphereMesh {
    /// Vertices in interleaved struct-of-arrays order.
    pub vertices: Vec<SphereVertex>,
    /// 32-bit indices in clockwise winding (matches the Unity / D3D-style `FrontFace::Cw`
    /// convention the renderer pipelines expect).
    pub indices: Vec<u32>,
}

impl SphereMesh {
    /// Generates a unit sphere with `latitude` rings (excluding the poles, which are added)
    /// and `longitude` segments around the equator.
    ///
    /// Choosing `latitude = 16, longitude = 24` yields ~624 verts / ~1152 tris which is small
    /// enough to fit comfortably in any IPC ring while still showing recognizable shading.
    pub fn generate(latitude: u32, longitude: u32) -> Self {
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
                vertices.push(SphereVertex {
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
        Self { vertices, indices }
    }
}

#[cfg(test)]
mod tests {
    use super::SphereMesh;

    #[test]
    fn generates_expected_counts() {
        let m = SphereMesh::generate(16, 24);
        assert_eq!(m.vertices.len(), (16 + 1) * (24 + 1));
        assert_eq!(m.indices.len(), 16 * 24 * 6);
    }

    #[test]
    fn all_uvs_in_unit_range() {
        let m = SphereMesh::generate(8, 12);
        for v in &m.vertices {
            assert!((0.0..=1.0).contains(&v.uv[0]));
            assert!((0.0..=1.0).contains(&v.uv[1]));
        }
    }

    #[test]
    fn all_positions_on_unit_sphere() {
        let m = SphereMesh::generate(8, 12);
        for v in &m.vertices {
            let r = v.position[2]
                .mul_add(
                    v.position[2],
                    v.position[0].mul_add(v.position[0], v.position[1] * v.position[1]),
                )
                .sqrt();
            assert!((r - 1.0).abs() < 1e-4, "position not on unit sphere: r={r}");
        }
    }

    #[test]
    fn all_indices_in_vertex_range() {
        let m = SphereMesh::generate(8, 12);
        let n = m.vertices.len() as u32;
        for &idx in &m.indices {
            assert!(idx < n, "index {idx} out of range for {n} vertices");
        }
    }

    #[test]
    fn all_normals_have_unit_length() {
        let m = SphereMesh::generate(8, 12);
        for v in &m.vertices {
            let len = v.normal[0]
                .mul_add(
                    v.normal[0],
                    v.normal[1].mul_add(v.normal[1], v.normal[2] * v.normal[2]),
                )
                .sqrt();
            assert!(
                (len - 1.0).abs() < 1e-3,
                "normal not unit length: |n|={len}"
            );
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let a = SphereMesh::generate(8, 12);
        let b = SphereMesh::generate(8, 12);
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.indices, b.indices);
    }

    #[test]
    fn pole_vertices_are_at_expected_y() {
        let m = SphereMesh::generate(8, 12);
        let lon = 12u32;
        let row = (lon + 1) as usize;
        for v in &m.vertices[..row] {
            assert!(
                (v.position[1] - 1.0).abs() < 1e-4,
                "top ring not at y=+1: {:?}",
                v.position
            );
        }
        let last = m.vertices.len();
        for v in &m.vertices[last - row..] {
            assert!(
                (v.position[1] + 1.0).abs() < 1e-4,
                "bottom ring not at y=-1: {:?}",
                v.position
            );
        }
    }
}
