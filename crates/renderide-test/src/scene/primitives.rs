//! Additional deterministic procedural primitives for visual integration cases.

use glam::{Vec2, Vec3};
use renderide_shared::shared::RenderBoundingBox;

use super::mesh::{Mesh, Vertex};

/// Builds a unit cube with hard face normals and per-face UVs.
pub fn generate_cube() -> Mesh {
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let faces = [
        (
            Vec3::Z,
            [
                Vec3::new(-0.5, -0.5, 0.5),
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(-0.5, 0.5, 0.5),
            ],
        ),
        (
            Vec3::NEG_Z,
            [
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(-0.5, 0.5, -0.5),
                Vec3::new(0.5, 0.5, -0.5),
            ],
        ),
        (
            Vec3::X,
            [
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(0.5, 0.5, -0.5),
                Vec3::new(0.5, 0.5, 0.5),
            ],
        ),
        (
            Vec3::NEG_X,
            [
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(-0.5, -0.5, 0.5),
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(-0.5, 0.5, -0.5),
            ],
        ),
        (
            Vec3::Y,
            [
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(0.5, 0.5, -0.5),
                Vec3::new(-0.5, 0.5, -0.5),
            ],
        ),
        (
            Vec3::NEG_Y,
            [
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(-0.5, -0.5, 0.5),
            ],
        ),
    ];
    let uvs = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    for (normal, positions) in faces {
        let base = vertices.len() as u32;
        for (position, uv) in positions.into_iter().zip(uvs) {
            vertices.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: uv.to_array(),
            });
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
    Mesh { vertices, indices }
}

/// Builds a unit quad in the XY plane facing the camera at `-Z`.
pub fn generate_quad() -> Mesh {
    Mesh {
        vertices: vec![
            vertex([-0.5, -0.5, 0.0], [0.0, 0.0]),
            vertex([0.5, -0.5, 0.0], [1.0, 0.0]),
            vertex([0.5, 0.5, 0.0], [1.0, 1.0]),
            vertex([-0.5, 0.5, 0.0], [0.0, 1.0]),
        ],
        indices: vec![0, 2, 1, 0, 3, 2],
    }
}

/// Bounds for a mesh centered at the origin with known half-extents.
pub fn centered_bounds(extents: Vec3) -> RenderBoundingBox {
    RenderBoundingBox {
        center: Vec3::ZERO,
        extents,
    }
}

fn vertex(position: [f32; 3], uv: [f32; 2]) -> Vertex {
    Vertex {
        position,
        normal: [0.0, 0.0, -1.0],
        uv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_has_hard_normal_vertices() {
        let mesh = generate_cube();
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        for vertex in &mesh.vertices {
            let normal = Vec3::from_array(vertex.normal);
            assert!((0.999..1.001).contains(&normal.length()));
        }
    }

    #[test]
    fn quad_is_two_triangles() {
        let mesh = generate_quad();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, vec![0, 2, 1, 0, 3, 2]);
    }
}
