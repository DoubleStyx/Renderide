//! Reserved fallback cube geometry when mesh data is missing (currently unused).

use bytemuck::{Pod, Zeroable};

use super::types::{VertexPosNormal, VertexWithUv};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
}

/// Fallback cube with position+normal for normal debug pipeline (8 vertices, 12 triangles, 36 indices).
/// Reserved for fallback rendering when mesh geometry is missing.
#[allow(dead_code)]
fn fallback_cube_pos_normal() -> (Vec<VertexPosNormal>, Vec<u16>) {
    let s = 0.5f32;
    let n = [0.0f32, 1.0, 0.0];
    let vertices = vec![
        VertexPosNormal {
            position: [-s, -s, -s],
            normal: n,
        },
        VertexPosNormal {
            position: [s, -s, -s],
            normal: n,
        },
        VertexPosNormal {
            position: [s, s, -s],
            normal: n,
        },
        VertexPosNormal {
            position: [-s, s, -s],
            normal: n,
        },
        VertexPosNormal {
            position: [-s, -s, s],
            normal: n,
        },
        VertexPosNormal {
            position: [s, -s, s],
            normal: n,
        },
        VertexPosNormal {
            position: [s, s, s],
            normal: n,
        },
        VertexPosNormal {
            position: [-s, s, s],
            normal: n,
        },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3, 0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Fallback cube mesh (8 vertices, 12 triangles, 36 indices).
/// Reserved for fallback rendering when mesh geometry is missing.
#[allow(dead_code)]
fn fallback_cube() -> (Vec<Vertex>, Vec<u16>) {
    let s = 0.5f32;
    let vertices = vec![
        Vertex {
            position: [-s, -s, -s],
        },
        Vertex {
            position: [s, -s, -s],
        },
        Vertex {
            position: [s, s, -s],
        },
        Vertex {
            position: [-s, s, -s],
        },
        Vertex {
            position: [-s, -s, s],
        },
        Vertex {
            position: [s, -s, s],
        },
        Vertex {
            position: [s, s, s],
        },
        Vertex {
            position: [-s, s, s],
        },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3, 0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Fallback cube with UVs for UV debug shader.
/// Reserved for fallback rendering when mesh geometry is missing.
#[allow(dead_code)]
fn fallback_cube_with_uv() -> (Vec<VertexWithUv>, Vec<u16>) {
    let s = 0.5f32;
    let vertices = vec![
        VertexWithUv {
            position: [-s, -s, -s],
            uv: [0.0, 0.0],
        },
        VertexWithUv {
            position: [s, -s, -s],
            uv: [1.0, 0.0],
        },
        VertexWithUv {
            position: [s, s, -s],
            uv: [1.0, 1.0],
        },
        VertexWithUv {
            position: [-s, s, -s],
            uv: [0.0, 1.0],
        },
        VertexWithUv {
            position: [-s, -s, s],
            uv: [0.0, 0.0],
        },
        VertexWithUv {
            position: [s, -s, s],
            uv: [1.0, 0.0],
        },
        VertexWithUv {
            position: [s, s, s],
            uv: [1.0, 1.0],
        },
        VertexWithUv {
            position: [-s, s, s],
            uv: [0.0, 1.0],
        },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3, 0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}
