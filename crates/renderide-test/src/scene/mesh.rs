//! Procedural mesh primitives shared by all scene-DSL cases.
//!
//! Vertex layout matches what [`super::mesh_payload::pack_mesh_upload`] expects: position
//! (`[f32; 3]`), normal (`[f32; 3]`), and UV (`[f32; 2]`). Both [`super::sphere::SphereMesh`]
//! and [`super::torus::TorusMesh`] expose [`Mesh`] aliases so they share one packing path.

use bytemuck::{Pod, Zeroable};

/// One vertex of a procedural test mesh.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Smooth shading normal (unit length, except where the geometry calls for facet shading).
    pub normal: [f32; 3],
    /// UV in `[0, 1] x [0, 1]`.
    pub uv: [f32; 2],
}

/// Triangle-list mesh in clockwise winding (matches the renderer's Unity/D3D `FrontFace::Cw`).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// Interleaved-friendly vertex array.
    pub vertices: Vec<Vertex>,
    /// 32-bit indices.
    pub indices: Vec<u32>,
}
