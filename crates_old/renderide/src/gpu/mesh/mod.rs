//! wgpu mesh rendering with debug texture.
//!
//! Extension point for mesh buffers, vertex formats.

mod buffers;
mod create;
mod decode;
mod fallback;
mod skinned;
mod types;

pub use buffers::GpuMeshBuffers;
pub use create::{compute_vertex_stride_from_mesh, create_mesh_buffers};
pub use types::{
    BlendshapeOffset, VertexPosNormal, VertexPosNormalUv, VertexSkinned, VertexUiCanvas,
    VertexWithUv,
};
