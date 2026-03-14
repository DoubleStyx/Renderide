//! Render backend: GPU resources and presentation.
//!
//! Re-exports from gpu_mesh for the mesh pipeline. The present logic lives in main.rs
//! and consumes draw batches from Session.

pub use crate::gpu_mesh::{create_mesh_buffers, GpuMeshBuffers, MeshPipeline};
