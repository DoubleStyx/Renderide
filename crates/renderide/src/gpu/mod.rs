//! GPU state, pipelines, and mesh rendering.

pub mod mesh;
pub mod pipeline;
pub mod registry;
pub mod state;

pub use mesh::{GpuMeshBuffers, compute_vertex_stride_from_mesh, create_mesh_buffers};
pub use pipeline::{RenderPipeline, UniformData};
pub use registry::{PipelineKey, PipelineManager, PipelineRegistry, PipelineVariant};
pub use state::{GpuState, ensure_depth_texture, init_gpu};
