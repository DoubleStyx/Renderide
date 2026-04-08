//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::assets::material::MaterialPropertyStore), the compiled
//! [`CompiledRenderGraph`](crate::render_graph::CompiledRenderGraph) after attach, and code paths
//! that turn shared-memory asset payloads into resident GPU resources. [`light_gpu`](crate::backend::light_gpu)
//! packs scene [`ResolvedLight`](crate::scene::ResolvedLight) values for future storage-buffer upload. It does **not**
//! own IPC queues, [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state;
//! callers pass those in where a command requires both transport and GPU work.

mod cluster_gpu;
mod debug_draw;
mod embedded_material_bind;
mod frame_gpu;
mod light_gpu;
mod mesh_deform_scratch;
mod render_backend;

pub use cluster_gpu::{
    ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE,
    MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub use debug_draw::DebugDrawResources;
pub use embedded_material_bind::EmbeddedMaterialBindResources;
pub use frame_gpu::{empty_material_bind_group_layout, EmptyMaterialBindGroup, FrameGpuResources};
pub use light_gpu::{order_lights_for_clustered_shading, GpuLight, MAX_LIGHTS};
pub use mesh_deform_scratch::{advance_slab_cursor, MeshDeformScratch};
pub use render_backend::{
    RenderBackend, MAX_PENDING_MATERIAL_BATCHES, MAX_PENDING_MESH_UPLOADS,
    MAX_PENDING_TEXTURE_UPLOADS,
};
