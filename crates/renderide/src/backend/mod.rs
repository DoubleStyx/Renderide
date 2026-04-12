//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::assets::material::MaterialPropertyStore), the compiled
//! [`CompiledRenderGraph`](crate::render_graph::CompiledRenderGraph) after attach, and code paths
//! that turn shared-memory asset payloads into resident GPU resources. [`light_gpu`](crate::backend::light_gpu)
//! packs scene [`ResolvedLight`](crate::scene::ResolvedLight) values for future storage-buffer upload. It does **not**
//! own IPC queues, [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state;
//! callers pass those in where a command requires both transport and GPU work.

mod asset_transfer_queue;
mod cluster_gpu;
mod debug_draw;
#[cfg(feature = "debug-hud")]
mod debug_hud_bundle;
mod embedded_material_bind;
mod frame_gpu;
mod frame_resource_manager;
mod light_gpu;
mod material_system;
pub mod mesh_deform;
mod mesh_deform_scratch;
mod occlusion;
mod render_backend;

pub use asset_transfer_queue::AssetTransferQueue;
pub use cluster_gpu::{
    ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE,
    MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub use debug_draw::DebugDrawResources;
#[cfg(feature = "debug-hud")]
pub use debug_hud_bundle::DebugHudBundle;
pub use embedded_material_bind::EmbeddedMaterialBindResources;
pub use frame_gpu::{empty_material_bind_group_layout, EmptyMaterialBindGroup, FrameGpuResources};
pub use frame_resource_manager::FrameResourceManager;
pub use light_gpu::{order_lights_for_clustered_shading, GpuLight, MAX_LIGHTS};
pub use material_system::{MaterialSystem, MAX_PENDING_MATERIAL_BATCHES};
pub use mesh_deform::{
    plan_blendshape_bind_chunks, write_per_draw_uniform_slab, MeshPreprocessPipelines,
    PaddedPerDrawUniforms, INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE,
};
pub use mesh_deform_scratch::{advance_slab_cursor, MeshDeformScratch};
pub use occlusion::OcclusionSystem;
pub use render_backend::{
    RenderBackend, MAX_DEFERRED_MESH_UPLOADS, MAX_PENDING_MESH_UPLOADS,
    MAX_PENDING_TEXTURE_UPLOADS, MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL,
};
