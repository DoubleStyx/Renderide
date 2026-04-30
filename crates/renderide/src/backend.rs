//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::materials::host_data::MaterialPropertyStore), the compiled
//! [`CompiledRenderGraph`](crate::render_graph::CompiledRenderGraph) after attach, and code paths
//! that turn shared-memory asset payloads into resident GPU resources. [`light_gpu`](crate::backend::light_gpu)
//! packs scene [`ResolvedLight`](crate::scene::ResolvedLight) values for future storage-buffer upload. It does **not**
//! own IPC queues, [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state;
//! callers pass those in where a command requires both transport and GPU work.

mod cluster_gpu;
mod coordinator;
mod debug_hud_bundle;
mod embedded;
pub(crate) mod frame_gpu;
mod frame_gpu_bindings;
mod frame_gpu_error;
mod frame_resource_manager;
pub(crate) mod gpu_jobs;
mod history_registry;
mod light_gpu;
pub(crate) mod material_property_reader;
mod material_system;
mod per_draw_resources;
mod per_view_resource_map;
mod view_resource_registry;

pub use crate::assets::AssetTransferQueue;
pub use cluster_gpu::{
    CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, ClusterBufferCache, ClusterBufferRefs,
    MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub(crate) use coordinator::{ExtractedFrameShared, WorldMeshForwardEncodeRefs};
pub use coordinator::{
    MAX_ASSET_INTEGRATION_QUEUED, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
    RenderBackend, RenderBackendAttachDesc, RenderBackendAttachError,
};
pub use debug_hud_bundle::DebugHudBundle;
pub use embedded::{
    EmbeddedMaterialBindError, EmbeddedMaterialBindResources, EmbeddedTexturePools,
};
pub use frame_gpu::{
    EmptyMaterialBindGroup, FrameGpuResources, FrameSceneSnapshotTextureViews,
    empty_material_bind_group_layout,
};
pub use frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
pub use frame_gpu_error::FrameGpuInitError;
pub use frame_resource_manager::{
    FrameGpuBindContext, FrameResourceManager, PerViewFrameState, PreRecordViewResourceLayout,
};
pub(crate) use gpu_jobs::{
    GpuJobResources, GpuReadbackJobs, GpuReadbackOutcomes, SubmittedReadbackJob,
};
pub use history_registry::{
    BufferHistorySlot, BufferHistorySpec, HistoryRegistry, HistoryRegistryError,
    HistoryResourceScope, HistoryTexture, HistoryTextureMipViews, TextureHistorySlot,
    TextureHistorySpec,
};
pub use light_gpu::{
    GpuLight, MAX_LIGHTS, order_lights_for_clustered_shading,
    order_lights_for_clustered_shading_in_place,
};
pub use material_system::{MAX_PENDING_MATERIAL_BATCHES, MaterialSystem};
pub use per_draw_resources::PerDrawResources;
pub(crate) use view_resource_registry::ViewResourceRegistry;
