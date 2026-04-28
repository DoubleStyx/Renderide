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
mod debug_hud_bundle;
mod embedded;
mod frame_gpu;
mod frame_gpu_bindings;
mod frame_gpu_error;
mod frame_resource_manager;
mod history_registry;
mod light_gpu;
mod material_system;
pub mod mesh_deform;
mod occlusion;
mod per_draw_resources;
mod per_view_resource_map;
mod reflection_probe_sh2;
mod render_backend;
mod skybox_specular;

pub use crate::assets::AssetTransferQueue;
pub use cluster_gpu::{
    ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE,
    MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub use debug_hud_bundle::DebugHudBundle;
pub use embedded::{
    EmbeddedMaterialBindError, EmbeddedMaterialBindResources, EmbeddedTexturePools,
};
pub use frame_gpu::{
    empty_material_bind_group_layout, EmptyMaterialBindGroup, FrameGpuResources,
    FrameSceneSnapshotTextureViews,
};
pub use frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
pub use frame_gpu_error::FrameGpuInitError;
pub use frame_resource_manager::{
    FrameGpuBindContext, FrameResourceManager, PerViewFrameState, PreRecordViewResourceLayout,
};
pub use history_registry::{
    BufferHistorySlot, BufferHistorySpec, HistoryRegistry, HistoryRegistryError,
    HistoryResourceScope, HistoryTexture, HistoryTextureMipViews, TextureHistorySlot,
    TextureHistorySpec,
};
pub use light_gpu::{
    order_lights_for_clustered_shading, order_lights_for_clustered_shading_in_place, GpuLight,
    MAX_LIGHTS,
};
pub use material_system::{MaterialSystem, MAX_PENDING_MATERIAL_BATCHES};
pub use mesh_deform::{
    advance_slab_cursor, blendshape_sparse_buffers_fit_device, plan_blendshape_scatter_chunks,
    write_per_draw_uniform_slab, MeshDeformScratch, MeshPreprocessPipelines, PaddedPerDrawUniforms,
    WgslMat3x3, INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE,
};
pub(crate) use occlusion::HiZBuildInput;
pub use occlusion::OcclusionSystem;
pub use per_draw_resources::PerDrawResources;
pub(crate) use reflection_probe_sh2::ReflectionProbeSh2System;
pub(crate) use render_backend::WorldMeshForwardEncodeRefs;
pub use render_backend::{
    RenderBackend, RenderBackendAttachDesc, RenderBackendAttachError, MAX_ASSET_INTEGRATION_QUEUED,
    MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
};
pub(crate) use skybox_specular::resolve_active_main_skybox_specular_environment;
