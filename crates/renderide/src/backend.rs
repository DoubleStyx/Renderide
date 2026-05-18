//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::materials::host_data::MaterialPropertyStore), the compiled
//! [`CompiledRenderGraph`](crate::render_graph::CompiledRenderGraph) after attach, and code paths
//! that turn shared-memory asset payloads into resident GPU resources. [`light_gpu`](crate::backend::light_gpu)
//! packs scene [`ResolvedLight`](crate::scene::ResolvedLight) values for future storage-buffer upload. It does **not**
//! own IPC queues, [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state;
//! callers pass those in where a command requires both transport and GPU work.

pub(crate) mod asset_transfers;
mod cluster_gpu;
mod debug_hud_bundle;
mod facade;
pub(crate) mod frame_gpu;
mod frame_gpu_bindings;
mod frame_gpu_error;
mod frame_resource_manager;
pub(crate) mod gpu_jobs;
pub(crate) mod graph;
mod light_gpu;
mod per_draw_resources;
mod per_view_resource_map;
mod secondary_rt_scratch;
mod view_resource_registry;
mod world_mesh_frame_plan;

pub(crate) use crate::render_graph::HistoryRegistry;
pub(crate) use asset_transfers::{AssetIntegrationDrainSummary, AssetTransferQueue};
pub(crate) use facade::ExtractedFrameShared;
#[expect(
    unused_imports,
    reason = "intentional public re-export of attach result error type"
)]
pub use facade::RenderBackendAttachError;
pub use facade::{RenderBackend, RenderBackendAttachDesc};
pub use frame_gpu_bindings::FrameGpuBindingsError;
pub(crate) use frame_resource_manager::FrameLightViewDesc;
pub use frame_resource_manager::FrameResourceManager;
pub(crate) use gpu_jobs::{
    GpuJobResources, GpuReadbackJobs, GpuReadbackOutcomes, SubmittedReadbackJob,
};
pub(crate) use view_resource_registry::ViewResourceRegistry;
pub(crate) use world_mesh_frame_plan::{
    BackendWorldMeshFramePlanner, WorldMeshDrawPlanSlot, prepare_world_mesh_view_blackboard,
};
