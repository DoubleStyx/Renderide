//! Concrete render passes registered on a [`super::CompiledRenderGraph`].
//!
//! Phase 2 can add G-buffer, lighting, post, and UI passes here.

mod clustered_light;
mod hi_z_build;
mod mesh_deform;
mod swapchain_clear;
mod world_mesh_forward;

pub use clustered_light::{ClusteredLightGraphResources, ClusteredLightPass};
pub use hi_z_build::{HiZBuildGraphResources, HiZBuildPass};
pub use mesh_deform::MeshDeformPass;
pub use swapchain_clear::SwapchainClearPass;
pub use world_mesh_forward::{
    WorldMeshDepthSnapshotPass, WorldMeshForwardDepthResolvePass, WorldMeshForwardGraphResources,
    WorldMeshForwardIntersectPass, WorldMeshForwardOpaquePass, WorldMeshForwardPass,
    WorldMeshForwardPreparePass,
};
