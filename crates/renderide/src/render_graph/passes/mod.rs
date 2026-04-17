//! Concrete render passes registered on a [`super::CompiledRenderGraph`].
//!
//! Phase 2 can add G-buffer, lighting, post, and UI passes here.

mod clustered_light;
mod hi_z_build;
mod mesh_deform;
mod swapchain_clear;
mod world_mesh_forward;

pub use clustered_light::{ClusteredLightModule, ClusteredLightPass};
pub use hi_z_build::{HiZBuildModule, HiZBuildPass};
pub use mesh_deform::{MeshDeformModule, MeshDeformPass};
pub use swapchain_clear::{SwapchainClearModule, SwapchainClearPass};
pub use world_mesh_forward::{WorldMeshForwardModule, WorldMeshForwardPass};
