//! Concrete render passes registered on a [`super::CompiledRenderGraph`].
//!
//! Phase 2 can add G-buffer, lighting, post, and UI passes here.

mod mesh_deform;
mod swapchain_clear;
mod world_mesh_forward;

pub use mesh_deform::MeshDeformPass;
pub use swapchain_clear::SwapchainClearPass;
pub use world_mesh_forward::WorldMeshForwardPass;
