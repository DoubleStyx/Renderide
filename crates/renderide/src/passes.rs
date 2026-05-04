//! Concrete render passes registered on a [`crate::render_graph::CompiledRenderGraph`].
//!
//! Each pass implements one of the four typed pass traits:
//! - [`crate::render_graph::pass::RasterPass`] -- raster render passes
//! - [`crate::render_graph::pass::ComputePass`] -- encoder-driven compute
//! - [`crate::render_graph::pass::CopyPass`] -- copy-only work
//! - [`crate::render_graph::pass::CallbackPass`] -- CPU callbacks with no encoder

mod clustered_light;
mod helpers;
mod hi_z_build;
mod mesh_deform;
pub mod post_processing;
mod scene_color_compose;
mod swapchain_clear;
mod world_mesh_forward;

pub use clustered_light::{ClusteredLightGraphResources, ClusteredLightPass};
pub use hi_z_build::{HiZBuildGraphResources, HiZBuildPass};
pub use mesh_deform::MeshDeformPass;
pub use post_processing::{
    AcesTonemapEffect, AcesTonemapGraphResources, AcesTonemapPass, BloomEffect, GtaoEffect,
};
pub use scene_color_compose::{SceneColorComposeGraphResources, SceneColorComposePass};
pub use swapchain_clear::SwapchainClearPass;
pub(crate) use world_mesh_forward::{
    GTAO_VIEW_NORMAL_FORMAT, PipelineVariantKey, PrefetchedWorldMeshDrawsSlot,
};
pub use world_mesh_forward::{
    WorldMeshColorSnapshotPass, WorldMeshDepthSnapshotPass,
    WorldMeshForwardColorResolveGraphResources, WorldMeshForwardColorResolvePass,
    WorldMeshForwardDepthResolvePass, WorldMeshForwardGraphResources,
    WorldMeshForwardIntersectPass, WorldMeshForwardNormalGraphResources,
    WorldMeshForwardNormalPass, WorldMeshForwardOpaquePass, WorldMeshForwardPreparePass,
    WorldMeshForwardTransparentPass,
};
