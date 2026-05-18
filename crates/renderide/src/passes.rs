//! Concrete render passes registered on a [`crate::render_graph::CompiledRenderGraph`].
//!
//! Each pass implements one of the typed pass traits:
//! - [`crate::render_graph::pass::RasterPass`] -- raster render passes
//! - [`crate::render_graph::pass::ComputePass`] -- encoder-driven compute
//! - [`crate::render_graph::pass::EncoderPass`] -- encoder-driven mixed copy/render work

mod clustered_light;
mod helpers;
mod hi_z_build;
mod mesh_deform;
pub(crate) mod post_processing;
mod scene_color_compose;
mod world_mesh_forward;

pub(crate) use clustered_light::{ClusteredLightGraphResources, ClusteredLightPass};
pub(crate) use hi_z_build::{HiZBuildGraphResources, HiZBuildPass};
pub(crate) use mesh_deform::MeshDeformPass;
pub(crate) use post_processing::{
    AcesTonemapEffect, AgxTonemapEffect, AutoExposureEffect, BloomEffect, GtaoEffect,
    MotionBlurEffect,
};
pub(crate) use scene_color_compose::{SceneColorComposeGraphResources, SceneColorComposePass};
pub(crate) use world_mesh_forward::{
    GTAO_VIEW_NORMAL_FORMAT, MaterialBatchBoundary, PreparedWorldMeshForwardFrame,
    WorldMeshDepthSnapshotPass, WorldMeshForwardDepthPrepass,
    WorldMeshForwardDepthPrepassGraphResources, WorldMeshForwardDepthResolvePass,
    WorldMeshForwardEncodeRefs, WorldMeshForwardGraphResources, WorldMeshForwardIntersectPass,
    WorldMeshForwardNormalGraphResources, WorldMeshForwardNormalPass, WorldMeshForwardOpaquePass,
    WorldMeshForwardPlanSlot, WorldMeshForwardPrepareContext, WorldMeshForwardSkyboxRenderer,
    WorldMeshForwardTransparentSequencePass, prepare_world_mesh_forward_frame,
};
