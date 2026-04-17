//! GPU resources that exist only after a successful [`super::RenderBackend::attach`].
//!
//! Frame `@group(0/1/2)` binds live in [`super::FrameResourceManager`]; this module holds mesh
//! deform scratch, skin arenas, optional preprocess pipelines, and optional MSAA depth resolve.

use std::sync::Arc;

use super::embedded::EmbeddedMaterialBindError;
use super::frame_gpu_bindings::FrameGpuBindingsError;
use super::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::gpu::MsaaDepthResolveResources;

/// Mesh deform scratch, skin arenas, optional preprocess pipelines, and optional MSAA depth resolve.
///
/// Installed on [`super::RenderBackend`] when GPU attach succeeds; frame `@group(0/1/2)` binds live in
/// [`super::FrameResourceManager::gpu_binds`](super::FrameResourceManager).
pub struct GpuAttached {
    /// Scratch buffers for mesh deformation compute.
    pub(crate) mesh_deform_scratch: MeshDeformScratch,
    /// Arena-backed deformed vertex streams keyed by renderable.
    pub(crate) gpu_skin_cache: GpuSkinCache,
    /// Optional skinning / blendshape compute pipelines when shader creation succeeded.
    pub(crate) mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// MSAA depth → R32F → single-sample depth resolve when supported.
    pub(crate) msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
}

/// Failure to bring [`super::RenderBackend`] to a GPU-backed state (no partial critical install).
#[derive(Debug, thiserror::Error)]
pub enum RenderBackendAttachError {
    /// `@group(0)` / `@group(2)` frame or per-draw bind bundle could not be created.
    #[error("frame GPU bind bundle: {0}")]
    FrameBinds(#[from] FrameGpuBindingsError),
    /// Embedded `@group(1)` resources are required for mesh forward.
    #[error("embedded material binds: {0}")]
    EmbeddedMaterial(#[from] EmbeddedMaterialBindError),
}
