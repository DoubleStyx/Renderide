//! Errors for graph build, pass setup, pass execution, and frame submission.

use crate::present::PresentClearError;

use super::ids::PassId;
use super::resources::{BufferHandle, ImportedBufferHandle, ImportedTextureHandle, TextureHandle};

/// Setup-time validation errors reported by a [`super::RenderPass`].
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    /// Raster passes must declare at least one attachment.
    #[error("raster pass declared no color or depth attachments")]
    RasterWithoutAttachments,

    /// Compute/copy passes cannot declare color or depth attachments.
    #[error("non-raster pass declared a color or depth attachment")]
    NonRasterPassHasAttachment,

    /// Callback passes are reserved for side-effect-only work and cannot declare graph accesses.
    #[error("callback pass declared graph resource accesses")]
    CallbackPassHasAccesses,

    /// A pass referenced a transient texture handle unknown to the graph.
    #[error("unknown transient texture handle {0:?}")]
    UnknownTexture(TextureHandle),

    /// A pass referenced a transient buffer handle unknown to the graph.
    #[error("unknown transient buffer handle {0:?}")]
    UnknownBuffer(BufferHandle),

    /// A pass referenced an imported texture handle unknown to the graph.
    #[error("unknown imported texture handle {0:?}")]
    UnknownImportedTexture(ImportedTextureHandle),

    /// A pass referenced an imported buffer handle unknown to the graph.
    #[error("unknown imported buffer handle {0:?}")]
    UnknownImportedBuffer(ImportedBufferHandle),

    /// Pass-specific setup failure.
    #[error("{0}")]
    Message(String),
}

/// Errors that can occur when building a render graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphBuildError {
    /// The graph contains a cycle; topological sort is impossible.
    #[error("cycle detected in render graph")]
    CycleDetected,

    /// A pass id in an explicit edge is outside this builder.
    #[error("edge references pass outside graph: {from:?} -> {to:?}")]
    InvalidEdge {
        /// Source pass.
        from: PassId,
        /// Destination pass.
        to: PassId,
    },

    /// A pass reads a transient resource that no earlier pass produces.
    #[error("pass {pass:?} reads transient resource `{resource}` but no earlier pass writes it")]
    MissingDependency {
        /// Pass that requires the missing dependency.
        pass: PassId,
        /// Human-readable resource label.
        resource: String,
    },

    /// Pass setup failed.
    #[error("setup failed for pass {pass:?} `{name}`: {source}")]
    Setup {
        /// Pass id.
        pass: PassId,
        /// Pass name.
        name: String,
        /// Setup validation error.
        source: SetupError,
    },
}

/// Failure inside a single [`super::RenderPass::execute`] call.
#[derive(Debug, thiserror::Error)]
pub enum RenderPassError {
    /// A pass that writes or samples the swapchain target ran without an acquired backbuffer view.
    #[error("pass `{pass}` requires swapchain view but none was provided")]
    MissingBackbuffer {
        /// Pass name from [`super::RenderPass::name`].
        pass: String,
    },

    /// A pass that writes depth ran without a depth attachment view.
    #[error("pass `{pass}` requires depth view but none was provided")]
    MissingDepth {
        /// Pass name from [`super::RenderPass::name`].
        pass: String,
    },

    /// Frame params (scene/backend) were not supplied for a mesh pass.
    #[error("pass `{pass}` requires FrameRenderParams but none was provided")]
    MissingFrameParams {
        /// Pass name from [`super::RenderPass::name`].
        pass: String,
    },
}

/// Frame-level failure when recording or presenting the compiled graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphExecuteError {
    /// No compiled graph was installed (e.g. GPU attach failed before graph build).
    #[error("no frame graph configured on render backend")]
    NoFrameGraph,

    /// Surface acquisition or recovery failed after retry.
    #[error(transparent)]
    Present(#[from] PresentClearError),

    /// Main depth attachment could not be ensured for the current surface extent.
    #[error("GPU depth attachment unavailable")]
    DepthTarget,

    /// A [`super::FrameViewTarget::Swapchain`] view was scheduled without an acquired surface texture.
    #[error("swapchain backbuffer missing for swapchain view")]
    MissingSwapchainView,

    /// A graph-managed raster pass could not resolve one of its declared attachments.
    #[error("pass `{pass}` could not resolve graph attachment `{resource}`")]
    MissingGraphAttachment {
        /// Pass name.
        pass: String,
        /// Resource label.
        resource: String,
    },

    /// A pass opted into graph-managed raster execution but has no raster template.
    #[error("pass `{pass}` opted into graph-managed raster execution without a raster template")]
    MissingRasterTemplate {
        /// Pass name.
        pass: String,
    },

    /// A pass returned an error while recording.
    #[error("pass execution failed: {0}")]
    Pass(#[from] RenderPassError),
}
