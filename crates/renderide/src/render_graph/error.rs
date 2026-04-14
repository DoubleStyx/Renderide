//! Errors for graph build, pass execution, and frame submission.

use crate::present::PresentClearError;

use super::ids::PassId;
use super::resources::ResourceSlot;

/// Errors that can occur when building a render graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphBuildError {
    /// The graph contains a cycle; topological sort is impossible.
    #[error("cycle detected in render graph")]
    CycleDetected,

    /// A pass reads a resource slot that no earlier pass produces.
    #[error("pass {pass:?} reads {slot:?} but no earlier pass writes it")]
    MissingDependency {
        /// Pass that requires the missing dependency.
        pass: PassId,
        /// Resource slot that has no producer.
        slot: ResourceSlot,
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

    /// A pass returned an error while recording.
    #[error("pass execution failed: {0}")]
    Pass(#[from] RenderPassError),
}
