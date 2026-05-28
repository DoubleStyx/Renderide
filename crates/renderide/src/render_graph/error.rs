//! Errors for graph build, pass setup, pass execution, and frame submission.

use crate::present::PresentClearError;

use super::ids::PassId;
use super::resources::{
    BufferHandle, HistorySlotId, ImportedBufferHandle, ImportedTextureHandle, SubresourceHandle,
    TextureHandle,
};
use crate::camera::ViewId;

/// Setup-time validation errors reported by a [`super::RenderPass`].
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    /// Raster passes must declare at least one attachment.
    #[error("raster pass declared no color or depth attachments")]
    RasterWithoutAttachments,

    /// Compute passes cannot declare color or depth attachments.
    #[error("non-raster pass declared a color or depth attachment")]
    NonRasterPassHasAttachment,

    /// Scheduler records through wgpu's single queue and cannot require real async compute.
    #[error(
        "async compute was required, but the current wgpu scheduler only records single-queue work"
    )]
    AsyncComputeRequiredUnsupported,

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

    /// A pass referenced a transient texture subresource handle unknown to the graph.
    #[error("unknown transient texture subresource handle {0:?}")]
    UnknownSubresource(SubresourceHandle),
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

    /// Internal invariant violated while taking ownership of compiled passes (should never happen).
    #[error("render graph build pass ownership invariant violated: {message}")]
    PassOwnershipInvariant {
        /// Short invariant description.
        message: &'static str,
    },

    /// A declared texture subresource points outside its parent texture.
    #[error("invalid texture subresource {handle:?}: {reason}")]
    InvalidSubresource {
        /// Invalid subresource handle.
        handle: SubresourceHandle,
        /// Short validation reason.
        reason: &'static str,
    },

    /// An imported frame target has a final access that requires a retained writer.
    #[error(
        "import `{label}` requires final access `{final_access}` but no retained pass writes it"
    )]
    MissingImportedFinalWriter {
        /// Import declaration label.
        label: &'static str,
        /// Final access category.
        final_access: &'static str,
    },

    /// The compiled execution schedule violated scheduler invariants.
    #[error("invalid render graph schedule: {source}")]
    InvalidSchedule {
        /// Schedule validation failure.
        source: super::schedule::ScheduleValidationError,
    },

    /// Graph validation found declaration issues in strict mode.
    #[error("render graph validation failed: {report}")]
    Validation {
        /// Validation report containing the diagnostics.
        report: super::validation::GraphValidationReport,
    },
}

/// Failure inside a single [`super::RenderPass::execute`] call.
#[derive(Debug, thiserror::Error)]
pub enum RenderPassError {
    /// Frame params (scene/backend) were not supplied for a mesh pass.
    #[error("pass `{pass}` requires GraphPassFrame but none was provided")]
    FrameParamsRequired {
        /// Pass name from [`super::RenderPass::name`].
        pass: String,
    },

    /// A compute/copy pass expected a resolved imported texture but none was available.
    #[error("pass `{pass}` requires imported texture `{resource}` but it was not resolved")]
    UnresolvedImportedTexture {
        /// Pass name from [`super::RenderPass::name`].
        pass: String,
        /// Human-readable imported resource label.
        resource: &'static str,
    },

    /// A compute/copy pass expected a ping-pong history texture but got a non-history import.
    #[error(
        "pass `{pass}` requires history texture `{resource}` but the import has no history backing"
    )]
    HistoryImportWithoutBacking {
        /// Pass name from [`super::RenderPass::name`].
        pass: String,
        /// Human-readable imported resource label.
        resource: &'static str,
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
    #[error("GPU depth attachment unavailable: {0}")]
    DepthTarget(&'static str),

    /// A [`super::FrameViewTarget::Swapchain`] view was scheduled without an acquired surface texture.
    #[error("swapchain backbuffer missing for swapchain view")]
    MissingSwapchainView,

    /// A [`super::FrameViewTarget::Swapchain`] view was scheduled but no winit window was provided
    /// (the headless offscreen path passes [`None`] here and must not target the swapchain).
    #[error("swapchain view requires a winit window but none was provided (headless mode)")]
    SwapchainRequiresWindow,

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

    /// Multi-view execution was invoked with no views while frame-global work was required.
    #[error("no views in batch for frame-global graph execution")]
    NoViewsInBatch,

    /// Transient GPU pool could not produce a lease (internal invariant violated).
    #[error(transparent)]
    TransientPool(#[from] super::pool::TransientPoolError),

    /// History registry rejected a slot registration.
    #[error(transparent)]
    HistoryRegistry(#[from] super::HistoryRegistryError),

    /// Per-view recording looked up a transient-resource key that the pre-resolve step did not populate.
    ///
    /// Raised as an internal invariant violation when [`super::compiled::CompiledRenderGraph`]
    /// drives per-view encoding after `pre_resolve_transients_for_views` was expected to have
    /// inserted an entry for every unique [`super::compiled::GraphResolveKey`].
    #[error("per-view record missing pre-resolved transient resources")]
    MissingTransientResources,

    /// A per-view frame resource required for command recording was not prepared.
    #[error("per-view {resource} resources were not prepared for view {view_id:?}")]
    MissingPerViewResources {
        /// View whose resource was missing.
        view_id: ViewId,
        /// Short resource name, such as `frame` or `per-draw`.
        resource: &'static str,
    },

    /// A graph ping-pong texture import referenced a history slot that was not registered.
    #[error("history texture slot `{slot}` was not registered for graph import `{import_label}`")]
    MissingHistoryTexture {
        /// Slot id referenced by the import.
        slot: &'static str,
        /// Import declaration label.
        import_label: &'static str,
    },

    /// A graph ping-pong buffer import referenced a history slot that was not registered.
    #[error("history buffer slot `{slot}` was not registered for graph import `{import_label}`")]
    MissingHistoryBuffer {
        /// Slot id referenced by the import.
        slot: &'static str,
        /// Import declaration label.
        import_label: &'static str,
    },

    /// A registered history texture slot was selected before its two halves were allocated.
    #[error("history texture slot `{slot}` has no allocated {half} half")]
    UnallocatedHistoryTexture {
        /// Slot id referenced by the import.
        slot: &'static str,
        /// Selected half, either "current" or "previous".
        half: &'static str,
    },

    /// A registered history buffer slot was selected before its two halves were allocated.
    #[error("history buffer slot `{slot}` has no allocated {half} half")]
    UnallocatedHistoryBuffer {
        /// Slot id referenced by the import.
        slot: &'static str,
        /// Selected half, either "current" or "previous".
        half: &'static str,
    },

    /// A pass declared a required blackboard slot that was missing when the pass recorded.
    #[error("pass `{pass}` requires blackboard slot `{slot}` but it was not present")]
    MissingBlackboardSlot {
        /// Pass name.
        pass: String,
        /// Slot that was missing.
        slot: &'static str,
    },

    /// A pass accessed a blackboard slot without declaring the access during setup.
    #[error("pass `{pass}` performed undeclared blackboard {access} access to slot `{slot}`")]
    UndeclaredBlackboardAccess {
        /// Pass name.
        pass: String,
        /// Slot that was accessed.
        slot: &'static str,
        /// Access kind observed at runtime.
        access: &'static str,
    },
}

impl GraphExecuteError {
    /// Builds a missing texture-history error from a strongly typed slot id.
    pub(crate) fn missing_history_texture(slot: HistorySlotId, import_label: &'static str) -> Self {
        Self::MissingHistoryTexture {
            slot: slot.name(),
            import_label,
        }
    }

    /// Builds a missing buffer-history error from a strongly typed slot id.
    pub(crate) fn missing_history_buffer(slot: HistorySlotId, import_label: &'static str) -> Self {
        Self::MissingHistoryBuffer {
            slot: slot.name(),
            import_label,
        }
    }

    /// Builds an unallocated texture-history error from a strongly typed slot id.
    pub(crate) fn unallocated_history_texture(slot: HistorySlotId, half: &'static str) -> Self {
        Self::UnallocatedHistoryTexture {
            slot: slot.name(),
            half,
        }
    }

    /// Builds an unallocated buffer-history error from a strongly typed slot id.
    pub(crate) fn unallocated_history_buffer(slot: HistorySlotId, half: &'static str) -> Self {
        Self::UnallocatedHistoryBuffer {
            slot: slot.name(),
            half,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_error_display_includes_unknown_handles() {
        assert_eq!(
            SetupError::UnknownTexture(TextureHandle(3)).to_string(),
            "unknown transient texture handle TextureHandle(3)"
        );
        assert_eq!(
            SetupError::UnknownImportedBuffer(ImportedBufferHandle(4)).to_string(),
            "unknown imported buffer handle ImportedBufferHandle(4)"
        );
    }

    #[test]
    fn graph_build_error_display_includes_pass_and_resource_context() {
        let error = GraphBuildError::MissingDependency {
            pass: PassId(2),
            resource: String::from("scene-color"),
        };

        assert_eq!(
            error.to_string(),
            "pass PassId(2) reads transient resource `scene-color` but no earlier pass writes it"
        );
    }

    #[test]
    fn history_error_constructors_use_slot_names() {
        let slot = HistorySlotId::new("temporal");

        assert_eq!(
            GraphExecuteError::missing_history_texture(slot, "history-color").to_string(),
            "history texture slot `temporal` was not registered for graph import `history-color`"
        );
        assert_eq!(
            GraphExecuteError::unallocated_history_buffer(slot, "previous").to_string(),
            "history buffer slot `temporal` has no allocated previous half"
        );
    }

    #[test]
    fn missing_per_view_resources_display_includes_view_and_resource() {
        assert_eq!(
            GraphExecuteError::MissingPerViewResources {
                view_id: ViewId::Main,
                resource: "frame",
            }
            .to_string(),
            "per-view frame resources were not prepared for view Main"
        );
    }
}
