//! Render-graph error crash context categories.

use crate::render_graph::GraphExecuteError;

/// Last render-graph error category recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum GraphErrorKind {
    /// No graph error has been recorded.
    None = 0,
    /// Graph did not contain frame work.
    NoFrameGraph = 1,
    /// Presentation target acquisition or configuration failed.
    Present = 2,
    /// Depth target setup failed.
    DepthTarget = 3,
    /// Swapchain view was unavailable.
    MissingSwapchainView = 4,
    /// Swapchain rendering required a window that was unavailable.
    SwapchainRequiresWindow = 5,
    /// Graph attachment lookup failed.
    MissingGraphAttachment = 6,
    /// Raster pipeline template lookup failed.
    MissingRasterTemplate = 7,
    /// Render pass execution failed.
    Pass = 8,
    /// Graph execution was asked to process an empty view batch.
    NoViewsInBatch = 9,
    /// Transient pool allocation or aliasing failed.
    TransientPool = 10,
    /// History resource registry failed.
    HistoryRegistry = 11,
    /// Required transient resources were missing.
    MissingTransientResources = 12,
    /// Required per-view resources were missing.
    MissingPerViewResources = 13,
    /// Required history texture was missing.
    MissingHistoryTexture = 14,
    /// Required history buffer was missing.
    MissingHistoryBuffer = 15,
    /// History texture was not allocated.
    UnallocatedHistoryTexture = 16,
    /// History buffer was not allocated.
    UnallocatedHistoryBuffer = 17,
    /// Required blackboard slot was missing at pass execution.
    MissingBlackboardSlot = 18,
    /// Error type did not map to a narrower category.
    Other = 19,
}

impl GraphErrorKind {
    /// Converts an atomic storage value into a graph error kind.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::NoFrameGraph,
            2 => Self::Present,
            3 => Self::DepthTarget,
            4 => Self::MissingSwapchainView,
            5 => Self::SwapchainRequiresWindow,
            6 => Self::MissingGraphAttachment,
            7 => Self::MissingRasterTemplate,
            8 => Self::Pass,
            9 => Self::NoViewsInBatch,
            10 => Self::TransientPool,
            11 => Self::HistoryRegistry,
            12 => Self::MissingTransientResources,
            13 => Self::MissingPerViewResources,
            14 => Self::MissingHistoryTexture,
            15 => Self::MissingHistoryBuffer,
            16 => Self::UnallocatedHistoryTexture,
            17 => Self::UnallocatedHistoryBuffer,
            18 => Self::MissingBlackboardSlot,
            19 => Self::Other,
            _ => Self::None,
        }
    }

    /// Returns the stable crash-context label for this graph error kind.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NoFrameGraph => "no-frame-graph",
            Self::Present => "present",
            Self::DepthTarget => "depth-target",
            Self::MissingSwapchainView => "missing-swapchain-view",
            Self::SwapchainRequiresWindow => "swapchain-requires-window",
            Self::MissingGraphAttachment => "missing-graph-attachment",
            Self::MissingRasterTemplate => "missing-raster-template",
            Self::Pass => "pass",
            Self::NoViewsInBatch => "no-views-in-batch",
            Self::TransientPool => "transient-pool",
            Self::HistoryRegistry => "history-registry",
            Self::MissingTransientResources => "missing-transient-resources",
            Self::MissingPerViewResources => "missing-per-view-resources",
            Self::MissingHistoryTexture => "missing-history-texture",
            Self::MissingHistoryBuffer => "missing-history-buffer",
            Self::UnallocatedHistoryTexture => "unallocated-history-texture",
            Self::UnallocatedHistoryBuffer => "unallocated-history-buffer",
            Self::MissingBlackboardSlot => "missing-blackboard-slot",
            Self::Other => "other",
        }
    }
}

/// Maps a render-graph execution error into the compact crash-context category.
pub(crate) fn graph_error_kind(error: &GraphExecuteError) -> GraphErrorKind {
    match error {
        GraphExecuteError::NoFrameGraph => GraphErrorKind::NoFrameGraph,
        GraphExecuteError::Present(_) => GraphErrorKind::Present,
        GraphExecuteError::DepthTarget(_) => GraphErrorKind::DepthTarget,
        GraphExecuteError::MissingSwapchainView => GraphErrorKind::MissingSwapchainView,
        GraphExecuteError::SwapchainRequiresWindow => GraphErrorKind::SwapchainRequiresWindow,
        GraphExecuteError::MissingGraphAttachment { .. } => GraphErrorKind::MissingGraphAttachment,
        GraphExecuteError::MissingRasterTemplate { .. } => GraphErrorKind::MissingRasterTemplate,
        GraphExecuteError::MissingBlackboardSlot { .. } => GraphErrorKind::MissingBlackboardSlot,
        GraphExecuteError::Pass(_) => GraphErrorKind::Pass,
        GraphExecuteError::NoViewsInBatch => GraphErrorKind::NoViewsInBatch,
        GraphExecuteError::TransientPool(_) => GraphErrorKind::TransientPool,
        GraphExecuteError::HistoryRegistry(_) => GraphErrorKind::HistoryRegistry,
        GraphExecuteError::MissingTransientResources => GraphErrorKind::MissingTransientResources,
        GraphExecuteError::MissingPerViewResources { .. } => {
            GraphErrorKind::MissingPerViewResources
        }
        GraphExecuteError::MissingHistoryTexture { .. } => GraphErrorKind::MissingHistoryTexture,
        GraphExecuteError::MissingHistoryBuffer { .. } => GraphErrorKind::MissingHistoryBuffer,
        GraphExecuteError::UnallocatedHistoryTexture { .. } => {
            GraphErrorKind::UnallocatedHistoryTexture
        }
        GraphExecuteError::UnallocatedHistoryBuffer { .. } => {
            GraphErrorKind::UnallocatedHistoryBuffer
        }
    }
}
