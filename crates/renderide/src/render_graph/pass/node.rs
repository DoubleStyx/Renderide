//! [`PassNode`] enum: the union type for all four pass kinds stored in the render graph.
//!
//! The graph stores `Vec<PassNode>` instead of `Vec<Box<dyn RenderPass>>`. The executor matches
//! on the variant to dispatch to the correct context type and recording path without a runtime
//! `graph_managed_raster()` toggle.

use super::{CallbackPass, ComputePass, CopyPass, RasterPass};
use crate::render_graph::compiled::{DepthAttachmentTemplate, RenderPassTemplate};
use crate::render_graph::context::{
    CallbackCtx, ComputePassCtx, CopyPassCtx, PostSubmitContext, RasterPassCtx,
};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::builder::PassBuilder;

/// Command domain for a compiled pass.
///
/// Mirrors the [`PassNode`] variant and is stored in [`crate::render_graph::compiled::CompiledPassInfo`]
/// for diagnostics and validation without holding a trait-object reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// Raster render pass opened by the graph.
    Raster,
    /// Encoder-driven compute pass.
    Compute,
    /// Encoder-driven copy-only pass.
    Copy,
    /// CPU callback with no encoder (planning, uploads, blackboard mutations).
    Callback,
}

/// Scheduling phase: when in the multi-view loop a pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassPhase {
    /// Runs exactly once per tick before any per-view passes (e.g. mesh deform compute).
    FrameGlobal,
    /// Runs once per [`crate::render_graph::compiled::FrameView`] in the view loop.
    PerView,
}

/// Execution scope for a group (mirrors [`PassPhase`] at the group level).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupScope {
    /// Runs once per tick.
    FrameGlobal,
    /// Runs once per view.
    PerView,
}

impl From<PassPhase> for GroupScope {
    fn from(value: PassPhase) -> Self {
        match value {
            PassPhase::FrameGlobal => Self::FrameGlobal,
            PassPhase::PerView => Self::PerView,
        }
    }
}

/// Backend hint describing whether a pass is safe to merge with an adjacent pass that reads
/// the same attachments.
///
/// Populated by passes at setup time via [`crate::render_graph::pass::PassBuilder::merge_hint`].
/// The current wgpu executor ignores the hint (each pass opens its own render pass), so populating
/// the hint on existing passes is a no-op today. It exists as scaffolding for a future
/// subpass-aware backend (tile-based mobile GPUs, Vulkan subpasses, Metal tile shading) that can
/// merge adjacent raster passes sharing attachments to preserve tile memory and avoid redundant
/// load/store traffic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PassMergeHint {
    /// When `true`, adjacent passes writing to the same attachments may reuse the render-pass
    /// encoder without resolving / storing attachment contents in between. Safe when the next
    /// pass in the group will read or continue writing the same attachments.
    pub attachment_reuse: bool,
    /// When `true`, the pass should prefer keeping attachment data in on-chip tile memory across
    /// a merge boundary. Used on tiled-GPU backends to skip the tile-store step between merged
    /// subpasses.
    pub tile_memory_preferred: bool,
}

/// One node in the compiled render graph.
///
/// Wraps one of the four pass kinds, each with its own trait object. The executor matches on
/// this enum to open the correct pass type and context.
pub enum PassNode {
    /// Graph-managed raster pass.
    Raster(Box<dyn RasterPass>),
    /// Encoder-driven compute pass.
    Compute(Box<dyn ComputePass>),
    /// Encoder-driven copy-only pass.
    Copy(Box<dyn CopyPass>),
    /// CPU callback pass (no encoder).
    Callback(Box<dyn CallbackPass>),
}

impl PassNode {
    /// Stable name for logging and error messages.
    pub fn name(&self) -> &str {
        match self {
            Self::Raster(p) => p.name(),
            Self::Compute(p) => p.name(),
            Self::Copy(p) => p.name(),
            Self::Callback(p) => p.name(),
        }
    }

    /// Command kind for this node.
    pub fn kind(&self) -> PassKind {
        match self {
            Self::Raster(_) => PassKind::Raster,
            Self::Compute(_) => PassKind::Compute,
            Self::Copy(_) => PassKind::Copy,
            Self::Callback(_) => PassKind::Callback,
        }
    }

    /// Scheduling phase.
    pub fn phase(&self) -> PassPhase {
        match self {
            Self::Raster(p) => p.phase(),
            Self::Compute(p) => p.phase(),
            Self::Copy(p) => p.phase(),
            Self::Callback(p) => p.phase(),
        }
    }

    /// Calls the inner pass's `setup` method using `name` for builder context.
    ///
    /// `name` should match [`Self::name()`]; it is passed separately so callers can supply a
    /// `&str` with the required lifetime for [`PassBuilder`].
    pub(crate) fn call_setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        match self {
            Self::Raster(p) => p.setup(builder),
            Self::Compute(p) => p.setup(builder),
            Self::Copy(p) => p.setup(builder),
            Self::Callback(p) => p.setup(builder),
        }
    }

    /// Runs [`CallbackPass::run`]. Returns `Ok(())` for non-callback variants (no-op).
    pub(crate) fn run_callback(
        &self,
        ctx: &mut CallbackCtx<'_, '_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Callback(p) => p.run(ctx),
            _ => Ok(()),
        }
    }

    /// Records compute commands into the encoder held in `ctx`. Returns `Ok(())` for non-compute variants.
    pub(crate) fn record_compute(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Compute(p) => p.record(ctx),
            _ => Ok(()),
        }
    }

    /// Records copy commands into the encoder held in `ctx`. Returns `Ok(())` for non-copy variants.
    pub(crate) fn record_copy(
        &self,
        ctx: &mut CopyPassCtx<'_, '_, '_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Copy(p) => p.record(ctx),
            _ => Ok(()),
        }
    }

    /// Records raster draw commands into an already-open render pass.
    /// Returns `Ok(())` for non-raster variants (no-op).
    pub(crate) fn record_raster(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Raster(p) => p.record(ctx, rpass),
            _ => Ok(()),
        }
    }

    /// Returns whether a raster pass should be opened for this view. Returns `true` for non-raster variants.
    pub(crate) fn should_record_raster(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
    ) -> Result<bool, RenderPassError> {
        match self {
            Self::Raster(p) => p.should_record(ctx),
            _ => Ok(true),
        }
    }

    /// Runtime multiview mask override for raster passes. Returns the template's mask for others.
    pub(crate) fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<std::num::NonZeroU32> {
        match self {
            Self::Raster(p) => p.multiview_mask_override(ctx, template),
            _ => template.multiview_mask,
        }
    }

    /// Runtime stencil ops override for raster passes. Returns template default for others.
    pub(crate) fn stencil_ops_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        depth: &DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        match self {
            Self::Raster(p) => p.stencil_ops_override(ctx, depth),
            _ => depth.stencil,
        }
    }

    /// Dispatches `post_submit` to the correct inner trait.
    pub(crate) fn post_submit(
        &mut self,
        ctx: &mut PostSubmitContext<'_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Raster(p) => p.post_submit(ctx),
            Self::Compute(p) => p.post_submit(ctx),
            Self::Copy(p) => p.post_submit(ctx),
            Self::Callback(p) => p.post_submit(ctx),
        }
    }

    /// Releases view-scoped caches for views that are no longer active.
    pub(crate) fn release_view_resources(&mut self, retired_views: &[crate::render_graph::ViewId]) {
        match self {
            Self::Raster(p) => p.release_view_resources(retired_views),
            Self::Compute(p) => p.release_view_resources(retired_views),
            Self::Copy(p) => p.release_view_resources(retired_views),
            Self::Callback(p) => p.release_view_resources(retired_views),
        }
    }
}

// SAFETY: every variant payload (`RasterPass`, `ComputePass`, …) is a `Box<dyn Trait + Send>`;
// therefore every field of `PassNode` is `Send`, and the enum as a whole may be safely shared
// across rayon worker threads when recording per-view encoders.
unsafe impl Send for PassNode {}
