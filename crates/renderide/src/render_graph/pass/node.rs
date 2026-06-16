//! [`PassNode`] enum: the union type for pass kinds stored in the render graph.
//!
//! The graph stores `Vec<PassNode>` instead of `Vec<Box<dyn RenderPass>>`. The executor matches
//! on the variant to dispatch to the correct context type and recording path without a runtime
//! `graph_managed_raster()` toggle.

use std::borrow::Cow;

use super::{ComputePass, EncoderPass, RasterPass};
use crate::graph_inputs::FrameGlobalResourcePass;
use crate::render_graph::context::{
    ComputePassCtx, EncoderPassCtx, PostSubmitContext, RasterPassCtx,
};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::builder::PassBuilder;
use crate::render_graph::pass::{DepthAttachmentTemplate, RenderPassTemplate};

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
    /// Encoder-driven pass that may interleave copies, resolves, and manually opened passes.
    Encoder,
}

/// Scheduler-visible workload and execution policy flags for one pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PassWorkloadFlags {
    bits: u16,
}

impl PassWorkloadFlags {
    /// No workload or policy flags.
    pub const EMPTY: Self = Self { bits: 0 };
    /// Pass uses graph-managed rasterization on the graphics queue.
    pub const RASTER: Self = Self { bits: 1 << 0 };
    /// Pass uses compute work on the graphics queue.
    pub const COMPUTE: Self = Self { bits: 1 << 1 };
    /// Pass records copy or mixed encoder commands directly.
    pub const COPY_ENCODER: Self = Self { bits: 1 << 2 };
    /// Pass and its producers are kept even when no graph export consumes them.
    pub const NEVER_CULL: Self = Self { bits: 1 << 3 };
    /// Pass must not be folded into an adjacent render-pass merge group.
    pub const NEVER_MERGE: Self = Self { bits: 1 << 4 };
    /// Pass must stay on the main recording path even when parallel recording is available.
    pub const NEVER_PARALLEL: Self = Self { bits: 1 << 5 };
    /// Compute pass is eligible for async-compute scheduling on a future multi-queue backend.
    pub const ASYNC_COMPUTE_CAPABLE: Self = Self { bits: 1 << 6 };

    /// Returns whether every flag in `other` is present.
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Returns a copy of this set with `other` inserted.
    pub const fn with(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Inserts `other` into this set.
    pub fn insert(&mut self, other: Self) {
        self.bits |= other.bits;
    }

    /// Returns the scheduler workload bit implied by a pass kind.
    pub const fn for_kind(kind: PassKind) -> Self {
        match kind {
            PassKind::Raster => Self::RASTER,
            PassKind::Compute => Self::COMPUTE,
            PassKind::Encoder => Self::COPY_ENCODER,
        }
    }
}

impl std::ops::BitOr for PassWorkloadFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.with(rhs)
    }
}

impl std::ops::BitOrAssign for PassWorkloadFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
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
/// Scheduler uses the hint when building conservative merge groups. The wgpu executor
/// materializes compatible groups into one render pass when load/store, multiview, and stencil
/// state allow it.
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
/// Wraps one of the pass kinds, each with its own trait object. The executor matches on this enum
/// to open the correct pass type and context.
pub enum PassNode {
    /// Graph-managed raster pass.
    Raster(Box<dyn RasterPass>),
    /// Encoder-driven compute pass.
    Compute(Box<dyn ComputePass>),
    /// Encoder-driven mixed pass.
    Encoder(Box<dyn EncoderPass>),
}

impl PassNode {
    /// Stable name for logging and error messages.
    pub fn name(&self) -> &str {
        match self {
            Self::Raster(p) => p.name(),
            Self::Compute(p) => p.name(),
            Self::Encoder(p) => p.name(),
        }
    }

    /// Human-readable label for profiler markers.
    pub fn profiling_label(&self) -> Cow<'_, str> {
        match self {
            Self::Raster(p) => p.profiling_label(),
            Self::Compute(p) => p.profiling_label(),
            Self::Encoder(p) => p.profiling_label(),
        }
    }

    /// Command kind for this node.
    pub fn kind(&self) -> PassKind {
        match self {
            Self::Raster(_) => PassKind::Raster,
            Self::Compute(_) => PassKind::Compute,
            Self::Encoder(_) => PassKind::Encoder,
        }
    }

    /// Scheduling phase.
    pub fn phase(&self) -> PassPhase {
        match self {
            Self::Raster(p) => p.phase(),
            Self::Compute(p) => p.phase(),
            Self::Encoder(p) => p.phase(),
        }
    }

    /// Typed backend resource hook for renderer-owned frame-global pass resources.
    pub fn frame_global_resource_pass(&self) -> Option<FrameGlobalResourcePass> {
        match self {
            Self::Raster(_) => None,
            Self::Compute(p) => p.frame_global_resource_pass(),
            Self::Encoder(p) => p.frame_global_resource_pass(),
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
            Self::Encoder(p) => p.setup(builder),
        }
    }

    /// Records compute commands into the encoder held in `ctx`. Returns `Ok(())` for non-compute variants.
    pub(crate) fn record_compute(
        &self,
        ctx: &mut ComputePassCtx<'_, '_, '_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Compute(p) => p.record(ctx),
            Self::Raster(_) | Self::Encoder(_) => Ok(()),
        }
    }

    /// Returns whether a compute pass should be recorded for this view. Returns `true` for non-compute variants.
    pub(crate) fn should_record_compute(
        &self,
        ctx: &ComputePassCtx<'_, '_, '_>,
    ) -> Result<bool, RenderPassError> {
        match self {
            Self::Compute(p) => p.should_record(ctx),
            Self::Raster(_) | Self::Encoder(_) => Ok(true),
        }
    }

    /// Records encoder commands into the encoder held in `ctx`. Returns `Ok(())` for non-encoder variants.
    pub(crate) fn record_encoder(
        &self,
        ctx: &mut EncoderPassCtx<'_, '_, '_>,
    ) -> Result<(), RenderPassError> {
        match self {
            Self::Encoder(p) => p.record(ctx),
            Self::Raster(_) | Self::Compute(_) => Ok(()),
        }
    }

    /// Returns whether an encoder pass should be recorded for this view. Returns `true` for non-encoder variants.
    pub(crate) fn should_record_encoder(
        &self,
        ctx: &EncoderPassCtx<'_, '_, '_>,
    ) -> Result<bool, RenderPassError> {
        match self {
            Self::Encoder(p) => p.should_record(ctx),
            Self::Raster(_) | Self::Compute(_) => Ok(true),
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
            Self::Compute(_) | Self::Encoder(_) => Ok(()),
        }
    }

    /// Returns whether a raster pass should be opened for this view. Returns `true` for non-raster variants.
    pub(crate) fn should_record_raster(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
    ) -> Result<bool, RenderPassError> {
        match self {
            Self::Raster(p) => p.should_record(ctx),
            Self::Compute(_) | Self::Encoder(_) => Ok(true),
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
            Self::Compute(_) | Self::Encoder(_) => template.multiview_mask,
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
            Self::Compute(_) | Self::Encoder(_) => depth.stencil,
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
            Self::Encoder(p) => p.post_submit(ctx),
        }
    }

    /// Releases view-scoped caches for views that are no longer active.
    pub(crate) fn release_view_resources(&mut self, retired_views: &[crate::camera::ViewId]) {
        match self {
            Self::Raster(p) => p.release_view_resources(retired_views),
            Self::Compute(p) => p.release_view_resources(retired_views),
            Self::Encoder(p) => p.release_view_resources(retired_views),
        }
    }
}
