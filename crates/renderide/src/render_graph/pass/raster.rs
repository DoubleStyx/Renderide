//! [`RasterPass`] trait for graph-managed raster render passes.
//!
//! The graph opens the [`wgpu::RenderPass`] from the compiled [`crate::render_graph::RenderPassTemplate`]
//! and attachment views resolved from the transient pool. The pass records draw commands into the
//! already-open render pass via [`RasterPass::record`].

use std::num::NonZeroU32;

use crate::render_graph::compiled::{DepthAttachmentTemplate, RenderPassTemplate};
use crate::render_graph::context::{PostSubmitContext, RasterPassCtx};
use crate::render_graph::error::{RenderPassError, SetupError};

use super::builder::PassBuilder;
use super::node::PassPhase;

/// A graph node whose GPU work is a raster render pass opened and closed by the graph executor.
///
/// The executor resolves all attachments from the pass's [`PassSetup`] template, opens
/// `wgpu::RenderPass`, calls [`Self::record`], then drops the pass (closing it). Multiview and
/// stencil overrides may be supplied through the optional hook methods.
pub trait RasterPass: Send + Sync {
    /// Stable name for logging, profiling, and error messages.
    fn name(&self) -> &str;

    /// Declares resource accesses and raster attachment intent.
    ///
    /// The implementor must call `builder.raster()` and declare at least one color or depth
    /// attachment; the builder enforces this in [`PassBuilder::finish`].
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Records GPU commands into the graph-opened render pass.
    ///
    /// Takes `&self` so per-view passes can be recorded on rayon workers concurrently.
    /// Passes that hold mutable recording state must use interior mutability (e.g. `Mutex`).
    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError>;

    /// Optional: runtime multiview mask override for the render pass.
    ///
    /// Defaults to the mask baked into the compiled attachment template. Passes that select
    /// multiview based on runtime VR state (e.g. [`crate::render_graph::passes::WorldMeshForwardOpaquePass`])
    /// return `Some` when stereo is active.
    fn multiview_mask_override(
        &self,
        _ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        template.multiview_mask
    }

    /// Optional: runtime stencil ops override for the depth/stencil attachment.
    ///
    /// Defaults to the stencil ops baked into the compiled template.
    fn stencil_ops_override(
        &self,
        _ctx: &RasterPassCtx<'_, '_>,
        depth: &DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        depth.stencil
    }

    /// Scheduling phase for multi-view execution. Defaults to [`PassPhase::PerView`].
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }

    /// Runs after the encoder containing this pass is submitted to the queue.
    ///
    /// Used for passes that need to start `map_async` on staging buffers written this frame.
    /// Default is a no-op.
    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}
