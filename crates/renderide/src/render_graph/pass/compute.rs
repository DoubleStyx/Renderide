//! [`ComputePass`] trait for encoder-driven compute and mixed compute/copy work.
//!
//! Unlike [`super::raster::RasterPass`], the graph does not open any GPU pass object for compute
//! passes -- the implementor receives the context (which includes the [`wgpu::CommandEncoder`])
//! and dispatches compute workgroups or uses the encoder API directly.

use std::borrow::Cow;

use crate::camera::ViewId;
use crate::graph_inputs::FrameGlobalResourcePass;
use crate::render_graph::context::{ComputePassCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};

use super::builder::PassBuilder;
use super::node::PassPhase;

/// A graph node whose GPU work is encoder-driven compute (compute shaders, pipeline barriers,
/// compute dispatch, or mixed compute/copy operations).
pub trait ComputePass: Send + Sync {
    /// Stable name for logging, profiling, and error messages.
    fn name(&self) -> &str;

    /// Human-readable label for GPU profiler markers.
    ///
    /// Defaults to [`Self::name`]. Pass families that register multiple instances in one graph
    /// should include an instance discriminator here so Tracy can distinguish them.
    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.name())
    }

    /// Declares resource accesses and compute intent.
    ///
    /// The implementor must call `builder.compute()`.
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Records GPU compute commands.
    ///
    /// The [`wgpu::CommandEncoder`] is accessible via `ctx.encoder`. The pass opens and closes
    /// compute sub-passes on it directly.
    ///
    /// Takes `&self` so per-view passes can be recorded on rayon workers concurrently.
    /// Passes that hold mutable recording state must use interior mutability (e.g. `Mutex`).
    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError>;

    /// Optional: returns whether the executor should record this compute pass for the current view.
    ///
    /// Runs before [`Self::record`], so per-view passes with no work can skip opening compute
    /// passes, bind-group churn, and timestamp queries. Defaults to recording.
    fn should_record(&self, _ctx: &ComputePassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(true)
    }

    /// Scheduling phase. Defaults to [`PassPhase::PerView`].
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }

    /// Typed backend resource hook used by frame-global passes with renderer-owned resources.
    ///
    /// Defaults to [`None`] for ordinary graph passes. Resource-backed frame-global passes return
    /// a stable enum so the backend does not route by diagnostic pass name strings.
    fn frame_global_resource_pass(&self) -> Option<FrameGlobalResourcePass> {
        None
    }

    /// Releases any cached state owned by views that are no longer active.
    ///
    /// Default is a no-op for passes that keep no view-scoped resources across frames.
    fn release_view_resources(&mut self, _retired_views: &[ViewId]) {}

    /// Runs after the encoder containing this pass is submitted.
    ///
    /// Default is a no-op.
    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}
