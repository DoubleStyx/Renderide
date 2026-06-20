//! [`EncoderPass`] trait for passes that need explicit encoder-level ordering.
//!
//! Unlike [`super::raster::RasterPass`], the graph does not open a render pass for encoder
//! passes. Unlike [`super::compute::ComputePass`], the command stream is not expected to be a
//! compute dispatch; implementations may interleave copies, resolves, and manually opened render
//! passes while still declaring their resource accesses to the graph.

use std::borrow::Cow;

use crate::camera::ViewId;
use crate::graph_inputs::FrameGlobalResourcePass;
use crate::render_graph::context::{EncoderPassCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};

use super::builder::PassBuilder;
use super::node::PassPhase;

/// A graph node whose GPU work is recorded directly into the command encoder.
pub trait EncoderPass: Send + Sync {
    /// Stable name for logging, profiling, and error messages.
    fn name(&self) -> &str;

    /// Human-readable label for GPU profiler markers.
    ///
    /// Defaults to [`Self::name`]. Pass families that register multiple instances in one graph
    /// should include an instance discriminator here so Tracy can distinguish them.
    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.name())
    }

    /// Declares resource accesses and encoder intent.
    ///
    /// The implementor must call [`PassBuilder::encoder`].
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Records encoder-level GPU commands.
    ///
    /// Takes `&self` so per-view passes can be recorded on rayon workers concurrently.
    /// Passes that hold mutable recording state must use interior mutability.
    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError>;

    /// Optional: returns whether the executor should record this encoder pass for the current view.
    ///
    /// Runs before [`Self::record`], so per-view passes with no work can skip render-pass and
    /// copy setup. Defaults to recording.
    fn should_record(&self, _ctx: &EncoderPassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
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
