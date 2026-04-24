//! [`CallbackPass`] trait for CPU-only graph passes.
//!
//! A callback pass has no encoder and records no GPU commands. It runs as a CPU callback during
//! graph execution, typically to:
//! - Perform draw collection, CPU culling, uniform packing, and [`wgpu::Queue::write_buffer`]
//!   uploads (e.g. the world-mesh forward plan pass).
//! - Mutate the per-view [`super::super::blackboard::Blackboard`] with data subsequent raster or
//!   compute passes will consume.
//!
//! Declaring a pass as callback (via `builder.callback()` in `setup`) means the graph compiler
//! expects no resource accesses — the pass is cull-exempt by default since its side effects (Queue
//! writes, blackboard mutations) are not visible through graph resource declarations.

use crate::render_graph::context::{CallbackCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};

use super::builder::PassBuilder;
use super::node::PassPhase;

/// A graph node that runs as a CPU callback without a command encoder.
///
/// Callback passes are responsible for declaring themselves as callback via
/// `builder.callback()` in [`Self::setup`], and for calling [`PassBuilder::cull_exempt`] if
/// they have side effects not visible through graph resource declarations (which is almost always
/// the case for planning / upload passes).
pub trait CallbackPass: Send + Sync {
    /// Stable name for logging, profiling, and error messages.
    fn name(&self) -> &str;

    /// Declares the pass kind and any cull-exempt flag.
    ///
    /// Callback passes must call `builder.callback()`. They must NOT declare any resource
    /// accesses (the builder validates this in [`PassBuilder::finish`]).
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Runs as a CPU callback during graph execution.
    ///
    /// No encoder is provided. The pass may issue [`wgpu::Queue::write_buffer`] calls via
    /// `ctx.queue`, read scene data via `ctx.frame`, and mutate `ctx.blackboard`.
    ///
    /// Takes `&self` so per-view passes can be recorded on rayon workers concurrently.
    /// Passes that hold mutable recording state must use interior mutability (e.g. `Mutex`).
    fn run(&self, ctx: &mut CallbackCtx<'_, '_>) -> Result<(), RenderPassError>;

    /// Scheduling phase. Defaults to [`PassPhase::PerView`].
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }

    /// Runs after the encoder containing passes from the same phase group is submitted.
    ///
    /// Default is a no-op. Callback passes that trigger `map_async` after the frame submits
    /// should override this.
    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}
