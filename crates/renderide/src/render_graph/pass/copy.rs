//! [`CopyPass`] trait for encoder-driven copy-only work.
//!
//! Copy passes share the same context as [`super::compute::ComputePass`] but are semantically
//! restricted to copy operations (texture/buffer copies, `clear_buffer`, `clear_texture`,
//! staging uploads). The distinction keeps pass intent explicit and allows future scheduling
//! or validation to identify copy-only work.

use crate::render_graph::context::{CopyPassCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};

use super::builder::PassBuilder;
use super::node::PassPhase;

/// A graph node whose GPU work consists solely of copy and clear operations.
pub trait CopyPass: Send + Sync {
    /// Stable name for logging, profiling, and error messages.
    fn name(&self) -> &str;

    /// Declares resource accesses and copy intent.
    ///
    /// The implementor must call `builder.copy()`.
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Records GPU copy operations.
    ///
    /// The [`wgpu::CommandEncoder`] is accessible via `ctx.encoder`.
    ///
    /// Takes `&self` so per-view passes can be recorded on rayon workers concurrently.
    /// Passes that hold mutable recording state must use interior mutability (e.g. `Mutex`).
    fn record(&self, ctx: &mut CopyPassCtx<'_, '_, '_>) -> Result<(), RenderPassError>;

    /// Scheduling phase. Defaults to [`PassPhase::PerView`].
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }

    /// Runs after the encoder containing this pass is submitted.
    ///
    /// Default is a no-op.
    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}
