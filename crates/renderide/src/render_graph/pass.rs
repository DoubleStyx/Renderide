//! [`RenderPass`] trait: metadata and command recording hook.

use super::context::RenderPassContext;
use super::error::RenderPassError;
use super::resources::PassResources;

/// Whether a render pass runs once per frame or once per view in a multi-view tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassPhase {
    /// Runs once per frame regardless of view count (e.g. mesh deform).
    FrameGlobal,
    /// Runs once per view (e.g. clustered light compute, forward raster, Hi-Z build).
    PerView,
}

/// One node in the DAG: declares resource flow and records GPU commands.
///
/// Implementations are typically stateless or hold pass-local configuration (clear color, etc.).
/// The graph owns passes as [`Box<dyn RenderPass + Send>`] after [`super::GraphBuilder::build`].
pub trait RenderPass: Send {
    /// Stable name for logging and errors.
    fn name(&self) -> &str;

    /// Declared reads and writes used for topological validation at compile time.
    fn resources(&self) -> PassResources;

    /// Records GPU commands for this pass into `ctx.encoder`.
    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError>;

    /// Scheduling phase for multi-view execution. Defaults to per-view.
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }
}
