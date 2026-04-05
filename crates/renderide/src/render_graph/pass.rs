//! [`RenderPass`] trait: metadata and command recording hook.

use super::context::RenderPassContext;
use super::error::RenderPassError;
use super::resources::PassResources;

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
}
