//! [`RenderPass`] trait for graph leaf nodes.

use crate::render::pass::error::RenderPassError;

use super::context::RenderPassContext;
use super::resources::PassResources;

/// Trait for render passes that can be executed by the render graph.
pub trait RenderPass {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Declares which resource slots this pass reads and writes.
    /// Default: empty reads and writes.
    fn resources(&self) -> PassResources {
        PassResources::default()
    }

    /// Executes the pass, recording commands into the context's encoder.
    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError>;
}
