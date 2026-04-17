//! Subsystem registration for render graph construction.
//!
//! Each [`RenderModule`] adds one or more [`super::pass::RenderPass`] nodes, typically binding
//! [`SharedRenderHandles`] for cross-pass resource flow.

use super::builder::GraphBuilder;
use super::handles::SharedRenderHandles;

/// Registers one or more passes belonging to a renderer subsystem.
pub trait RenderModule: Send {
    /// Stable module name (logging).
    fn name(&self) -> &str;

    /// Adds passes to `builder` using shared logical resource handles.
    fn register(self: Box<Self>, builder: &mut GraphBuilder, handles: &SharedRenderHandles);
}

/// Runs [`RenderModule::register`] on each module in order.
pub fn register_modules(
    builder: &mut GraphBuilder,
    handles: &SharedRenderHandles,
    modules: Vec<Box<dyn RenderModule>>,
) {
    for m in modules {
        m.register(builder, handles);
    }
}
