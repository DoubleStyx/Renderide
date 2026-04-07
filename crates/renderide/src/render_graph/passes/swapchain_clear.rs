//! Full-screen clear of the swapchain target.

use crate::present::{record_swapchain_clear_pass, SWAPCHAIN_CLEAR_COLOR};

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};

/// Clears the acquired backbuffer to a solid color (default [`SWAPCHAIN_CLEAR_COLOR`]).
#[derive(Debug)]
pub struct SwapchainClearPass {
    /// Clear color for the swapchain load op.
    pub clear_color: wgpu::Color,
}

impl SwapchainClearPass {
    /// Default clear color matches [`SWAPCHAIN_CLEAR_COLOR`].
    pub fn new() -> Self {
        Self {
            clear_color: SWAPCHAIN_CLEAR_COLOR,
        }
    }

    /// Full control over the clear color (HDR or branding).
    pub fn with_clear_color(clear_color: wgpu::Color) -> Self {
        Self { clear_color }
    }
}

impl Default for SwapchainClearPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for SwapchainClearPass {
    fn name(&self) -> &str {
        "SwapchainClear"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: vec![ResourceSlot::Backbuffer],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(view) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        record_swapchain_clear_pass(ctx.encoder, view, self.clear_color, Some("swapchain-clear"));
        Ok(())
    }
}
