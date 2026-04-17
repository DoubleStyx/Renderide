//! Full-screen clear of the swapchain target.

use crate::present::{record_swapchain_clear_pass, SWAPCHAIN_CLEAR_COLOR};

use crate::render_graph::context::{GraphRasterPassContext, RenderPassContext};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RenderPass};
use crate::render_graph::resources::ImportedTextureHandle;

/// Clears the acquired backbuffer to a solid color (default [`SWAPCHAIN_CLEAR_COLOR`]).
#[derive(Debug)]
pub struct SwapchainClearPass {
    /// Clear color for the swapchain load op.
    pub clear_color: wgpu::Color,
    target: ImportedTextureHandle,
}

impl SwapchainClearPass {
    /// Default clear color matches [`SWAPCHAIN_CLEAR_COLOR`].
    pub fn new(target: ImportedTextureHandle) -> Self {
        Self {
            clear_color: SWAPCHAIN_CLEAR_COLOR,
            target,
        }
    }

    /// Full control over the clear color (HDR or branding).
    pub fn with_clear_color(target: ImportedTextureHandle, clear_color: wgpu::Color) -> Self {
        Self {
            clear_color,
            target,
        }
    }
}

impl RenderPass for SwapchainClearPass {
    fn name(&self) -> &str {
        "SwapchainClear"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        let mut r = b.raster();
        r.color(
            self.target,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(self.clear_color),
                store: wgpu::StoreOp::Store,
            },
            Option::<ImportedTextureHandle>::None,
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(view) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        record_swapchain_clear_pass(ctx.encoder, view, self.clear_color, Some("swapchain-clear"));
        Ok(())
    }

    fn graph_managed_raster(&self) -> bool {
        true
    }

    fn execute_graph_raster(
        &mut self,
        _ctx: &mut GraphRasterPassContext<'_, '_>,
        _rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        Ok(())
    }
}
