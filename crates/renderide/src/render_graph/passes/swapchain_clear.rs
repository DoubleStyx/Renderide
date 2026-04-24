//! Full-screen clear of the swapchain target.
//!
//! The graph opens a render pass with [`wgpu::LoadOp::Clear`] applied to the swapchain color
//! attachment. The pass's `record` implementation is a deliberate no-op: the clear is performed
//! by the load operation when the render pass begins.

use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::ImportedTextureHandle;

/// Clears the acquired backbuffer to a solid color (default [`crate::present::SWAPCHAIN_CLEAR_COLOR`]).
#[derive(Debug)]
pub struct SwapchainClearPass {
    /// Clear color applied via the raster attachment's `LoadOp::Clear`.
    pub clear_color: wgpu::Color,
    target: ImportedTextureHandle,
}

impl SwapchainClearPass {
    /// Default clear color matches [`crate::present::SWAPCHAIN_CLEAR_COLOR`].
    pub fn new(target: ImportedTextureHandle) -> Self {
        Self {
            clear_color: crate::present::SWAPCHAIN_CLEAR_COLOR,
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

impl RasterPass for SwapchainClearPass {
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

    fn record(
        &self,
        _ctx: &mut RasterPassCtx<'_, '_>,
        _rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        // No-op: the clear is performed by the render pass LoadOp::Clear(self.clear_color)
        // declared in setup. No draw calls are needed.
        Ok(())
    }
}
