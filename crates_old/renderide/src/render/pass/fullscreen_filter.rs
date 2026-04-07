//! Placeholder for Unity-style fullscreen filter passes (blur, LUT, color grading, …).
//!
//! When [`RenderConfig::fullscreen_filter_hook`](crate::config::RenderConfig::fullscreen_filter_hook) is enabled on the
//! non-MRT graph, [`FullscreenFilterPlaceholderPass`] runs between the mesh and overlay passes. It
//! performs a no-op color attachment pass (load + store, no draws) so the render graph reserves a
//! stable insertion point for future filter materials.

use super::{PassResources, RenderPass, RenderPassContext, RenderPassError, ResourceSlot};

/// No-op pass that reserves the mesh → filter → overlay ordering for future fullscreen effects.
pub struct FullscreenFilterPlaceholderPass;

impl FullscreenFilterPlaceholderPass {
    /// Creates the placeholder pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FullscreenFilterPlaceholderPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for FullscreenFilterPlaceholderPass {
    fn name(&self) -> &str {
        "fullscreen_filter_placeholder"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::Color],
            writes: vec![ResourceSlot::Color],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fullscreen filter hook (no-op)"),
            timestamp_writes: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.render_target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
        Ok(())
    }
}
