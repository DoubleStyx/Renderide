//! Builds a CPU-readable hierarchical depth pyramid from the main depth attachment after the forward pass.

use crate::backend::HiZBuildInput;
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RenderPass};
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, ImportedTextureHandle, StorageAccess, TextureAccess,
};

/// Compute + copy pass that samples main depth and stages mips for next-frame occlusion.
#[derive(Debug)]
pub struct HiZBuildPass {
    resources: HiZBuildGraphResources,
}

/// Graph resources used by [`HiZBuildPass`].
#[derive(Clone, Copy, Debug)]
pub struct HiZBuildGraphResources {
    /// Imported single-sample depth texture for this view.
    pub depth: ImportedTextureHandle,
    /// Imported ping-pong Hi-Z pyramid output.
    pub hi_z_current: ImportedTextureHandle,
    /// Transient staging/readback buffer.
    pub readback_staging: BufferHandle,
}

impl HiZBuildPass {
    /// Creates a Hi-Z build pass instance.
    pub fn new(resources: HiZBuildGraphResources) -> Self {
        Self { resources }
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str {
        "HiZBuild"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.import_texture(
            self.resources.depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        b.import_texture(
            self.resources.hi_z_current,
            TextureAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.write_buffer(self.resources.readback_staging, BufferAccess::CopyDst);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(_depth) = ctx.depth_view else {
            return Ok(());
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };
        let depth_sample_view = frame
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("hi_z_depth_sample_view"),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });
        let mode = frame.output_depth_mode();
        let view_id = frame.occlusion_view;
        let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
        frame.backend.occlusion.encode_hi_z_build_pass(
            ctx.device,
            &queue,
            ctx.encoder,
            HiZBuildInput {
                depth_view: &depth_sample_view,
                extent: frame.viewport_px,
                mode,
                view: view_id,
            },
        );
        Ok(())
    }
}
