//! Builds a CPU-readable hierarchical depth pyramid from the main depth attachment after the forward pass.

use crate::backend::HiZBuildInput;
use crate::render_graph::context::{ComputePassCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{ComputePass, PassBuilder};
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

impl ComputePass for HiZBuildPass {
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

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("hi_z::encode_pyramid");
        if ctx.depth_view.is_none() {
            return Ok(());
        }
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };
        let Some(depth_sample_view) = frame.view.depth_sample_view.as_ref() else {
            return Ok(());
        };
        let mode = frame.output_depth_mode();
        frame.shared.occlusion.encode_hi_z_build_pass(
            ctx.device,
            ctx.queue.as_ref(),
            ctx.encoder,
            frame.view.hi_z_slot.as_ref(),
            HiZBuildInput {
                depth_view: depth_sample_view,
                extent: frame.view.viewport_px,
                mode,
            },
            ctx.profiler,
        );
        Ok(())
    }

    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        // Hi-Z staging-buffer `map_async` now runs from a
        // [`wgpu::Queue::on_submitted_work_done`] callback installed in
        // [`crate::render_graph::compiled::exec::CompiledRenderGraph::execute_multi_view`],
        // so this post-submit hook is a no-op on the main thread.
        Ok(())
    }
}
