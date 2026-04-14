//! Builds a CPU-readable hierarchical depth pyramid from the main depth attachment after the forward pass.

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};

/// Compute + copy pass that samples main depth and stages mips for next-frame occlusion.
#[derive(Debug, Default)]
pub struct HiZBuildPass;

impl HiZBuildPass {
    /// Creates a Hi-Z build pass instance.
    pub fn new() -> Self {
        Self
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str {
        "HiZBuild"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::Depth],
            writes: vec![],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(depth) = ctx.depth_view else {
            return Ok(());
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };
        let mode = frame.output_depth_mode();
        let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
        frame.backend.occlusion.encode_hi_z_build_pass(
            ctx.device,
            &queue,
            ctx.encoder,
            depth,
            frame.viewport_px,
            mode,
        );
        Ok(())
    }
}
