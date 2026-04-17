//! Builds a CPU-readable hierarchical depth pyramid from the main depth attachment after the forward pass.

use crate::backend::HiZBuildInput;
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::handles::ResourceId;
use crate::render_graph::module::RenderModule;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::PassResources;
use crate::render_graph::{GraphBuilder, SharedRenderHandles};

/// Compute + copy pass that samples main depth and stages mips for next-frame occlusion.
#[derive(Debug)]
pub struct HiZBuildPass {
    depth: ResourceId,
}

impl HiZBuildPass {
    /// Creates a Hi-Z build pass bound to the main depth logical resource.
    pub fn new(depth: ResourceId) -> Self {
        Self { depth }
    }
}

/// Registers [`HiZBuildPass`] on the main frame graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct HiZBuildModule;

impl RenderModule for HiZBuildModule {
    fn name(&self) -> &str {
        "hi_z_build"
    }

    fn register(self: Box<Self>, builder: &mut GraphBuilder, handles: &SharedRenderHandles) {
        builder.add_pass(Box::new(HiZBuildPass::new(handles.depth)));
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str {
        "HiZBuild"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![self.depth],
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
        let view_id = frame.occlusion_view;
        let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
        frame.backend.occlusion.encode_hi_z_build_pass(
            ctx.device,
            &queue,
            ctx.encoder,
            HiZBuildInput {
                depth_view: depth,
                extent: frame.viewport_px,
                mode,
                view: view_id,
            },
        );
        Ok(())
    }
}
