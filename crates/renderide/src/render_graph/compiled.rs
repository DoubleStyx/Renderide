//! Compiled DAG: immutable pass order and per-frame execution.

use winit::window::Window;

use crate::gpu::GpuContext;
use crate::present::{acquire_surface_outcome, SurfaceFrameOutcome};

use super::context::RenderPassContext;
use super::error::GraphExecuteError;
use super::pass::RenderPass;

/// Statistics emitted when building a [`CompiledRenderGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileStats {
    /// Number of passes in the flattened schedule.
    pub pass_count: usize,
    /// Number of topological levels (waves); a parallelism hint for future scheduling.
    pub topo_levels: usize,
}

/// Immutable execution schedule produced by [`super::GraphBuilder::build`].
///
/// After build, pass order and [`Self::needs_surface_acquire`] do not change. Per-frame work is
/// only [`Self::execute`], which records into one command encoder and submits once (v1).
///
/// Phase 2 may add subgraph expansion, multi-encoder recording, and explicit barrier insertion.
pub struct CompiledRenderGraph {
    pub(super) passes: Vec<Box<dyn RenderPass>>,
    /// `true` when any pass writes [`super::resources::ResourceSlot::Backbuffer`] — frame execution
    /// acquires the swapchain once and presents after submit.
    pub needs_surface_acquire: bool,
    /// Build-time stats for tests and future profiling hooks.
    pub compile_stats: CompileStats,
}

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Records all passes and submits. Matches [`crate::present::present_clear_frame`] recovery
    /// behavior for surface acquire (timeout/occluded skip, validation reconfigure).
    pub fn execute(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        let (frame, backbuffer_view_holder): (
            Option<wgpu::SurfaceTexture>,
            Option<wgpu::TextureView>,
        ) = if self.needs_surface_acquire {
            match acquire_surface_outcome(gpu, window)? {
                SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => {
                    return Ok(());
                }
                SurfaceFrameOutcome::Acquired(tex) => {
                    let view = tex
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    (Some(tex), Some(view))
                }
            }
        } else {
            (None, None)
        };

        let device = gpu.device().as_ref();
        let backbuffer_ref = backbuffer_view_holder.as_ref();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph"),
        });

        let mut ctx = RenderPassContext {
            device,
            queue: gpu.queue(),
            encoder: &mut encoder,
            backbuffer: backbuffer_ref,
        };

        for pass in &mut self.passes {
            pass.execute(&mut ctx)?;
        }

        gpu.queue()
            .lock()
            .expect("queue mutex poisoned")
            .submit(std::iter::once(encoder.finish()));

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }
}
