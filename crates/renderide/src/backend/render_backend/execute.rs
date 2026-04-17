//! Compiled render graph execution (desktop, multiview, offscreen).

use crate::gpu::GpuContext;
use crate::render_graph::{
    CompiledRenderGraph, ExternalFrameTargets, FrameView, GraphExecuteError,
    OffscreenSingleViewExecuteSpec,
};
use crate::scene::SceneCoordinator;
use winit::window::Window;

use super::RenderBackend;

impl RenderBackend {
    /// Runs `run` with a taken [`CompiledRenderGraph`], restoring it afterward.
    ///
    /// When `skip_hi_z_begin_readback` is `false`, drains Hi-Z `map_async` readbacks first
    /// ([`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`]). Set to `true` when the
    /// caller already invoked readback this tick (e.g. [`Self::execute_multi_view_frame`] after prefetch).
    fn with_compiled_graph<R>(
        &mut self,
        gpu: &mut GpuContext,
        skip_hi_z_begin_readback: bool,
        run: impl FnOnce(
            &mut CompiledRenderGraph,
            &mut GpuContext,
            &mut RenderBackend,
        ) -> Result<R, GraphExecuteError>,
    ) -> Result<R, GraphExecuteError> {
        if !skip_hi_z_begin_readback {
            self.occlusion.hi_z_begin_frame_readback(gpu.device());
        }
        let Some(mut graph) = self.frame_graph.take() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = run(&mut graph, gpu, self);
        self.frame_graph = Some(graph);
        res
    }

    /// Records and presents one frame using the compiled render graph (deform compute + forward mesh pass).
    ///
    /// Returns [`GraphExecuteError::NoFrameGraph`] if graph build failed during [`crate::backend::RenderBackend::attach`].
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, false, |graph, gpu_ctx, backend| {
            graph.execute(gpu_ctx, window, scene, backend, host_camera)
        })
    }

    /// Renders the frame graph to pre-acquired OpenXR multiview array targets (no surface present).
    ///
    /// When `skip_hi_z_begin_readback` is `true`, the caller has already drained Hi-Z readbacks
    /// this tick.
    pub fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
        external: ExternalFrameTargets<'_>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, skip_hi_z_begin_readback, |graph, gpu_ctx, backend| {
            graph.execute_external_multiview(gpu_ctx, window, scene, backend, host_camera, external)
        })
    }

    /// Unified multi-view entry: one Hi-Z readback (unless skipped), one encoder, one submit.
    pub fn execute_multi_view_frame(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        views: Vec<FrameView<'_>>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, skip_hi_z_begin_readback, |graph, gpu_ctx, backend| {
            graph.execute_multi_view(gpu_ctx, window, scene, backend, views)
        })
    }

    /// Renders the default graph to a single-view render texture (secondary camera).
    ///
    /// When `spec.prefetched_world_mesh_draws` is [`Some`], the world mesh forward pass skips CPU draw
    /// collection and uses the provided list (see [`crate::render_graph::FrameRenderParams::prefetched_world_mesh_draws`]).
    pub fn execute_frame_graph_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        spec: OffscreenSingleViewExecuteSpec<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, false, |graph, gpu_ctx, backend| {
            graph.execute_offscreen_single_view(gpu_ctx, backend, spec)
        })
    }
}
