//! Compiled render graph execution (desktop, multiview, offscreen).

use crate::gpu::GpuContext;
use crate::render_graph::{
    build_main_graph, CompiledRenderGraph, ExternalFrameTargets, FrameView, FrameViewTarget,
    GraphCacheKey, GraphExecuteError, OffscreenSingleViewExecuteSpec,
};
use crate::scene::SceneCoordinator;
use winit::window::Window;

use super::RenderBackend;

fn graph_cache_key_from_gpu(gpu: &GpuContext, multiview_stereo: bool) -> GraphCacheKey {
    let msaa = if multiview_stereo {
        gpu.swapchain_msaa_effective_stereo()
    } else {
        gpu.swapchain_msaa_effective()
    };
    GraphCacheKey {
        surface_extent: gpu.surface_extent_px(),
        msaa_sample_count: msaa.min(255) as u8,
        multiview_stereo,
        surface_format: gpu.config_format(),
    }
}

impl RenderBackend {
    /// Runs `run` with a compiled graph from [`super::RenderBackend::graph_cache`], restoring it afterward.
    ///
    /// When `skip_hi_z_begin_readback` is `false`, drains Hi-Z `map_async` readbacks first
    /// ([`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`]). Set to `true` when the
    /// caller already invoked readback this tick (e.g. [`Self::execute_multi_view_frame`] after prefetch).
    fn with_graph_for<R>(
        &mut self,
        gpu: &mut GpuContext,
        key: GraphCacheKey,
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
        self.graph_cache
            .ensure(key, || build_main_graph(key))
            .map_err(GraphExecuteError::from)?;
        let Some(mut graph) = self.graph_cache.take_graph() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = run(&mut graph, gpu, self);
        self.graph_cache.restore_graph(graph);
        res
    }

    /// Records and presents one frame using the compiled render graph (deform compute + forward mesh pass).
    ///
    /// Returns [`GraphExecuteError::NoFrameGraph`] if the graph could not be built for the current cache key.
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        let key = graph_cache_key_from_gpu(gpu, false);
        self.with_graph_for(gpu, key, false, |graph, gpu_ctx, backend| {
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
        let key = graph_cache_key_from_gpu(gpu, true);
        self.with_graph_for(
            gpu,
            key,
            skip_hi_z_begin_readback,
            |graph, gpu_ctx, backend| {
                graph.execute_external_multiview(
                    gpu_ctx,
                    window,
                    scene,
                    backend,
                    host_camera,
                    external,
                )
            },
        )
    }

    /// Unified multi-view entry: frame-global submit then one submit per view.
    pub fn execute_multi_view_frame(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        views: Vec<FrameView<'_>>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        let multiview_stereo = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::ExternalMultiview(_)));
        let key = graph_cache_key_from_gpu(gpu, multiview_stereo);
        self.with_graph_for(
            gpu,
            key,
            skip_hi_z_begin_readback,
            |graph, gpu_ctx, backend| {
                graph.execute_multi_view(gpu_ctx, window, scene, backend, views)
            },
        )
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
        let key = graph_cache_key_from_gpu(gpu, false);
        self.with_graph_for(gpu, key, false, |graph, gpu_ctx, backend| {
            graph.execute_offscreen_single_view(gpu_ctx, backend, spec)
        })
    }
}
