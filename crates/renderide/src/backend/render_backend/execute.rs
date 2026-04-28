//! Compiled render graph execution (multiview entry point).

use crate::gpu::GpuContext;
use crate::render_graph::{FrameView, GraphExecuteError};
use crate::scene::SceneCoordinator;

use super::RenderBackend;

impl RenderBackend {
    /// Unified multi-view entry: one Hi-Z readback (unless skipped), one encoder, one submit.
    ///
    /// When `skip_hi_z_begin_readback` is `false`, drains Hi-Z `map_async` readbacks first
    /// ([`crate::backend::OcclusionSystem::hi_z_begin_frame_readback`]). Set to `true` when the
    /// caller already invoked readback this tick (e.g. the runtime drains Hi-Z once at the top
    /// of [`crate::app::RenderideApp::tick_frame`] via
    /// [`crate::runtime::RendererRuntime::drain_hi_z_readback`]).
    ///
    /// `views` is not consumed; callers can clear and repopulate the same [`Vec`] each frame to
    /// retain capacity. Each [`FrameView`] routes to its own target — desktop swapchain, external
    /// OpenXR multiview, or host render-texture offscreen — without changing the backend entry
    /// point.
    pub fn execute_multi_view_frame(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        views: &mut Vec<FrameView<'_>>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("backend::execute_multi_view_frame");
        if !skip_hi_z_begin_readback {
            self.hi_z_begin_frame_readback(gpu.device());
        }
        self.history_registry.advance_frame();
        // Live HUD edits to `[post_processing]` only take effect when the graph is rebuilt; check
        // each tick so signature flips (effect added or removed) take effect on the next frame.
        // Parameter-only edits do not flip the signature and avoid the rebuild cost.
        let multiview_stereo = views.iter().any(FrameView::is_multiview_stereo_active);
        self.ensure_frame_graph_in_sync(multiview_stereo);
        let Some(mut graph) = self.frame_graph_cache.take_graph() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = graph.execute_multi_view(gpu, scene, self, views.as_mut_slice());
        self.frame_graph_cache.restore_graph(graph);
        res
    }
}
