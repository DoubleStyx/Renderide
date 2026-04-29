//! Render-graph cache synchronization and frame-shape invalidation policy.

use crate::config::{ClusterAssignmentMode, PostProcessingSettings};
use crate::render_graph::post_processing::PostProcessChainSignature;
use crate::render_graph::GraphCacheKey;

use super::RenderBackend;

/// The graph-shaping inputs that matter at the current renderer stage.
///
/// `surface_extent` intentionally stays outside this shape today because the main graph resolves
/// frame extents dynamically via frame-view targets and transient extent policies. The cache key
/// still stores a placeholder extent so the shared graph cache remains uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameGraphShape {
    /// Effective MSAA sample count for the main frame.
    msaa_sample_count: u8,
    /// `true` when at least one frame view uses stereo multiview execution.
    multiview_stereo: bool,
    /// Main surface format used by display-target composition.
    surface_format: wgpu::TextureFormat,
    /// HDR scene-color format resolved from live renderer settings.
    scene_color_format: wgpu::TextureFormat,
    /// Active post-processing topology.
    post_processing: PostProcessChainSignature,
    /// Clustered-light assignment backend compiled into the cluster pass.
    cluster_assignment: ClusterAssignmentMode,
}

impl FrameGraphShape {
    /// Converts the stage-local shape into the shared render-graph cache key.
    fn into_cache_key(self) -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (1, 1),
            msaa_sample_count: self.msaa_sample_count,
            multiview_stereo: self.multiview_stereo,
            surface_format: self.surface_format,
            scene_color_format: self.scene_color_format,
            post_processing: self.post_processing,
            cluster_assignment: self.cluster_assignment,
        }
    }
}

impl RenderBackend {
    /// Builds the current main-graph shape from live settings and the execution mode for this
    /// frame.
    pub(super) fn frame_graph_shape_for(
        &self,
        post_processing: &PostProcessingSettings,
        msaa_sample_count: u8,
        multiview_stereo: bool,
        cluster_assignment: ClusterAssignmentMode,
    ) -> FrameGraphShape {
        FrameGraphShape {
            msaa_sample_count,
            multiview_stereo,
            surface_format: self
                .surface_format
                .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb),
            scene_color_format: self.scene_color_format_wgpu(),
            post_processing: PostProcessChainSignature::from_settings(post_processing),
            cluster_assignment,
        }
    }

    /// Ensures the compiled main frame graph matches the supplied graph-shaping settings.
    ///
    /// Graph-build failures are logged and leave the cache empty so the runtime can surface a
    /// recoverable [`crate::render_graph::GraphExecuteError::NoFrameGraph`] path.
    pub(super) fn sync_frame_graph_cache(
        &mut self,
        post_processing: &PostProcessingSettings,
        shape: FrameGraphShape,
    ) {
        let key = shape.into_cache_key();
        let previous_key = self.frame_graph_cache.last_key();
        if let Some(previous_key) = previous_key.filter(|previous| *previous != key) {
            logger::info!(
                "graph inputs changed (post-processing {:?} -> {:?}, msaa {}x -> {}x, multiview {} -> {}, surface {:?} -> {:?}, scene color {:?} -> {:?}, cluster assignment {:?} -> {:?}); rebuilding render graph",
                previous_key.post_processing,
                key.post_processing,
                previous_key.msaa_sample_count,
                key.msaa_sample_count,
                previous_key.multiview_stereo,
                key.multiview_stereo,
                previous_key.surface_format,
                key.surface_format,
                previous_key.scene_color_format,
                key.scene_color_format,
                previous_key.cluster_assignment,
                key.cluster_assignment,
            );
        }
        if let Err(error) = self.frame_graph_cache.ensure(key, || {
            crate::render_graph::build_main_graph(key, post_processing)
        }) {
            logger::warn!("render graph build failed: {error}");
        }
    }

    /// Rebuilds the main graph when live settings or the execution multiview shape changed.
    pub(crate) fn ensure_frame_graph_in_sync(&mut self, multiview_stereo: bool) {
        let Some(handle) = self.renderer_settings.as_ref() else {
            return;
        };
        let (live_parallelism, live_settings, live_msaa, live_cluster_assignment) =
            match handle.read() {
                Ok(guard) => (
                    guard.rendering.record_parallelism,
                    guard.post_processing.clone(),
                    guard.rendering.msaa.as_count() as u8,
                    guard.rendering.cluster_assignment,
                ),
                Err(_) => return,
            };
        self.set_record_parallelism(live_parallelism);
        let shape = self.frame_graph_shape_for(
            &live_settings,
            live_msaa,
            multiview_stereo,
            live_cluster_assignment,
        );
        self.sync_frame_graph_cache(&live_settings, shape);
    }
}
