//! Render-graph cache synchronization and frame-shape invalidation policy.

use crate::backend::graph::build_main_graph_with_resources;
use crate::config::PostProcessingSettings;
use crate::passes::post_processing::gpu_supports_gtao;
use crate::render_graph::post_process_chain::PostProcessChainSignature;
use crate::render_graph::{GraphCacheEnsureResult, GraphCacheKey};

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
        }
    }
}

impl RenderBackend {
    /// Applies device-capability fallbacks to post-processing topology before graph build.
    pub(super) fn effective_post_processing_settings_for_graph(
        &self,
        settings: &PostProcessingSettings,
    ) -> PostProcessingSettings {
        let mut effective = settings.clone();
        if self.headless {
            effective.enabled = false;
            return effective;
        }
        if effective.gtao.enabled
            && let Some(limits) = self.gpu_limits()
            && !gpu_supports_gtao(limits.as_ref())
        {
            effective.gtao.enabled = false;
        }
        effective
    }

    /// Builds the current main-graph shape from live settings and the execution mode for this
    /// frame.
    pub(super) fn frame_graph_shape_for(
        &self,
        post_processing: &PostProcessingSettings,
        msaa_sample_count: u8,
        multiview_stereo: bool,
    ) -> FrameGraphShape {
        FrameGraphShape {
            msaa_sample_count,
            multiview_stereo,
            surface_format: self
                .surface_format
                .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb),
            scene_color_format: self.scene_color_format_wgpu(),
            post_processing: PostProcessChainSignature::from_settings(post_processing),
        }
    }

    /// Ensures the compiled main frame graph has a cached variant for the supplied shape.
    ///
    /// Graph-build failures are logged and clear only the active graph selection so the runtime
    /// can surface a recoverable [`crate::render_graph::GraphExecuteError::NoFrameGraph`] path.
    pub(super) fn sync_frame_graph_cache(
        &mut self,
        post_processing: &PostProcessingSettings,
        shape: FrameGraphShape,
    ) {
        let key = shape.into_cache_key();
        let previous_key = self.graph_state.frame_graph_cache.last_key();
        let key_cached = self.graph_state.frame_graph_cache.contains_key(key);
        if let Some(previous_key) = previous_key.filter(|previous| *previous != key && !key_cached)
        {
            logger::info!(
                "graph inputs changed (post-processing {:?} -> {:?}, msaa {}x -> {}x, multiview {} -> {}, surface {:?} -> {:?}, scene color {:?} -> {:?}); building render graph variant",
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
            );
        } else if previous_key.is_some_and(|previous| previous != key) {
            logger::debug!(
                "render graph cache switched active variant: previous={previous_key:?} key={key:?}"
            );
        }
        let post_processing_resources = self.graph_state.post_processing_resources().clone();
        match self.graph_state.frame_graph_cache.ensure(key, || {
            build_main_graph_with_resources(key, post_processing, &post_processing_resources)
        }) {
            Ok(GraphCacheEnsureResult::Hit) => {}
            Ok(GraphCacheEnsureResult::Built) => {
                self.graph_state.reset_upload_arena();
                if let Some(stats) = self.graph_state.frame_graph_cache.compile_stats() {
                    logger::info!(
                        "render graph ready: passes={} topo_levels={} culled={} transient_textures={} texture_slots={} transient_buffers={} buffer_slots={} imported_textures={} imported_buffers={} key={:?}",
                        stats.pass_count,
                        stats.topo_levels,
                        stats.culled_count,
                        stats.transient_texture_count,
                        stats.transient_texture_slots,
                        stats.transient_buffer_count,
                        stats.transient_buffer_slots,
                        stats.imported_texture_count,
                        stats.imported_buffer_count,
                        key,
                    );
                }
            }
            Err(error) => {
                self.graph_state.reset_upload_arena();
                logger::warn!("render graph build failed: {error}");
            }
        }
    }

    /// Rebuilds the main graph when live settings or the execution multiview shape changed.
    pub(crate) fn ensure_frame_graph_in_sync(&mut self, multiview_stereo: bool) {
        let Some(handle) = self.renderer_settings.as_ref() else {
            return;
        };
        let (live_settings, live_msaa) = match handle.read() {
            Ok(guard) => (
                guard.post_processing.clone(),
                guard.rendering.msaa.as_count() as u8,
            ),
            Err(_) => return,
        };
        let graph_settings = self.effective_post_processing_settings_for_graph(&live_settings);
        let shape = self.frame_graph_shape_for(&graph_settings, live_msaa, multiview_stereo);
        self.sync_frame_graph_cache(&graph_settings, shape);
    }
}
