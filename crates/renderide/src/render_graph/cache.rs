//! Compile cache for [`super::CompiledRenderGraph`] keyed by inputs that change schedule or targets.

use wgpu::TextureFormat;

use crate::config::ClusterAssignmentMode;

use super::compiled::CompiledRenderGraph;
use super::error::GraphBuildError;
use super::post_processing::PostProcessChainSignature;
use crate::camera::ViewId;

/// Inputs that invalidate a compiled main graph (extent, MSAA, multiview, surface format,
/// post-processing chain topology).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphCacheKey {
    /// Main surface extent in physical pixels.
    pub surface_extent: (u32, u32),
    /// Effective MSAA sample count for the main swapchain path (`1` = off).
    pub msaa_sample_count: u8,
    /// OpenXR / stereo multiview targets (affects cluster buffer layout in practice).
    pub multiview_stereo: bool,
    /// Swapchain / main color format.
    pub surface_format: TextureFormat,
    /// Forward scene-color HDR format ([`crate::config::SceneColorFormat`] at runtime).
    pub scene_color_format: TextureFormat,
    /// Active post-processing chain topology (which effects are wired into the graph). Changes to
    /// effect parameters that only update uniforms do not flip this signature; only adding or
    /// removing a pass invalidates the cached graph.
    pub post_processing: PostProcessChainSignature,
    /// Clustered-light assignment backend compiled into the cluster pass.
    pub cluster_assignment: ClusterAssignmentMode,
}

/// Holds the last successfully built graph and its cache key.
#[derive(Default)]
pub struct GraphCache {
    last_key: Option<GraphCacheKey>,
    graph: Option<CompiledRenderGraph>,
}

impl GraphCache {
    /// Ensures a graph is compiled for `key`, rebuilding when the key changes or no graph exists.
    ///
    /// Logs at `info` when a rebuild occurs.
    pub fn ensure(
        &mut self,
        key: GraphCacheKey,
        build: impl FnOnce() -> Result<CompiledRenderGraph, GraphBuildError>,
    ) -> Result<(), GraphBuildError> {
        let need_rebuild = self.last_key != Some(key) || self.graph.is_none();
        if need_rebuild {
            let g = match build() {
                Ok(g) => g,
                Err(e) => {
                    self.last_key = None;
                    self.graph = None;
                    return Err(e);
                }
            };
            if self.last_key.is_some() {
                logger::info!("render graph rebuilt (cache key changed)");
            }
            self.last_key = Some(key);
            self.graph = Some(g);
        }
        Ok(())
    }

    /// Cache key of the last successfully compiled graph, if any.
    #[must_use]
    pub fn last_key(&self) -> Option<GraphCacheKey> {
        self.last_key
    }

    /// Takes the compiled graph out for recording (matches prior `frame_graph.take()`).
    ///
    /// Graph execution borrows this cache, [`crate::backend::BackendGraphAccess`] (orchestration, transient pool,
    /// HUD overlay), and per-pass [`crate::render_graph::GraphPassFrame`] built via
    /// [`crate::backend::BackendGraphAccess::split_for_graph_frame_params`].
    #[must_use]
    pub fn take_graph(&mut self) -> Option<CompiledRenderGraph> {
        self.graph.take()
    }

    /// Restores the graph after [`Self::take_graph`].
    pub fn restore_graph(&mut self, graph: CompiledRenderGraph) {
        self.graph = Some(graph);
    }

    /// Releases view-scoped pass caches for views that are no longer active.
    pub fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        if let Some(graph) = self.graph.as_mut() {
            graph.release_view_resources(retired_views);
        }
    }

    /// Pass count for diagnostics when a graph is cached.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(0, CompiledRenderGraph::pass_count)
    }

    /// DAG wave count from [`super::CompileStats::topo_levels`] when a graph is cached, else `0`.
    #[must_use]
    pub fn topo_levels(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(0, |g| g.compile_stats.topo_levels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_with_post(sig: PostProcessChainSignature) -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (1280, 720),
            msaa_sample_count: 1,
            multiview_stereo: false,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            scene_color_format: TextureFormat::Rgba16Float,
            post_processing: sig,
            cluster_assignment: ClusterAssignmentMode::Auto,
        }
    }

    #[test]
    fn post_processing_signature_change_changes_cache_key_equality() {
        let off = key_with_post(PostProcessChainSignature::default());
        let on = key_with_post(PostProcessChainSignature {
            aces_tonemap: true,
            bloom: false,
            bloom_max_mip_dimension: 0,
            gtao: false,
            gtao_denoise_passes: 0,
        });
        assert_ne!(off, on);
        assert_eq!(off, key_with_post(PostProcessChainSignature::default()));
    }
}
