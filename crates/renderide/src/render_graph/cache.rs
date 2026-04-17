//! Compile cache for [`super::CompiledRenderGraph`] keyed by inputs that change schedule or targets.

use wgpu::TextureFormat;

use super::compiled::CompiledRenderGraph;
use super::error::GraphBuildError;

/// Inputs that invalidate a compiled main graph (extent, MSAA, multiview, surface format).
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

    /// Takes the compiled graph out for recording (matches prior `frame_graph.take()`).
    #[must_use]
    pub fn take_graph(&mut self) -> Option<CompiledRenderGraph> {
        self.graph.take()
    }

    /// Restores the graph after [`Self::take_graph`].
    pub fn restore_graph(&mut self, graph: CompiledRenderGraph) {
        self.graph = Some(graph);
    }

    /// Pass count for diagnostics when a graph is cached.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(0, CompiledRenderGraph::pass_count)
    }
}
