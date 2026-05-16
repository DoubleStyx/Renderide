//! Compile cache for [`super::CompiledRenderGraph`] keyed by inputs that change schedule or targets.

use std::collections::VecDeque;

use hashbrown::HashMap;
use wgpu::TextureFormat;

use super::super::error::GraphBuildError;
use super::super::post_process_chain::PostProcessChainSignature;
use super::CompileStats;
use super::CompiledRenderGraph;
use crate::camera::ViewId;

/// Maximum number of compiled graph variants retained by the main graph cache.
const GRAPH_CACHE_CAPACITY: usize = 4;

/// Inputs that invalidate a compiled main graph (extent, MSAA, multiview, surface format,
/// scene-color format, and post-processing chain topology).
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
}

/// Result of ensuring a graph variant is available for a cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphCacheEnsureResult {
    /// The requested graph was already cached and has been made active.
    Hit,
    /// The requested graph was built and inserted into the cache.
    Built,
}

/// Holds recently built graph variants and tracks the active key for frame execution.
#[derive(Default)]
pub struct GraphCache {
    /// Cache key selected for the next graph execution.
    active_key: Option<GraphCacheKey>,
    /// Compiled graph variants keyed by the shape that produced them.
    graphs: HashMap<GraphCacheKey, CompiledRenderGraph>,
    /// Least-recently-used ordering for bounded variant retention.
    usage_order: VecDeque<GraphCacheKey>,
}

impl GraphCache {
    /// Ensures a graph is compiled for `key`, building only when that exact key is absent.
    pub fn ensure(
        &mut self,
        key: GraphCacheKey,
        build: impl FnOnce() -> Result<CompiledRenderGraph, GraphBuildError>,
    ) -> Result<GraphCacheEnsureResult, GraphBuildError> {
        if self.graphs.contains_key(&key) {
            self.active_key = Some(key);
            self.touch_key(key);
            return Ok(GraphCacheEnsureResult::Hit);
        }

        let graph = match build() {
            Ok(graph) => graph,
            Err(error) => {
                self.active_key = None;
                return Err(error);
            }
        };
        self.graphs.insert(key, graph);
        self.active_key = Some(key);
        self.touch_key(key);
        self.evict_excess_graphs();
        Ok(GraphCacheEnsureResult::Built)
    }

    /// Returns `true` when a graph variant is already cached for `key`.
    #[must_use]
    pub fn contains_key(&self, key: GraphCacheKey) -> bool {
        self.graphs.contains_key(&key)
    }

    /// Marks `key` as the most recently used graph variant.
    fn touch_key(&mut self, key: GraphCacheKey) {
        self.usage_order.retain(|existing| *existing != key);
        self.usage_order.push_back(key);
    }

    /// Evicts least-recently-used inactive graph variants until the cache is within capacity.
    fn evict_excess_graphs(&mut self) {
        while self.graphs.len() > GRAPH_CACHE_CAPACITY {
            let mut evicted = false;
            let scan_count = self.usage_order.len();
            for _ in 0..scan_count {
                let Some(candidate) = self.usage_order.pop_front() else {
                    break;
                };
                if Some(candidate) == self.active_key {
                    self.usage_order.push_back(candidate);
                    continue;
                }
                if self.graphs.remove(&candidate).is_some() {
                    evicted = true;
                    break;
                }
            }
            if !evicted {
                break;
            }
        }
    }

    /// Cache key of the graph currently selected for execution, if any.
    #[must_use]
    pub fn last_key(&self) -> Option<GraphCacheKey> {
        self.active_key
    }

    /// Takes the active compiled graph out for recording.
    ///
    /// Graph execution borrows this cache, a [`crate::render_graph::GraphExecutionBackend`]
    /// implementation, and per-pass [`crate::render_graph::GraphPassFrame`] values built from
    /// graph-facing resource traits.
    #[must_use]
    pub fn take_graph(&mut self) -> Option<CompiledRenderGraph> {
        let key = self.active_key?;
        self.graphs.remove(&key)
    }

    /// Restores the graph after [`Self::take_graph`].
    pub fn restore_graph(&mut self, graph: CompiledRenderGraph) {
        let Some(key) = self.active_key else {
            logger::warn!("render graph restored without an active cache key; dropping graph");
            return;
        };
        self.graphs.insert(key, graph);
        self.touch_key(key);
        self.evict_excess_graphs();
    }

    /// Releases view-scoped pass caches for views that are no longer active.
    pub fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        for graph in self.graphs.values_mut() {
            graph.release_view_resources(retired_views);
        }
    }

    /// Returns the active graph variant without changing cache ownership.
    fn active_graph(&self) -> Option<&CompiledRenderGraph> {
        let key = self.active_key?;
        self.graphs.get(&key)
    }

    /// Pass count for diagnostics when a graph is cached.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.active_graph()
            .map_or(0, CompiledRenderGraph::pass_count)
    }

    /// DAG wave count from [`super::CompileStats::topo_levels`] when a graph is cached, else `0`.
    #[must_use]
    pub fn topo_levels(&self) -> usize {
        self.active_graph()
            .map_or(0, |g| g.compile_stats.topo_levels)
    }

    /// Compile stats for diagnostics when a graph is cached.
    #[must_use]
    pub fn compile_stats(&self) -> Option<CompileStats> {
        self.active_graph().map(|g| g.compile_stats)
    }

    /// Number of retained graph variants for unit tests.
    #[cfg(test)]
    #[must_use]
    pub fn variant_count_for_tests(&self) -> usize {
        self.graphs.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::render_graph::context::ComputePassCtx;
    use crate::render_graph::error::{RenderPassError, SetupError};
    use crate::render_graph::pass::{ComputePass, PassBuilder, PassNode};
    use crate::render_graph::schedule::FrameSchedule;

    fn key_with_post(sig: PostProcessChainSignature) -> GraphCacheKey {
        GraphCacheKey {
            surface_extent: (1280, 720),
            msaa_sample_count: 1,
            multiview_stereo: false,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            scene_color_format: TextureFormat::Rgba16Float,
            post_processing: sig,
        }
    }

    #[test]
    fn post_processing_signature_change_changes_cache_key_equality() {
        let off = key_with_post(PostProcessChainSignature::default());
        let on = key_with_post(PostProcessChainSignature {
            aces_tonemap: true,
            agx_tonemap: false,
            auto_exposure: false,
            bloom: false,
            bloom_max_mip_dimension: 0,
            gtao: false,
            gtao_denoise_passes: 0,
        });
        assert_ne!(off, on);
        assert_eq!(off, key_with_post(PostProcessChainSignature::default()));
    }

    /// Test pass that records how many retired views reached it.
    struct ReleaseCountingPass {
        /// Shared retirement count.
        releases: Arc<AtomicUsize>,
    }

    impl ComputePass for ReleaseCountingPass {
        fn name(&self) -> &str {
            "release_counting"
        }

        fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            builder.compute();
            Ok(())
        }

        fn record(&self, _ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
            Ok(())
        }

        fn release_view_resources(&mut self, retired_views: &[ViewId]) {
            self.releases
                .fetch_add(retired_views.len(), Ordering::Relaxed);
        }
    }

    /// Builds a minimal compiled graph containing one release-counting pass.
    fn graph_with_release_counter(releases: Arc<AtomicUsize>) -> CompiledRenderGraph {
        CompiledRenderGraph {
            passes: vec![PassNode::Compute(Box::new(ReleaseCountingPass {
                releases,
            }))],
            needs_surface_acquire: false,
            compile_stats: CompileStats {
                pass_count: 1,
                topo_levels: 1,
                ..CompileStats::default()
            },
            pass_info: Vec::new(),
            transient_textures: Vec::new(),
            transient_buffers: Vec::new(),
            subresources: Vec::new(),
            imported_textures: Vec::new(),
            imported_buffers: Vec::new(),
            schedule: FrameSchedule::empty(),
            main_graph_msaa_transient_handles: None,
        }
    }

    /// Releases retired view resources from every retained graph variant.
    #[test]
    fn release_view_resources_reaches_all_retained_variants() {
        let mono_key = key_with_post(PostProcessChainSignature::default());
        let mut stereo_key = mono_key;
        stereo_key.multiview_stereo = true;
        let mono_releases = Arc::new(AtomicUsize::new(0));
        let stereo_releases = Arc::new(AtomicUsize::new(0));
        let mut cache = GraphCache::default();

        assert_eq!(
            cache
                .ensure(mono_key, || {
                    Ok(graph_with_release_counter(Arc::clone(&mono_releases)))
                })
                .expect("mono graph"),
            GraphCacheEnsureResult::Built
        );
        assert_eq!(
            cache
                .ensure(stereo_key, || {
                    Ok(graph_with_release_counter(Arc::clone(&stereo_releases)))
                })
                .expect("stereo graph"),
            GraphCacheEnsureResult::Built
        );

        cache.release_view_resources(&[ViewId::Main]);

        assert_eq!(mono_releases.load(Ordering::Relaxed), 1);
        assert_eq!(stereo_releases.load(Ordering::Relaxed), 1);
    }
}
