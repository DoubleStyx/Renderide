//! Cache of [`wgpu::RenderPipeline`] per [`RasterPipelineKind`] + permutation + attachment formats.
//!
//! Lookup keys intentionally **do not** reflect WGSL on every cache probe: doing so would dominate
//! CPU cost. The material asset graph supplies a source generation for each embedded target, and
//! that generation is included in the key so development reloads and future dynamic shader sources
//! cannot reuse stale pipelines.
//!
//! The cache is LRU-bounded to avoid unbounded growth when many format/permutation combinations appear.

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::{Arc, OnceLock};

use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use lru::LruCache;
use parking_lot::Mutex;

use crate::gpu_resource::AtomicCacheCounters;
use crate::materials::ShaderPermutation;
use crate::materials::embedded::stem_metadata::{
    EmbeddedRasterPipelineSource, build_embedded_wgsl, create_embedded_render_pipelines,
    embedded_required_features_for_permutation,
};
use crate::materials::null_pipeline::{build_null_wgsl, create_null_render_pipeline};
use crate::materials::raster_pipeline::ShaderModuleBuildRefs;
use crate::materials::{
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
    RasterPrimitiveTopology,
};

use super::pipeline_build_error::PipelineBuildError;
use super::raster_pipeline::MaterialPipelineDesc;

/// Maximum raster pipelines retained (LRU eviction).
const MAX_CACHED_PIPELINES: usize = 512;

/// Number of shards across which the pipeline cache, pending-build set, and failed-build map are
/// split. Must be a power of two so shard routing collapses to a bitmask. Sized to keep each
/// recording worker on a distinct shard for the common case where N rayon workers issue
/// concurrent cache probes.
const PIPELINE_CACHE_SHARDS: usize = 16;

/// Per-shard LRU capacity so the sharded total still hits [`MAX_CACHED_PIPELINES`].
fn per_shard_cap() -> NonZeroUsize {
    NonZeroUsize::new(MAX_CACHED_PIPELINES.div_ceil(PIPELINE_CACHE_SHARDS).max(1))
        .unwrap_or(NonZeroUsize::MIN)
}

/// Material-driven pipeline variant: selectors that affect [`wgpu::RenderPipeline`] state but are
/// not derived from [`MaterialPipelineDesc`] attachment formats.
///
/// Bundled together so registry / cache lookups carry a single argument instead of five
/// loose scalars, and so any future axis (e.g. additional shader permutations) lands here without
/// growing call signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineVariantSpec {
    /// Stereo multiview / single-view permutation for the pipeline.
    pub permutation: ShaderPermutation,
    /// Material-level blend override for stems without explicit pass directives.
    pub blend_mode: MaterialBlendMode,
    /// Material-level stencil and color write state.
    pub render_state: MaterialRenderState,
    /// Front-face winding for draw transforms in this pipeline bucket.
    pub front_face: RasterFrontFace,
    /// Primitive topology baked into [`wgpu::PrimitiveState::topology`] for this pipeline bucket.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Key for [`MaterialPipelineCache`] lookups (no WGSL parse -- see module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineCacheKey {
    /// Which WGSL program backs the pipeline (embedded stem or null fallback).
    pub kind: RasterPipelineKind,
    /// Source generation from the material asset graph.
    ///
    /// Development hot reload and future dynamic shader compiler paths bump this value so stale
    /// pipelines cannot be reused for a changed WGSL source.
    pub shader_source_generation: u64,
    /// Stereo multiview / single-view permutation for the pipeline.
    pub permutation: ShaderPermutation,
    /// Color attachment format (swapchain or offscreen).
    pub surface_format: wgpu::TextureFormat,
    /// Depth/stencil format when depth attachment is used.
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// MSAA sample count for the color target.
    pub sample_count: u32,
    /// OpenXR / multiview view mask when compiling multiview pipelines.
    pub multiview_mask: Option<NonZeroU32>,
    /// Material-level blend override for stems without explicit pass directives.
    pub blend_mode: MaterialBlendMode,
    /// Material-level stencil and color write state.
    pub render_state: MaterialRenderState,
    /// Front-face winding for draw transforms in this pipeline bucket.
    pub front_face: RasterFrontFace,
    /// Primitive topology baked into [`wgpu::PrimitiveState::topology`] for this pipeline bucket.
    ///
    /// `wgpu::RenderPipeline` immutably bakes its primitive topology, so two draws of the same
    /// shader/material that differ in topology must build separate pipelines.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// One or more pipelines for a material entry (one per declared `//#pass`).
///
/// Materials without pass directives have `len == 1`; OverlayFresnel and other multi-pass shaders
/// have `len >= 2`. The forward encode loop dispatches every pipeline in order for each draw.
pub type MaterialPipelineSet = Arc<[wgpu::RenderPipeline]>;

/// Nonblocking cache lookup result.
pub(super) enum MaterialPipelineLookup {
    /// The requested pipeline set is available for this frame.
    Ready(MaterialPipelineSet),
    /// A background worker is building the requested pipeline set.
    Pending,
    /// The requested pipeline failed to build; callers may use a fallback.
    Failed(String),
}

/// Plain-data diagnostic snapshot for the material pipeline cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MaterialPipelineCacheDiagnosticSnapshot {
    /// Ready cached pipeline entries across all shards.
    pub(crate) ready_entries: usize,
    /// Pipeline entries currently compiling on the background worker.
    pub(crate) pending_entries: usize,
    /// Pipeline entries that failed compilation.
    pub(crate) failed_entries: usize,
    /// Cache hit counter.
    pub(crate) hits: u64,
    /// Cache miss counter.
    pub(crate) misses: u64,
    /// Cache insertion counter.
    pub(crate) insertions: u64,
    /// Cache eviction counter.
    pub(crate) evictions: u64,
}

struct PipelineBuildRequest {
    key: MaterialPipelineCacheKey,
    kind: RasterPipelineKind,
    desc: MaterialPipelineDesc,
    variant: MaterialPipelineVariantSpec,
    /// Optional WGSL source override loaded by development hot reload.
    wgsl_override: Option<Arc<str>>,
    device: Arc<wgpu::Device>,
    limits: Arc<crate::gpu::GpuLimits>,
    tx: crossbeam_channel::Sender<PipelineBuildOutcome>,
}

struct PipelineBuildOutcome {
    key: MaterialPipelineCacheKey,
    kind: RasterPipelineKind,
    result: Result<MaterialPipelineSet, String>,
}

/// One LRU slab plus the pending/failed sets for entries whose hash routes here.
///
/// Co-locating the three under one [`Mutex`] keeps the hot probe path on a single lock acquire:
/// the cache hit returns the [`MaterialPipelineSet`] without touching any other shard, and the
/// miss path stays on the same lock to record / observe pending and failed builds.
struct PipelineCacheShard {
    pipelines: LruCache<MaterialPipelineCacheKey, MaterialPipelineSet>,
    pending: HashSet<MaterialPipelineCacheKey>,
    failed: HashMap<MaterialPipelineCacheKey, String>,
}

impl PipelineCacheShard {
    fn new(cap: NonZeroUsize) -> Self {
        Self {
            pipelines: LruCache::new(cap),
            pending: HashSet::new(),
            failed: HashMap::new(),
        }
    }
}

/// Lazily built pipeline sets; LRU-evicted when over [`MAX_CACHED_PIPELINES`].
///
/// Cache state is split across [`PIPELINE_CACHE_SHARDS`] shards. Each shard bundles the
/// per-key LRU slab, the in-flight build set, and the failed-build map under one
/// [`Mutex`], so a concurrent probe from any rayon recording worker acquires exactly one lock.
pub struct MaterialPipelineCache {
    device: Arc<wgpu::Device>,
    limits: Arc<crate::gpu::GpuLimits>,
    shards: Box<[Mutex<PipelineCacheShard>]>,
    hasher: RandomState,
    pipeline_build_tx: crossbeam_channel::Sender<PipelineBuildOutcome>,
    pipeline_build_rx: crossbeam_channel::Receiver<PipelineBuildOutcome>,
    stats: AtomicCacheCounters,
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device` with the device's effective [`crate::gpu::GpuLimits`].
    pub fn new(device: Arc<wgpu::Device>, limits: Arc<crate::gpu::GpuLimits>) -> Self {
        let (pipeline_build_tx, pipeline_build_rx) = crossbeam_channel::unbounded();
        let cap = per_shard_cap();
        let shards: Box<[Mutex<PipelineCacheShard>]> = (0..PIPELINE_CACHE_SHARDS)
            .map(|_| Mutex::new(PipelineCacheShard::new(cap)))
            .collect();
        Self {
            device,
            limits,
            shards,
            hasher: RandomState::new(),
            pipeline_build_tx,
            pipeline_build_rx,
            stats: AtomicCacheCounters::default(),
        }
    }

    #[inline]
    fn shard_for(&self, key: &MaterialPipelineCacheKey) -> &Mutex<PipelineCacheShard> {
        let idx = (self.hasher.hash_one(key) as usize) & (PIPELINE_CACHE_SHARDS - 1);
        &self.shards[idx]
    }

    /// Returns the cached pipeline set or queues a background build for a miss.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only on the worker.
    ///
    /// Recording paths invoke this concurrently from rayon workers; the per-call hot path takes
    /// one shard lock and walks pipelines/pending/failed under it. Callers must invoke
    /// [`Self::drain_pipeline_build_completions`] once per frame before recording so freshly-built
    /// pipelines land in the cache.
    pub(super) fn get_or_queue(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
        shader_source_generation: u64,
        wgsl_override: Option<Arc<str>>,
    ) -> MaterialPipelineLookup {
        profiling::scope!("materials::get_or_create_pipeline");
        let key = Self::cache_key(kind, desc, variant, shader_source_generation);

        {
            let mut shard = self.shard_for(&key).lock();
            if let Some(hit) = shard.pipelines.get(&key).cloned() {
                drop(shard);
                self.stats.note_hit();
                return MaterialPipelineLookup::Ready(hit);
            }
            if let Some(err) = shard.failed.get(&key).cloned() {
                drop(shard);
                return MaterialPipelineLookup::Failed(err);
            }
            if !shard.pending.insert(key.clone()) {
                return MaterialPipelineLookup::Pending;
            }
        }

        self.queue_pipeline_build(key, kind.clone(), *desc, variant, wgsl_override);
        MaterialPipelineLookup::Pending
    }

    /// Queues a pipeline build if the exact key is not already ready, pending, or failed.
    pub(super) fn queue_warmup(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
        shader_source_generation: u64,
        wgsl_override: Option<Arc<str>>,
    ) {
        match self.get_or_queue(kind, desc, variant, shader_source_generation, wgsl_override) {
            MaterialPipelineLookup::Ready(_)
            | MaterialPipelineLookup::Pending
            | MaterialPipelineLookup::Failed(_) => {}
        }
    }

    fn cache_key(
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
        shader_source_generation: u64,
    ) -> MaterialPipelineCacheKey {
        let MaterialPipelineVariantSpec {
            permutation,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        } = variant;
        MaterialPipelineCacheKey {
            kind: kind.clone(),
            shader_source_generation,
            permutation,
            surface_format: desc.surface_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            multiview_mask: desc.multiview_mask,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        }
    }

    fn queue_pipeline_build(
        &self,
        key: MaterialPipelineCacheKey,
        kind: RasterPipelineKind,
        desc: MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
        wgsl_override: Option<Arc<str>>,
    ) {
        self.stats.note_miss();

        let request = PipelineBuildRequest {
            key: key.clone(),
            kind: kind.clone(),
            desc,
            variant,
            wgsl_override,
            device: self.device.clone(),
            limits: self.limits.clone(),
            tx: self.pipeline_build_tx.clone(),
        };
        if let Err(e) = spawn_pipeline_build(request) {
            let mut shard = self.shard_for(&key).lock();
            shard.pending.remove(&key);
            shard.failed.insert(key, e.clone());
            drop(shard);
            logger::warn!("MaterialPipelineCache: could not queue {kind:?} pipeline build: {e}");
        }
    }

    /// Drains the background-build completion channel into the pipeline cache.
    ///
    /// Must be called once per frame before per-view recording starts. Pulling the channel off
    /// the hot path keeps [`Self::get_or_queue`] from contending the pending/failed sets on
    /// every cache probe.
    pub(super) fn drain_pipeline_build_completions(&self) {
        profiling::scope!("materials::drain_pipeline_build_completions");
        self.drain_completed_pipeline_builds();
    }

    fn drain_completed_pipeline_builds(&self) {
        while let Ok(outcome) = self.pipeline_build_rx.try_recv() {
            match outcome.result {
                Ok(set) => self.insert_completed_pipeline_set(outcome.key, set),
                Err(e) => {
                    logger::warn!(
                        "MaterialPipelineCache: async pipeline build failed for {:?}: {e}",
                        outcome.kind
                    );
                    let mut shard = self.shard_for(&outcome.key).lock();
                    shard.pending.remove(&outcome.key);
                    shard.failed.insert(outcome.key, e);
                }
            }
        }
    }

    fn build_pipeline_set_for(
        device: Arc<wgpu::Device>,
        limits: Arc<crate::gpu::GpuLimits>,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
        wgsl_override: Option<Arc<str>>,
    ) -> Result<MaterialPipelineSet, PipelineBuildError> {
        let MaterialPipelineVariantSpec {
            permutation,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        } = variant;
        let wgsl = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => {
                validate_embedded_required_features(&device, stem, permutation)?;
                if let Some(source) = wgsl_override.as_ref() {
                    source.to_string()
                } else {
                    build_embedded_wgsl(stem, permutation)?
                }
            }
            RasterPipelineKind::Null => {
                if let Some(source) = wgsl_override.as_ref() {
                    source.to_string()
                } else {
                    build_null_wgsl(permutation)?
                }
            }
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster_material_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        let pipelines: Vec<wgpu::RenderPipeline> = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => create_embedded_render_pipelines(
                EmbeddedRasterPipelineSource {
                    stem: stem.clone(),
                    permutation,
                    blend_mode,
                    render_state,
                    front_face,
                    primitive_topology,
                },
                ShaderModuleBuildRefs {
                    device: &device,
                    limits: &limits,
                    module: &module,
                    desc,
                    wgsl_source: &wgsl,
                },
            )?,
            RasterPipelineKind::Null => {
                vec![create_null_render_pipeline(
                    &device,
                    &limits,
                    &module,
                    desc,
                    &wgsl,
                    front_face,
                    primitive_topology,
                )?]
            }
        };
        Ok(Arc::from(pipelines.into_boxed_slice()))
    }

    fn insert_completed_pipeline_set(
        &self,
        key: MaterialPipelineCacheKey,
        set: MaterialPipelineSet,
    ) {
        self.stats.note_insertion();
        let evicted = {
            let mut shard = self.shard_for(&key).lock();
            shard.pending.remove(&key);
            shard.failed.remove(&key);
            shard.pipelines.push(key, set)
        };

        if let Some((_evicted_key, evicted)) = evicted {
            drop(evicted);
            self.stats.note_eviction();
            let stats = self.stats.snapshot();
            logger::trace!(
                "MaterialPipelineCache: evicted LRU pipeline entry hits={} misses={} insertions={} evictions={}",
                stats.hits,
                stats.misses,
                stats.insertions,
                stats.evictions
            );
        }
    }

    /// Captures ready/pending/failed entry counts and cache counters.
    pub(super) fn diagnostic_snapshot(&self) -> MaterialPipelineCacheDiagnosticSnapshot {
        let mut ready_entries = 0usize;
        let mut pending_entries = 0usize;
        let mut failed_entries = 0usize;
        for shard in &self.shards {
            let shard = shard.lock();
            ready_entries = ready_entries.saturating_add(shard.pipelines.len());
            pending_entries = pending_entries.saturating_add(shard.pending.len());
            failed_entries = failed_entries.saturating_add(shard.failed.len());
        }
        let stats = self.stats.snapshot();
        MaterialPipelineCacheDiagnosticSnapshot {
            ready_entries,
            pending_entries,
            failed_entries,
            hits: stats.hits,
            misses: stats.misses,
            insertions: stats.insertions,
            evictions: stats.evictions,
        }
    }
}

/// Ensures the active device can compile an embedded material target before module creation.
fn validate_embedded_required_features(
    device: &wgpu::Device,
    stem: &Arc<str>,
    permutation: ShaderPermutation,
) -> Result<(), PipelineBuildError> {
    let required = embedded_required_features_for_permutation(stem, permutation);
    let missing = required - device.features();
    if missing.is_empty() {
        return Ok(());
    }
    Err(PipelineBuildError::MissingDeviceFeatures {
        stem: stem.to_string(),
        missing,
    })
}

fn spawn_pipeline_build(request: PipelineBuildRequest) -> Result<(), String> {
    let pool = material_pipeline_compile_pool()?;
    pool.spawn(move || {
        profiling::scope!("materials::async_pipeline_compile");
        let PipelineBuildRequest {
            key,
            kind,
            desc,
            variant,
            wgsl_override,
            device,
            limits,
            tx,
        } = request;
        let result = MaterialPipelineCache::build_pipeline_set_for(
            device,
            limits,
            &kind,
            &desc,
            variant,
            wgsl_override,
        )
        .map_err(|e| e.to_string());
        let _ = tx.send(PipelineBuildOutcome { key, kind, result });
    });
    Ok(())
}

fn material_pipeline_compile_pool() -> Result<&'static rayon::ThreadPool, String> {
    static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "material-pipeline-worker".to_string())
            .build()
            .map_err(|e| format!("material pipeline worker pool creation failed: {e}"))
    })
    .as_ref()
    .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{MaterialPipelineCache, MaterialPipelineVariantSpec};
    use crate::materials::{
        MaterialBlendMode, MaterialPipelineDesc, MaterialRenderState, RasterFrontFace,
        RasterPipelineKind, RasterPrimitiveTopology, ShaderPermutation,
    };

    fn base_desc() -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Rgba16Float,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
            multiview_mask: None,
        }
    }

    fn base_variant() -> MaterialPipelineVariantSpec {
        MaterialPipelineVariantSpec {
            permutation: ShaderPermutation::default(),
            blend_mode: MaterialBlendMode::Opaque,
            render_state: MaterialRenderState::default(),
            front_face: RasterFrontFace::default(),
            primitive_topology: RasterPrimitiveTopology::default(),
        }
    }

    #[test]
    fn cache_key_includes_shader_source_generation() {
        let kind = RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default"));

        let first = MaterialPipelineCache::cache_key(&kind, &base_desc(), base_variant(), 1);
        let second = MaterialPipelineCache::cache_key(&kind, &base_desc(), base_variant(), 2);

        assert_ne!(first, second);
        assert_eq!(first.shader_source_generation, 1);
        assert_eq!(second.shader_source_generation, 2);
    }
}
