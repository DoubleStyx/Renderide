//! Cache of [`wgpu::RenderPipeline`] per [`RasterPipelineKind`] + permutation + attachment formats.
//!
//! Lookup keys intentionally **do not** include a WGSL layout fingerprint: reflecting the full
//! shader on every cache probe would dominate CPU cost. Embedded targets are stable per
//! `(kind, permutation, [`MaterialPipelineDesc`])`. If hot-reload or dynamic WGSL is introduced,
//! extend the key with a content hash or version.
//!
//! The cache is LRU-bounded to avoid unbounded growth when many format/permutation combinations appear.

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use parking_lot::Mutex;

use crate::materials::ShaderPermutation;
use crate::materials::embedded_raster_pipeline::{
    EmbeddedRasterPipelineSource, build_embedded_wgsl, create_embedded_render_pipelines,
};
use crate::materials::null_pipeline::{build_null_wgsl, create_null_render_pipeline};
use crate::materials::raster_pipeline::ShaderModuleBuildRefs;
use crate::materials::{
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
};

use super::family::MaterialPipelineDesc;
use super::pipeline_build_error::PipelineBuildError;

/// Maximum raster pipelines retained (LRU eviction).
const MAX_CACHED_PIPELINES: usize = 512;

/// Non-zero raster pipeline cache capacity.
fn max_cached_pipelines() -> NonZeroUsize {
    NonZeroUsize::new(MAX_CACHED_PIPELINES).unwrap_or(NonZeroUsize::MIN)
}

/// Key for [`MaterialPipelineCache`] lookups (no WGSL parse — see module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineCacheKey {
    /// Which WGSL program backs the pipeline (embedded stem or null fallback).
    pub kind: RasterPipelineKind,
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
    /// Whether this pipeline renders point-list geometry.
    pub point_topology: bool,
}

impl MaterialPipelineCacheKey {
    /// Rehydrates the descriptor used by raster pipeline builders.
    fn pass_desc(&self) -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: self.surface_format,
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
            multiview_mask: self.multiview_mask,
        }
    }
}

/// One or more pipelines for a material entry (one per declared `//#pass`).
///
/// Materials without pass directives have `len == 1`; OverlayFresnel and other multi-pass shaders
/// have `len >= 2`. The forward encode loop dispatches every pipeline in order for each draw.
pub type MaterialPipelineSet = Arc<[wgpu::RenderPipeline]>;

/// Lazily built pipeline sets; LRU-evicted when over [`MAX_CACHED_PIPELINES`].
#[derive(Debug)]
pub struct MaterialPipelineCache {
    device: Arc<wgpu::Device>,
    limits: Arc<crate::gpu::GpuLimits>,
    pipelines: Mutex<LruCache<MaterialPipelineCacheKey, MaterialPipelineSet>>,
    stats: MaterialPipelineCacheStatsAtomic,
}

/// Snapshot of material pipeline cache counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaterialPipelineCacheStats {
    /// Cache hits that returned an already-built pipeline set.
    pub hits: u64,
    /// Cache misses that had to compose WGSL and build pipelines.
    pub misses: u64,
    /// Newly inserted pipeline sets.
    pub insertions: u64,
    /// LRU evictions caused by cache capacity pressure.
    pub evictions: u64,
}

/// Atomic backing counters for [`MaterialPipelineCacheStats`].
#[derive(Debug, Default)]
struct MaterialPipelineCacheStatsAtomic {
    /// Cache hits that returned an already-built pipeline set.
    hits: AtomicU64,
    /// Cache misses that had to compose WGSL and build pipelines.
    misses: AtomicU64,
    /// Newly inserted pipeline sets.
    insertions: AtomicU64,
    /// LRU evictions caused by cache capacity pressure.
    evictions: AtomicU64,
}

impl MaterialPipelineCacheStatsAtomic {
    /// Captures a relaxed diagnostic snapshot.
    fn snapshot(&self) -> MaterialPipelineCacheStats {
        MaterialPipelineCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            insertions: self.insertions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device` with the device's effective [`crate::gpu::GpuLimits`].
    pub fn new(device: Arc<wgpu::Device>, limits: Arc<crate::gpu::GpuLimits>) -> Self {
        Self {
            device,
            limits,
            pipelines: Mutex::new(LruCache::new(max_cached_pipelines())),
            stats: MaterialPipelineCacheStatsAtomic::default(),
        }
    }

    /// Device used for `create_shader_module` / `create_render_pipeline`.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Effective device limits used to validate reflected material layouts.
    pub fn limits(&self) -> &Arc<crate::gpu::GpuLimits> {
        &self.limits
    }

    /// Returns a diagnostic snapshot of cache hit/miss/eviction counters.
    pub fn stats(&self) -> MaterialPipelineCacheStats {
        self.stats.snapshot()
    }

    /// Returns or builds the pipeline set for one fully-specified cache key.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only when inserting a new entry.
    pub fn get_or_create(
        &self,
        key: MaterialPipelineCacheKey,
    ) -> Result<MaterialPipelineSet, PipelineBuildError> {
        profiling::scope!("materials::get_or_create_pipeline");
        let desc = key.pass_desc();
        //perf xlinka: a hit is real use; promote it so hot pipelines do not get evicted.
        if let Some(hit) = self.pipelines.lock().get(&key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(hit.clone());
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let wgsl = match &key.kind {
            RasterPipelineKind::EmbeddedStem(stem) => build_embedded_wgsl(stem, key.permutation)?,
            RasterPipelineKind::Null => build_null_wgsl(key.permutation)?,
        };
        let device = self.device.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster_material_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        let pipelines: Vec<wgpu::RenderPipeline> = match &key.kind {
            RasterPipelineKind::EmbeddedStem(stem) => create_embedded_render_pipelines(
                EmbeddedRasterPipelineSource {
                    stem: stem.clone(),
                    permutation: key.permutation,
                    blend_mode: key.blend_mode,
                    render_state: key.render_state,
                    front_face: key.front_face,
                    point_topology: key.point_topology,
                },
                ShaderModuleBuildRefs {
                    device: &device,
                    limits: &self.limits,
                    module: &module,
                    desc: &desc,
                    wgsl_source: &wgsl,
                },
            )?,
            RasterPipelineKind::Null => {
                vec![create_null_render_pipeline(
                    &device,
                    &self.limits,
                    &module,
                    &desc,
                    &wgsl,
                    key.front_face,
                    key.point_topology,
                )?]
            }
        };
        let set: MaterialPipelineSet = Arc::from(pipelines.into_boxed_slice());
        let mut cache = self.pipelines.lock();
        if let Some(existing) = cache.get(&key) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(existing.clone());
        }
        self.stats.insertions.fetch_add(1, Ordering::Relaxed);
        if let Some(evicted) = cache.put(key, set.clone()) {
            drop(evicted);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            logger::trace!("MaterialPipelineCache: evicted LRU pipeline entry");
        }
        drop(cache);
        Ok(set)
    }
}
