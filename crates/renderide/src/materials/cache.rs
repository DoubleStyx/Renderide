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

use lru::LruCache;
use parking_lot::Mutex;

use crate::materials::embedded_raster_pipeline::{
    build_embedded_wgsl, create_embedded_render_pipelines, EmbeddedRasterPipelineSource,
};
use crate::materials::raster_pipeline::ShaderModuleBuildRefs;
use crate::materials::{
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
};
use crate::pipelines::raster::null::{build_null_wgsl, create_null_render_pipeline};
use crate::pipelines::ShaderPermutation;

use super::family::MaterialPipelineDesc;
use super::pipeline_build_error::PipelineBuildError;

/// Maximum raster pipelines retained (LRU eviction).
const MAX_CACHED_PIPELINES: usize = 512;

const MAX_CACHED_PIPELINES_NZ: NonZeroUsize = {
    match NonZeroUsize::new(MAX_CACHED_PIPELINES) {
        Some(n) => n,
        None => panic!("MAX_CACHED_PIPELINES must be non-zero"),
    }
};

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
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device` with the device's effective [`crate::gpu::GpuLimits`].
    pub fn new(device: Arc<wgpu::Device>, limits: Arc<crate::gpu::GpuLimits>) -> Self {
        Self {
            device,
            limits,
            pipelines: Mutex::new(LruCache::new(MAX_CACHED_PIPELINES_NZ)),
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

    /// Returns or builds the pipeline set for `kind`, `desc`, and `permutation`.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only when inserting a new entry.
    pub fn get_or_create(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
        front_face: RasterFrontFace,
    ) -> Result<MaterialPipelineSet, PipelineBuildError> {
        profiling::scope!("materials::get_or_create_pipeline");
        let key = MaterialPipelineCacheKey {
            kind: kind.clone(),
            permutation,
            surface_format: desc.surface_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            multiview_mask: desc.multiview_mask,
            blend_mode,
            render_state,
            front_face,
        };
        //perf xlinka: a hit is real use; promote it so hot pipelines do not get evicted.
        if let Some(hit) = self.pipelines.lock().get(&key) {
            return Ok(hit.clone());
        }
        let wgsl = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => build_embedded_wgsl(stem, permutation)?,
            RasterPipelineKind::Null => build_null_wgsl(permutation)?,
        };
        let device = self.device.clone();
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
                },
                ShaderModuleBuildRefs {
                    device: &device,
                    limits: &self.limits,
                    module: &module,
                    desc,
                    wgsl_source: &wgsl,
                },
            )?,
            RasterPipelineKind::Null => {
                vec![create_null_render_pipeline(
                    &device,
                    &self.limits,
                    &module,
                    desc,
                    &wgsl,
                    front_face,
                )?]
            }
        };
        let set: MaterialPipelineSet = Arc::from(pipelines.into_boxed_slice());
        let mut cache = self.pipelines.lock();
        if let Some(existing) = cache.get(&key) {
            return Ok(existing.clone());
        }
        if let Some(evicted) = cache.put(key, set.clone()) {
            drop(evicted);
            logger::trace!("MaterialPipelineCache: evicted LRU pipeline entry");
        }
        drop(cache);
        Ok(set)
    }
}
