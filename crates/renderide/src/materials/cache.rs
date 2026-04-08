//! Cache of [`wgpu::RenderPipeline`] per [`RasterPipelineKind`] + permutation + attachment formats.
//!
//! Lookup keys intentionally **do not** include a WGSL layout fingerprint: reflecting the full
//! shader on every cache probe would dominate CPU cost. Embedded targets are stable per
//! `(kind, permutation, [`MaterialPipelineDesc`])`. If hot-reload or dynamic WGSL is introduced,
//! extend the key with a content hash or version.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::materials::embedded_raster_pipeline::{
    build_embedded_wgsl, create_embedded_render_pipeline,
};
use crate::materials::RasterPipelineKind;
use crate::pipelines::raster::debug_world_normals::{
    build_debug_world_normals_wgsl, create_debug_world_normals_render_pipeline,
};
use crate::pipelines::ShaderPermutation;

use super::family::MaterialPipelineDesc;

/// Key for [`MaterialPipelineCache`] lookups (no WGSL parse — see module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineCacheKey {
    /// Which WGSL program backs the pipeline (embedded stem or debug fallback).
    pub kind: RasterPipelineKind,
    pub permutation: ShaderPermutation,
    pub surface_format: wgpu::TextureFormat,
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    pub sample_count: u32,
    pub multiview_mask: Option<NonZeroU32>,
}

/// Lazily built pipelines; safe to retain for the [`wgpu::Device`] lifetime.
#[derive(Debug)]
pub struct MaterialPipelineCache {
    device: Arc<wgpu::Device>,
    pipelines: HashMap<MaterialPipelineCacheKey, wgpu::RenderPipeline>,
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device`.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            pipelines: HashMap::new(),
        }
    }

    /// Device used for `create_shader_module` / `create_render_pipeline`.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Returns or builds a pipeline for `kind`, `desc`, and `permutation`.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only when inserting a new entry.
    pub fn get_or_create(
        &mut self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
    ) -> &wgpu::RenderPipeline {
        let key = MaterialPipelineCacheKey {
            kind: kind.clone(),
            permutation,
            surface_format: desc.surface_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            multiview_mask: desc.multiview_mask,
        };
        self.pipelines.entry(key).or_insert_with(|| {
            let wgsl = match kind {
                RasterPipelineKind::EmbeddedStem(stem) => build_embedded_wgsl(stem, permutation),
                RasterPipelineKind::DebugWorldNormals => {
                    build_debug_world_normals_wgsl(permutation)
                }
            };
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("raster_material_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
                });
            match kind {
                RasterPipelineKind::EmbeddedStem(_) => {
                    create_embedded_render_pipeline(&self.device, &module, desc, &wgsl)
                }
                RasterPipelineKind::DebugWorldNormals => {
                    create_debug_world_normals_render_pipeline(&self.device, &module, desc, &wgsl)
                }
            }
        })
    }
}
