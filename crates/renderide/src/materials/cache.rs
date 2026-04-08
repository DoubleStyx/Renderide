//! Cache of [`wgpu::RenderPipeline`] per material family + permutation + attachment formats.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::pipelines::ShaderPermutation;

use super::family::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use super::wgsl_reflect::reflect_raster_material_wgsl;

/// Key for [`MaterialPipelineCache`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineCacheKey {
    pub family_id: MaterialFamilyId,
    /// Present for [`super::MANIFEST_RASTER_FAMILY_ID`] so distinct manifest stems do not share a pipeline.
    pub manifest_stem: Option<Arc<str>>,
    pub permutation: ShaderPermutation,
    /// From [`super::reflect_raster_material_wgsl`] when the shader matches the frame-globals contract; `0` if reflection failed (e.g. no `@group(0)`).
    pub layout_fingerprint: u64,
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

    /// Returns or builds a pipeline for `family`, `desc`, and `permutation`.
    pub fn get_or_create(
        &mut self,
        family: &dyn MaterialPipelineFamily,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
    ) -> &wgpu::RenderPipeline {
        let wgsl = family.build_wgsl(permutation);
        let layout_fingerprint = reflect_raster_material_wgsl(&wgsl)
            .map(|r| r.layout_fingerprint)
            .unwrap_or(0);
        let key = MaterialPipelineCacheKey {
            family_id: family.family_id(),
            manifest_stem: family.manifest_stem(),
            permutation,
            layout_fingerprint,
            surface_format: desc.surface_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            multiview_mask: desc.multiview_mask,
        };
        self.pipelines.entry(key).or_insert_with(|| {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("material_family_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
                });
            family.create_render_pipeline(&self.device, &module, desc, &wgsl)
        })
    }
}
