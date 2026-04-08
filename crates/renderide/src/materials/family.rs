//! [`MaterialPipelineFamily`]: WGSL + render pipeline layout for one material class.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::pipelines::ShaderPermutation;

/// Opaque id for cache keys and routing (stable across runs for builtins).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialFamilyId(pub u32);

/// Swapchain-relevant state needed to build a [`wgpu::RenderPipeline`].
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelineDesc {
    /// Primary color attachment format (for example swapchain format).
    pub surface_format: wgpu::TextureFormat,
    /// Optional depth attachment (meshes / MRT later).
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// MSAA sample count (1 = off).
    pub sample_count: u32,
    /// When set, must match the render pass and pipeline (e.g. `0b11` for two multiview layers).
    pub multiview_mask: Option<NonZeroU32>,
}

/// One WGSL material program and how to compile it into a [`wgpu::RenderPipeline`].
///
/// Implementations are typically small, own no GPU handles, and return static-ish layouts from
/// `create_render_pipeline`.
pub trait MaterialPipelineFamily: Send + Sync {
    /// Stable id used in [`super::MaterialPipelineCacheKey`].
    fn family_id(&self) -> MaterialFamilyId;

    /// Full WGSL program (all entry points) after applying `permutation` (include patches via
    /// [`super::compose_wgsl`].
    fn build_wgsl(&self, permutation: ShaderPermutation) -> String;

    /// Compiles `module` into a raster pipeline for `desc` (layouts, targets, depth, MSAA).
    ///
    /// `wgsl_source` is the same string as [`Self::build_wgsl`]; reflective families use it with
    /// [`super::reflect_raster_material_wgsl`] to derive bind group layouts.
    fn create_render_pipeline(
        &self,
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        desc: &MaterialPipelineDesc,
        wgsl_source: &str,
    ) -> wgpu::RenderPipeline;

    /// When [`Some`], [`super::MaterialPipelineCache`] keys pipelines by this stem for manifest raster shaders.
    fn manifest_stem(&self) -> Option<Arc<str>> {
        None
    }
}
