//! [`MaterialRegistry`]: [`MaterialRouter`], [`super::MaterialPipelineCache`], and shader route updates.

use std::sync::Arc;

use super::cache::{
    MaterialPipelineCache, MaterialPipelineLookup, MaterialPipelineSet, MaterialPipelineVariantSpec,
};
use super::pipeline_kind::RasterPipelineKind;
use super::raster_pipeline::MaterialPipelineDesc;
use super::router::MaterialRouter;

/// Pipeline set paired with the concrete raster kind that produced it.
#[derive(Clone)]
pub(crate) struct MaterialPipelineResolution {
    /// Raster pipeline kind whose layout matches [`Self::pipelines`].
    pub(crate) kind: RasterPipelineKind,
    /// Pipeline set for the resolved raster kind.
    pub(crate) pipelines: MaterialPipelineSet,
}

impl MaterialPipelineResolution {
    /// Builds a resolved pipeline value from a concrete kind and ready pipeline set.
    fn new(kind: RasterPipelineKind, pipelines: MaterialPipelineSet) -> Self {
        Self { kind, pipelines }
    }
}

/// Full cache lookup request for one material pipeline variant.
struct PipelineLookupRequest<'a> {
    /// Host shader asset id for diagnostics, or [`None`] for direct-kind lookups.
    shader_asset_id: Option<i32>,
    /// Raster pipeline kind to resolve.
    kind: &'a RasterPipelineKind,
    /// Attachment formats and sample count.
    desc: &'a MaterialPipelineDesc,
    /// Material-driven pipeline-state selectors (permutation, blend, render state, front face,
    /// primitive topology).
    variant: MaterialPipelineVariantSpec,
}

/// Owning table of material routing and pipeline cache.
pub struct MaterialRegistry {
    /// Shader asset id -> pipeline family and resolved asset-name routing.
    pub router: MaterialRouter,
    cache: MaterialPipelineCache,
}

impl MaterialRegistry {
    fn try_pipeline_with_fallback(
        &self,
        request: PipelineLookupRequest<'_>,
    ) -> Option<MaterialPipelineResolution> {
        let PipelineLookupRequest {
            shader_asset_id,
            kind,
            desc,
            variant,
        } = request;
        match self.cache.get_or_queue(kind, desc, variant) {
            MaterialPipelineLookup::Ready(p) => {
                Some(MaterialPipelineResolution::new(kind.clone(), p))
            }
            MaterialPipelineLookup::Pending if matches!(kind, RasterPipelineKind::Null) => None,
            MaterialPipelineLookup::Pending => self.null_pipeline_fallback(desc, variant),
            MaterialPipelineLookup::Failed(err) if matches!(kind, RasterPipelineKind::Null) => {
                match shader_asset_id {
                    Some(id) => {
                        logger::error!("Null pipeline build failed (shader_asset_id={id}): {err}");
                    }
                    None => {
                        logger::error!("Null pipeline build failed: {err}");
                    }
                }
                None
            }
            MaterialPipelineLookup::Failed(err) => {
                match shader_asset_id {
                    Some(id) => {
                        logger::warn!(
                            "material pipeline build failed (shader_asset_id={id}, kind={kind:?}): {err}; falling back to Null"
                        );
                    }
                    None => {
                        logger::warn!(
                            "material pipeline build failed (kind={kind:?}): {err}; falling back to Null"
                        );
                    }
                }
                self.null_pipeline_fallback(desc, variant)
            }
        }
    }

    fn null_pipeline_fallback(
        &self,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Option<MaterialPipelineResolution> {
        let fallback = RasterPipelineKind::Null;
        match self.cache.get_or_queue(&fallback, desc, variant) {
            MaterialPipelineLookup::Ready(p) => Some(MaterialPipelineResolution::new(fallback, p)),
            MaterialPipelineLookup::Pending => None,
            MaterialPipelineLookup::Failed(e) => {
                logger::error!("fallback Null pipeline build failed: {e}");
                None
            }
        }
    }

    /// Resolves the Null fallback pipeline for a caller that must abandon a routed embedded draw.
    pub(crate) fn null_pipeline_for_variant(
        &self,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Option<MaterialPipelineResolution> {
        self.null_pipeline_fallback(desc, variant)
    }

    /// Builds a registry whose router falls back to [`RasterPipelineKind::Null`] for unknown shader assets.
    pub fn with_default_families(
        device: Arc<wgpu::Device>,
        limits: Arc<crate::gpu::GpuLimits>,
    ) -> Self {
        Self {
            router: MaterialRouter::new(RasterPipelineKind::Null),
            cache: MaterialPipelineCache::new(device, limits),
        }
    }

    /// Inserts a host shader id -> pipeline mapping and optional resolved AssetBundle shader asset name.
    pub fn map_shader_route(
        &mut self,
        shader_asset_id: i32,
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
        shader_variant_bits: Option<u32>,
    ) {
        self.router.set_shader_route(
            shader_asset_id,
            pipeline.clone(),
            shader_asset_name,
            shader_variant_bits,
        );
        match &pipeline {
            RasterPipelineKind::EmbeddedStem(s) => {
                self.router.set_shader_stem(shader_asset_id, s.to_string());
            }
            RasterPipelineKind::Null => {
                self.router.remove_shader_stem(shader_asset_id);
            }
        }
    }

    /// Removes routing for a host shader id [`crate::shared::ShaderUnload`].
    pub fn unmap_shader(&mut self, shader_asset_id: i32) {
        self.router.remove_shader_route(shader_asset_id);
    }

    /// Resolves a cached or new pipeline for an already-resolved raster pipeline kind.
    pub(crate) fn pipeline_for_resolved_kind(
        &self,
        shader_asset_id: i32,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Option<MaterialPipelineResolution> {
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: Some(shader_asset_id),
            kind,
            desc,
            variant,
        })
    }

    /// Shader routes for the debug HUD (`shader_asset_id`, [`RasterPipelineKind`], optional AssetBundle shader metadata), sorted.
    pub fn shader_routes_for_hud(
        &self,
    ) -> Vec<(i32, RasterPipelineKind, Option<String>, Option<u32>)> {
        self.router.routes_sorted_for_hud()
    }

    /// Resolved composed WGSL stem for a host shader id, when [`Self::map_shader_route`] recorded one.
    pub fn stem_for_shader_asset(&self, shader_asset_id: i32) -> Option<&str> {
        self.router.stem_for_shader_asset(shader_asset_id)
    }

    /// Froox shader variant bitmask for a host shader id, when one was parsed.
    pub fn variant_bits_for_shader_asset(&self, shader_asset_id: i32) -> Option<u32> {
        self.router.variant_bits_for_shader_asset(shader_asset_id)
    }

    /// Drains finished background pipeline builds into the cache.
    ///
    /// Invoked once per frame from the renderer's tick before per-view recording so worker
    /// threads never touch the completion channel or the pending/failed mutexes during draw.
    pub fn drain_pipeline_build_completions(&self) {
        self.cache.drain_pipeline_build_completions();
    }
}
