//! [`MaterialRegistry`]: [`MaterialRouter`], [`super::MaterialPipelineCache`], and shader route updates.

use std::sync::Arc;

use hashbrown::HashSet;
use parking_lot::Mutex;

use crate::passes::PipelineVariantKey;

use super::cache::{
    MaterialPipelineCache, MaterialPipelineCacheStats, MaterialPipelineSet,
    MaterialPipelineVariantSpec,
};
use super::family::MaterialPipelineDesc;
use super::pipeline_kind::RasterPipelineKind;
use super::resolve_raster::resolve_raster_pipeline;
use super::router::MaterialRouter;

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
    device: Arc<wgpu::Device>,
    /// Shader asset id -> pipeline family and resolved asset-name routing.
    pub router: MaterialRouter,
    cache: MaterialPipelineCache,
    /// Cross-frame set of pipeline variants the pre-warm path has already requested. Lets
    /// `pre_warm_pipeline_cache_for_views` skip the rayon dispatch (and the per-call cache mutex
    /// acquisition) for variants that are already cached in steady state. The set is best-effort:
    /// LRU eviction in [`MaterialPipelineCache`] may cause an entry here to outlive its cached
    /// pipeline, but the lazy compile inside the record path still matches today's behavior.
    warmed_variants: Mutex<HashSet<PipelineVariantKey>>,
}

impl MaterialRegistry {
    fn try_pipeline_with_fallback(
        &self,
        request: PipelineLookupRequest<'_>,
    ) -> Option<MaterialPipelineSet> {
        let PipelineLookupRequest {
            shader_asset_id,
            kind,
            desc,
            variant,
        } = request;
        let err = match self.cache.get_or_create(kind, desc, variant) {
            Ok(p) => return Some(p),
            Err(e) => e,
        };
        if matches!(kind, RasterPipelineKind::Null) {
            match shader_asset_id {
                Some(id) => {
                    logger::error!("Null pipeline build failed (shader_asset_id={id}): {err}");
                }
                None => {
                    logger::error!("Null pipeline build failed: {err}");
                }
            }
            return None;
        }
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
        let fallback = RasterPipelineKind::Null;
        match self.cache.get_or_create(&fallback, desc, variant) {
            Ok(p) => Some(p),
            Err(e2) => {
                logger::error!("fallback Null pipeline build failed: {e2}");
                None
            }
        }
    }

    /// Builds a registry whose router falls back to [`RasterPipelineKind::Null`] for unknown shader assets.
    pub fn with_default_families(
        device: Arc<wgpu::Device>,
        limits: Arc<crate::gpu::GpuLimits>,
    ) -> Self {
        Self {
            device: device.clone(),
            router: MaterialRouter::new(RasterPipelineKind::Null),
            cache: MaterialPipelineCache::new(device, limits),
            warmed_variants: Mutex::new(HashSet::new()),
        }
    }

    /// Inserts a host shader id -> pipeline mapping and optional resolved AssetBundle shader asset name.
    pub fn map_shader_route(
        &mut self,
        shader_asset_id: i32,
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
    ) {
        self.router
            .set_shader_route(shader_asset_id, pipeline.clone(), shader_asset_name);
        match &pipeline {
            RasterPipelineKind::EmbeddedStem(s) => {
                self.router.set_shader_stem(shader_asset_id, s.to_string());
            }
            RasterPipelineKind::Null => {
                self.router.remove_shader_stem(shader_asset_id);
            }
        }
    }

    /// Inserts a host shader id -> pipeline mapping without an AssetBundle shader asset name.
    pub fn map_shader_pipeline(&mut self, shader_asset_id: i32, pipeline: RasterPipelineKind) {
        self.map_shader_route(shader_asset_id, pipeline, None);
    }

    /// Removes routing for a host shader id [`crate::shared::ShaderUnload`].
    pub fn unmap_shader(&mut self, shader_asset_id: i32) {
        self.router.remove_shader_route(shader_asset_id);
        self.warmed_variants
            .lock()
            .retain(|v| v.shader_asset_id != shader_asset_id);
    }

    /// Returns `true` when `key` has already been requested through the pre-warm path. See
    /// [`Self::warmed_variants`].
    pub(crate) fn is_pipeline_variant_warmed(&self, key: &PipelineVariantKey) -> bool {
        self.warmed_variants.lock().contains(key)
    }

    /// Marks `keys` as warmed so the next frame's pre-warm walk can skip them.
    pub(crate) fn mark_pipeline_variants_warmed<I>(&self, keys: I)
    where
        I: IntoIterator<Item = PipelineVariantKey>,
    {
        let mut set = self.warmed_variants.lock();
        set.extend(keys);
    }

    /// Resolves a cached or new pipeline for a host shader asset (via router + embedded stem when applicable).
    pub fn pipeline_for_shader_asset(
        &self,
        shader_asset_id: i32,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Option<MaterialPipelineSet> {
        let kind = resolve_raster_pipeline(shader_asset_id, &self.router);
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: Some(shader_asset_id),
            kind: &kind,
            desc,
            variant,
        })
    }

    /// Looks up a pipeline by explicit kind (for example tests or tools that do not use a host shader id).
    pub fn pipeline_for_kind(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Option<MaterialPipelineSet> {
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: None,
            kind,
            desc,
            variant,
        })
    }

    /// Low-level cache access keyed by [`RasterPipelineKind`].
    pub fn get_or_create_pipeline(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Option<MaterialPipelineSet> {
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: None,
            kind,
            desc,
            variant,
        })
    }

    /// Borrow the wgpu device held by this registry.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Diagnostic snapshot of the material pipeline cache.
    pub fn pipeline_cache_stats(&self) -> MaterialPipelineCacheStats {
        self.cache.stats()
    }

    /// Shader routes for the debug HUD (`shader_asset_id`, [`RasterPipelineKind`], optional AssetBundle shader asset name), sorted.
    pub fn shader_routes_for_hud(&self) -> Vec<(i32, RasterPipelineKind, Option<String>)> {
        self.router.routes_sorted_for_hud()
    }

    /// Resolved composed WGSL stem for a host shader id, when [`Self::map_shader_route`] recorded one.
    pub fn stem_for_shader_asset(&self, shader_asset_id: i32) -> Option<&str> {
        self.router.stem_for_shader_asset(shader_asset_id)
    }
}
