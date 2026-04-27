//! [`MaterialRegistry`]: [`MaterialRouter`], [`super::MaterialPipelineCache`], and shader route updates.

use std::sync::Arc;

use crate::pipelines::ShaderPermutation;

use super::cache::{MaterialPipelineCache, MaterialPipelineSet};
use super::embedded_shader_stem::embedded_default_stem_for_unity_name;
use super::family::MaterialPipelineDesc;
use super::material_passes::MaterialBlendMode;
use super::pipeline_kind::RasterPipelineKind;
use super::render_state::{MaterialRenderState, RasterFrontFace};
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
    /// Shader permutation for mono vs stereo targets.
    permutation: ShaderPermutation,
    /// Material blend mode for pipeline materialization.
    blend_mode: MaterialBlendMode,
    /// Runtime material render state for pipeline materialization.
    render_state: MaterialRenderState,
    /// Front-face winding selected from draw transform handedness.
    front_face: RasterFrontFace,
}

/// Owning table of material routing and pipeline cache.
pub struct MaterialRegistry {
    device: Arc<wgpu::Device>,
    /// Shader asset id → pipeline family and display name routing.
    pub router: MaterialRouter,
    cache: MaterialPipelineCache,
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
            permutation,
            blend_mode,
            render_state,
            front_face,
        } = request;
        let err = match self.cache.get_or_create(
            kind,
            desc,
            permutation,
            blend_mode,
            render_state,
            front_face,
        ) {
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
        match self.cache.get_or_create(
            &fallback,
            desc,
            permutation,
            blend_mode,
            render_state,
            front_face,
        ) {
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
        }
    }

    /// Inserts a host shader id → pipeline mapping and optional HUD display name (Unity-style logical name or upload field).
    ///
    /// When `display_name` normalizes to an embedded `{key}_default` WGSL target, records the stem on
    /// [`MaterialRouter::stem_for_shader_asset`].
    pub fn map_shader_route(
        &mut self,
        shader_asset_id: i32,
        pipeline: RasterPipelineKind,
        display_name: Option<String>,
    ) {
        let stem_from_display = display_name
            .as_deref()
            .and_then(embedded_default_stem_for_unity_name);
        self.router
            .set_shader_route(shader_asset_id, pipeline.clone(), display_name);
        match &pipeline {
            RasterPipelineKind::EmbeddedStem(s) => {
                self.router.set_shader_stem(shader_asset_id, s.to_string());
            }
            RasterPipelineKind::Null => {
                if let Some(s) = stem_from_display {
                    self.router.set_shader_stem(shader_asset_id, s);
                } else {
                    self.router.remove_shader_stem(shader_asset_id);
                }
            }
        }
    }

    /// Inserts a host shader id → pipeline mapping without a HUD display name.
    pub fn map_shader_pipeline(&mut self, shader_asset_id: i32, pipeline: RasterPipelineKind) {
        self.map_shader_route(shader_asset_id, pipeline, None);
    }

    /// Removes routing for a host shader id [`crate::shared::ShaderUnload`].
    pub fn unmap_shader(&mut self, shader_asset_id: i32) {
        self.router.remove_shader_route(shader_asset_id);
    }

    /// Resolves a cached or new pipeline for a host shader asset (via router + embedded stem when applicable).
    pub fn pipeline_for_shader_asset(
        &self,
        shader_asset_id: i32,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
        front_face: RasterFrontFace,
    ) -> Option<MaterialPipelineSet> {
        let kind = resolve_raster_pipeline(shader_asset_id, &self.router);
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: Some(shader_asset_id),
            kind: &kind,
            desc,
            permutation,
            blend_mode,
            render_state,
            front_face,
        })
    }

    /// Looks up a pipeline by explicit kind (for example tests or tools that do not use a host shader id).
    pub fn pipeline_for_kind(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
        front_face: RasterFrontFace,
    ) -> Option<MaterialPipelineSet> {
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: None,
            kind,
            desc,
            permutation,
            blend_mode,
            render_state,
            front_face,
        })
    }

    /// Low-level cache access keyed by [`RasterPipelineKind`].
    pub fn get_or_create_pipeline(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
        front_face: RasterFrontFace,
    ) -> Option<MaterialPipelineSet> {
        self.try_pipeline_with_fallback(PipelineLookupRequest {
            shader_asset_id: None,
            kind,
            desc,
            permutation,
            blend_mode,
            render_state,
            front_face,
        })
    }

    /// Borrow the wgpu device held by this registry.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shader routes for the debug HUD (`shader_asset_id`, [`RasterPipelineKind`], optional display name), sorted.
    pub fn shader_routes_for_hud(&self) -> Vec<(i32, RasterPipelineKind, Option<String>)> {
        self.router.routes_sorted_for_hud()
    }

    /// Resolved composed WGSL stem for a host shader id, when [`Self::map_shader_route`] recorded one.
    pub fn stem_for_shader_asset(&self, shader_asset_id: i32) -> Option<&str> {
        self.router.stem_for_shader_asset(shader_asset_id)
    }
}
