//! [`MaterialRegistry`]: [`MaterialRouter`], [`super::MaterialPipelineCache`], and shader route updates.

use std::sync::Arc;

use crate::pipelines::ShaderPermutation;

use super::cache::{MaterialPipelineCache, MaterialPipelineSet};
use super::embedded_shader_stem::embedded_default_stem_for_unity_name;
use super::family::MaterialPipelineDesc;
use super::material_passes::MaterialBlendMode;
use super::pipeline_kind::RasterPipelineKind;
use super::render_state::MaterialRenderState;
use super::resolve_raster::resolve_raster_pipeline;
use super::router::MaterialRouter;

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
        shader_asset_id: Option<i32>,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
    ) -> Option<MaterialPipelineSet> {
        let err = match self
            .cache
            .get_or_create(kind, desc, permutation, blend_mode, render_state)
        {
            Ok(p) => return Some(p),
            Err(e) => e,
        };
        if matches!(kind, RasterPipelineKind::DebugWorldNormals) {
            match shader_asset_id {
                Some(id) => {
                    logger::error!(
                        "DebugWorldNormals pipeline build failed (shader_asset_id={id}): {err}"
                    );
                }
                None => {
                    logger::error!("DebugWorldNormals pipeline build failed: {err}");
                }
            }
            return None;
        }
        match shader_asset_id {
            Some(id) => {
                logger::warn!(
                    "material pipeline build failed (shader_asset_id={id}, kind={kind:?}): {err}; falling back to DebugWorldNormals"
                );
            }
            None => {
                logger::warn!(
                    "material pipeline build failed (kind={kind:?}): {err}; falling back to DebugWorldNormals"
                );
            }
        }
        let fallback = RasterPipelineKind::DebugWorldNormals;
        match self
            .cache
            .get_or_create(&fallback, desc, permutation, blend_mode, render_state)
        {
            Ok(p) => Some(p),
            Err(e2) => {
                logger::error!("fallback DebugWorldNormals pipeline build failed: {e2}");
                None
            }
        }
    }

    /// Builds a registry whose router falls back to [`RasterPipelineKind::DebugWorldNormals`] for unknown shader assets.
    pub fn with_default_families(device: Arc<wgpu::Device>) -> Self {
        Self {
            device: device.clone(),
            router: MaterialRouter::new(RasterPipelineKind::DebugWorldNormals),
            cache: MaterialPipelineCache::new(device),
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
            RasterPipelineKind::DebugWorldNormals => {
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
    ) -> Option<MaterialPipelineSet> {
        let kind = resolve_raster_pipeline(shader_asset_id, &self.router);
        self.try_pipeline_with_fallback(
            Some(shader_asset_id),
            &kind,
            desc,
            permutation,
            blend_mode,
            render_state,
        )
    }

    /// Looks up a pipeline by explicit kind (for example tests or tools that do not use a host shader id).
    pub fn pipeline_for_kind(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
    ) -> Option<MaterialPipelineSet> {
        self.try_pipeline_with_fallback(None, kind, desc, permutation, blend_mode, render_state)
    }

    /// Low-level cache access keyed by [`RasterPipelineKind`].
    pub fn get_or_create_pipeline(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
        blend_mode: MaterialBlendMode,
        render_state: MaterialRenderState,
    ) -> Option<MaterialPipelineSet> {
        self.try_pipeline_with_fallback(None, kind, desc, permutation, blend_mode, render_state)
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

#[cfg(test)]
mod wgpu_cache_tests {
    use std::sync::Arc;

    use super::MaterialRegistry;
    use crate::materials::family::MaterialPipelineDesc;
    use crate::materials::{MaterialBlendMode, MaterialRenderState, RasterPipelineKind};
    use crate::pipelines::ShaderPermutation;

    async fn device_with_adapter() -> Option<Arc<wgpu::Device>> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        instance_desc.flags = wgpu::InstanceFlags::empty();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        let (device, _) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("material_registry_test"),
                required_features: wgpu::Features::empty(),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Arc::new(device))
    }

    fn ci_expects_wgpu_adapter() -> bool {
        matches!(std::env::var("CI").as_deref(), Ok("true" | "1"))
    }

    /// Headless wgpu smoke: cache returns the same pipeline pointer for identical keys.
    ///
    /// On CI (`CI=true` / `CI=1`), missing adapters **fail** the test so runners without Vulkan
    /// (e.g. Lavapipe) are caught. Locally, missing adapters log a warning and skip.
    #[test]
    fn debug_world_normals_pipeline_cache_hits() {
        let Some(device) = pollster::block_on(device_with_adapter()) else {
            assert!(
                !ci_expects_wgpu_adapter(),
                "wgpu adapter required when CI is set (install Vulkan / Mesa or Lavapipe on Linux)"
            );
            logger::warn!("skipping debug_world_normals_pipeline_cache_hits: no wgpu adapter");
            return;
        };
        let reg = MaterialRegistry::with_default_families(device);
        let desc = MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_stencil_format: None,
            sample_count: 1,
            multiview_mask: None,
        };
        let addr = {
            let p = reg
                .pipeline_for_kind(
                    &RasterPipelineKind::DebugWorldNormals,
                    &desc,
                    ShaderPermutation(0),
                    MaterialBlendMode::StemDefault,
                    MaterialRenderState::default(),
                )
                .expect("pipeline");
            std::ptr::from_ref(&p[0])
        };
        let addr2 = {
            let p = reg
                .pipeline_for_kind(
                    &RasterPipelineKind::DebugWorldNormals,
                    &desc,
                    ShaderPermutation(0),
                    MaterialBlendMode::StemDefault,
                    MaterialRenderState::default(),
                )
                .expect("pipeline");
            std::ptr::from_ref(&p[0])
        };
        assert_eq!(addr, addr2);
    }

    #[test]
    #[ignore = "wgpu/GPU stack (may SIGSEGV in sandbox CI); run with --ignored"]
    fn permutation_bit_changes_pipeline() {
        let Some(device) = pollster::block_on(device_with_adapter()) else {
            logger::warn!("skipping permutation_bit_changes_pipeline: no wgpu adapter");
            return;
        };
        let reg = MaterialRegistry::with_default_families(device);
        let desc = MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_stencil_format: None,
            sample_count: 1,
            multiview_mask: None,
        };
        let addr0 = {
            let p = reg
                .pipeline_for_kind(
                    &RasterPipelineKind::DebugWorldNormals,
                    &desc,
                    ShaderPermutation(0),
                    MaterialBlendMode::StemDefault,
                    MaterialRenderState::default(),
                )
                .expect("pipeline");
            std::ptr::from_ref(&p[0])
        };
        let addr1 = {
            let p = reg
                .pipeline_for_kind(
                    &RasterPipelineKind::DebugWorldNormals,
                    &desc,
                    ShaderPermutation(1),
                    MaterialBlendMode::StemDefault,
                    MaterialRenderState::default(),
                )
                .expect("pipeline");
            std::ptr::from_ref(&p[0])
        };
        assert_ne!(addr0, addr1);
    }
}
