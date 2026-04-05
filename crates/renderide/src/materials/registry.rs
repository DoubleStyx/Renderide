//! Registered [`MaterialPipelineFamily`] implementations and shared [`super::MaterialPipelineCache`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::pipelines::ShaderPermutation;

use super::builtin_solid::SolidColorFamily;
use super::cache::MaterialPipelineCache;
use super::family::{MaterialFamilyId, MaterialPipelineDesc, MaterialPipelineFamily};
use super::router::MaterialRouter;

/// Owning table of material families, routing, and pipeline cache.
pub struct MaterialRegistry {
    device: Arc<wgpu::Device>,
    families: HashMap<MaterialFamilyId, Arc<dyn MaterialPipelineFamily>>,
    pub router: MaterialRouter,
    cache: MaterialPipelineCache,
}

impl MaterialRegistry {
    /// Registers builtin families and routes all unknown shader assets to [`SolidColorFamily`].
    pub fn with_default_families(device: Arc<wgpu::Device>) -> Self {
        let mut registry = Self {
            device: device.clone(),
            families: HashMap::new(),
            router: MaterialRouter::new(super::builtin_solid::SOLID_COLOR_FAMILY_ID),
            cache: MaterialPipelineCache::new(device),
        };
        registry.register_family(Arc::new(SolidColorFamily));
        registry
    }

    /// Adds a family (replaces if `family_id` matches an existing entry).
    pub fn register_family(&mut self, family: Arc<dyn MaterialPipelineFamily>) {
        self.families.insert(family.family_id(), family);
    }

    /// Inserts a host shader id → family mapping (for when shader names are known).
    pub fn map_shader_to_family(&mut self, shader_asset_id: i32, family: MaterialFamilyId) {
        self.router.set_shader_family(shader_asset_id, family);
    }

    /// Removes routing for a host shader id [`crate::shared::ShaderUnload`].
    pub fn unmap_shader(&mut self, shader_asset_id: i32) {
        self.router.remove_shader_family(shader_asset_id);
    }

    /// Resolves a pipeline for a host shader asset (via static or default router).
    pub fn pipeline_for_shader_asset(
        &mut self,
        shader_asset_id: i32,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
    ) -> Option<&wgpu::RenderPipeline> {
        let id = self.router.family_for_shader_asset(shader_asset_id);
        self.pipeline_for_family(id, desc, permutation)
    }

    /// Looks up `family_id` and returns a cached or new pipeline.
    pub fn pipeline_for_family(
        &mut self,
        family_id: MaterialFamilyId,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
    ) -> Option<&wgpu::RenderPipeline> {
        let family = self.families.get(&family_id)?.clone();
        Some(self.cache.get_or_create(family.as_ref(), desc, permutation))
    }

    /// Low-level cache access (family object instead of id).
    pub fn get_or_create_pipeline(
        &mut self,
        family: &dyn MaterialPipelineFamily,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
    ) -> &wgpu::RenderPipeline {
        self.cache.get_or_create(family, desc, permutation)
    }

    /// Borrow the wgpu device held by this registry.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }
}

#[cfg(test)]
mod wgpu_cache_tests {
    use std::sync::Arc;

    use super::MaterialRegistry;
    use crate::materials::family::MaterialPipelineDesc;
    use crate::materials::SOLID_COLOR_FAMILY_ID;
    use crate::pipelines::ShaderPermutation;

    async fn device_with_adapter() -> Option<Arc<wgpu::Device>> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
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

    /// Real device; run `cargo test -p renderide wgpu_cache -- --ignored` locally.
    #[test]
    #[ignore = "wgpu/GPU stack (may SIGSEGV in sandbox CI); run with --ignored"]
    fn solid_color_pipeline_cache_hits() {
        let Some(device) = pollster::block_on(device_with_adapter()) else {
            eprintln!("skipping solid_color_pipeline_cache_hits: no wgpu adapter");
            return;
        };
        let mut reg = MaterialRegistry::with_default_families(device);
        let desc = MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_stencil_format: None,
            sample_count: 1,
        };
        let addr = {
            let p = reg
                .pipeline_for_family(SOLID_COLOR_FAMILY_ID, &desc, ShaderPermutation(0))
                .expect("builtin family");
            std::ptr::from_ref(p)
        };
        let addr2 = {
            let p = reg
                .pipeline_for_family(SOLID_COLOR_FAMILY_ID, &desc, ShaderPermutation(0))
                .expect("cache hit");
            std::ptr::from_ref(p)
        };
        assert_eq!(addr, addr2);
    }

    #[test]
    #[ignore = "wgpu/GPU stack (may SIGSEGV in sandbox CI); run with --ignored"]
    fn permutation_bit_changes_pipeline() {
        let Some(device) = pollster::block_on(device_with_adapter()) else {
            eprintln!("skipping permutation_bit_changes_pipeline: no wgpu adapter");
            return;
        };
        let mut reg = MaterialRegistry::with_default_families(device);
        let desc = MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_stencil_format: None,
            sample_count: 1,
        };
        let addr0 = {
            let p = reg
                .pipeline_for_family(SOLID_COLOR_FAMILY_ID, &desc, ShaderPermutation(0))
                .expect("perm 0");
            std::ptr::from_ref(p)
        };
        let addr1 = {
            let p = reg
                .pipeline_for_family(SOLID_COLOR_FAMILY_ID, &desc, ShaderPermutation(1))
                .expect("perm 1");
            std::ptr::from_ref(p)
        };
        assert_ne!(addr0, addr1);
    }
}
