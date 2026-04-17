//! Material property store, shader routing, pipeline registry, and embedded `@group(1)` bind resources.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::assets::material::{
    parse_materials_update_batch_into_store, MaterialPropertyStore, ParseMaterialBatchOptions,
    PropertyIdRegistry,
};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;

use super::embedded::{EmbeddedMaterialBindError, EmbeddedMaterialBindResources};
use crate::shared::{MaterialsUpdateBatch, MaterialsUpdateBatchResult, RendererCommand};

/// Max queued [`MaterialsUpdateBatch`] when shared memory is not available.
pub const MAX_PENDING_MATERIAL_BATCHES: usize = 256;

/// GPU material registry and embedded `@group(1)` resources after [`MaterialSystem::try_attach_gpu`].
pub(crate) struct MaterialGpuState {
    /// Material families, router, and pipeline cache.
    pub(crate) registry: crate::materials::MaterialRegistry,
    /// Embedded raster materials (`@group(1)` textures/uniforms).
    pub(crate) embedded: EmbeddedMaterialBindResources,
}

/// Host material tables, GPU registry/cache, embedded bind builder, and deferred shader routes.
pub struct MaterialSystem {
    /// Host material property batches (`MaterialsUpdateBatch`); separate maps for materials vs blocks.
    material_property_store: MaterialPropertyStore,
    /// Stable ids for [`crate::shared::MaterialPropertyIdRequest`] / batch `property_id` keys.
    property_id_registry: Arc<PropertyIdRegistry>,
    /// Batches received before shared memory is available.
    pending_material_batches: VecDeque<MaterialsUpdateBatch>,
    /// GPU registry + embedded binds after a successful [`Self::try_attach_gpu`].
    pub(crate) material_gpu: Option<MaterialGpuState>,
    /// Shader asset id → pipeline kind and optional HUD label when uploads arrive before GPU attach.
    pending_shader_routes: HashMap<i32, (RasterPipelineKind, Option<String>)>,
}

impl Default for MaterialSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialSystem {
    /// Empty store and registry; no GPU resources until [`Self::try_attach_gpu`].
    pub fn new() -> Self {
        Self {
            material_property_store: MaterialPropertyStore::new(),
            property_id_registry: Arc::new(PropertyIdRegistry::new()),
            pending_material_batches: VecDeque::new(),
            material_gpu: None,
            pending_shader_routes: HashMap::new(),
        }
    }

    /// Creates [`MaterialRegistry`] and [`EmbeddedMaterialBindResources`] after the device is bound.
    ///
    /// Returns an error if embedded binds cannot be created; the caller must not treat the backend as attached.
    pub fn try_attach_gpu(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: &Arc<Mutex<wgpu::Queue>>,
    ) -> Result<(), EmbeddedMaterialBindError> {
        let mut registry =
            crate::materials::MaterialRegistry::with_default_families(device.clone());
        for (asset_id, (pipeline, display_name)) in self.pending_shader_routes.drain() {
            registry.map_shader_route(asset_id, pipeline, display_name);
        }
        let embedded =
            EmbeddedMaterialBindResources::new(device, Arc::clone(&self.property_id_registry))?;
        if let Ok(q) = queue.lock() {
            embedded.write_default_white(&q);
        }
        self.material_gpu = Some(MaterialGpuState { registry, embedded });
        Ok(())
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        &self.material_property_store
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(&mut self) -> &mut MaterialPropertyStore {
        &mut self.material_property_store
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &PropertyIdRegistry {
        self.property_id_registry.as_ref()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.material_gpu.as_ref().map(|g| &g.registry)
    }

    /// Mutable registry (pipeline cache and shader routes).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.material_gpu.as_mut().map(|g| &mut g.registry)
    }

    /// Embedded material bind groups (world Unlit, etc.) after GPU attach.
    pub fn embedded_material_bind(&self) -> Option<&EmbeddedMaterialBindResources> {
        self.material_gpu.as_ref().map(|g| &g.embedded)
    }

    /// Maps shader asset to raster pipeline kind and optional HUD display name, or defers until GPU attach.
    pub fn register_shader_route(
        &mut self,
        asset_id: i32,
        pipeline: RasterPipelineKind,
        display_name: Option<String>,
    ) {
        if let Some(g) = self.material_gpu.as_mut() {
            g.registry
                .map_shader_route(asset_id, pipeline, display_name);
        } else {
            self.pending_shader_routes
                .insert(asset_id, (pipeline, display_name));
        }
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.pending_shader_routes.remove(&asset_id);
        if let Some(g) = self.material_gpu.as_mut() {
            g.registry.unmap_shader(asset_id);
        }
    }

    /// Drain pending material batches using the given shared memory and IPC.
    pub fn flush_pending_material_batches(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        let batches: Vec<MaterialsUpdateBatch> = self.pending_material_batches.drain(..).collect();
        for batch in batches {
            self.apply_materials_update_batch(batch, shm, ipc);
        }
    }

    /// Queue a materials batch when shared memory is not yet available. Returns `false` if queue full.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) -> bool {
        if self.pending_material_batches.len() >= MAX_PENDING_MATERIAL_BATCHES {
            logger::warn!(
                "materials update batch {} dropped: pending queue full (no shared memory)",
                batch.update_batch_id
            );
            return false;
        }
        self.pending_material_batches.push_back(batch);
        true
    }

    /// Apply one host materials batch (shared memory must be valid for the batch descriptors).
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        let update_batch_id = batch.update_batch_id;
        let opts = ParseMaterialBatchOptions::default();
        parse_materials_update_batch_into_store(
            shm,
            &batch,
            &mut self.material_property_store,
            &opts,
        );
        ipc.send_background(RendererCommand::MaterialsUpdateBatchResult(
            MaterialsUpdateBatchResult { update_batch_id },
        ));
    }

    /// Remove material / property-block entries from the host store.
    pub fn on_unload_material(&mut self, asset_id: i32) {
        self.material_property_store.remove_material(asset_id);
    }

    /// Remove a property block from the host store.
    pub fn on_unload_material_property_block(&mut self, asset_id: i32) {
        self.material_property_store.remove_property_block(asset_id);
    }
}
