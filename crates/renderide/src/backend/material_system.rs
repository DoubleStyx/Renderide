//! Material property store, shader routing, pipeline registry, and embedded `@group(1)` bind resources.

use hashbrown::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
use crate::materials::host_data::{
    MaterialPropertyStore, ParseMaterialBatchOptions, PropertyIdRegistry,
    parse_materials_update_batch_into_store_with_instance_changed,
};

use super::embedded::{EmbeddedMaterialBindError, EmbeddedMaterialBindResources};
use crate::shared::bit_span::BitSpanMut;
use crate::shared::{MaterialsUpdateBatch, MaterialsUpdateBatchResult, RendererCommand};

/// Max queued [`MaterialsUpdateBatch`] when shared memory is not available.
pub const MAX_PENDING_MATERIAL_BATCHES: usize = 256;

/// Host material tables, GPU registry/cache, embedded bind builder, and deferred shader routes.
pub struct MaterialSystem {
    /// Host material property batches (`MaterialsUpdateBatch`); separate maps for materials vs blocks.
    material_property_store: MaterialPropertyStore,
    /// Stable ids for [`crate::shared::MaterialPropertyIdRequest`] / batch `property_id` keys.
    property_id_registry: Arc<PropertyIdRegistry>,
    /// Batches received before shared memory is ready.
    pending_material_batches: VecDeque<MaterialsUpdateBatch>,
    /// GPU material families, router, and pipeline cache (after GPU attach).
    pub(crate) material_registry: Option<crate::materials::MaterialRegistry>,
    /// Shader asset id -> pipeline kind and optional AssetBundle shader asset name before GPU attach.
    pending_shader_routes: HashMap<i32, (RasterPipelineKind, Option<String>)>,
    /// Embedded raster materials (`@group(1)` textures/uniforms), after GPU attach.
    pub(crate) embedded_material_bind: Option<EmbeddedMaterialBindResources>,
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
            material_registry: None,
            pending_shader_routes: HashMap::new(),
            embedded_material_bind: None,
        }
    }

    /// Creates [`MaterialRegistry`] and [`EmbeddedMaterialBindResources`] after the device is bound.
    ///
    /// Fails if embedded `@group(1)` resources cannot be built; on failure, no GPU material state is left
    /// installed (registry and embedded remain unset).
    pub fn try_attach_gpu(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        limits: Arc<crate::gpu::GpuLimits>,
    ) -> Result<(), EmbeddedMaterialBindError> {
        let embedded = EmbeddedMaterialBindResources::new(
            device.clone(),
            Arc::clone(&self.property_id_registry),
        )?;
        self.material_registry = Some(crate::materials::MaterialRegistry::with_default_families(
            device, limits,
        ));
        if let Some(reg) = self.material_registry.as_mut() {
            for (asset_id, (pipeline, shader_asset_name)) in self.pending_shader_routes.drain() {
                reg.map_shader_route(asset_id, pipeline, shader_asset_name);
            }
        }
        embedded.write_default_white(queue.as_ref());
        self.embedded_material_bind = Some(embedded);
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
        self.material_registry.as_ref()
    }

    /// Mutable registry (pipeline cache and shader routes).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.material_registry.as_mut()
    }

    /// Embedded material bind groups (world Unlit, etc.) after GPU attach.
    pub fn embedded_material_bind(&self) -> Option<&EmbeddedMaterialBindResources> {
        self.embedded_material_bind.as_ref()
    }

    /// Maps shader asset to raster pipeline kind and optional AssetBundle shader asset name, or defers until GPU attach.
    pub fn register_shader_route(
        &mut self,
        asset_id: i32,
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
    ) {
        if let Some(reg) = self.material_registry.as_mut() {
            reg.map_shader_route(asset_id, pipeline, shader_asset_name);
        } else {
            self.pending_shader_routes
                .insert(asset_id, (pipeline, shader_asset_name));
        }
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.pending_shader_routes.remove(&asset_id);
        if let Some(reg) = self.material_registry.as_mut() {
            reg.unmap_shader(asset_id);
        }
    }

    /// Drain pending material batches using the given shared memory and IPC.
    pub fn flush_pending_material_batches(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        profiling::scope!("material::flush_pending_batches");
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
    ///
    /// Writes per-target instance-changed bits into [`MaterialsUpdateBatch::instance_changed_buffer`]
    /// before sending the [`MaterialsUpdateBatchResult`] ack so the host's completion callback reads
    /// fresh values when it dispatches its per-material/per-property-block "asset updated" signal.
    /// Without this, every bit reads as `false`, the host always takes its `AssetUpdated()` branch
    /// instead of `AssetCreated()`/`Reinitialize()`, and property blocks (e.g. font atlases) are
    /// never re-emitted to the renderers using them.
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        profiling::scope!("material::apply_update_batch");
        let update_batch_id = batch.update_batch_id;
        let opts = ParseMaterialBatchOptions {
            render_type_property_id: Some(self.property_id_registry.intern("_RenderType")),
            render_queue_property_id: Some(self.property_id_registry.intern("_RenderQueue")),
            persist_extended_payloads: true,
            ..ParseMaterialBatchOptions::default()
        };

        // Capacity in bits derived from the host-allocated u32 slab; padding bits past the actual
        // material+PB count are harmless (they sit in the trailing element of the bitspan and
        // never get read by `MaterialUpdateData.RunCompleted`).
        let bit_capacity = (batch.instance_changed_buffer.length.max(0) as usize / 4) * 32;
        let mut instance_changed = vec![false; bit_capacity];

        parse_materials_update_batch_into_store_with_instance_changed(
            shm,
            &batch,
            &mut self.material_property_store,
            &opts,
            &mut instance_changed,
        );

        if batch.instance_changed_buffer.length > 0 {
            profiling::scope!("material::write_instance_changed_bits");
            let descriptor = batch.instance_changed_buffer;
            let written = shm.access_mut::<u32, _>(&descriptor, |slab| {
                let mut bits = BitSpanMut::new(slab);
                bits.clear();
                for (i, &flag) in instance_changed.iter().enumerate() {
                    if flag {
                        bits.set(i, true);
                    }
                }
            });
            if !written {
                logger::warn!(
                    "materials update batch {update_batch_id}: failed to write instance_changed_buffer (descriptor offset={} length={})",
                    descriptor.offset,
                    descriptor.length,
                );
            }
        }

        let _ = ipc.send_background(RendererCommand::MaterialsUpdateBatchResult(
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
