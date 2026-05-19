//! Material property store, shader routing, pipeline registry, and embedded `@group(1)` bind resources.

use hashbrown::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::diagnostics::log_throttle::LogThrottle;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
use crate::materials::host_data::{
    MaterialBatchParseReport, MaterialPropertyStore, ParseMaterialBatchOptions, PropertyIdRegistry,
    parse_materials_update_batch_into_store_with_instance_changed,
};

use crate::materials::PipelinePropertyResolver;
use crate::materials::embedded::{EmbeddedMaterialBindError, EmbeddedMaterialBindResources};
use crate::materials::{
    MaterialPipelineCacheDiagnosticSnapshot, MaterialPipelineDesc, MaterialPipelineVariantSpec,
};
use crate::materials::{MaterialShaderGraphDiagnosticSnapshot, MaterialShaderHotReloadReport};
use crate::shared::bit_span::BitSpanMut;
use crate::shared::{MaterialsUpdateBatch, MaterialsUpdateBatchResult, RendererCommand};

/// Deferred [`MaterialsUpdateBatch`] count that emits queue-pressure diagnostics.
pub const PENDING_MATERIAL_BATCH_WARN_THRESHOLD: usize = 256;

/// Parsed update row count that emits a single debug summary for unusually large batches.
const LARGE_MATERIAL_BATCH_UPDATE_THRESHOLD: usize = 2048;

/// Parsed target count that emits a single debug summary for unusually broad batches.
const LARGE_MATERIAL_BATCH_TARGET_THRESHOLD: usize = 512;

/// Throttle for recoverable material batch parse anomaly logs.
static MATERIAL_BATCH_PARSE_ANOMALY_LOG: LogThrottle = LogThrottle::new();

/// Throttle for unusually large material batch summaries.
static LARGE_MATERIAL_BATCH_LOG: LogThrottle = LogThrottle::new();

/// Snapshot of deferred material work and GPU material-system attachment state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MaterialSystemDiagnosticSnapshot {
    /// Host material batches waiting for shared-memory availability.
    pub(crate) pending_material_batches: usize,
    /// Shader routes captured before GPU material registry attachment.
    pub(crate) pending_shader_routes: usize,
    /// Whether the GPU material registry has been attached.
    pub(crate) material_registry_attached: bool,
    /// Whether embedded material bind resources have been attached.
    pub(crate) embedded_bind_attached: bool,
    /// Shader/material graph diagnostics.
    pub(crate) shader_graph: MaterialShaderGraphDiagnosticSnapshot,
    /// Material pipeline cache diagnostics.
    pub(crate) pipeline_cache: MaterialPipelineCacheDiagnosticSnapshot,
}

/// Host material tables, GPU registry/cache, embedded bind builder, and deferred shader routes.
pub struct MaterialSystem {
    /// Host material property batches (`MaterialsUpdateBatch`); separate maps for materials vs blocks.
    material_property_store: MaterialPropertyStore,
    /// Stable ids for [`crate::shared::MaterialPropertyIdRequest`] / batch `property_id` keys.
    property_id_registry: Arc<PropertyIdRegistry>,
    /// Cached `MaterialPipelinePropertyIds` over `property_id_registry`.
    pipeline_property_resolver: PipelinePropertyResolver,
    /// Batches received before shared memory is ready.
    pending_material_batches: VecDeque<MaterialsUpdateBatch>,
    /// GPU material families, router, and pipeline cache (after GPU attach).
    pub(crate) material_registry: Option<crate::materials::MaterialRegistry>,
    /// Shader asset id -> pipeline kind and optional AssetBundle shader asset name before GPU attach.
    pending_shader_routes: HashMap<i32, (RasterPipelineKind, Option<String>, Option<u32>)>,
    /// Embedded raster materials (`@group(1)` textures/uniforms), after GPU attach.
    pub(crate) embedded_material_bind: Option<EmbeddedMaterialBindResources>,
    /// Reusable scratch for `MaterialUpdateData.RunCompleted`'s `instance_changed` bit slab.
    ///
    /// Sized once per batch via `clear` + `resize` to retain capacity across calls; replaces a
    /// per-batch `vec![false; bit_capacity]` allocation in
    /// [`Self::apply_materials_update_batch`].
    instance_changed_scratch: Vec<bool>,
}

impl Default for MaterialSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialSystem {
    /// Empty store and registry; no GPU resources until [`Self::try_attach_gpu`].
    pub fn new() -> Self {
        let property_id_registry = Arc::new(PropertyIdRegistry::new());
        let pipeline_property_resolver =
            PipelinePropertyResolver::new(Arc::clone(&property_id_registry));
        Self {
            material_property_store: MaterialPropertyStore::new(),
            property_id_registry,
            pipeline_property_resolver,
            pending_material_batches: VecDeque::new(),
            material_registry: None,
            pending_shader_routes: HashMap::new(),
            embedded_material_bind: None,
            instance_changed_scratch: Vec::new(),
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
            limits.clone(),
        )?;
        self.material_registry = Some(crate::materials::MaterialRegistry::with_default_families(
            device, limits,
        ));
        if let Some(reg) = self.material_registry.as_mut() {
            for (asset_id, (pipeline, shader_asset_name, shader_variant_bits)) in
                self.pending_shader_routes.drain()
            {
                reg.map_shader_route(asset_id, pipeline, shader_asset_name, shader_variant_bits);
            }
        }
        embedded.write_default_textures(queue.as_ref());
        self.embedded_material_bind = Some(embedded);
        Ok(())
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        &self.material_property_store
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &PropertyIdRegistry {
        self.property_id_registry.as_ref()
    }

    /// Cached resolver for [`crate::materials::MaterialPipelinePropertyIds`] over the same
    /// registry. Hot paths (`frame_packet`, draw collection) clone this once instead of
    /// re-interning ~14 property names per frame.
    pub fn pipeline_property_resolver(&self) -> &PipelinePropertyResolver {
        &self.pipeline_property_resolver
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.material_registry.as_ref()
    }

    /// Drains finished background pipeline builds into the cache (no-op before GPU attach).
    ///
    /// Recording threads call into the cache concurrently; pulling completions here keeps the
    /// per-draw lookup from touching the completion channel or pending/failed mutexes.
    pub fn drain_pipeline_build_completions(&self) {
        if let Some(reg) = self.material_registry.as_ref() {
            reg.drain_pipeline_build_completions();
        }
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
        shader_variant_bits: Option<u32>,
    ) {
        if let Some(reg) = self.material_registry.as_mut() {
            reg.map_shader_route(asset_id, pipeline, shader_asset_name, shader_variant_bits);
        } else {
            self.pending_shader_routes
                .insert(asset_id, (pipeline, shader_asset_name, shader_variant_bits));
        }
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.pending_shader_routes.remove(&asset_id);
        if let Some(reg) = self.material_registry.as_mut() {
            reg.unmap_shader(asset_id);
        }
    }

    /// Queue a materials batch when shared memory is not yet available.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) {
        logger::trace!(
            "materials update batch {} deferred: pending_no_shm_queue={}",
            batch.update_batch_id,
            self.pending_material_batches.len() + 1,
        );
        self.pending_material_batches.push_back(batch);
        self.log_pending_material_batch_pressure();
    }

    fn log_pending_material_batch_pressure(&self) {
        let pending = self.pending_material_batches.len();
        if pending == PENDING_MATERIAL_BATCH_WARN_THRESHOLD
            || (pending > PENDING_MATERIAL_BATCH_WARN_THRESHOLD
                && pending.is_multiple_of(PENDING_MATERIAL_BATCH_WARN_THRESHOLD))
        {
            logger::warn!(
                "materials: deferred update batch backlog high: pending={} threshold={} reason=shared memory unavailable",
                pending,
                PENDING_MATERIAL_BATCH_WARN_THRESHOLD
            );
        }
    }

    /// Whether any material batches are waiting for shared memory.
    pub fn has_pending_material_batches(&self) -> bool {
        !self.pending_material_batches.is_empty()
    }

    /// Returns a compact snapshot for lifecycle diagnostics.
    pub(crate) fn diagnostic_snapshot(&self) -> MaterialSystemDiagnosticSnapshot {
        MaterialSystemDiagnosticSnapshot {
            pending_material_batches: self.pending_material_batches.len(),
            pending_shader_routes: self.pending_shader_routes.len(),
            material_registry_attached: self.material_registry.is_some(),
            embedded_bind_attached: self.embedded_material_bind.is_some(),
            shader_graph: self
                .material_registry
                .as_ref()
                .map_or_else(MaterialShaderGraphDiagnosticSnapshot::default, |registry| {
                    registry.shader_graph_diagnostics()
                }),
            pipeline_cache: self.material_registry.as_ref().map_or_else(
                MaterialPipelineCacheDiagnosticSnapshot::default,
                |registry| registry.pipeline_cache_diagnostics(),
            ),
        }
    }

    /// Queues a material pipeline warmup for a prepared draw batch.
    pub(crate) fn queue_material_pipeline_warmup(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) {
        if let Some(registry) = self.material_registry.as_ref() {
            registry.queue_pipeline_warmup(kind, desc, variant);
        }
    }

    /// Pre-warms a reflected embedded material layout when group-1 resources are attached.
    pub(crate) fn pre_warm_embedded_material_layout(&self, stem: &str) {
        let Some(embedded) = self.embedded_material_bind.as_ref() else {
            return;
        };
        if let Err(error) = embedded.embedded_material_bind_group_layout(stem) {
            logger::trace!("materials: embedded layout pre-warm failed for {stem}: {error}");
        }
    }

    /// Enables or disables development WGSL hot reload.
    pub(crate) fn set_dev_shader_hot_reload_enabled(&mut self, enabled: bool) {
        if let Some(registry) = self.material_registry.as_mut() {
            registry.set_dev_shader_hot_reload_enabled(enabled);
        }
    }

    /// Polls development WGSL hot reload and invalidates affected material-side caches.
    pub(crate) fn poll_dev_shader_hot_reload(&mut self) -> MaterialShaderHotReloadReport {
        let Some(registry) = self.material_registry.as_mut() else {
            return MaterialShaderHotReloadReport::default();
        };
        let report = registry.poll_dev_shader_hot_reload();
        if report.is_empty() {
            return report;
        }
        if let Some(embedded) = self.embedded_material_bind.as_ref() {
            for stem in &report.reloaded_stems {
                embedded.invalidate_stem_layout(stem);
            }
        }
        report
    }

    /// Moves material batches that were waiting for shared memory out for cooperative integration.
    pub fn take_pending_material_batches(&mut self) -> Vec<MaterialsUpdateBatch> {
        self.pending_material_batches.drain(..).collect()
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
        self.apply_materials_update_batch_inner(batch, shm, Some(ipc));
    }

    /// Apply one host materials batch without sending an IPC acknowledgement.
    pub fn apply_materials_update_batch_no_ack(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
    ) {
        self.apply_materials_update_batch_inner(batch, shm, None);
    }

    fn apply_materials_update_batch_inner(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
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
        // Pool the bit slab across batches: a fresh `vec![false; bit_capacity]` per batch
        // allocated even when zero materials changed; clear+resize retains capacity instead.
        let instance_changed = &mut self.instance_changed_scratch;
        instance_changed.clear();
        instance_changed.resize(bit_capacity, false);

        let parse_report: MaterialBatchParseReport =
            parse_materials_update_batch_into_store_with_instance_changed(
                shm,
                &batch,
                &mut self.material_property_store,
                &opts,
                instance_changed,
            );
        logger::trace!(
            "materials update batch {update_batch_id}: material_updates={} material_count={} int_buffers={} float_buffers={} float4_buffers={} matrix_buffers={} instance_changed_bits={}",
            batch.material_updates.len(),
            batch.material_update_count,
            batch.int_buffers.len(),
            batch.float_buffers.len(),
            batch.float4_buffers.len(),
            batch.matrix_buffers.len(),
            instance_changed.iter().filter(|&&flag| flag).count(),
        );
        if parse_report.has_anomaly() {
            if let Some(occurrence) = MATERIAL_BATCH_PARSE_ANOMALY_LOG.should_log(8, 128) {
                logger::warn!(
                    "materials update batch {update_batch_id}: parse anomalies update_end_seen={} missing_payload_reads={} missing_ints={} missing_floats={} missing_float4s={} missing_matrices={} updates_read={} select_targets={} instance_changed_bits={}/{} occurrence={}",
                    parse_report.update_batch_end_seen,
                    parse_report.missing_payload_reads(),
                    parse_report.missing_ints,
                    parse_report.missing_floats,
                    parse_report.missing_float4s,
                    parse_report.missing_matrices,
                    parse_report.updates_read,
                    parse_report.select_targets,
                    parse_report.instance_changed_set_bits,
                    parse_report.instance_changed_capacity_bits,
                    occurrence,
                );
            }
        } else if material_batch_is_large(parse_report.updates_read, parse_report.select_targets)
            && logger::enabled(logger::LogLevel::Debug)
            && let Some(occurrence) = LARGE_MATERIAL_BATCH_LOG.should_log(4, 64)
        {
            logger::debug!(
                "materials update batch {update_batch_id}: large batch parsed updates_read={} select_targets={} ints={} floats={} float4s={} matrices={} instance_changed_bits={}/{} occurrence={}",
                parse_report.updates_read,
                parse_report.select_targets,
                parse_report.ints_read,
                parse_report.floats_read,
                parse_report.float4s_read,
                parse_report.matrices_read,
                parse_report.instance_changed_set_bits,
                parse_report.instance_changed_capacity_bits,
                occurrence,
            );
        }

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

        if let Some(ipc) = ipc {
            let ack_queued =
                ipc.send_background_reliable(RendererCommand::MaterialsUpdateBatchResult(
                    MaterialsUpdateBatchResult { update_batch_id },
                ));
            if !ack_queued {
                logger::warn!(
                    "materials update batch {update_batch_id}: failed to enqueue reliable background ack"
                );
            }
        }
    }

    /// Remove material / property-block entries from the host store.
    pub fn on_unload_material(&mut self, asset_id: i32) {
        if let Some(embedded) = self.embedded_material_bind.as_ref() {
            embedded.purge_material_asset(asset_id);
        }
        self.material_property_store.remove_material(asset_id);
    }

    /// Remove a property block from the host store.
    pub fn on_unload_material_property_block(&mut self, asset_id: i32) {
        if let Some(embedded) = self.embedded_material_bind.as_ref() {
            embedded.purge_property_block_asset(asset_id);
        }
        self.material_property_store.remove_property_block(asset_id);
    }

    /// Drops embedded bind groups that may retain texture views for unloaded texture assets.
    pub(crate) fn purge_texture_reference_caches(&self) {
        if let Some(embedded) = self.embedded_material_bind.as_ref() {
            embedded.purge_texture_reference_caches();
        }
    }
}

/// Returns whether a parsed materials batch is large enough for summary diagnostics.
fn material_batch_is_large(updates_read: usize, select_targets: usize) -> bool {
    updates_read >= LARGE_MATERIAL_BATCH_UPDATE_THRESHOLD
        || select_targets >= LARGE_MATERIAL_BATCH_TARGET_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::{
        LARGE_MATERIAL_BATCH_TARGET_THRESHOLD, LARGE_MATERIAL_BATCH_UPDATE_THRESHOLD,
        material_batch_is_large,
    };

    #[test]
    fn material_batch_large_thresholds_are_inclusive() {
        assert!(!material_batch_is_large(
            LARGE_MATERIAL_BATCH_UPDATE_THRESHOLD - 1,
            LARGE_MATERIAL_BATCH_TARGET_THRESHOLD - 1,
        ));
        assert!(material_batch_is_large(
            LARGE_MATERIAL_BATCH_UPDATE_THRESHOLD,
            0,
        ));
        assert!(material_batch_is_large(
            0,
            LARGE_MATERIAL_BATCH_TARGET_THRESHOLD,
        ));
    }
}
