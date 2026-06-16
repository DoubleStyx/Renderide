//! Plain-data backend snapshot consumed by [`crate::diagnostics::FrameDiagnosticsSnapshot`] and
//! [`crate::diagnostics::RendererInfoSnapshot`].
//!
//! Captured by [`crate::backend::RenderBackend::snapshot_for_diagnostics`] before the diagnostics
//! HUD layer runs, so `diagnostics/` never borrows `&RenderBackend` directly.

use crate::frame_upload_batch::FrameUploadBatchStats;
use crate::hud_contract::WorldMeshViewHudStats;
use crate::materials::{
    EmbeddedMaterialBindCacheDiagnosticSnapshot, MaterialPipelineCacheDiagnosticSnapshot,
    MaterialShaderGraphDiagnosticSnapshot, RasterPipelineKind,
};
use crate::passes::WorldMeshForwardInstancePlanCacheStats;
use crate::world_mesh::{
    RenderWorldMaintenanceStats, WorldMeshCommandCacheStats, WorldMeshDrawStateRow,
    WorldMeshDrawStats,
};

/// Asset streaming, worker, and deferred-work diagnostics for the HUD.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssetDiagnosticsSnapshot {
    /// Renderer-main-thread integration tasks waiting to run.
    pub main_queued: usize,
    /// Urgent upload integration tasks waiting to run.
    pub high_priority_queued: usize,
    /// Render-lane integration tasks waiting to run.
    pub render_queued: usize,
    /// Standard-priority upload integration tasks waiting to run.
    pub normal_priority_queued: usize,
    /// Dynamic-buffer and particle integration tasks waiting to run.
    pub particle_queued: usize,
    /// Total queued integration tasks.
    pub total_queued: usize,
    /// Highest total queued integration depth observed since startup.
    pub peak_queued: usize,
    /// Mesh uploads deferred on prerequisites.
    pub pending_mesh_uploads: usize,
    /// Texture2D uploads deferred on prerequisites.
    pub pending_texture_uploads: usize,
    /// Texture3D uploads deferred on prerequisites.
    pub pending_texture3d_uploads: usize,
    /// Cubemap uploads deferred on prerequisites.
    pub pending_cubemap_uploads: usize,
    /// Video texture loads deferred on prerequisites.
    pub pending_video_texture_loads: usize,
    /// Asset-worker jobs waiting in the bounded queue.
    pub worker_queued: usize,
    /// Asset-worker jobs currently executing.
    pub worker_running: usize,
    /// Highest observed asset-worker queue depth.
    pub worker_max_queued: usize,
    /// Asset-worker jobs accepted by the dispatch path.
    pub worker_spawned: u64,
    /// Asset-worker jobs completed on worker threads.
    pub worker_completed: u64,
    /// Asset-worker jobs executed inline.
    pub worker_inline_executed: u64,
    /// Asset-worker queue saturation events.
    pub worker_saturated: u64,
    /// Material update batches deferred on shared-memory availability.
    pub pending_material_batches: usize,
    /// Shader routes captured before GPU registry attachment.
    pub pending_shader_routes: usize,
    /// Whether the GPU material registry is attached.
    pub material_registry_attached: bool,
    /// Whether embedded material bind resources are attached.
    pub embedded_bind_attached: bool,
}

/// Light packing and influence-volume culling diagnostics for the HUD.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightDiagnosticsSnapshot {
    /// Packed lights in the default frame-light buffer.
    pub packed_default_lights: usize,
    /// Retained per-view light-pack count.
    pub per_view_light_packs: usize,
    /// Largest retained per-view light-pack length.
    pub max_per_view_lights: usize,
    /// Whether signed scene color is active for negative direct lights.
    pub signed_scene_color_active: bool,
    /// Render spaces visited while resolving view light packs.
    pub visibility_space_count: usize,
    /// Render spaces prepared without a culling descriptor.
    pub visibility_cull_disabled_spaces: usize,
    /// Resolved lights before contribution and culling filters.
    pub visibility_lights_before_cull: usize,
    /// Resolved lights discarded because they cannot contribute visible direct lighting.
    pub visibility_non_contributing_lights: usize,
    /// Light influence volumes with finite bounds tested against active views.
    pub visibility_indexed_lights: usize,
    /// Lights kept conservatively because influence bounds were unavailable.
    pub visibility_fallback_lights: usize,
    /// Bounded light influence volumes rejected before clustered-light packing.
    pub visibility_rejected_lights: usize,
    /// Lights kept after contribution and frustum filters, before `MAX_LIGHTS` truncation.
    pub visibility_lights_after_cull: usize,
    /// Lights retained in packed GPU light arrays after `MAX_LIGHTS` truncation.
    pub visibility_packed_lights: usize,
    /// Lights kept by culling but dropped because the GPU light buffer reached `MAX_LIGHTS`.
    pub visibility_max_lights_culled: usize,
    /// Space-level light BVH traversals used during the latest light prep.
    pub visibility_bvh_queries: usize,
    /// Space-level linear light scans used during the latest light prep.
    pub visibility_linear_queries: usize,
    /// Per-light AABB frustum tests executed by linear runs or BVH leaves.
    pub visibility_light_aabb_tests: usize,
    /// BVH node AABB frustum tests executed before leaf light tests.
    pub visibility_bvh_node_tests: usize,
    /// BVH nodes rejected as a group before testing their contained lights.
    pub visibility_bvh_nodes_culled: usize,
}

/// One host-shader -> renderer-pipeline routing row captured for the **Shader routes** HUD tab.
#[derive(Clone, Debug)]
pub struct ShaderRouteSnapshot {
    /// Host-assigned shader asset id.
    pub shader_asset_id: i32,
    /// Resolved renderer pipeline kind for the route.
    pub pipeline: RasterPipelineKind,
    /// Shader asset filename if known.
    pub shader_asset_name: Option<String>,
    /// Froox shader variant bitmask parsed from the serialized Shader name suffix.
    pub shader_variant_bits: Option<u32>,
}

/// Persistent upload arena pressure and fallback counters from the most recent graph submit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameUploadArenaSnapshot {
    /// Deferred buffer writes drained by the latest frame upload batch.
    pub writes: usize,
    /// Deferred upload payload bytes drained by the latest frame upload batch.
    pub bytes: usize,
    /// Writes replayed through [`wgpu::Queue::write_buffer`] because staging was not usable.
    pub fallback_writes: usize,
    /// Bytes staged through persistent upload arena slots.
    pub persistent_staging_bytes: u64,
    /// Persistent upload arena slot reuse count.
    pub persistent_slot_reuses: usize,
    /// Persistent upload arena allocation or growth count.
    pub persistent_slot_grows: usize,
    /// Bytes staged through one-frame temporary fallback buffers.
    pub temporary_staging_bytes: u64,
    /// Temporary fallback buffer count caused by unavailable persistent slots.
    pub temporary_staging_fallbacks: usize,
    /// Staged writes replayed through queue writes because no staging buffer fit.
    pub oversized_queue_fallback_writes: usize,
    /// Total bytes allocated across persistent upload arena slots.
    pub arena_capacity_bytes: u64,
    /// Persistent upload arena slots currently mapped and free.
    pub arena_free_slots: usize,
    /// Persistent upload arena slots referenced by submitted GPU work.
    pub arena_in_flight_slots: usize,
    /// Persistent upload arena slots waiting for `map_async` completion.
    pub arena_remapping_slots: usize,
}

impl From<FrameUploadBatchStats> for FrameUploadArenaSnapshot {
    fn from(stats: FrameUploadBatchStats) -> Self {
        Self {
            writes: stats.writes,
            bytes: stats.bytes,
            fallback_writes: stats.fallback_writes,
            persistent_staging_bytes: stats.persistent_staging_bytes,
            persistent_slot_reuses: stats.persistent_slot_reuses,
            persistent_slot_grows: stats.persistent_slot_grows,
            temporary_staging_bytes: stats.temporary_staging_bytes,
            temporary_staging_fallbacks: stats.temporary_staging_fallbacks,
            oversized_queue_fallback_writes: stats.oversized_queue_fallback_writes,
            arena_capacity_bytes: stats.arena_capacity_bytes,
            arena_free_slots: stats.arena_free_slots,
            arena_in_flight_slots: stats.arena_in_flight_slots,
            arena_remapping_slots: stats.arena_remapping_slots,
        }
    }
}

/// Plain-data view of `RenderBackend` fields the diagnostics HUD reads.
///
/// This breaks the diagnostics-to-backend borrow: `diagnostics/` consumes this snapshot rather
/// than borrowing `&RenderBackend` directly, which keeps backend internals private and lets the
/// HUD layer evolve independently of backend visibility.
#[derive(Clone, Debug, Default)]
pub struct BackendDiagSnapshot {
    /// CPU-side host texture format registrations.
    pub texture_format_registration_count: usize,
    /// CPU-side host textures with mip 0 ready.
    pub texture_mip0_ready_count: usize,
    /// Resident GPU textures in the renderer's [`crate::gpu_pools::TexturePool`].
    pub texture_pool_resident_count: usize,
    /// Resident host render textures in the renderer's [`crate::gpu_pools::RenderTexturePool`].
    pub render_texture_pool_len: usize,
    /// Resident mesh entries in the renderer's [`crate::gpu_pools::MeshPool`].
    pub mesh_pool_entry_count: usize,
    /// Host-shader -> pipeline routing rows for the **Shader routes** HUD tab.
    pub shader_routes: Vec<ShaderRouteSnapshot>,
    /// Latest world-mesh draw stats published by the previous frame.
    pub last_world_mesh_draw_stats: WorldMeshDrawStats,
    /// Latest per-view world-mesh draw stats published by the previous frame.
    pub last_world_mesh_view_stats: Vec<WorldMeshViewHudStats>,
    /// Latest world-mesh draw-state rows published by the previous frame.
    pub last_world_mesh_draw_state_rows: Vec<WorldMeshDrawStateRow>,
    /// Retained render-world maintenance counters from the latest backend extraction.
    pub render_world_maintenance: RenderWorldMaintenanceStats,
    /// Retained arranged draw command-list cache counters.
    pub world_mesh_command_cache: WorldMeshCommandCacheStats,
    /// Retained world-mesh forward instance-plan cache counters.
    pub world_mesh_instance_plan_cache: WorldMeshForwardInstancePlanCacheStats,
    /// Allocated material property uniform slots.
    pub material_property_slots: usize,
    /// Allocated material property block slots.
    pub property_block_slots: usize,
    /// Distinct shader binding sets registered for materials.
    pub material_shader_bindings: usize,
    /// Shader/material graph diagnostics.
    pub material_shader_graph: MaterialShaderGraphDiagnosticSnapshot,
    /// Material pipeline cache diagnostics.
    pub material_pipeline_cache: MaterialPipelineCacheDiagnosticSnapshot,
    /// Embedded material bind-group cache diagnostics.
    pub embedded_material_bind_cache: EmbeddedMaterialBindCacheDiagnosticSnapshot,
    /// Pass count in the compiled main render graph.
    pub frame_graph_pass_count: usize,
    /// Pass count before compile-time render graph culling.
    pub frame_graph_registered_pass_count: usize,
    /// Kahn-style DAG wave count from the compiled main render graph.
    pub frame_graph_topo_levels: usize,
    /// Passes culled because no retained consumer or import needed them.
    pub frame_graph_culled_pass_count: usize,
    /// Passes intentionally omitted before graph construction.
    pub frame_graph_compile_skipped_pass_count: usize,
    /// Attachment resolve declarations retained by the graph.
    pub frame_graph_attachment_resolve_count: usize,
    /// Retained transient attachment stores.
    pub frame_graph_transient_store_count: usize,
    /// Retained transient attachment discards.
    pub frame_graph_transient_discard_count: usize,
    /// Coarse compile-time attachment bandwidth estimate in bytes.
    pub frame_graph_estimated_bandwidth_bytes: u64,
    /// Packed lights after the latest [`crate::backend::RenderBackend::prepare_lights_from_scene`].
    pub gpu_light_count: usize,
    /// Whether signed scene-color HDR is active for the current packed light set.
    pub signed_scene_color_active: bool,
    /// Latest persistent upload arena pressure and fallback counters.
    pub upload_arena: FrameUploadArenaSnapshot,
    /// Latest graph command-recording diagnostics.
    pub command_encoding: crate::render_graph::CommandEncodingHudSnapshot,
    /// Latest asset streaming and worker diagnostics.
    pub assets: AssetDiagnosticsSnapshot,
    /// Latest light packing and light influence-volume culling diagnostics.
    pub lights: LightDiagnosticsSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_arena_snapshot_copies_pressure_and_fallback_counters() {
        let snapshot = FrameUploadArenaSnapshot::from(FrameUploadBatchStats {
            writes: 7,
            bytes: 1024,
            fallback_writes: 2,
            persistent_staging_bytes: 512,
            persistent_slot_reuses: 3,
            persistent_slot_grows: 4,
            temporary_staging_bytes: 256,
            temporary_staging_fallbacks: 5,
            oversized_queue_fallback_writes: 6,
            arena_capacity_bytes: 4096,
            arena_free_slots: 1,
            arena_in_flight_slots: 2,
            arena_remapping_slots: 3,
            ..FrameUploadBatchStats::default()
        });

        assert_eq!(snapshot.writes, 7);
        assert_eq!(snapshot.bytes, 1024);
        assert_eq!(snapshot.fallback_writes, 2);
        assert_eq!(snapshot.persistent_staging_bytes, 512);
        assert_eq!(snapshot.persistent_slot_reuses, 3);
        assert_eq!(snapshot.persistent_slot_grows, 4);
        assert_eq!(snapshot.temporary_staging_bytes, 256);
        assert_eq!(snapshot.temporary_staging_fallbacks, 5);
        assert_eq!(snapshot.oversized_queue_fallback_writes, 6);
        assert_eq!(snapshot.arena_capacity_bytes, 4096);
        assert_eq!(snapshot.arena_free_slots, 1);
        assert_eq!(snapshot.arena_in_flight_slots, 2);
        assert_eq!(snapshot.arena_remapping_slots, 3);
    }
}
