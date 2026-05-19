//! [`RenderBackend`] -- thin facade for frame execution and IPC-facing GPU work.
//!
//! Core subsystems live in [`super::MaterialSystem`], [`crate::backend::AssetTransferQueue`],
//! [`super::FrameResourceManager`], and [`crate::occlusion::OcclusionSystem`]; this type wires attach,
//! the compiled render graph, mesh deform preprocess, and debug HUD.
//!
//! Graph execution lives in the `execute` submodule; IPC-facing asset handlers in `asset_ipc`.

mod asset_ipc;
mod attach;
mod diagnostics;
mod draw_preparation;
mod execute;
mod frame_packet;
mod frame_services;
mod graph_access;
mod graph_cache;
mod graph_state;
mod hud_methods;
#[cfg(test)]
mod post_processing_rebuild_tests;
mod reflection_services;

pub use attach::{RenderBackendAttachDesc, RenderBackendAttachError};

use std::sync::Arc;

use crate::backend::AssetTransferQueue;
use crate::config::{RendererSettingsHandle, SceneColorFormat};
use crate::gpu::GpuLimits;
use crate::gpu_pools::{MeshPool, RenderTexturePool, TexturePool};
use crate::materials::host_data::MaterialPropertyStore;
use crate::render_graph::TransientPool;

use super::FrameResourceManager;
use super::secondary_rt_scratch::{SecondaryRtScratchCache, SecondaryRtScratchTargets};
use crate::materials::MaterialSystem;
use crate::occlusion::OcclusionSystem;
use diagnostics::BackendDiagnostics;
use draw_preparation::BackendDrawPreparation;
use frame_services::BackendFrameServices;
pub(crate) use graph_access::BackendGraphAccess;
use graph_state::RenderGraphState;
use reflection_services::ReflectionProbeServices;

pub(crate) use frame_packet::ExtractedFrameShared;

fn scene_color_usage_supported(format: wgpu::TextureFormat, limits: &GpuLimits) -> bool {
    limits.texture_usage_supported(
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

fn scene_color_format_supports_signed_rgb(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
    )
}

fn effective_scene_color_format(
    requested: wgpu::TextureFormat,
    limits: &GpuLimits,
    signed_rgb_required: bool,
) -> wgpu::TextureFormat {
    if signed_rgb_required && !scene_color_format_supports_signed_rgb(requested) {
        let signed_default = SceneColorFormat::Rgba16Float.wgpu_format();
        if scene_color_usage_supported(signed_default, limits) {
            return signed_default;
        }
    }
    if scene_color_usage_supported(requested, limits) {
        return requested;
    }
    let default = SceneColorFormat::default().wgpu_format();
    if scene_color_usage_supported(default, limits) {
        return default;
    }
    wgpu::TextureFormat::Rgba8Unorm
}

/// Coordinates materials, asset uploads, per-frame GPU binds, occlusion, optional deform + ImGui HUD, and the render graph.
pub struct RenderBackend {
    /// Material property store, shader routes, pipeline registry, embedded `@group(1)` binds.
    pub(crate) materials: MaterialSystem,
    /// Mesh/texture upload queues, budgets, format tables, pools, and GPU device/queue for uploads.
    pub(crate) asset_transfers: AssetTransferQueue,
    /// Per-frame bind groups, mesh deformation services, skin cache, and MSAA depth resolve resources.
    frame_services: BackendFrameServices,
    /// CPU draw-preparation caches and material-batch caches.
    draw_preparation: BackendDrawPreparation,
    /// Backend-owned world-mesh forward frame planning caches.
    world_mesh_frame_planner: super::BackendWorldMeshFramePlanner,
    /// Dear ImGui overlay and diagnostics snapshot state.
    diagnostics: BackendDiagnostics,
    /// Nonblocking reflection-probe projection, bake, cache, and selection services.
    reflection_probes: ReflectionProbeServices,
    /// Reusable color/depth targets for partial secondary render-texture camera viewports.
    secondary_rt_scratch: SecondaryRtScratchCache,
    /// Render-graph cache, transient pool, history registry, and view-scoped graph resource ownership.
    graph_state: RenderGraphState,
    /// Hierarchical depth pyramid, CPU readback, and temporal cull state for occlusion culling.
    pub(crate) occlusion: OcclusionSystem,
    /// Swapchain or primary output color format used for frame-graph cache identity.
    surface_format: Option<wgpu::TextureFormat>,
    /// Live settings for per-frame graph parameters (scene HDR format, etc.); set in [`Self::attach`].
    renderer_settings: Option<RendererSettingsHandle>,
    /// Whether this backend is attached to a headless offscreen target.
    headless: bool,
}

impl Default for RenderBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderBackend {
    /// Empty pools and material store; no GPU until [`Self::attach`].
    pub fn new() -> Self {
        Self {
            materials: MaterialSystem::new(),
            asset_transfers: AssetTransferQueue::new(),
            frame_services: BackendFrameServices::new(),
            draw_preparation: BackendDrawPreparation::new(),
            world_mesh_frame_planner: super::BackendWorldMeshFramePlanner::new(),
            diagnostics: BackendDiagnostics::new(),
            reflection_probes: ReflectionProbeServices::new(),
            secondary_rt_scratch: SecondaryRtScratchCache::new(),
            graph_state: RenderGraphState::new(),
            occlusion: OcclusionSystem::new(),
            surface_format: None,
            renderer_settings: None,
            headless: false,
        }
    }

    /// Requested HDR scene-color [`wgpu::TextureFormat`] from [`crate::config::RenderingSettings`].
    ///
    /// Falls back to [`SceneColorFormat::default`] when settings are unavailable (pre-attach).
    fn requested_scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map_or_else(
                || SceneColorFormat::default().wgpu_format(),
                |s| s.rendering.scene_color_format.wgpu_format(),
            )
    }

    /// Effective HDR scene-color [`wgpu::TextureFormat`] supported by the active device.
    pub(crate) fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        let signed_rgb_required = self
            .frame_services
            .frame_resources
            .signed_scene_color_required();
        let requested = match self.requested_scene_color_format_wgpu() {
            format if signed_rgb_required && !scene_color_format_supports_signed_rgb(format) => {
                SceneColorFormat::Rgba16Float.wgpu_format()
            }
            format => format,
        };
        self.gpu_limits().map_or(requested, |limits| {
            effective_scene_color_format(requested, limits, signed_rgb_required)
        })
    }

    /// Returns true when negative lights force signed scene-color HDR for the current frame.
    pub(crate) fn signed_scene_color_active(&self) -> bool {
        self.frame_services
            .frame_resources
            .signed_scene_color_required()
            && scene_color_format_supports_signed_rgb(self.scene_color_format_wgpu())
    }

    /// Snapshot of the live GTAO settings for the current frame.
    ///
    /// Seeded into each view's blackboard as [`crate::render_graph::post_process_settings::GtaoSettingsSlot`]
    /// so the shader UBO reflects slider changes without rebuilding the compiled render graph
    /// (the chain signature only tracks enable booleans, so parameter edits wouldn't otherwise
    /// reach the pass).
    pub(crate) fn live_gtao_settings(&self) -> crate::config::GtaoSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.gtao)
            .unwrap_or_default()
    }

    /// Snapshot of the live bloom settings for the current frame.
    ///
    /// Seeded into each view's blackboard as [`crate::render_graph::post_process_settings::BloomSettingsSlot`]
    /// so the first downsample's params UBO and the upsample blend constants reflect slider
    /// changes without rebuilding the compiled render graph. The effective `max_mip_dimension`
    /// is the one exception -- it drives mip-chain texture sizes, so it lives on the chain
    /// signature and triggers a rebuild instead.
    pub(crate) fn live_bloom_settings(&self) -> crate::config::BloomSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.bloom)
            .unwrap_or_default()
    }

    /// Snapshot of the live motion-blur settings for the current frame.
    ///
    /// Seeded into each view's blackboard as
    /// [`crate::render_graph::post_process_settings::MotionBlurSettingsSlot`] so blur samples,
    /// shutter scale, and clamp edits take effect without rebuilding the compiled graph.
    pub(crate) fn live_motion_blur_settings(&self) -> crate::config::MotionBlurSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.motion_blur)
            .unwrap_or_default()
    }

    /// Snapshot of the live auto-exposure settings for the current frame.
    ///
    /// Seeded into each view's blackboard as
    /// [`crate::render_graph::post_process_settings::AutoExposureSettingsSlot`] so histogram
    /// settings and adaptation speed edits take effect without rebuilding the compiled graph.
    pub(crate) fn live_auto_exposure_settings(&self) -> crate::config::AutoExposureSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.auto_exposure)
            .unwrap_or_default()
    }

    /// Snapshot of the live reflection-probe SH2 experimental toggle.
    pub(crate) fn reflection_probe_sh2_enabled(&self) -> bool {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.experimental.reflection_probe_sh2_enabled)
            .unwrap_or(crate::config::ExperimentalSettings::default().reflection_probe_sh2_enabled)
    }

    /// Snapshot of the live development WGSL material hot-reload toggle.
    pub(crate) fn material_shader_hot_reload_enabled(&self) -> bool {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.experimental.material_shader_hot_reload_enabled)
            .unwrap_or(
                crate::config::ExperimentalSettings::default().material_shader_hot_reload_enabled,
            )
    }

    /// Applies development WGSL hot-reload settings and polls for changed local material targets.
    pub(crate) fn sync_material_shader_hot_reload(&mut self) {
        let enabled = self.material_shader_hot_reload_enabled();
        self.materials.set_dev_shader_hot_reload_enabled(enabled);
        let report = self.materials.poll_dev_shader_hot_reload();
        if report.is_empty() {
            return;
        }
        for stem in &report.reloaded_stems {
            logger::info!("materials: development WGSL reloaded for {stem}");
        }
        for error in &report.errors {
            logger::warn!("materials: development WGSL reload failed: {error}");
        }
    }

    /// Count of host Texture2D asset ids that have received a [`crate::shared::SetTexture2DFormat`] (CPU-side table).
    pub fn texture_format_registration_count(&self) -> usize {
        self.asset_transfers.texture_format_registration_count()
    }

    /// Count of GPU-resident textures with `mip_levels_resident > 0` (at least mip0 uploaded).
    pub fn texture_mip0_ready_count(&self) -> usize {
        self.asset_transfers
            .texture_pool()
            .iter()
            .filter(|t| t.mip_levels_resident > 0)
            .count()
    }

    /// Resets per-tick light prep flags, mesh deform coalescing, and advances the skin cache frame counter.
    ///
    /// Call once per winit tick before IPC and frame work (see [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn reset_light_prep_for_tick(&mut self) {
        self.frame_services.reset_for_tick();
    }

    /// GPU limits snapshot after [`Self::attach`], if attach succeeded.
    pub fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.asset_transfers.gpu_limits()
    }

    /// Mutable frame resources for runtime draw-preparation handoffs.
    pub(crate) fn frame_resources_mut(&mut self) -> &mut FrameResourceManager {
        &mut self.frame_services.frame_resources
    }

    /// Drains latest video clock-error samples produced by asset integration.
    pub(crate) fn take_pending_video_clock_errors(
        &mut self,
    ) -> Vec<crate::shared::VideoTextureClockErrorState> {
        self.asset_transfers.take_pending_video_clock_errors()
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        self.asset_transfers.mesh_pool()
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        self.asset_transfers.texture_pool()
    }

    /// Host render texture targets (secondary cameras, material sampling).
    pub fn render_texture_pool(&self) -> &RenderTexturePool {
        self.asset_transfers.render_texture_pool()
    }

    /// Returns a reusable scratch target for a partial secondary camera viewport.
    pub(crate) fn secondary_render_rect_scratch(
        &mut self,
        device: &wgpu::Device,
        extent_px: (u32, u32),
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Option<SecondaryRtScratchTargets> {
        self.secondary_rt_scratch
            .get_or_create(device, extent_px, color_format, depth_format)
    }

    /// Answers host SH2 task rows for the latest frame submit without blocking GPU readback.
    pub(crate) fn answer_reflection_probe_sh2_tasks(
        &mut self,
        shm: &mut crate::ipc::SharedMemoryAccessor,
        scene: &crate::scene::SceneCoordinator,
        data: &crate::shared::FrameSubmitData,
    ) {
        self.reflection_probes.answer_sh2_frame_submit_tasks(
            shm,
            scene,
            &self.asset_transfers,
            data,
        );
    }

    /// Registers a completed OnChanges runtime reflection-probe cubemap capture.
    pub(crate) fn register_runtime_reflection_probe_capture(
        &mut self,
        capture: crate::reflection_probes::specular::RuntimeReflectionProbeCapture,
    ) {
        self.reflection_probes.register_runtime_capture(capture);
    }

    /// Advances nonblocking SH2 GPU jobs and schedules queued projection work.
    pub(crate) fn maintain_reflection_probe_sh2_jobs(&mut self, gpu: &mut crate::gpu::GpuContext) {
        self.reflection_probes
            .maintain_sh2_jobs(gpu, &self.asset_transfers);
    }

    /// Advances reflection-probe specular IBL jobs and syncs frame-global probe bindings.
    pub(crate) fn maintain_reflection_probe_specular_jobs(
        &mut self,
        gpu: &mut crate::gpu::GpuContext,
        scene: &crate::scene::SceneCoordinator,
        render_context: crate::shared::RenderingContext,
    ) {
        let reflection_probe_sh2_enabled = self.reflection_probe_sh2_enabled();
        let resources = self.reflection_probes.maintain_specular_jobs(
            gpu,
            scene,
            &self.asset_transfers,
            render_context,
            reflection_probe_sh2_enabled,
        );
        let _ = self
            .frame_services
            .frame_resources
            .sync_reflection_probe_specular_resources(gpu.device(), resources);
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        self.materials.material_property_store()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::materials::host_data::PropertyIdRegistry {
        self.materials.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.materials.material_registry()
    }

    /// Drains finished background pipeline builds into the cache (no-op before GPU attach).
    ///
    /// The renderer's per-tick render entry calls this before per-view recording starts so
    /// worker threads stay off the completion channel and pending/failed mutexes on the hot path.
    pub fn drain_pipeline_build_completions(&self) {
        self.materials.drain_pipeline_build_completions();
    }

    /// Number of schedules passes in the compiled frame graph, or `0` if none.
    pub fn frame_graph_pass_count(&self) -> usize {
        self.graph_state.frame_graph_cache.pass_count()
    }

    /// Compile-time topological wave count for the cached frame graph, or `0` if none has been built yet.
    pub fn frame_graph_topo_levels(&self) -> usize {
        self.graph_state.frame_graph_cache.topo_levels()
    }

    /// Upload arena generation used by graph-cache reset-policy unit tests.
    #[cfg(test)]
    pub(crate) fn upload_arena_generation_for_tests(&self) -> u64 {
        self.graph_state.upload_arena_generation_for_tests()
    }

    /// Plain-data backend snapshot consumed by the diagnostics HUD.
    ///
    /// Returns a [`crate::diagnostics::BackendDiagSnapshot`] capturing the fields
    /// `FrameDiagnosticsSnapshot::capture` and `RendererInfoSnapshot::capture` need, so the
    /// diagnostics layer never borrows `&RenderBackend` directly.
    pub fn snapshot_for_diagnostics(&self) -> crate::diagnostics::BackendDiagSnapshot {
        let store = self.material_property_store();
        let shader_routes = self
            .material_registry()
            .map(|reg| {
                reg.shader_routes_for_hud()
                    .into_iter()
                    .map(|(id, pipeline, name, shader_variant_bits)| {
                        crate::diagnostics::ShaderRouteSnapshot {
                            shader_asset_id: id,
                            pipeline,
                            shader_asset_name: name,
                            shader_variant_bits,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let material_diagnostics = self.materials.diagnostic_snapshot();
        crate::diagnostics::BackendDiagSnapshot {
            texture_format_registration_count: self.texture_format_registration_count(),
            texture_mip0_ready_count: self.texture_mip0_ready_count(),
            texture_pool_resident_count: self.texture_pool().len(),
            render_texture_pool_len: self.render_texture_pool().len(),
            mesh_pool_entry_count: self.mesh_pool().len(),
            shader_routes,
            last_world_mesh_draw_stats: self.last_world_mesh_draw_stats(),
            last_world_mesh_draw_state_rows: self.last_world_mesh_draw_state_rows(),
            material_property_slots: store.material_property_slot_count(),
            property_block_slots: store.property_block_slot_count(),
            material_shader_bindings: store.material_shader_binding_count(),
            material_shader_graph: material_diagnostics.shader_graph,
            material_pipeline_cache: material_diagnostics.pipeline_cache,
            frame_graph_pass_count: self.frame_graph_pass_count(),
            frame_graph_topo_levels: self.frame_graph_topo_levels(),
            gpu_light_count: self.frame_services.frame_resources.frame_lights().len(),
            signed_scene_color_active: self.signed_scene_color_active(),
        }
    }

    /// Mutable render-graph transient resource pool.
    pub(crate) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        self.graph_state.transient_pool_mut()
    }

    /// Synchronizes backend view-scoped resource ownership against the runtime's active view list.
    pub(crate) fn sync_active_views<I>(&mut self, active_views: I)
    where
        I: IntoIterator<Item = crate::camera::ViewId>,
    {
        let retired = self.graph_state.sync_active_views(active_views);
        if retired.is_empty() {
            return;
        }
        logger::debug!(
            "retiring {} inactive view-scoped resource sets",
            retired.len()
        );
        self.world_mesh_frame_planner
            .release_view_resources(&retired);
        for view_id in retired {
            self.frame_services.frame_resources.retire_view(view_id);
            self.graph_state.history_registry_mut().retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
    }

    /// Releases resources for one-shot views that were never part of the active-view registry.
    pub(crate) fn retire_one_shot_views(&mut self, retired: &[crate::camera::ViewId]) {
        if retired.is_empty() {
            return;
        }
        self.graph_state.release_view_resources(retired);
        self.world_mesh_frame_planner
            .release_view_resources(retired);
        for &view_id in retired {
            self.frame_services.frame_resources.retire_view(view_id);
            self.graph_state.history_registry_mut().retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
    }

    /// Builds the narrow graph-execution access packet from disjoint backend owners.
    pub(crate) fn graph_access(&mut self) -> BackendGraphAccess<'_> {
        let scene_color_format = self.scene_color_format_wgpu();
        let gpu_limits = self.gpu_limits().cloned();
        let msaa_depth_resolve = self.frame_services.msaa_depth_resolve();
        let live_gtao_settings = self.live_gtao_settings();
        let live_bloom_settings = self.live_bloom_settings();
        let live_motion_blur_settings = self.live_motion_blur_settings();
        let live_auto_exposure_settings = self.live_auto_exposure_settings();
        let wall_frame_time_ms = self.debug_frame_time_ms();
        let (transient_pool, history_registry, upload_arena) =
            self.graph_state.execution_resources_mut();
        let (frame_resources, mesh_preprocess, mesh_deform_scratch, skin_cache) =
            self.frame_services.graph_access_slices();
        BackendGraphAccess {
            occlusion: &mut self.occlusion,
            frame_resources,
            materials: &self.materials,
            asset_transfers: &mut self.asset_transfers,
            mesh_preprocess,
            mesh_deform_scratch,
            skin_cache,
            world_mesh_frame_planner: &self.world_mesh_frame_planner,
            transient_pool,
            history_registry,
            upload_arena,
            debug_hud: self.diagnostics.bundle_mut(),
            scene_color_format,
            gpu_limits,
            msaa_depth_resolve,
            live_gtao_settings,
            live_bloom_settings,
            live_motion_blur_settings,
            live_auto_exposure_settings,
            wall_frame_time_ms,
        }
    }
}
