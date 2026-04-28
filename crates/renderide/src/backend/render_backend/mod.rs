//! [`RenderBackend`] — thin coordinator for frame execution and IPC-facing GPU work.
//!
//! Core subsystems live in [`super::MaterialSystem`], [`crate::assets::AssetTransferQueue`],
//! [`super::FrameResourceManager`], and [`super::OcclusionSystem`]; this type wires attach,
//! the compiled render graph, mesh deform preprocess, and debug HUD.
//!
//! Graph execution lives in the `execute` submodule; IPC-facing asset handlers in `asset_ipc`.

mod asset_ipc;
mod execute;
mod frame_packet;
mod graph_cache;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

use crate::assets::asset_transfer_queue::{self as asset_uploads, AssetTransferQueue};
use crate::assets::material::MaterialPropertyStore;
use crate::backend::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::config::{PostProcessingSettings, RendererSettingsHandle, SceneColorFormat};
use crate::diagnostics::{DebugHudEncodeError, DebugHudInput, SceneTransformsSnapshot};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::{MaterialRouter, RasterPipelineKind};
use crate::render_graph::{
    FrameMaterialBatchCache, GraphCache, PerViewHudConfig, PerViewHudOutputs, TransientPool,
    WorldMeshDrawStateRow, WorldMeshDrawStats,
};
use crate::resources::{
    CubemapPool, MeshPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};

use super::debug_hud_bundle::DebugHudBundle;
use super::embedded::{EmbeddedMaterialBindError, EmbeddedTexturePools};
use super::material_system::MaterialSystem;
use super::occlusion::OcclusionSystem;
use super::FrameGpuBindingsError;
use super::FrameResourceManager;
use super::ViewResourceRegistry;

/// Disjoint backend slices assembled into [`crate::render_graph::FrameRenderParams`].
type GraphFrameParamsSplit<'a> = (
    &'a OcclusionSystem,
    &'a FrameResourceManager,
    &'a MaterialSystem,
    &'a AssetTransferQueue,
    Option<&'a MeshPreprocessPipelines>,
    Option<&'a mut MeshDeformScratch>,
    Option<&'a mut GpuSkinCache>,
    Option<Arc<GpuLimits>>,
    Option<Arc<MsaaDepthResolveResources>>,
    PerViewHudConfig,
);

pub use crate::assets::asset_transfer_queue::{
    MAX_ASSET_INTEGRATION_QUEUED, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
};
pub(crate) use frame_packet::ExtractedFrameShared;

/// GPU attach failed for frame binds (`@group(0/1/2)`) or embedded materials (`@group(1)`).
#[derive(Debug, Error)]
pub enum RenderBackendAttachError {
    /// Frame / empty material / per-draw allocation failed atomically.
    #[error(transparent)]
    FrameGpuBindings(#[from] FrameGpuBindingsError),
    /// Embedded raster `@group(1)` bind resources could not be created.
    #[error(transparent)]
    EmbeddedMaterialBind(#[from] EmbeddedMaterialBindError),
}

/// Device, queue, and settings passed to [`RenderBackend::attach`] (shared-memory flush is passed separately for borrow reasons).
pub struct RenderBackendAttachDesc {
    /// Logical device for uploads and graph encoding.
    pub device: Arc<wgpu::Device>,
    /// Queue used for submits and GPU writes.
    pub queue: Arc<wgpu::Queue>,
    /// Shared GPU queue access gate cloned from [`crate::gpu::GpuContext`]; acquired by
    /// upload, submit, and OpenXR queue-access paths. See [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: crate::gpu::GpuQueueAccessGate,
    /// Capabilities for buffer sizing and MSAA.
    pub gpu_limits: Arc<GpuLimits>,
    /// Swapchain / main surface format for HUD and pipelines.
    pub surface_format: wgpu::TextureFormat,
    /// Live renderer settings (HUD, VR budgets, etc.).
    pub renderer_settings: RendererSettingsHandle,
    /// Path for persisting HUD/config from the debug overlay.
    pub config_save_path: PathBuf,
    /// When `true`, the ImGui config window must not write `config.toml` (startup extract failed).
    pub suppress_renderer_config_disk_writes: bool,
}

/// Coordinates materials, asset uploads, per-frame GPU binds, occlusion, optional deform + ImGui HUD, and the render graph.
pub struct RenderBackend {
    /// Material property store, shader routes, pipeline registry, embedded `@group(1)` binds.
    pub(crate) materials: MaterialSystem,
    /// Mesh/texture upload queues, budgets, format tables, pools, and GPU device/queue for uploads.
    pub(crate) asset_transfers: AssetTransferQueue,
    /// Fallback router used before any embedded-material registry is available.
    null_material_router: MaterialRouter,
    /// Optional mesh skinning / blendshape compute pipelines (after [`Self::attach`]).
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Cached compiled frame graph keyed by the shared render-graph cache inputs.
    frame_graph_cache: GraphCache,
    /// Scratch buffers for mesh deformation compute (after [`Self::attach`]).
    mesh_deform_scratch: Option<MeshDeformScratch>,
    /// Arena-backed deformed vertex streams (after [`Self::attach`]); sibling to [`Self::frame_resources`] for borrow splitting.
    skin_cache: Option<GpuSkinCache>,
    /// MSAA depth -> R32F -> single-sample depth resolve resources when supported.
    msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Per-frame bind groups, light staging, and debug draw slab.
    pub(crate) frame_resources: super::FrameResourceManager,
    /// Dear ImGui overlay and capture state.
    debug_hud: DebugHudBundle,
    /// Hierarchical depth pyramid, CPU readback, and temporal cull state for occlusion culling.
    pub(crate) occlusion: OcclusionSystem,
    /// Render-graph transient texture/buffer pool retained across frames.
    pub(crate) transient_pool: TransientPool,
    /// Swapchain or primary output color format used for frame-graph cache identity.
    surface_format: Option<wgpu::TextureFormat>,
    /// Live settings for per-frame graph parameters (scene HDR format, etc.); set in [`Self::attach`].
    renderer_settings: Option<RendererSettingsHandle>,
    /// Whether per-view encoder recording runs on rayon workers or sequentially on the main thread.
    ///
    /// Defaults to [`crate::config::RecordParallelism::PerViewParallel`]. Switch via
    /// `[rendering] record_parallelism` in the renderer config once
    /// in the renderer config once per-view pass state is fully validated as `Send`-safe.
    pub(crate) record_parallelism: crate::config::RecordParallelism,
    /// Persistent resolved-material cache, refreshed once per frame before per-view draw
    /// collection. Entries invalidate against
    /// [`crate::assets::material::MaterialPropertyStore`] and
    /// [`crate::materials::MaterialRouter`] generation counters, so steady-state refresh cost is
    /// proportional to the number of mutated materials rather than the total material count.
    pub(crate) material_batch_cache: FrameMaterialBatchCache,
    /// Registry of persistent ping-pong resources used by graph history slots
    /// (`ImportSource::PingPong` / `BufferImportSource::PingPong`).
    pub(crate) history_registry: super::HistoryRegistry,
    /// Nonblocking reflection-probe SH2 GPU projection service.
    pub(crate) reflection_probe_sh2: super::ReflectionProbeSh2System,
    /// Retained logical-view ownership for every backend cache that lives beyond one frame.
    view_resources: ViewResourceRegistry,
}

/// Disjoint borrows of [`MaterialSystem`], [`AssetTransferQueue`], and the GPU skin cache for world mesh forward encoding.
///
/// Obtained from [`crate::render_graph::FrameRenderParams::world_mesh_forward_encode_refs`] so the raster
/// encoder never holds `&mut RenderBackend` while also borrowing the deform cache.
pub(crate) struct WorldMeshForwardEncodeRefs<'a> {
    /// Material registry, embedded binds, and property store.
    pub(crate) materials: &'a MaterialSystem,
    /// Mesh and texture pools.
    pub(crate) asset_transfers: &'a AssetTransferQueue,
    /// Arena-backed deformed positions and normals keyed by renderable (after [`RenderBackend::attach`]).
    pub(crate) skin_cache: Option<&'a GpuSkinCache>,
}

impl<'a> WorldMeshForwardEncodeRefs<'a> {
    /// Builds encode refs from disjoint [`crate::render_graph::FrameRenderParams`] slices.
    pub fn from_frame_params(
        materials: &'a MaterialSystem,
        asset_transfers: &'a AssetTransferQueue,
        skin_cache: Option<&'a GpuSkinCache>,
    ) -> Self {
        Self {
            materials,
            asset_transfers,
            skin_cache,
        }
    }

    /// Mesh pool for draw recording after any required lazy stream uploads were pre-warmed.
    pub(crate) fn mesh_pool(&self) -> &MeshPool {
        &self.asset_transfers.mesh_pool
    }

    /// Pool views for embedded `@group(1)` texture resolution.
    pub(crate) fn embedded_texture_pools(&self) -> EmbeddedTexturePools<'_> {
        EmbeddedTexturePools {
            texture: &self.asset_transfers.texture_pool,
            texture3d: &self.asset_transfers.texture3d_pool,
            cubemap: &self.asset_transfers.cubemap_pool,
            render_texture: &self.asset_transfers.render_texture_pool,
            video_texture: &self.asset_transfers.video_texture_pool,
        }
    }
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
            null_material_router: MaterialRouter::new(RasterPipelineKind::Null),
            mesh_preprocess: None,
            frame_graph_cache: GraphCache::default(),
            mesh_deform_scratch: None,
            skin_cache: None,
            msaa_depth_resolve: None,
            frame_resources: super::FrameResourceManager::new(),
            debug_hud: DebugHudBundle::new(),
            occlusion: OcclusionSystem::new(),
            transient_pool: TransientPool::new(),
            surface_format: None,
            renderer_settings: None,
            record_parallelism: crate::config::RecordParallelism::PerViewParallel,
            material_batch_cache: FrameMaterialBatchCache::new(),
            history_registry: super::HistoryRegistry::new(),
            reflection_probe_sh2: super::ReflectionProbeSh2System::new(),
            view_resources: ViewResourceRegistry::new(),
        }
    }

    /// Returns a mutable reference to the persistent history registry.
    ///
    /// Subsystems register ping-pong slots here before graph execution. Hi-Z uses view-scoped
    /// texture history through this path while [`OcclusionSystem`] keeps CPU snapshots, temporal
    /// cull data, and readback policy.
    pub fn history_registry_mut(&mut self) -> &mut super::HistoryRegistry {
        &mut self.history_registry
    }

    /// Shared reference to the persistent history registry.
    pub fn history_registry(&self) -> &super::HistoryRegistry {
        &self.history_registry
    }

    /// Effective HDR scene-color [`wgpu::TextureFormat`] from [`crate::config::RenderingSettings`].
    ///
    /// Falls back to [`SceneColorFormat::default`] when settings are unavailable (pre-attach).
    pub(crate) fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.rendering.scene_color_format.wgpu_format())
            .unwrap_or_else(|| SceneColorFormat::default().wgpu_format())
    }

    /// Snapshot of the live GTAO settings for the current frame.
    ///
    /// Seeded into each view's blackboard as [`crate::render_graph::frame_params::GtaoSettingsSlot`]
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
    /// Seeded into each view's blackboard as [`crate::render_graph::frame_params::BloomSettingsSlot`]
    /// so the first downsample's params UBO and the upsample blend constants reflect slider
    /// changes without rebuilding the compiled render graph. `max_mip_dimension` is the one
    /// exception — it drives mip-chain texture sizes, so it lives on the chain signature and
    /// triggers a rebuild instead.
    pub(crate) fn live_bloom_settings(&self) -> crate::config::BloomSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.bloom)
            .unwrap_or_default()
    }

    /// Count of host Texture2D asset ids that have received a [`crate::shared::SetTexture2DFormat`] (CPU-side table).
    pub fn texture_format_registration_count(&self) -> usize {
        self.asset_transfers.texture_formats.len()
    }

    /// Count of GPU-resident textures with `mip_levels_resident > 0` (at least mip0 uploaded).
    pub fn texture_mip0_ready_count(&self) -> usize {
        self.asset_transfers
            .texture_pool
            .textures()
            .values()
            .filter(|t| t.mip_levels_resident > 0)
            .count()
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        self.mesh_preprocess.as_ref()
    }

    /// Arena-backed deformed vertex streams shared by mesh deform compute and mesh forward draws.
    pub fn skin_cache(&self) -> Option<&GpuSkinCache> {
        self.skin_cache.as_ref()
    }

    /// Shared occlusion system view for per-view recording.
    pub(crate) fn occlusion(&self) -> &OcclusionSystem {
        &self.occlusion
    }

    /// Shared frame-resource manager view for per-view recording.
    pub(crate) fn frame_resources(&self) -> &FrameResourceManager {
        &self.frame_resources
    }

    /// Shared material system view for per-view recording.
    pub(crate) fn materials(&self) -> &MaterialSystem {
        &self.materials
    }

    /// Shared asset-transfer queues and pools for per-view recording.
    pub(crate) fn asset_transfers(&self) -> &AssetTransferQueue {
        &self.asset_transfers
    }

    /// Shared debug HUD view for per-view recording.
    pub(crate) fn per_view_hud_config(&self) -> PerViewHudConfig {
        PerViewHudConfig {
            main_enabled: self.debug_hud.main_enabled(),
            textures_enabled: self.debug_hud.textures_enabled(),
        }
    }

    /// MSAA depth resolve resources snapshot for per-view recording.
    pub(crate) fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>> {
        self.msaa_depth_resolve.clone()
    }

    /// Mutable skin cache for mesh deform compute and cache sweeps.
    pub fn skin_cache_mut(&mut self) -> Option<&mut GpuSkinCache> {
        self.skin_cache.as_mut()
    }

    /// Resets per-tick light prep flags, mesh deform coalescing, and advances the skin cache frame counter.
    ///
    /// Call once per winit tick before IPC and frame work (see [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn reset_light_prep_for_tick(&mut self) {
        self.frame_resources.reset_light_prep_for_tick();
        if let Some(ref mut cache) = self.skin_cache {
            cache.advance_frame();
        }
    }

    /// GPU limits snapshot after [`Self::attach`], if attach succeeded.
    pub fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.asset_transfers.gpu_limits.as_ref()
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        &self.asset_transfers.mesh_pool
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.asset_transfers.mesh_pool
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        &self.asset_transfers.texture_pool
    }

    /// Resident Texture3D table.
    pub fn texture3d_pool(&self) -> &Texture3dPool {
        &self.asset_transfers.texture3d_pool
    }

    /// Resident cubemap table.
    pub fn cubemap_pool(&self) -> &CubemapPool {
        &self.asset_transfers.cubemap_pool
    }

    /// Host render texture targets (secondary cameras, material sampling).
    pub fn render_texture_pool(&self) -> &RenderTexturePool {
        &self.asset_transfers.render_texture_pool
    }

    /// Resident video texture table.
    pub fn video_texture_pool(&self) -> &VideoTexturePool {
        &self.asset_transfers.video_texture_pool
    }

    /// Answers host SH2 task rows for the latest frame submit without blocking GPU readback.
    pub(crate) fn answer_reflection_probe_sh2_tasks(
        &mut self,
        shm: &mut crate::ipc::SharedMemoryAccessor,
        scene: &crate::scene::SceneCoordinator,
        data: &crate::shared::FrameSubmitData,
    ) {
        self.reflection_probe_sh2.answer_frame_submit_tasks(
            shm,
            scene,
            &self.materials,
            &self.asset_transfers,
            data,
        );
    }

    /// Advances nonblocking SH2 GPU jobs and schedules queued projection work.
    pub(crate) fn maintain_reflection_probe_sh2_jobs(&mut self, gpu: &crate::gpu::GpuContext) {
        self.reflection_probe_sh2
            .maintain_gpu_jobs(gpu, &self.asset_transfers);
    }

    /// Borrowed view of all texture pools used for embedded material `@group(1)` bind resolution.
    pub fn embedded_texture_pools(&self) -> EmbeddedTexturePools<'_> {
        EmbeddedTexturePools {
            texture: self.texture_pool(),
            texture3d: self.texture3d_pool(),
            cubemap: self.cubemap_pool(),
            render_texture: self.render_texture_pool(),
            video_texture: self.video_texture_pool(),
        }
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut TexturePool {
        &mut self.asset_transfers.texture_pool
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        self.materials.material_property_store()
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(&mut self) -> &mut MaterialPropertyStore {
        self.materials.material_property_store_mut()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::assets::material::PropertyIdRegistry {
        self.materials.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.materials.material_registry()
    }

    /// Mutable registry (pipeline cache and shader routes).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.materials.material_registry_mut()
    }

    /// Merges one deferred per-view HUD payload into the live debug HUD bundle.
    pub(crate) fn apply_per_view_hud_outputs(&mut self, outputs: &PerViewHudOutputs) {
        self.debug_hud.apply_per_view_outputs(outputs);
    }

    /// Embedded material bind groups (world Unlit, etc.) after [`Self::attach`].
    pub fn embedded_material_bind(
        &self,
    ) -> Option<&super::embedded::EmbeddedMaterialBindResources> {
        self.materials.embedded_material_bind()
    }

    /// Number of schedules passes in the compiled frame graph, or `0` if none.
    pub fn frame_graph_pass_count(&self) -> usize {
        self.frame_graph_cache.pass_count()
    }

    /// Compile-time topological wave count for the cached frame graph, or `0` if none has been built yet.
    pub fn frame_graph_topo_levels(&self) -> usize {
        self.frame_graph_cache.topo_levels()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// Wires device/queue into uploads, allocates frame binds and materials, and builds the default graph.
    /// `shm` flushes pending mesh/texture payloads that require shared-memory reads; omit when none is
    /// available yet (uploads stay queued).
    ///
    /// On error, CPU-side asset queues may already be partially configured; GPU draws must not run until
    /// a successful attach.
    pub fn attach(
        &mut self,
        desc: RenderBackendAttachDesc,
        shm: Option<&mut crate::ipc::SharedMemoryAccessor>,
    ) -> Result<(), RenderBackendAttachError> {
        let RenderBackendAttachDesc {
            device,
            queue,
            gpu_queue_access_gate,
            gpu_limits,
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        } = desc;
        self.renderer_settings = Some(renderer_settings.clone());
        self.surface_format = Some(surface_format);
        self.asset_transfers.gpu_device = Some(device.clone());
        self.asset_transfers.gpu_queue = Some(queue.clone());
        self.asset_transfers.gpu_queue_access_gate = Some(gpu_queue_access_gate);
        self.asset_transfers.gpu_limits = Some(Arc::clone(&gpu_limits));
        {
            let s = renderer_settings
                .read()
                .map(|g| g.clone())
                .unwrap_or_default();
            self.asset_transfers.render_texture_hdr_color = s.rendering.render_texture_hdr_color;
            self.asset_transfers.texture_vram_budget_bytes =
                u64::from(s.rendering.texture_vram_budget_mib).saturating_mul(1024 * 1024);
        }
        let max_buffer_size = gpu_limits.max_buffer_size();
        self.mesh_deform_scratch = Some(MeshDeformScratch::new(device.as_ref(), max_buffer_size));
        self.frame_resources
            .attach(device.as_ref(), queue.as_ref(), Arc::clone(&gpu_limits))?;
        self.skin_cache = Some(GpuSkinCache::new(device.as_ref(), max_buffer_size));
        self.debug_hud.attach(
            device.as_ref(),
            queue.as_ref(),
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        );
        match MeshPreprocessPipelines::new(device.as_ref()) {
            Ok(p) => self.mesh_preprocess = Some(p),
            Err(e) => {
                logger::warn!("mesh preprocess compute pipelines not created: {e}");
                self.mesh_preprocess = None;
            }
        }
        self.materials
            .try_attach_gpu(device.clone(), &queue, Arc::clone(&gpu_limits))?;
        asset_uploads::attach_flush_pending_asset_uploads(&mut self.asset_transfers, &device, shm);

        self.msaa_depth_resolve = MsaaDepthResolveResources::try_new(device.as_ref()).map(Arc::new);

        let (post_processing_settings, msaa_sample_count) = self
            .renderer_settings
            .as_ref()
            .and_then(|h| {
                h.read()
                    .ok()
                    .map(|g| (g.post_processing.clone(), g.rendering.msaa.as_count() as u8))
            })
            .unwrap_or_else(|| (PostProcessingSettings::default(), 1));
        let shape = self.frame_graph_shape_for(&post_processing_settings, msaa_sample_count, false);
        self.sync_frame_graph_cache(&post_processing_settings, shape);
        Ok(())
    }

    /// Updates the per-view record parallelism mode from live [`crate::config::RenderingSettings`].
    ///
    /// On the first frame after the effective mode changes, logs the new mode at `info!`. Runtime
    /// changes take effect on the next `execute_multi_view` call. See
    /// [`crate::render_graph::CompiledRenderGraph::execute_multi_view`] for the parallel branch.
    pub fn set_record_parallelism(&mut self, mode: crate::config::RecordParallelism) {
        if self.record_parallelism != mode {
            logger::info!(
                "record parallelism mode change: {:?} -> {:?}",
                self.record_parallelism,
                mode
            );
            self.record_parallelism = mode;
        }
    }

    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_debug_hud_main_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_main_enabled(enabled);
    }

    /// Updates whether texture HUD diagnostics run.
    pub(crate) fn set_debug_hud_textures_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_textures_enabled(enabled);
    }

    /// Clears the current-view Texture2D set before collecting this frame's submitted draws.
    pub(crate) fn clear_debug_hud_current_view_texture_2d_asset_ids(&mut self) {
        self.debug_hud.clear_current_view_texture_2d_asset_ids();
    }

    /// Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn debug_hud_current_view_texture_2d_asset_ids(&self) -> &BTreeSet<i32> {
        self.debug_hud.current_view_texture_2d_asset_ids()
    }

    /// Updates pointer state and frame delta for the ImGui overlay.
    pub fn set_debug_hud_frame_data(&mut self, input: DebugHudInput, frame_time_ms: f64) {
        self.debug_hud.set_frame_data(input, frame_time_ms);
    }

    /// Last inter-frame time in milliseconds supplied by the app for HUD FPS.
    pub(crate) fn debug_frame_time_ms(&self) -> f64 {
        self.debug_hud.frame_time_ms()
    }

    /// [`imgui::Io::want_capture_mouse`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.debug_hud.last_want_capture_mouse()
    }

    /// [`imgui::Io::want_capture_keyboard`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.debug_hud.last_want_capture_keyboard()
    }

    /// Stores [`crate::diagnostics::RendererInfoSnapshot`] for the next HUD frame.
    pub(crate) fn set_debug_hud_snapshot(
        &mut self,
        snapshot: crate::diagnostics::RendererInfoSnapshot,
    ) {
        self.debug_hud.set_snapshot(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_diagnostics(
        &mut self,
        snapshot: crate::diagnostics::FrameDiagnosticsSnapshot,
    ) {
        self.debug_hud.set_frame_diagnostics(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_timing(
        &mut self,
        snapshot: crate::diagnostics::FrameTimingHudSnapshot,
    ) {
        self.debug_hud.set_frame_timing(snapshot);
    }

    /// Pushes the latest flattened GPU pass timings into the debug HUD's **GPU passes** tab.
    pub(crate) fn set_debug_hud_gpu_pass_timings(
        &mut self,
        timings: Vec<crate::profiling::GpuPassEntry>,
    ) {
        self.debug_hud.set_gpu_pass_timings(timings);
    }

    /// Clears Stats / Shader routes payloads only (not frame timing or scene transforms).
    pub(crate) fn clear_debug_hud_stats_snapshots(&mut self) {
        self.debug_hud.clear_stats_snapshots();
    }

    /// Clears the **Scene transforms** HUD payload.
    pub(crate) fn clear_debug_hud_scene_transforms_snapshot(&mut self) {
        self.debug_hud.clear_scene_transforms_snapshot();
    }

    pub(crate) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.debug_hud.last_world_mesh_draw_stats()
    }

    pub(crate) fn last_world_mesh_draw_state_rows(&self) -> Vec<WorldMeshDrawStateRow> {
        self.debug_hud.last_world_mesh_draw_state_rows()
    }

    /// Updates the **Scene transforms** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_scene_transforms_snapshot(
        &mut self,
        snapshot: SceneTransformsSnapshot,
    ) {
        self.debug_hud.set_scene_transforms_snapshot(snapshot);
    }

    /// Updates the **Textures** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_texture_debug_snapshot(
        &mut self,
        snapshot: crate::diagnostics::TextureDebugSnapshot,
    ) {
        self.debug_hud.set_texture_debug_snapshot(snapshot);
    }

    /// Clears the **Textures** HUD payload.
    pub(crate) fn clear_debug_hud_texture_debug_snapshot(&mut self) {
        self.debug_hud.clear_texture_debug_snapshot();
    }

    /// Composites the debug HUD with `LoadOp::Load` onto the swapchain in `encoder`.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
    ) -> Result<(), DebugHudEncodeError> {
        profiling::scope!("hud::encode");
        self.debug_hud
            .encode_overlay(device, queue, encoder, backbuffer, extent)
    }

    /// Mutable render-graph transient resource pool.
    pub(crate) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        &mut self.transient_pool
    }

    /// Synchronizes backend view-scoped resource ownership against the runtime's active view list.
    pub(crate) fn sync_active_views<I>(&mut self, active_views: I)
    where
        I: IntoIterator<Item = crate::render_graph::ViewId>,
    {
        let retired = self.view_resources.sync_active_views(active_views);
        if retired.is_empty() {
            return;
        }
        logger::debug!(
            "retiring {} inactive view-scoped resource sets",
            retired.len()
        );
        self.frame_graph_cache.release_view_resources(&retired);
        for view_id in retired {
            self.frame_resources.retire_view(view_id);
            self.history_registry.retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
    }

    /// Disjoint mutable borrows and attach-time snapshots for [`crate::render_graph::FrameRenderParams`].
    ///
    /// Centralizes the split so the graph executor can build per-view frame parameters without
    /// violating the borrow checker on field-by-field struct literals.
    pub(crate) fn split_for_graph_frame_params(&mut self) -> GraphFrameParamsSplit<'_> {
        let gpu_limits = self.gpu_limits().cloned();
        let msaa_depth_resolve = self.msaa_depth_resolve.clone();
        let per_view_hud_config = self.per_view_hud_config();
        (
            &self.occlusion,
            &self.frame_resources,
            &self.materials,
            &self.asset_transfers,
            self.mesh_preprocess.as_ref(),
            self.mesh_deform_scratch.as_mut(),
            self.skin_cache.as_mut(),
            gpu_limits,
            msaa_depth_resolve,
            per_view_hud_config,
        )
    }

    /// Scratch buffers for mesh deformation (`MeshDeformPass`).
    pub fn mesh_deform_scratch_mut(&mut self) -> Option<&mut MeshDeformScratch> {
        self.mesh_deform_scratch.as_mut()
    }

    /// Compute preprocess pipelines + deform scratch (`MeshDeformPass`) as one disjoint borrow.
    pub fn mesh_deform_pre_and_scratch(
        &mut self,
    ) -> Option<(&MeshPreprocessPipelines, &mut MeshDeformScratch)> {
        let pre = self.mesh_preprocess.as_ref()?;
        let scratch = self.mesh_deform_scratch.as_mut()?;
        Some((pre, scratch))
    }

    /// Preprocess pipelines, deform scratch, and GPU skin cache as one disjoint borrow for [`MeshDeformPass`].
    ///
    /// Bundles [`Self::mesh_preprocess`], [`Self::mesh_deform_scratch`], and [`Self::skin_cache`].
    pub fn mesh_deform_pre_scratch_and_skin_cache(
        &mut self,
    ) -> Option<(
        &MeshPreprocessPipelines,
        &mut MeshDeformScratch,
        &mut GpuSkinCache,
    )> {
        let pre = self.mesh_preprocess.as_ref()?;
        let scratch = self.mesh_deform_scratch.as_mut()?;
        let skin = self.skin_cache.as_mut()?;
        Some((pre, scratch, skin))
    }
}

#[cfg(test)]
mod post_processing_rebuild_tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::config::{RendererSettings, TonemapMode, TonemapSettings};
    use crate::render_graph::{post_processing::PostProcessChainSignature, GraphCacheKey};

    fn settings_handle(post: PostProcessingSettings) -> RendererSettingsHandle {
        Arc::new(RwLock::new(RendererSettings {
            post_processing: post,
            ..Default::default()
        }))
    }

    /// Returns the current cached graph key.
    fn cached_graph_key(backend: &RenderBackend) -> GraphCacheKey {
        backend
            .frame_graph_cache
            .last_key()
            .expect("graph key should exist after sync")
    }

    /// First sync builds the graph and stores the live signature.
    #[test]
    fn first_sync_builds_graph_and_records_signature() {
        let mut backend = RenderBackend::new();
        let handle = settings_handle(PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        });
        backend.renderer_settings = Some(handle);
        backend.ensure_frame_graph_in_sync(false);
        assert!(
            backend.frame_graph_pass_count() > 0,
            "graph should be built"
        );
        assert_eq!(
            cached_graph_key(&backend).post_processing,
            PostProcessChainSignature {
                aces_tonemap: true,
                bloom: true,
                bloom_max_mip_dimension: 512,
                gtao: true,
            }
        );
    }

    /// Toggling the master enable flips the signature and rebuilds the graph with an extra pass.
    #[test]
    fn signature_change_triggers_rebuild() {
        let mut backend = RenderBackend::new();
        let handle = settings_handle(PostProcessingSettings {
            enabled: false,
            ..Default::default()
        });
        backend.renderer_settings = Some(Arc::clone(&handle));
        backend.ensure_frame_graph_in_sync(false);
        let initial_passes = backend.frame_graph_pass_count();
        let initial_signature = cached_graph_key(&backend).post_processing;

        if let Ok(mut g) = handle.write() {
            g.post_processing.enabled = true;
            g.post_processing.tonemap.mode = TonemapMode::AcesFitted;
        }
        backend.ensure_frame_graph_in_sync(false);

        assert_ne!(
            cached_graph_key(&backend).post_processing,
            initial_signature,
            "signature must update after rebuild"
        );
        assert!(
            backend.frame_graph_pass_count() > initial_passes,
            "enabling ACES should add a graph pass"
        );
    }

    /// Repeat sync without HUD edits is a no-op (no rebuild, signature and pass count unchanged).
    #[test]
    fn unchanged_signature_does_not_rebuild() {
        let mut backend = RenderBackend::new();
        let handle = settings_handle(PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        });
        backend.renderer_settings = Some(handle);
        backend.ensure_frame_graph_in_sync(false);
        let signature = cached_graph_key(&backend).post_processing;
        let pass_count = backend.frame_graph_pass_count();

        backend.ensure_frame_graph_in_sync(false);
        assert_eq!(cached_graph_key(&backend).post_processing, signature);
        assert_eq!(backend.frame_graph_pass_count(), pass_count);
    }

    /// Switching between mono and stereo multiview should flip the graph key in one place so the
    /// runtime does not rely on implicit backend assumptions when VR starts or stops.
    #[test]
    fn multiview_change_updates_graph_key() {
        let mut backend = RenderBackend::new();
        backend.renderer_settings = Some(settings_handle(PostProcessingSettings::default()));

        backend.ensure_frame_graph_in_sync(false);
        let mono_key = cached_graph_key(&backend);
        backend.ensure_frame_graph_in_sync(true);
        let stereo_key = cached_graph_key(&backend);

        assert!(!mono_key.multiview_stereo);
        assert!(stereo_key.multiview_stereo);
        assert_ne!(mono_key, stereo_key);
    }
}
