//! [`RenderBackend`] implementation.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::assets::material::MaterialPropertyStore;
use crate::config::RendererSettingsHandle;
use crate::gpu::{GpuContext, MeshPreprocessPipelines};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
#[cfg(feature = "debug-hud")]
use crate::render_graph::WorldMeshDrawStats;
use crate::render_graph::{CompiledRenderGraph, ExternalFrameTargets, GraphExecuteError};
use crate::resources::{MeshPool, TexturePool};
use crate::scene::SceneCoordinator;

#[cfg(feature = "debug-hud")]
use crate::diagnostics::{DebugHud, DebugHudInput, SceneTransformsSnapshot};

use super::asset_transfer_queue::{self as asset_uploads, AssetTransferQueue};
use super::material_system::MaterialSystem;
use super::mesh_deform_scratch::MeshDeformScratch;
use super::occlusion::OcclusionSystem;
use crate::shared::{
    MaterialsUpdateBatch, MeshUnload, MeshUploadData, SetTexture2DData, SetTexture2DFormat,
    SetTexture2DProperties, UnloadTexture2D,
};
use winit::window::Window;

pub use super::asset_transfer_queue::{
    MAX_DEFERRED_MESH_UPLOADS, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
    MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL,
};

/// GPU resource pools, material property data, and asset upload paths.
pub struct RenderBackend {
    /// Material property store, shader routes, pipeline registry, embedded `@group(1)` binds.
    pub(crate) materials: MaterialSystem,
    /// Mesh/texture upload queues, budgets, format tables, pools, and GPU device/queue for uploads.
    pub(crate) asset_transfers: AssetTransferQueue,
    /// Optional mesh skinning / blendshape compute pipelines (after [`Self::attach`]).
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Compiled DAG of render passes (after [`Self::attach`]); see [`crate::render_graph`].
    frame_graph: Option<CompiledRenderGraph>,
    /// Scratch buffers for mesh deformation compute (after [`Self::attach`]).
    mesh_deform_scratch: Option<MeshDeformScratch>,
    /// Per-frame bind groups, light staging, and debug draw slab.
    pub(crate) frame_resources: super::FrameResourceManager,
    #[cfg(feature = "debug-hud")]
    debug_hud: Option<DebugHud>,
    #[cfg(feature = "debug-hud")]
    debug_hud_input: DebugHudInput,
    #[cfg(feature = "debug-hud")]
    debug_frame_time_ms: f64,
    /// Last [`imgui::Io::want_capture_mouse`] after a successful debug HUD encode (for next tick IPC).
    #[cfg(feature = "debug-hud")]
    debug_hud_want_capture_mouse: bool,
    /// Last [`imgui::Io::want_capture_keyboard`] after a successful debug HUD encode (for next tick IPC).
    #[cfg(feature = "debug-hud")]
    debug_hud_want_capture_keyboard: bool,
    /// Last [`WorldMeshDrawStats`] from [`crate::render_graph::passes::WorldMeshForwardPass`].
    #[cfg(feature = "debug-hud")]
    last_world_mesh_draw_stats: WorldMeshDrawStats,
    /// Mirrors [`crate::config::DebugSettings::debug_hud_enabled`] at frame graph start (mesh-draw stats for HUD).
    #[cfg(feature = "debug-hud")]
    debug_hud_main_enabled: bool,
    /// Hierarchical depth pyramid, CPU readback, and temporal cull state for occlusion culling.
    pub(crate) occlusion: OcclusionSystem,
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
            mesh_preprocess: None,
            frame_graph: None,
            mesh_deform_scratch: None,
            frame_resources: super::FrameResourceManager::new(),
            #[cfg(feature = "debug-hud")]
            debug_hud: None,
            #[cfg(feature = "debug-hud")]
            debug_hud_input: DebugHudInput::default(),
            #[cfg(feature = "debug-hud")]
            debug_frame_time_ms: 0.0,
            #[cfg(feature = "debug-hud")]
            debug_hud_want_capture_mouse: false,
            #[cfg(feature = "debug-hud")]
            debug_hud_want_capture_keyboard: false,
            #[cfg(feature = "debug-hud")]
            last_world_mesh_draw_stats: WorldMeshDrawStats::default(),
            #[cfg(feature = "debug-hud")]
            debug_hud_main_enabled: false,
            occlusion: OcclusionSystem::new(),
        }
    }

    /// Count of host Texture2D asset ids that have received a [`SetTexture2DFormat`] (CPU-side table).
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

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        &self.asset_transfers.mesh_pool
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.asset_transfers.mesh_pool
    }

    /// Resets the per-[`crate::runtime::RendererRuntime::poll_ipc`] budget for non-high-priority mesh uploads.
    pub fn begin_ipc_poll_mesh_upload_budget(&mut self) {
        asset_uploads::begin_ipc_poll_mesh_upload_budget(&mut self.asset_transfers);
    }

    /// Drains mesh and texture uploads deferred when the non-high-priority budget was exhausted mid-batch.
    pub fn finish_ipc_poll_mesh_upload_deferred(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        match ipc {
            Some(ipc_ref) => {
                asset_uploads::drain_deferred_mesh_uploads_after_poll(
                    &mut self.asset_transfers,
                    shm,
                    Some(ipc_ref),
                );
                asset_uploads::drain_deferred_texture_uploads_after_poll(
                    &mut self.asset_transfers,
                    shm,
                    Some(ipc_ref),
                );
            }
            None => {
                asset_uploads::drain_deferred_mesh_uploads_after_poll(
                    &mut self.asset_transfers,
                    shm,
                    None,
                );
                asset_uploads::drain_deferred_texture_uploads_after_poll(
                    &mut self.asset_transfers,
                    shm,
                    None,
                );
            }
        }
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        &self.asset_transfers.texture_pool
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

    /// Embedded material bind groups (world Unlit, etc.) after [`Self::attach`].
    pub fn embedded_material_bind(
        &self,
    ) -> Option<&super::embedded_material_bind::EmbeddedMaterialBindResources> {
        self.materials.embedded_material_bind()
    }

    /// Number of schedules passes in the compiled frame graph, or `0` if none.
    pub fn frame_graph_pass_count(&self) -> usize {
        self.frame_graph.as_ref().map_or(0, |g| g.pass_count())
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// `shm` is used to flush pending mesh/texture payloads that require shared-memory reads; omit
    /// when none is available yet (uploads stay queued).
    pub fn attach(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<Mutex<wgpu::Queue>>,
        shm: Option<&mut SharedMemoryAccessor>,
        surface_format: wgpu::TextureFormat,
        renderer_settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) {
        self.asset_transfers.gpu_device = Some(device.clone());
        self.asset_transfers.gpu_queue = Some(queue.clone());
        self.mesh_deform_scratch = Some(MeshDeformScratch::new(device.as_ref()));
        self.frame_resources.attach(device.as_ref());
        #[cfg(feature = "debug-hud")]
        {
            let q = queue.lock().unwrap_or_else(|e| e.into_inner());
            self.debug_hud = Some(DebugHud::new(
                device.as_ref(),
                &q,
                surface_format,
                renderer_settings,
                config_save_path,
            ));
        }
        #[cfg(not(feature = "debug-hud"))]
        let _ = (surface_format, renderer_settings, config_save_path);
        match MeshPreprocessPipelines::new(device.as_ref()) {
            Ok(p) => self.mesh_preprocess = Some(p),
            Err(e) => {
                logger::warn!("mesh preprocess compute pipelines not created: {e}");
                self.mesh_preprocess = None;
            }
        }
        self.materials.attach_gpu(device.clone(), &queue);
        asset_uploads::attach_flush_pending_asset_uploads(&mut self.asset_transfers, &device, shm);

        self.frame_graph = match crate::render_graph::build_default_main_graph() {
            Ok(g) => Some(g),
            Err(e) => {
                logger::warn!("default render graph build failed: {e}");
                None
            }
        };
    }

    /// Records and presents one frame using the compiled render graph (deform compute + forward mesh pass).
    ///
    /// Returns [`GraphExecuteError::NoFrameGraph`] if graph build failed during [`Self::attach`].
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        self.occlusion.hi_z_begin_frame_readback(gpu.device());
        let Some(mut graph) = self.frame_graph.take() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = graph.execute(gpu, window, scene, self, host_camera);
        self.frame_graph = Some(graph);
        res
    }

    /// Renders the frame graph to pre-acquired OpenXR multiview array targets (no surface present).
    pub fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.occlusion.hi_z_begin_frame_readback(gpu.device());
        let Some(mut graph) = self.frame_graph.take() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = graph.execute_external_multiview(gpu, window, scene, self, host_camera, external);
        self.frame_graph = Some(graph);
        res
    }

    #[cfg(feature = "debug-hud")]
    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_debug_hud_main_enabled(&mut self, enabled: bool) {
        self.debug_hud_main_enabled = enabled;
    }

    #[cfg(feature = "debug-hud")]
    /// Whether main debug HUD is on (mesh-draw stats for [`crate::render_graph::passes::WorldMeshForwardPass`]).
    pub(crate) fn debug_hud_main_enabled(&self) -> bool {
        self.debug_hud_main_enabled
    }

    #[cfg(feature = "debug-hud")]
    /// Updates pointer state and frame delta for the optional ImGui overlay.
    pub fn set_debug_hud_frame_data(&mut self, input: DebugHudInput, frame_time_ms: f64) {
        self.debug_hud_input = input;
        self.debug_frame_time_ms = frame_time_ms;
    }

    #[cfg(feature = "debug-hud")]
    /// Last inter-frame time in milliseconds supplied by the app for HUD FPS.
    pub(crate) fn debug_frame_time_ms(&self) -> f64 {
        self.debug_frame_time_ms
    }

    #[cfg(feature = "debug-hud")]
    /// [`imgui::Io::want_capture_mouse`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.debug_hud_want_capture_mouse
    }

    #[cfg(feature = "debug-hud")]
    /// [`imgui::Io::want_capture_keyboard`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.debug_hud_want_capture_keyboard
    }

    #[cfg(feature = "debug-hud")]
    /// Stores [`crate::diagnostics::RendererInfoSnapshot`] for the next HUD frame.
    pub(crate) fn set_debug_hud_snapshot(
        &mut self,
        snapshot: crate::diagnostics::RendererInfoSnapshot,
    ) {
        if let Some(hud) = self.debug_hud.as_mut() {
            hud.set_snapshot(snapshot);
        }
    }

    #[cfg(feature = "debug-hud")]
    pub(crate) fn set_debug_hud_frame_diagnostics(
        &mut self,
        snapshot: crate::diagnostics::FrameDiagnosticsSnapshot,
    ) {
        if let Some(hud) = self.debug_hud.as_mut() {
            hud.set_frame_diagnostics(snapshot);
        }
    }

    #[cfg(feature = "debug-hud")]
    pub(crate) fn set_debug_hud_frame_timing(
        &mut self,
        snapshot: crate::diagnostics::FrameTimingHudSnapshot,
    ) {
        if let Some(hud) = self.debug_hud.as_mut() {
            hud.set_frame_timing(snapshot);
        }
    }

    #[cfg(feature = "debug-hud")]
    /// Clears Stats / Shader routes payloads only (not frame timing or scene transforms).
    pub(crate) fn clear_debug_hud_stats_snapshots(&mut self) {
        if let Some(hud) = self.debug_hud.as_mut() {
            hud.clear_stats_hud_payloads();
        }
    }

    #[cfg(feature = "debug-hud")]
    /// Clears the **Scene transforms** HUD payload.
    pub(crate) fn clear_debug_hud_scene_transforms_snapshot(&mut self) {
        if let Some(hud) = self.debug_hud.as_mut() {
            hud.clear_scene_transforms_snapshot();
        }
    }

    #[cfg(feature = "debug-hud")]
    pub(crate) fn set_last_world_mesh_draw_stats(&mut self, stats: WorldMeshDrawStats) {
        self.last_world_mesh_draw_stats = stats;
    }

    #[cfg(feature = "debug-hud")]
    pub(crate) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.last_world_mesh_draw_stats
    }

    /// Updates the **Scene transforms** Dear ImGui window payload for the next composite pass.
    #[cfg(feature = "debug-hud")]
    pub(crate) fn set_debug_hud_scene_transforms_snapshot(
        &mut self,
        snapshot: SceneTransformsSnapshot,
    ) {
        if let Some(hud) = self.debug_hud.as_mut() {
            hud.set_scene_transforms_snapshot(snapshot);
        }
    }

    #[cfg(feature = "debug-hud")]
    /// Composites the debug HUD with `LoadOp::Load` onto the swapchain in `encoder`.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
    ) -> Result<(), String> {
        let Some(hud) = self.debug_hud.as_mut() else {
            return Ok(());
        };
        match hud.encode_overlay(
            device,
            queue,
            encoder,
            backbuffer,
            extent,
            &self.debug_hud_input,
        ) {
            Ok((want_mouse, want_keyboard)) => {
                self.debug_hud_want_capture_mouse = want_mouse;
                self.debug_hud_want_capture_keyboard = want_keyboard;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(not(feature = "debug-hud"))]
    /// No-op without `debug-hud`.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _backbuffer: &wgpu::TextureView,
        _extent: (u32, u32),
    ) -> Result<(), String> {
        Ok(())
    }

    /// Scratch buffers for mesh deformation (`MeshDeformPass`).
    pub fn mesh_deform_scratch_mut(&mut self) -> Option<&mut MeshDeformScratch> {
        self.mesh_deform_scratch.as_mut()
    }

    /// Compute preprocess pipelines + deform scratch (`MeshDeformPass`) as one disjoint borrow.
    pub fn mesh_deform_pre_and_scratch(
        &mut self,
    ) -> Option<(&crate::gpu::MeshPreprocessPipelines, &mut MeshDeformScratch)> {
        let pre = self.mesh_preprocess.as_ref()?;
        let scratch = self.mesh_deform_scratch.as_mut()?;
        Some((pre, scratch))
    }

    /// Maps shader asset to raster pipeline kind and optional HUD display name, or defers until [`Self::attach`].
    pub fn register_shader_route(
        &mut self,
        asset_id: i32,
        pipeline: RasterPipelineKind,
        display_name: Option<String>,
    ) {
        self.materials
            .register_shader_route(asset_id, pipeline, display_name);
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.materials.unregister_shader_route(asset_id);
    }

    /// Drain pending material batches using the given shared memory and IPC.
    pub fn flush_pending_material_batches(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        self.materials.flush_pending_material_batches(shm, ipc);
    }

    /// Queue a materials batch when shared memory is not yet available. Returns `false` if queue full.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) -> bool {
        self.materials.enqueue_materials_batch_no_shm(batch)
    }

    /// Apply one host materials batch (shared memory must be valid for the batch descriptors).
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        self.materials.apply_materials_update_batch(batch, shm, ipc);
    }

    /// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
    pub fn on_set_texture_2d_format(
        &mut self,
        f: SetTexture2DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
    pub fn on_set_texture_2d_properties(
        &mut self,
        p: SetTexture2DProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_properties(&mut self.asset_transfers, p, ipc);
    }

    /// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
    /// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
    pub fn on_set_texture_2d_data(
        &mut self,
        d: SetTexture2DData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_data(&mut self.asset_transfers, d, shm, ipc);
    }

    /// Upload texture mips from shared memory and optionally notify the host on the background queue.
    ///
    /// `consume_texture_upload_budget` should be `true` for normal IPC handling; use `false` when
    /// draining deferred uploads or replaying pending uploads on GPU attach.
    pub fn try_texture_upload_with_device(
        &mut self,
        data: SetTexture2DData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
        consume_texture_upload_budget: bool,
    ) {
        asset_uploads::try_texture_upload_with_device(
            &mut self.asset_transfers,
            data,
            shm,
            ipc,
            consume_texture_upload_budget,
        );
    }

    /// Remove a texture asset from CPU tables and the pool.
    pub fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
        asset_uploads::on_unload_texture_2d(&mut self.asset_transfers, u);
    }

    /// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
    pub fn try_process_mesh_upload(
        &mut self,
        data: MeshUploadData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::try_process_mesh_upload(&mut self.asset_transfers, data, shm, ipc);
    }

    /// Remove a mesh from the pool.
    pub fn on_mesh_unload(&mut self, u: MeshUnload) {
        asset_uploads::on_mesh_unload(&mut self.asset_transfers, u);
    }

    /// Remove material / property-block entries from the host store.
    pub fn on_unload_material(&mut self, asset_id: i32) {
        self.materials.on_unload_material(asset_id);
    }

    /// Remove a property block from the host store.
    pub fn on_unload_material_property_block(&mut self, asset_id: i32) {
        self.materials.on_unload_material_property_block(asset_id);
    }
}
