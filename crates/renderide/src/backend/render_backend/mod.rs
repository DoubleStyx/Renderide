//! [`RenderBackend`] implementation.

mod uploads;

pub use uploads::{
    MAX_DEFERRED_MESH_UPLOADS, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
    MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL,
};

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::assets::material::{
    parse_materials_update_batch_into_store, MaterialPropertyStore, ParseMaterialBatchOptions,
    PropertyIdRegistry,
};
use crate::config::RendererSettingsHandle;
use crate::gpu::hi_z_build::{encode_hi_z_build, HiZGpuState};
use crate::gpu::{GpuContext, MeshPreprocessPipelines};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
#[cfg(feature = "debug-hud")]
use crate::render_graph::WorldMeshDrawStats;
use crate::render_graph::{
    capture_hi_z_temporal, HiZCullData, HiZTemporalState, OutputDepthMode, WorldMeshCullProjParams,
};
use crate::render_graph::{CompiledRenderGraph, ExternalFrameTargets, GraphExecuteError};
use crate::resources::{MeshPool, TexturePool};
use crate::scene::SceneCoordinator;

#[cfg(feature = "debug-hud")]
use crate::diagnostics::{DebugHud, DebugHudInput, SceneTransformsSnapshot};

use super::debug_draw::DebugDrawResources;
use super::embedded_material_bind::EmbeddedMaterialBindResources;
use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::light_gpu::{order_lights_for_clustered_shading, GpuLight};
use super::mesh_deform_scratch::MeshDeformScratch;
use crate::shared::{
    MaterialsUpdateBatch, MaterialsUpdateBatchResult, MeshUnload, MeshUploadData, RendererCommand,
    SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties, UnloadTexture2D,
};
use winit::window::Window;

/// Max queued [`MaterialsUpdateBatch`] when shared memory is not available.
pub const MAX_PENDING_MATERIAL_BATCHES: usize = 256;

/// GPU resource pools, material property data, and asset upload paths.
pub struct RenderBackend {
    /// Host material property batches (`MaterialsUpdateBatch`); separate maps for materials vs blocks.
    material_property_store: MaterialPropertyStore,
    /// Stable ids for [`crate::shared::MaterialPropertyIdRequest`] / batch `property_id` keys.
    property_id_registry: Arc<PropertyIdRegistry>,
    pending_material_batches: VecDeque<MaterialsUpdateBatch>,
    pub(crate) mesh_pool: MeshPool,
    texture_pool: TexturePool,
    /// Latest [`SetTexture2DFormat`] per asset (required before data upload).
    pub(super) texture_formats: HashMap<i32, SetTexture2DFormat>,
    /// Latest [`SetTexture2DProperties`] per asset (sampler metadata on [`GpuTexture2d`]).
    pub(super) texture_properties: HashMap<i32, SetTexture2DProperties>,
    /// Bound wgpu device after [`Self::attach`]; used by mesh/texture upload paths.
    pub(super) gpu_device: Option<Arc<wgpu::Device>>,
    /// Submission queue paired with [`Self::gpu_device`].
    pub(super) gpu_queue: Option<Arc<Mutex<wgpu::Queue>>>,
    /// Mesh payloads waiting for GPU or shared memory (drained on [`Self::attach`]).
    pub(super) pending_mesh_uploads: VecDeque<MeshUploadData>,
    /// Low-priority mesh uploads deferred when [`Self::mesh_upload_budget_this_poll`] is exhausted.
    pub(super) deferred_mesh_uploads: VecDeque<MeshUploadData>,
    /// Remaining non-high-priority mesh uploads allowed this [`crate::runtime::RendererRuntime::poll_ipc`] cycle.
    pub(super) mesh_upload_budget_this_poll: u32,
    /// Texture mip payloads waiting for GPU allocation or shared memory.
    pub(super) pending_texture_uploads: VecDeque<SetTexture2DData>,
    /// GPU material families, router, and pipeline cache (after [`Self::attach`]).
    pub(crate) material_registry: Option<crate::materials::MaterialRegistry>,
    /// Shader asset id → pipeline kind and optional HUD label when uploads arrive before GPU attach.
    pending_shader_routes: HashMap<i32, (RasterPipelineKind, Option<String>)>,
    /// Optional mesh skinning / blendshape compute pipelines (after [`Self::attach`]).
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Compiled DAG of render passes (after [`Self::attach`]); see [`crate::render_graph`].
    frame_graph: Option<CompiledRenderGraph>,
    /// Last packed lights for the frame (after [`Self::prepare_lights_from_scene`]).
    light_scratch: Vec<GpuLight>,
    /// Scratch buffers for mesh deformation compute (after [`Self::attach`]).
    mesh_deform_scratch: Option<MeshDeformScratch>,
    /// Per-frame `@group(0)` camera + lights (after [`Self::attach`]).
    pub(crate) frame_gpu: Option<FrameGpuResources>,
    /// Placeholder `@group(1)` for materials without per-material bindings.
    pub(crate) empty_material: Option<EmptyMaterialBindGroup>,
    /// Embedded raster materials (`@group(1)` textures/uniforms), after [`Self::attach`].
    pub(crate) embedded_material_bind: Option<EmbeddedMaterialBindResources>,
    /// Uniforms + bind group for debug mesh draws (`@group(2)` dynamic slab).
    pub(crate) debug_draw: Option<DebugDrawResources>,
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
    /// Hierarchical depth pyramid GPU/CPU state for occlusion culling (previous-frame readback).
    hi_z_gpu: HiZGpuState,
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
            material_property_store: MaterialPropertyStore::new(),
            property_id_registry: Arc::new(PropertyIdRegistry::new()),
            pending_material_batches: VecDeque::new(),
            mesh_pool: MeshPool::default_pool(),
            texture_pool: TexturePool::default_pool(),
            texture_formats: HashMap::new(),
            texture_properties: HashMap::new(),
            gpu_device: None,
            gpu_queue: None,
            pending_mesh_uploads: VecDeque::new(),
            deferred_mesh_uploads: VecDeque::new(),
            mesh_upload_budget_this_poll: uploads::MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL,
            pending_texture_uploads: VecDeque::new(),
            material_registry: None,
            pending_shader_routes: HashMap::new(),
            mesh_preprocess: None,
            frame_graph: None,
            light_scratch: Vec::new(),
            mesh_deform_scratch: None,
            frame_gpu: None,
            empty_material: None,
            embedded_material_bind: None,
            debug_draw: None,
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
            hi_z_gpu: HiZGpuState::default(),
        }
    }

    /// Count of host Texture2D asset ids that have received a [`SetTexture2DFormat`] (CPU-side table).
    pub fn texture_format_registration_count(&self) -> usize {
        self.texture_formats.len()
    }

    /// Count of GPU-resident textures with `mip_levels_resident > 0` (at least mip0 uploaded).
    pub fn texture_mip0_ready_count(&self) -> usize {
        self.texture_pool
            .textures()
            .values()
            .filter(|t| t.mip_levels_resident > 0)
            .count()
    }

    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Per-frame `@group(0)` bind group (camera + lights), after [`Self::attach`].
    pub fn frame_gpu(&self) -> Option<&FrameGpuResources> {
        self.frame_gpu.as_ref()
    }

    /// Mutable frame globals (cluster resize, uniform upload).
    pub fn frame_gpu_mut(&mut self) -> Option<&mut FrameGpuResources> {
        self.frame_gpu.as_mut()
    }

    /// Empty `@group(1)` bind group for shaders without per-material bindings.
    pub fn empty_material(&self) -> Option<&EmptyMaterialBindGroup> {
        self.empty_material.as_ref()
    }

    /// Cloned [`Arc`] bind groups for mesh forward (`@group(0)` frame + `@group(1)` empty material).
    ///
    /// Used when the pass also needs `&mut` access to other backend fields (avoids borrow conflicts).
    pub fn mesh_forward_frame_bind_groups(
        &self,
    ) -> Option<(
        std::sync::Arc<wgpu::BindGroup>,
        std::sync::Arc<wgpu::BindGroup>,
    )> {
        let f = self.frame_gpu.as_ref()?;
        let e = self.empty_material.as_ref()?;
        Some((f.bind_group.clone(), e.bind_group.clone()))
    }

    /// Fills [`Self::light_scratch`] from [`SceneCoordinator`] (all spaces, clustered ordering, cap [`crate::backend::MAX_LIGHTS`]).
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        self.light_scratch.clear();
        let mut all = Vec::new();
        for id in scene.render_space_ids() {
            all.extend(scene.resolve_lights_world(id));
        }
        let ordered = order_lights_for_clustered_shading(&all);
        self.light_scratch
            .extend(ordered.iter().map(GpuLight::from_resolved));
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        self.mesh_preprocess.as_ref()
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        &self.mesh_pool
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.mesh_pool
    }

    /// Resets the per-[`crate::runtime::RendererRuntime::poll_ipc`] budget for non-high-priority mesh uploads.
    pub fn begin_ipc_poll_mesh_upload_budget(&mut self) {
        uploads::begin_ipc_poll_mesh_upload_budget(self);
    }

    /// Drains mesh uploads deferred when the non-high-priority budget was exhausted mid-batch.
    pub fn finish_ipc_poll_mesh_upload_deferred(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        uploads::drain_deferred_mesh_uploads_after_poll(self, shm, ipc);
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        &self.texture_pool
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut TexturePool {
        &mut self.texture_pool
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

    /// Embedded material bind groups (world Unlit, etc.) after [`Self::attach`].
    pub fn embedded_material_bind(&self) -> Option<&EmbeddedMaterialBindResources> {
        self.embedded_material_bind.as_ref()
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
        self.gpu_device = Some(device.clone());
        self.gpu_queue = Some(queue.clone());
        self.mesh_deform_scratch = Some(MeshDeformScratch::new(device.as_ref()));
        self.frame_gpu = Some(FrameGpuResources::new(device.as_ref()));
        self.empty_material = Some(EmptyMaterialBindGroup::new(device.as_ref()));
        self.debug_draw = Some(DebugDrawResources::new(device.as_ref()));
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
        self.material_registry = Some(crate::materials::MaterialRegistry::with_default_families(
            device.clone(),
        ));
        if let Some(reg) = self.material_registry.as_mut() {
            for (asset_id, (pipeline, display_name)) in self.pending_shader_routes.drain() {
                reg.map_shader_route(asset_id, pipeline, display_name);
            }
        }
        match EmbeddedMaterialBindResources::new(
            device.clone(),
            Arc::clone(&self.property_id_registry),
        ) {
            Ok(m) => {
                if let Ok(q) = queue.lock() {
                    m.write_default_white(&q);
                }
                self.embedded_material_bind = Some(m);
            }
            Err(e) => {
                logger::warn!("embedded material bind resources not created: {e}");
                self.embedded_material_bind = None;
            }
        }
        uploads::attach_flush_pending_asset_uploads(self, &device, shm);

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
        let Some(mut graph) = self.frame_graph.take() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = graph.execute_external_multiview(gpu, window, scene, self, host_camera, external);
        self.frame_graph = Some(graph);
        res
    }

    /// Hi-Z occlusion data cloned from the **previous** frame’s pyramid readback, matching `mode`.
    pub(crate) fn hi_z_cull_data(&self, mode: OutputDepthMode) -> Option<HiZCullData> {
        match mode {
            OutputDepthMode::DesktopSingle => self
                .hi_z_gpu
                .desktop
                .as_ref()
                .map(|s| HiZCullData::Desktop(s.clone())),
            OutputDepthMode::StereoArray { .. } => {
                self.hi_z_gpu.stereo.as_ref().map(|s| HiZCullData::Stereo {
                    left: s.left.clone(),
                    right: s.right.clone(),
                })
            }
        }
    }

    /// Records Hi-Z GPU work into `encoder` (staging copy included).
    pub(crate) fn encode_hi_z_build_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        extent: (u32, u32),
        mode: OutputDepthMode,
    ) {
        encode_hi_z_build(
            device,
            queue,
            encoder,
            depth_view,
            extent,
            mode,
            &mut self.hi_z_gpu,
        );
    }

    /// Maps Hi-Z staging buffers after the queue submission for this frame has completed.
    pub(crate) fn hi_z_complete_pending_readback(&mut self, device: &wgpu::Device) {
        self.hi_z_gpu.complete_pending_readback(device);
    }

    /// View/projection snapshot from the **previous** world forward pass (for Hi-Z occlusion tests).
    pub(crate) fn hi_z_temporal_snapshot(&self) -> Option<HiZTemporalState> {
        self.hi_z_gpu.temporal.clone()
    }

    /// Records per-space views and cull params from **this** frame for Hi-Z tests on the **next** frame.
    pub(crate) fn capture_hi_z_temporal_for_next_frame(
        &mut self,
        scene: &SceneCoordinator,
        prev_cull: WorldMeshCullProjParams,
        viewport_px: (u32, u32),
    ) {
        self.hi_z_gpu.temporal = Some(capture_hi_z_temporal(scene, prev_cull, viewport_px));
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

    /// Per-draw debug mesh uniforms: 256-byte dynamic uniform slab ([`DebugDrawResources`]).
    pub fn debug_draw(&self) -> Option<&DebugDrawResources> {
        self.debug_draw.as_ref()
    }

    /// Maps shader asset to raster pipeline kind and optional HUD display name, or defers until [`Self::attach`].
    pub fn register_shader_route(
        &mut self,
        asset_id: i32,
        pipeline: RasterPipelineKind,
        display_name: Option<String>,
    ) {
        if let Some(reg) = self.material_registry.as_mut() {
            reg.map_shader_route(asset_id, pipeline, display_name);
        } else {
            self.pending_shader_routes
                .insert(asset_id, (pipeline, display_name));
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
        ipc.send_background(RendererCommand::materials_update_batch_result(
            MaterialsUpdateBatchResult { update_batch_id },
        ));
    }

    /// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
    pub fn on_set_texture_2d_format(
        &mut self,
        f: SetTexture2DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        uploads::on_set_texture_2d_format(self, f, ipc);
    }

    /// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
    pub fn on_set_texture_2d_properties(
        &mut self,
        p: SetTexture2DProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        uploads::on_set_texture_2d_properties(self, p, ipc);
    }

    /// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
    /// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
    pub fn on_set_texture_2d_data(
        &mut self,
        d: SetTexture2DData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        uploads::on_set_texture_2d_data(self, d, shm, ipc);
    }

    /// Upload texture mips from shared memory and optionally notify the host on the background queue.
    pub fn try_texture_upload_with_device(
        &mut self,
        data: SetTexture2DData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        uploads::try_texture_upload_with_device(self, data, shm, ipc);
    }

    /// Remove a texture asset from CPU tables and the pool.
    pub fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
        uploads::on_unload_texture_2d(self, u);
    }

    /// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
    pub fn try_process_mesh_upload(
        &mut self,
        data: MeshUploadData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        uploads::try_process_mesh_upload(self, data, shm, ipc);
    }

    /// Remove a mesh from the pool.
    pub fn on_mesh_unload(&mut self, u: MeshUnload) {
        uploads::on_mesh_unload(self, u);
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
