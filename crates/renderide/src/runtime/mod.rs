//! Renderer façade: orchestrates **frontend** (IPC / shared memory / lock-step), **scene** (host
//! logical state), and **backend** (GPU pools, material store, uploads).
//!
//! Phase order aligns with the historical session loop: drain incoming commands first, then emit
//! [`FrameStartData`](crate::shared::FrameStartData) when lock-step allows (see `app` `tick_frame`).
//! Asset integration between begin-frame and frame processing remains a stub here.
//!
//! Lock-step is driven by the `last_frame_index` field of [`FrameStartData`](crate::shared::FrameStartData)
//! on the **outgoing** `frame_start_data` the renderer sends from [`RendererRuntime::pre_frame`].
//! If the host sends [`RendererCommand::frame_start_data`](crate::shared::RendererCommand::frame_start_data),
//! optional payloads are trace-logged until consumers exist.

mod commands;
mod frame_submit;
mod host_camera_apply;
mod ipc_init_dispatch;
mod lights_ipc;
mod lockstep;
mod shader_material_ipc;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::assets::texture::supported_host_formats_for_init;
use crate::assets::AssetSubsystem;
use crate::backend::RenderBackend;
use crate::config::RendererSettingsHandle;
use crate::connection::{ConnectionParams, InitError};
use crate::frontend::RendererFrontend;
use crate::gpu::GpuContext;
use crate::output_device::head_output_device_wants_openxr;

#[cfg(feature = "debug-hud")]
use crate::diagnostics::{DebugHudInput, HostHudGatherer};
use glam::{Mat4, Quat, Vec3};

use crate::render_graph::{ExternalFrameTargets, GraphExecuteError, HostCameraFrame};

pub use crate::frontend::InitState;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::SceneCoordinator;
use crate::shared::{
    FrameSubmitData, HeadOutputDevice, InputState, LightsBufferRendererSubmission,
    MaterialsUpdateBatch, OutputState, RendererCommand, RendererInitData, RendererInitResult,
    ShaderUnload, ShaderUpload,
};
use winit::window::Window;

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    frontend: RendererFrontend,
    backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`FrameSubmitData`](crate::shared::FrameSubmitData).
    pub scene: SceneCoordinator,
    assets: AssetSubsystem,
    /// Last host clip / FOV / VR / ortho task state for [`crate::render_graph::FrameRenderParams`].
    pub host_camera: HostCameraFrame,
    /// Process-wide renderer settings (shared with the debug HUD and the frame loop).
    settings: RendererSettingsHandle,
    /// Target path for persisting [`Self::settings`] from the ImGui config window.
    config_save_path: PathBuf,
    /// Throttled host CPU/RAM sampling for the debug HUD.
    #[cfg(feature = "debug-hud")]
    host_hud: HostHudGatherer,
    /// [`FrameSubmitData::render_tasks`] length from the last applied frame submit (HUD).
    #[cfg(feature = "debug-hud")]
    last_submit_render_task_count: usize,
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(
        params: Option<ConnectionParams>,
        settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) -> Self {
        Self {
            frontend: RendererFrontend::new(params),
            backend: RenderBackend::new(),
            scene: SceneCoordinator::new(),
            assets: AssetSubsystem::default(),
            host_camera: HostCameraFrame::default(),
            settings,
            config_save_path,
            #[cfg(feature = "debug-hud")]
            host_hud: HostHudGatherer::default(),
            #[cfg(feature = "debug-hud")]
            last_submit_render_task_count: 0,
        }
    }

    /// Shared settings store ([`crate::config::RendererSettings`]).
    pub fn settings(&self) -> &RendererSettingsHandle {
        &self.settings
    }

    /// Path written by the **Renderer config** ImGui window and [`crate::config::save_renderer_settings`].
    pub fn config_save_path(&self) -> &PathBuf {
        &self.config_save_path
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&crate::backend::mesh_deform::MeshPreprocessPipelines> {
        self.backend.mesh_preprocess()
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.frontend.connect_ipc()
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.frontend.is_ipc_connected()
    }

    pub fn init_state(&self) -> InitState {
        self.frontend.init_state()
    }

    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    pub fn last_frame_data_processed(&self) -> bool {
        self.frontend.last_frame_data_processed
    }

    /// Current lock-step frame index echoed to the host.
    pub fn last_frame_index(&self) -> i32 {
        self.frontend.last_frame_index
    }

    pub fn shutdown_requested(&self) -> bool {
        self.frontend.shutdown_requested
    }

    pub fn fatal_error(&self) -> bool {
        self.frontend.fatal_error
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &crate::resources::MeshPool {
        self.backend.mesh_pool()
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut crate::resources::MeshPool {
        self.backend.mesh_pool_mut()
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &crate::resources::TexturePool {
        self.backend.texture_pool()
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut crate::resources::TexturePool {
        self.backend.texture_pool_mut()
    }

    /// Exposes asset subsystem hooks (upload queues, handle table) for future workers.
    pub fn assets_mut(&mut self) -> &mut AssetSubsystem {
        &mut self.assets
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &crate::assets::material::MaterialPropertyStore {
        self.backend.material_property_store()
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(
        &mut self,
    ) -> &mut crate::assets::material::MaterialPropertyStore {
        self.backend.material_property_store_mut()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::assets::material::PropertyIdRegistry {
        self.backend.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.backend.material_registry()
    }

    /// Mutable registry (pipeline cache and shader routes).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.backend.material_registry_mut()
    }

    /// Host [`RendererInitData`] after connect, before [`Self::take_pending_init`] consumes it.
    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.frontend.pending_init()
    }

    /// Applies pending init once a GPU/window stack exists (e.g. window title).
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.frontend.take_pending_init()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    pub fn attach_gpu(&mut self, gpu: &GpuContext) {
        let device = gpu.device().clone();
        let queue = Arc::clone(gpu.queue());
        let shm = self.frontend.shared_memory_mut();
        self.backend.attach(
            device,
            queue,
            shm,
            gpu.config_format(),
            Arc::clone(&self.settings),
            self.config_save_path.clone(),
        );
    }

    #[cfg(feature = "debug-hud")]
    /// Per-frame pointer state and timing for the optional ImGui overlay ([`diagnostics::DebugHud`]).
    pub fn set_debug_hud_frame_data(&mut self, input: DebugHudInput, frame_time_ms: f64) {
        self.backend.set_debug_hud_frame_data(input, frame_time_ms);
    }

    #[cfg(feature = "debug-hud")]
    /// Last ImGui `want_capture_mouse` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.backend.debug_hud_last_want_capture_mouse()
    }

    #[cfg(feature = "debug-hud")]
    /// Last ImGui `want_capture_keyboard` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.backend.debug_hud_last_want_capture_keyboard()
    }

    /// Records and presents one frame via the backend’s compiled render graph.
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        #[cfg(feature = "debug-hud")]
        self.sync_debug_hud_diagnostics_from_settings();
        let scene_ref: &SceneCoordinator = &self.scene;
        self.backend
            .execute_frame_graph(gpu, window, scene_ref, self.host_camera)
    }

    /// Renders to OpenXR multiview array targets (see [`RenderBackend::execute_frame_graph_external_multiview`]).
    pub fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.run_frame_graph_external_multiview(gpu, window, external)
    }

    fn run_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        #[cfg(feature = "debug-hud")]
        self.sync_debug_hud_diagnostics_from_settings();
        let scene_ref: &SceneCoordinator = &self.scene;
        self.backend.execute_frame_graph_external_multiview(
            gpu,
            window,
            scene_ref,
            self.host_camera,
            external,
        )
    }

    #[cfg(feature = "debug-hud")]
    /// Copies [`crate::config::DebugSettings::debug_hud_enabled`] into the backend before the render graph runs.
    fn sync_debug_hud_diagnostics_from_settings(&mut self) {
        let main = self
            .settings
            .read()
            .map(|s| s.debug.debug_hud_enabled)
            .unwrap_or(false);
        self.backend.set_debug_hud_main_enabled(main);
    }

    /// Updates debug HUD snapshots after [`crate::gpu::GpuContext::end_frame_timing`] for the winit tick.
    #[cfg(feature = "debug-hud")]
    pub fn capture_debug_hud_after_frame_end(&mut self, gpu: &GpuContext) {
        let frame_timing = crate::diagnostics::FrameTimingHudSnapshot::capture(
            gpu,
            self.backend.debug_frame_time_ms(),
        );
        self.backend.set_debug_hud_frame_timing(frame_timing);

        let (main_hud, transforms_hud) = self
            .settings
            .read()
            .map(|s| (s.debug.debug_hud_enabled, s.debug.debug_hud_transforms))
            .unwrap_or((false, false));

        if main_hud {
            let host = self.host_hud.snapshot();
            let frame_diag = crate::diagnostics::FrameDiagnosticsSnapshot::capture(
                gpu,
                self.backend.debug_frame_time_ms(),
                host,
                self.last_submit_render_task_count,
                &self.backend,
            );
            let snapshot = crate::diagnostics::RendererInfoSnapshot::capture(
                self.is_ipc_connected(),
                self.init_state(),
                self.last_frame_index(),
                gpu.adapter_info(),
                gpu.config_format(),
                gpu.surface_extent_px(),
                gpu.present_mode(),
                self.backend.debug_frame_time_ms(),
                &self.scene,
                &self.backend,
            );
            self.backend.set_debug_hud_snapshot(snapshot);
            self.backend.set_debug_hud_frame_diagnostics(frame_diag);
        } else {
            self.backend.clear_debug_hud_stats_snapshots();
        }

        if transforms_hud {
            let scene_transforms =
                crate::diagnostics::SceneTransformsSnapshot::capture(&self.scene);
            self.backend
                .set_debug_hud_scene_transforms_snapshot(scene_transforms);
        } else {
            self.backend.clear_debug_hud_scene_transforms_snapshot();
        }
    }

    /// Whether the next tick should build [`InputState`] and call [`Self::pre_frame`].
    pub fn should_send_begin_frame(&self) -> bool {
        self.frontend.should_send_begin_frame()
    }

    /// Records wall-clock spacing for host FPS metrics. Call at the very start of each winit tick,
    /// before [`Self::poll_ipc`], OpenXR, and [`Self::pre_frame`].
    pub fn tick_frame_wall_clock_begin(&mut self, now: Instant) {
        self.frontend.on_tick_frame_wall_clock(now);
    }

    /// Completes wall-clock timing for the tick: stores elapsed from `frame_start` for the next
    /// [`crate::shared::PerformanceState::render_time`]. Call once before every return from
    /// [`crate::app::RenderideApp::tick_frame`].
    pub fn tick_frame_wall_clock_end(&mut self, frame_start: Instant) {
        self.frontend
            .set_perf_last_total_us(frame_start.elapsed().as_micros() as u64);
    }

    /// Host [`OutputState::lock_cursor`] bit merged into packed mouse state.
    pub fn host_cursor_lock_requested(&self) -> bool {
        self.frontend.host_cursor_lock_requested()
    }

    /// If connected and init is complete, sends [`FrameStartData`] when we are ready for the next host frame.
    pub fn pre_frame(&mut self, inputs: InputState) {
        self.frontend.pre_frame(inputs);
    }

    /// Drains pending host window policy after [`Self::poll_ipc`].
    pub fn take_pending_output_state(&mut self) -> Option<OutputState> {
        self.frontend.take_pending_output_state()
    }

    /// Last [`OutputState`] from the host (for per-frame cursor lock / warp).
    pub fn last_output_state(&self) -> Option<&OutputState> {
        self.frontend.last_output_state()
    }

    /// Placeholder for bounded asset integration between begin-frame and frame processing (Unity:
    /// `RunAssetIntegration`).
    pub fn run_asset_integration_stub(&mut self, _budget: Duration) {
        let _ = self.assets.drain_pending_meta();
    }

    /// Drains IPC and dispatches commands. Each poll batch is ordered so `renderer_init_data` runs
    /// first, then frame submits, then the rest (see [`RendererFrontend::poll_commands`]).
    pub fn poll_ipc(&mut self) {
        self.backend.begin_ipc_poll_mesh_upload_budget();
        let batch = self.frontend.poll_commands();
        for cmd in batch {
            ipc_init_dispatch::dispatch_ipc_command(self, cmd);
        }
        let (shm, ipc) = self.frontend.transport_pair_mut();
        if let Some(shm) = shm {
            self.backend.finish_ipc_poll_mesh_upload_deferred(shm, ipc);
        }
    }

    pub(super) fn on_init_data(&mut self, d: RendererInitData) {
        self.host_camera.output_device = d.output_device;
        if let Some(ref prefix) = d.shared_memory_prefix {
            self.frontend
                .set_shared_memory(SharedMemoryAccessor::new(prefix.clone()));
            logger::info!("Shared memory prefix: {}", prefix);
            let (shm, ipc) = self.frontend.transport_pair_mut();
            if let (Some(shm), Some(ipc)) = (shm, ipc) {
                self.backend.flush_pending_material_batches(shm, ipc);
            }
        }
        self.frontend.set_pending_init(d.clone());
        if let Some(ref mut ipc) = self.frontend.ipc_mut() {
            send_renderer_init_result(ipc, d.output_device);
        }
        self.frontend.on_init_received();
    }

    pub(super) fn handle_running_command(&mut self, cmd: RendererCommand) {
        commands::handle_running_command(self, cmd);
    }

    fn on_shader_upload(&mut self, upload: ShaderUpload) {
        shader_material_ipc::on_shader_upload(&mut self.frontend, &mut self.backend, upload);
    }

    fn on_shader_unload(&mut self, unload: ShaderUnload) {
        shader_material_ipc::on_shader_unload(&mut self.backend, unload);
    }

    fn on_materials_update_batch(&mut self, batch: MaterialsUpdateBatch) {
        shader_material_ipc::on_materials_update_batch(
            &mut self.frontend,
            &mut self.backend,
            batch,
        );
    }

    fn on_lights_buffer_renderer_submission(&mut self, sub: LightsBufferRendererSubmission) {
        let buffer_id = sub.lights_buffer_unique_id;
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            logger::warn!("lights_buffer_renderer_submission: no shared memory (id={buffer_id})");
            return;
        };
        lights_ipc::apply_lights_buffer_submission(&mut self.scene, shm, ipc, sub);
    }

    fn on_frame_submit(&mut self, data: FrameSubmitData) {
        let prev_frame_index = self.host_camera.frame_index;
        lockstep::trace_duplicate_frame_index_if_interesting(data.frame_index, prev_frame_index);
        frame_submit::process_frame_submit(self, data);
    }
}

impl crate::xr::XrHostCameraSync for RendererRuntime {
    fn near_clip(&self) -> f32 {
        self.host_camera.near_clip
    }

    fn far_clip(&self) -> f32 {
        self.host_camera.far_clip
    }

    fn output_device(&self) -> HeadOutputDevice {
        self.host_camera.output_device
    }

    fn vr_active(&self) -> bool {
        self.host_camera.vr_active
    }

    fn scene_root_scale_for_clip(&self) -> Option<Vec3> {
        self.scene
            .active_main_space()
            .map(|space| space.root_transform.scale)
    }

    fn world_from_tracking(&self, center_pose_tracking: Option<(Vec3, Quat)>) -> Mat4 {
        self.scene
            .active_main_space()
            .map(|space| {
                crate::xr::tracking_space_to_world_matrix(
                    &space.root_transform,
                    &space.view_transform,
                    space.override_view_position,
                    center_pose_tracking,
                )
            })
            .unwrap_or(Mat4::IDENTITY)
    }

    fn set_head_output_transform(&mut self, transform: Mat4) {
        self.host_camera.head_output_transform = transform;
    }

    fn set_stereo_view_proj(&mut self, vp: Option<(Mat4, Mat4)>) {
        self.host_camera.stereo_view_proj = vp;
    }

    fn set_stereo_views(&mut self, views: Option<(Mat4, Mat4)>) {
        self.host_camera.stereo_views = views;
    }
}

impl crate::xr::XrMultiviewFrameRenderer for RendererRuntime {
    fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: crate::render_graph::ExternalFrameTargets<'_>,
    ) -> Result<(), crate::render_graph::GraphExecuteError> {
        self.run_frame_graph_external_multiview(gpu, window, external)
    }
}

pub(super) fn send_renderer_init_result(
    ipc: &mut crate::ipc::DualQueueIpc,
    output_device: HeadOutputDevice,
) {
    let stereo = if head_output_device_wants_openxr(output_device) {
        "OpenXR(multiview)"
    } else {
        "None"
    };
    let result = RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some("Renderide 0.1.0 (wgpu skeleton)".to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some(stereo.to_string()),
        max_texture_size: 8192,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: supported_host_formats_for_init(),
    };
    ipc.send_primary(RendererCommand::renderer_init_result(result));
}
