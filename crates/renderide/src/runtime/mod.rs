//! Renderer façade: orchestrates **frontend** (IPC / shared memory / lock-step), **scene** (host
//! logical state), and **backend** (GPU pools, material store, uploads).
//!
//! Phase order aligns with the historical session loop: drain incoming commands first, then emit
//! [`FrameStartData`](crate::shared::FrameStartData) when lock-step allows (see `app` `tick_frame`).
//! [`RendererRuntime::run_asset_integration`] time-slices cooperative mesh/texture uploads ([`crate::assets::asset_transfer_queue::drain_asset_tasks`]).
//!
//! Lock-step is driven by the `last_frame_index` field of [`FrameStartData`](crate::shared::FrameStartData)
//! on the **outgoing** `frame_start_data` the renderer sends from [`RendererRuntime::pre_frame`].
//! If the host sends [`RendererCommand::FrameStartData`](crate::shared::RendererCommand::FrameStartData),
//! optional payloads are trace-logged until consumers exist.

mod accessors;
mod command_dispatch;
mod commands;
mod debug_hud_frame;
mod frame_submit;
mod host_camera_apply;
mod ipc_init_dispatch;
mod lights_ipc;
mod lockstep;
mod renderer_command_kind;
mod secondary_cameras;
mod shader_material_ipc;
mod xr_impls;

use hashbrown::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::backend::RenderBackend;
use crate::config::RendererSettingsHandle;
use crate::connection::ConnectionParams;
use crate::frontend::RendererFrontend;
use crate::gpu::GpuContext;

use crate::render_graph::{ExternalFrameTargets, GraphExecuteError, HostCameraFrame};

pub use crate::frontend::InitState;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::SceneCoordinator;
use crate::shared::{
    FrameSubmitData, InputState, LightsBufferRendererSubmission, MaterialsUpdateBatch, OutputState,
    RendererCommand, RendererInitData, ShaderUnload, ShaderUpload,
};

/// Result of one [`RendererRuntime::tick_one_frame`] call.
///
/// `shutdown_requested` lets the calling driver exit its event loop; `fatal_error` triggers a
/// non-zero process exit. `graph_error` carries any failure from [`RendererRuntime::render_all_views`]
/// for the caller to decide whether to log + continue or escalate.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// Host requested an orderly shutdown via IPC during this tick.
    pub shutdown_requested: bool,
    /// IPC reported a fatal error during this tick (e.g. init dispatch protocol violation).
    pub fatal_error: bool,
    /// Render-graph execution error for this tick, if any.
    pub graph_error: Option<GraphExecuteError>,
}

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    frontend: RendererFrontend,
    backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`FrameSubmitData`](crate::shared::FrameSubmitData).
    pub scene: SceneCoordinator,
    /// Last host clip / FOV / VR / ortho task state for [`crate::render_graph::FrameRenderParams`].
    pub host_camera: HostCameraFrame,
    /// Process-wide renderer settings (shared with the debug HUD and the frame loop).
    settings: RendererSettingsHandle,
    /// Target path for persisting [`Self::settings`] from the ImGui config window.
    config_save_path: PathBuf,
    /// Throttled host CPU/RAM sampling for the debug HUD.
    host_hud: crate::diagnostics::HostHudGatherer,
    /// Rolling per-frame wall time history that feeds the Frame timing sparkline.
    frame_time_history: crate::diagnostics::FrameTimeHistory,
    /// [`FrameSubmitData::render_tasks`] length from the last applied frame submit (HUD).
    last_submit_render_task_count: usize,
    /// Cached full [`wgpu::AllocatorReport`] for the **GPU memory** HUD tab (refreshed on a timer).
    allocator_report_hud: Option<crate::diagnostics::GpuAllocatorReportHud>,
    /// Wall clock when a **GPU memory** tab refresh was last attempted (typically every 2s while the main debug HUD runs).
    allocator_report_last_refresh: Option<Instant>,
    /// Set when [`Self::run_asset_integration`] completed for the current winit tick (cleared in [`Self::tick_frame_wall_clock_begin`]).
    did_integrate_this_tick: bool,
    /// Count of failed [`SceneCoordinator::apply_frame_submit`] or [`SceneCoordinator::flush_world_caches`] after a host submit (HUD / drift).
    frame_submit_apply_failures: u64,
    /// Count of OpenXR `wait_frame` errors since startup (recoverable).
    xr_wait_frame_failures: u64,
    /// Count of OpenXR `locate_views` errors when `should_render` was true (recoverable).
    xr_locate_views_failures: u64,
    /// Running counts of post-init [`RendererCommand`] variants seen without a running handler.
    unhandled_ipc_command_counts: HashMap<&'static str, u64>,
    /// When `true`, ImGui and [`crate::config::save_renderer_settings_from_load`] must not overwrite `config.toml`.
    suppress_renderer_config_disk_writes: bool,
}

impl RendererRuntime {
    /// Drops transient-pool GPU textures for free-list entries whose MSAA sample count no longer
    /// matches the effective swapchain tier (avoids VRAM retention when toggling MSAA).
    pub(super) fn transient_evict_stale_msaa_tiers_if_changed(
        &mut self,
        prev_effective: u32,
        new_effective: u32,
    ) {
        if prev_effective == new_effective {
            return;
        }
        let eff = new_effective.max(1);
        self.backend
            .transient_pool_mut()
            .evict_texture_keys_where(|k| k.sample_count > 1 && k.sample_count != eff);
    }

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
            host_camera: HostCameraFrame::default(),
            settings,
            config_save_path,
            host_hud: crate::diagnostics::HostHudGatherer::default(),
            frame_time_history: crate::diagnostics::FrameTimeHistory::new(),
            last_submit_render_task_count: 0,
            allocator_report_hud: None,
            allocator_report_last_refresh: None,
            did_integrate_this_tick: false,
            frame_submit_apply_failures: 0,
            xr_wait_frame_failures: 0,
            xr_locate_views_failures: 0,
            unhandled_ipc_command_counts: HashMap::new(),
            suppress_renderer_config_disk_writes: false,
        }
    }

    /// Disables writing `config.toml` from the HUD when load-time Figment extraction failed.
    pub fn set_suppress_renderer_config_disk_writes(&mut self, value: bool) {
        self.suppress_renderer_config_disk_writes = value;
    }

    /// Whether disk persistence of renderer settings is blocked (bad on-disk config at startup).
    pub fn suppress_renderer_config_disk_writes(&self) -> bool {
        self.suppress_renderer_config_disk_writes
    }

    /// Total number of post-handshake IPC commands logged as unhandled (sum of per-variant counters).
    pub fn unhandled_ipc_command_event_total(&self) -> u64 {
        self.unhandled_ipc_command_counts.values().copied().sum()
    }

    pub(super) fn record_unhandled_renderer_command(&mut self, tag: &'static str) {
        *self.unhandled_ipc_command_counts.entry(tag).or_insert(0) += 1;
    }

    /// Records and presents one frame via the backend’s compiled render graph.
    pub fn execute_frame_graph(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();
        let requested_msaa = self
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        let prev_msaa = gpu.swapchain_msaa_effective();
        gpu.set_swapchain_msaa_requested(requested_msaa);
        self.transient_evict_stale_msaa_tiers_if_changed(prev_msaa, gpu.swapchain_msaa_effective());
        let scene_ref: &SceneCoordinator = &self.scene;
        self.backend
            .execute_frame_graph(gpu, scene_ref, self.host_camera)
    }

    /// Renders to OpenXR multiview array targets (see [`RenderBackend::execute_frame_graph_external_multiview`]).
    pub fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        external: ExternalFrameTargets<'_>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        self.run_frame_graph_external_multiview(gpu, external, skip_hi_z_begin_readback)
    }

    pub(super) fn run_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        external: ExternalFrameTargets<'_>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();
        let requested_msaa = self
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        let prev_stereo = gpu.swapchain_msaa_effective_stereo();
        gpu.set_swapchain_msaa_requested_stereo(requested_msaa);
        self.transient_evict_stale_msaa_tiers_if_changed(
            prev_stereo,
            gpu.swapchain_msaa_effective_stereo(),
        );
        let scene_ref: &SceneCoordinator = &self.scene;
        self.backend.execute_frame_graph_external_multiview(
            gpu,
            scene_ref,
            self.host_camera,
            external,
            skip_hi_z_begin_readback,
        )
    }

    /// Drains completed Hi-Z `map_async` readbacks into CPU snapshots (once per tick).
    ///
    /// Call at the top of the render-views phase so both the HMD and desktop paths share one drain.
    pub fn drain_hi_z_readback(&mut self, device: &wgpu::Device) {
        self.backend.occlusion.hi_z_begin_frame_readback(device);
    }

    /// Whether the next tick should build [`InputState`] and call [`Self::pre_frame`].
    pub fn should_send_begin_frame(&self) -> bool {
        self.frontend.should_send_begin_frame()
    }

    /// Records wall-clock spacing for host FPS metrics. Call at the very start of each winit tick,
    /// before [`Self::poll_ipc`], OpenXR, and [`Self::pre_frame`].
    pub fn tick_frame_wall_clock_begin(&mut self, now: Instant) {
        self.did_integrate_this_tick = false;
        self.frontend.reset_ipc_outbound_drop_tick_flags();
        self.backend.reset_light_prep_for_tick();
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

    /// If connected and init is complete, sends [`FrameStartData`](crate::shared::FrameStartData) when we are ready for the next host frame.
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

    /// Bounded cooperative mesh/texture asset integration (Unity `RunAssetIntegration`–style).
    /// Uses [`crate::config::RenderingSettings::asset_integration_budget_ms`] for the wall-clock slice.
    ///
    /// At most once per winit tick: a second call in the same tick is a no-op ([`Self::did_integrate_this_tick`]).
    pub fn run_asset_integration(&mut self) {
        if self.did_integrate_this_tick {
            return;
        }
        let budget_ms = self
            .settings
            .read()
            .map(|s| s.rendering.asset_integration_budget_ms)
            .unwrap_or(3);
        let budget_ms = budget_ms.max(1);
        let deadline = Instant::now() + Duration::from_millis(u64::from(budget_ms));
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            return;
        };
        let mut ipc_opt = ipc;
        self.backend.drain_asset_tasks(shm, &mut ipc_opt, deadline);
        self.did_integrate_this_tick = true;
    }

    /// Whether [`Self::run_asset_integration`] already ran this tick.
    pub fn did_integrate_assets_this_tick(&self) -> bool {
        self.did_integrate_this_tick
    }

    /// Drains IPC and dispatches commands. Each poll batch is ordered so `renderer_init_data` runs
    /// first, then frame submits, then the rest (see [`RendererFrontend::poll_commands`]).
    pub fn poll_ipc(&mut self) {
        let mut batch = self.frontend.poll_commands();
        for cmd in batch.drain(..) {
            ipc_init_dispatch::dispatch_ipc_command(self, cmd);
        }
        self.frontend.recycle_command_batch(batch);
    }

    /// Runs the canonical per-frame phase order shared between the winit-driven
    /// [`crate::app::RenderideApp::tick_frame`] (non-VR) and the headless interval driver
    /// [`crate::app::headless::run_headless`].
    ///
    /// Phases: drain IPC, dispatch asset integration, emit lock-step `FrameStartData` via
    /// [`Self::pre_frame`] (when allowed), and call [`Self::render_all_views`]. Mode-specific
    /// epilogue (HUD overlay encode + present in winit, PNG readback in headless) happens on
    /// the caller side after this returns.
    pub fn tick_one_frame(
        &mut self,
        gpu: &mut GpuContext,
        inputs: crate::shared::InputState,
    ) -> TickOutcome {
        self.poll_ipc();
        if self.shutdown_requested() {
            return TickOutcome {
                shutdown_requested: true,
                ..Default::default()
            };
        }
        if self.fatal_error() {
            return TickOutcome {
                fatal_error: true,
                ..Default::default()
            };
        }
        self.run_asset_integration();
        if self.should_send_begin_frame() {
            self.pre_frame(inputs);
        }
        let graph_error = self.render_all_views(gpu).err();
        TickOutcome {
            graph_error,
            ..Default::default()
        }
    }

    /// Same as [`Self::tick_one_frame`] but skips the render call.
    ///
    /// Used by the desktop VR path which runs its own HMD multiview submit + secondary cameras
    /// to render textures + mirror blit instead of [`Self::render_all_views`]. Phase order stays
    /// in this method so VR cannot drift from desktop / headless lock-step semantics.
    pub fn tick_one_frame_lockstep_only(
        &mut self,
        inputs: crate::shared::InputState,
    ) -> TickOutcome {
        self.poll_ipc();
        if self.shutdown_requested() {
            return TickOutcome {
                shutdown_requested: true,
                ..Default::default()
            };
        }
        if self.fatal_error() {
            return TickOutcome {
                fatal_error: true,
                ..Default::default()
            };
        }
        self.run_asset_integration();
        if self.should_send_begin_frame() {
            self.pre_frame(inputs);
        }
        TickOutcome::default()
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
            let settings = self.settings.read().map(|g| g.clone()).unwrap_or_default();
            if !ipc_init_dispatch::send_renderer_init_result(ipc, d.output_device, &settings, None)
            {
                logger::error!(
                    "IPC: RendererInitResult was not sent (primary queue full); stopping init handshake"
                );
                self.frontend.set_fatal_error(true);
                return;
            }
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

#[cfg(test)]
mod orchestration_tests;
