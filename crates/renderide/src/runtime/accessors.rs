//! Thin [`super::RendererRuntime`] accessors and forwards to the frontend, backend, and settings.

use crate::config::RendererSettingsHandle;
use crate::connection::InitError;
use crate::diagnostics::DebugHudInput;
use crate::frontend::InitState;
use crate::gpu::GpuContext;
use crate::shared::RendererInitData;

use super::RendererRuntime;

impl RendererRuntime {
    /// Updates the latest [`crate::shared::FrameSubmitData::render_tasks`] count for the HUD.
    pub(crate) fn set_last_submit_render_task_count(&mut self, n: usize) {
        self.diagnostics.set_last_submit_render_task_count(n);
    }

    /// Updates the current pending camera readback count surfaced on the diagnostics HUD.
    pub(crate) fn set_pending_camera_readbacks(&mut self, n: usize) {
        self.diagnostics.set_pending_camera_readbacks(n);
    }

    /// Adds camera readback completions/failures to cumulative diagnostics counters.
    pub(crate) fn note_camera_readback_results(&mut self, completed: u64, failed: u64) {
        self.diagnostics
            .note_camera_readback_results(completed, failed);
    }

    /// Increments the cumulative scene-apply failure counter surfaced on the diagnostics HUD.
    pub(crate) fn note_frame_submit_apply_failure(&mut self) {
        self.diagnostics.note_frame_submit_apply_failure();
    }

    /// Number of host camera readback tasks waiting for GPU processing.
    #[cfg(test)]
    pub(crate) fn pending_camera_render_task_count(&self) -> usize {
        self.tick_state.pending_camera_render_tasks.len()
    }

    /// Number of host reflection-probe bake tasks waiting for GPU processing.
    #[cfg(test)]
    pub(crate) fn pending_reflection_probe_render_task_count(&self) -> usize {
        self.tick_state.pending_reflection_probe_render_tasks.len()
    }

    /// Disables writing `config.toml` from the HUD when load-time Figment extraction failed.
    pub fn set_suppress_renderer_config_disk_writes(&mut self, value: bool) {
        self.config.set_suppress_renderer_config_disk_writes(value);
    }

    /// Shared settings store ([`crate::config::RendererSettings`]).
    pub fn settings(&self) -> &RendererSettingsHandle {
        &self.config.settings
    }

    /// Effective desktop foreground/background FPS caps for the winit app driver.
    pub(crate) fn desktop_frame_pacing_caps(&self) -> crate::runtime::DesktopFramePacingCaps {
        self.config.desktop_frame_pacing_caps()
    }

    /// Effective host-owned skin weight mode for mesh skinning.
    pub(crate) fn skin_weight_mode(&self) -> crate::shared::SkinWeightMode {
        self.config.skin_weight_mode()
    }

    /// Toggles the master ImGui overlay visibility setting and clears stale HUD input capture.
    pub fn toggle_imgui_visibility(&mut self) {
        if self.config.toggle_imgui_visibility().is_some() {
            self.backend.clear_debug_hud_input_capture();
        }
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.frontend.connect_ipc()
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.frontend.is_ipc_connected()
    }

    /// Host/renderer init handshake phase (see [`crate::frontend::RendererFrontend::init_state`]).
    pub fn init_state(&self) -> InitState {
        self.frontend.init_state()
    }

    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    #[cfg(test)]
    pub fn last_frame_data_processed(&self) -> bool {
        self.frontend.last_frame_data_processed()
    }

    /// Whether the host has enabled regular lockstep with `RendererEngineReady`.
    #[cfg(test)]
    pub fn host_lockstep_activated(&self) -> bool {
        self.frontend.host_lockstep_activated()
    }

    /// Whether an applied host frame still needs a renderer-side draw attempt.
    #[cfg(test)]
    pub fn pending_frame_submit_render(&self) -> bool {
        self.frontend.pending_frame_submit_render()
    }

    /// Current lock-step frame index echoed to the host.
    pub fn last_frame_index(&self) -> i32 {
        self.frontend.last_frame_index()
    }

    /// Host requested an orderly renderer shutdown over IPC.
    pub fn shutdown_requested(&self) -> bool {
        self.frontend.shutdown_requested()
    }

    /// Unrecoverable IPC/init error; begin-frame is suppressed until reset.
    pub fn fatal_error(&self) -> bool {
        self.frontend.fatal_error()
    }

    /// Whether the host last reported VR mode as active (see [`crate::camera::HostCameraFrame::vr_active`]).
    pub fn vr_active(&self) -> bool {
        self.host_camera.vr_active
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
    ///
    /// On attach failure, an error is logged; CPU-side work may continue but GPU rendering paths remain
    /// unconfigured until a successful attach.
    pub fn attach_gpu(&mut self, gpu: &GpuContext) {
        use std::sync::Arc;

        let device = gpu.device().clone();
        let queue = Arc::clone(gpu.queue());
        let gpu_queue_access_gate = gpu.gpu_queue_access_gate().clone();
        let renderer_settings = Arc::clone(&self.config.settings);
        let config_save_path = self.config.cloned_config_save_path();
        let suppress_renderer_config_disk_writes =
            self.config.suppress_renderer_config_disk_writes();
        self.backend.set_skin_weight_mode(self.skin_weight_mode());
        let (shm, ipc) = self.frontend.transport_pair_mut();
        if let Err(e) = self.backend.attach(
            crate::backend::RenderBackendAttachDesc {
                device,
                queue,
                gpu_queue_access_gate,
                gpu_limits: Arc::clone(gpu.limits()),
                mapped_buffer_health: gpu.mapped_buffer_health(),
                surface_format: gpu.config_format(),
                renderer_settings,
                config_save_path,
                suppress_renderer_config_disk_writes,
                headless: gpu.is_headless(),
            },
            shm,
            ipc,
        ) {
            logger::error!("GPU attach failed: {e}; CPU work continues, GPU draws disabled");
        }
    }

    /// Per-frame pointer state for the ImGui overlay ([`diagnostics::DebugHud`]).
    pub fn set_debug_hud_input(&mut self, input: DebugHudInput) {
        self.backend.set_debug_hud_input(input);
    }

    /// Wall-clock roundtrip (ms) between consecutive `tick_frame` starts; drives the HUD's
    /// FPS / Frame readout. Set in the tick prologue so the value cleanly reflects the
    /// roundtrip period.
    pub fn set_debug_hud_wall_frame_time_ms(&mut self, frame_time_ms: f64) {
        self.backend.set_debug_hud_wall_frame_time_ms(frame_time_ms);
    }

    /// Last ImGui `want_capture_mouse` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.backend.debug_hud_last_want_capture_mouse()
    }

    /// Last ImGui `want_capture_keyboard` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.backend.debug_hud_last_want_capture_keyboard()
    }

    /// Read-only scene state used by present-time queries (e.g. active `BlitToDisplay` lookup).
    pub fn scene(&self) -> &crate::scene::SceneCoordinator {
        &self.scene
    }

    /// Resolves a `BlitToDisplay.texture_id` into a sampleable 2D color view + texel size.
    ///
    /// Resolves only the texture kinds that can currently be sampled as fullscreen 2D blit
    /// sources. `Texture2D` and `RenderTexture` cover the common camera-preview and avatar-preview
    /// cases. `Texture3D` and `Cubemap` have no canonical 2D slice; `VideoTexture` and `Desktop`
    /// are returned as [`None`] until their pools expose stable 2D view dimensions.
    pub fn resolve_blit_to_display_texture(
        &self,
        packed_texture_id: i32,
    ) -> Option<(std::sync::Arc<wgpu::TextureView>, u32, u32)> {
        use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};
        let (asset_id, kind) = unpack_host_texture_packed(packed_texture_id)?;
        match kind {
            HostTextureAssetKind::Texture2D => {
                let asset = self.backend.texture_pool().get(asset_id)?;
                Some((
                    std::sync::Arc::clone(&asset.view),
                    asset.width,
                    asset.height,
                ))
            }
            HostTextureAssetKind::RenderTexture => {
                let asset = self.backend.render_texture_pool().get(asset_id)?;
                Some((
                    std::sync::Arc::clone(&asset.color_view),
                    asset.width,
                    asset.height,
                ))
            }
            HostTextureAssetKind::Texture3D
            | HostTextureAssetKind::Cubemap
            | HostTextureAssetKind::VideoTexture
            | HostTextureAssetKind::Desktop => None,
        }
    }
}

#[cfg(test)]
impl RendererRuntime {
    /// Installs a shared-memory accessor for tests that apply [`crate::shared::FrameSubmitData`].
    pub(crate) fn test_set_shared_memory(&mut self, prefix: impl Into<String>) {
        use crate::ipc::SharedMemoryAccessor;
        self.frontend
            .set_shared_memory(SharedMemoryAccessor::new(prefix.into()));
    }
}
