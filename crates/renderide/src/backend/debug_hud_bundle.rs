//! Dear ImGui overlay state and per-frame capture flags.

use std::path::PathBuf;

use crate::config::RendererSettingsHandle;
use crate::diagnostics::{
    DebugHud, DebugHudEncodeError, DebugHudInput, FrameDiagnosticsSnapshot, FrameTimingHudSnapshot,
    RendererInfoSnapshot, SceneTransformsSnapshot,
};
use crate::render_graph::WorldMeshDrawStats;

/// ImGui overlay, input/timing state, and mesh-draw stats for the diagnostics HUD.
pub struct DebugHudBundle {
    hud: Option<DebugHud>,
    input: DebugHudInput,
    frame_time_ms: f64,
    want_capture_mouse: bool,
    want_capture_keyboard: bool,
    last_world_mesh_draw_stats: WorldMeshDrawStats,
    main_enabled: bool,
}

impl Default for DebugHudBundle {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugHudBundle {
    /// Inert bundle until [`Self::attach`].
    pub fn new() -> Self {
        Self {
            hud: None,
            input: DebugHudInput::default(),
            frame_time_ms: 0.0,
            want_capture_mouse: false,
            want_capture_keyboard: false,
            last_world_mesh_draw_stats: WorldMeshDrawStats::default(),
            main_enabled: false,
        }
    }

    /// Creates GPU resources for the overlay.
    pub fn attach(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        renderer_settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) {
        self.hud = Some(DebugHud::new(
            device,
            queue,
            surface_format,
            renderer_settings,
            config_save_path,
        ));
    }

    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_main_enabled(&mut self, enabled: bool) {
        self.main_enabled = enabled;
    }

    /// Whether main debug HUD is on (mesh-draw stats for [`crate::render_graph::passes::WorldMeshForwardPass`]).
    pub(crate) fn main_enabled(&self) -> bool {
        self.main_enabled
    }

    /// Updates pointer state and frame delta for the optional ImGui overlay.
    pub fn set_frame_data(&mut self, input: DebugHudInput, frame_time_ms: f64) {
        self.input = input;
        self.frame_time_ms = frame_time_ms;
    }

    /// Last inter-frame time in milliseconds supplied by the app for HUD FPS.
    pub(crate) fn frame_time_ms(&self) -> f64 {
        self.frame_time_ms
    }

    /// [`imgui::Io::want_capture_mouse`] from the last successful HUD encode.
    pub(crate) fn last_want_capture_mouse(&self) -> bool {
        self.want_capture_mouse
    }

    /// [`imgui::Io::want_capture_keyboard`] from the last successful HUD encode.
    pub(crate) fn last_want_capture_keyboard(&self) -> bool {
        self.want_capture_keyboard
    }

    /// Stores [`RendererInfoSnapshot`] for the next HUD frame.
    pub(crate) fn set_snapshot(&mut self, snapshot: RendererInfoSnapshot) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_snapshot(snapshot);
        }
    }

    pub(crate) fn set_frame_diagnostics(&mut self, snapshot: FrameDiagnosticsSnapshot) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_frame_diagnostics(snapshot);
        }
    }

    pub(crate) fn set_frame_timing(&mut self, snapshot: FrameTimingHudSnapshot) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_frame_timing(snapshot);
        }
    }

    /// Clears Stats / Shader routes payloads only (not frame timing or scene transforms).
    pub(crate) fn clear_stats_snapshots(&mut self) {
        if let Some(hud) = self.hud.as_mut() {
            hud.clear_stats_hud_payloads();
        }
    }

    /// Clears the **Scene transforms** HUD payload.
    pub(crate) fn clear_scene_transforms_snapshot(&mut self) {
        if let Some(hud) = self.hud.as_mut() {
            hud.clear_scene_transforms_snapshot();
        }
    }

    pub(crate) fn set_last_world_mesh_draw_stats(&mut self, stats: WorldMeshDrawStats) {
        self.last_world_mesh_draw_stats = stats;
    }

    pub(crate) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.last_world_mesh_draw_stats
    }

    pub(crate) fn set_scene_transforms_snapshot(&mut self, snapshot: SceneTransformsSnapshot) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_scene_transforms_snapshot(snapshot);
        }
    }

    /// Composites the debug HUD with `LoadOp::Load` onto the swapchain in `encoder`.
    pub(crate) fn encode_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
    ) -> Result<(), DebugHudEncodeError> {
        let Some(hud) = self.hud.as_mut() else {
            return Ok(());
        };
        match hud.encode_overlay(device, queue, encoder, backbuffer, extent, &self.input) {
            Ok((want_mouse, want_keyboard)) => {
                self.want_capture_mouse = want_mouse;
                self.want_capture_keyboard = want_keyboard;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}
