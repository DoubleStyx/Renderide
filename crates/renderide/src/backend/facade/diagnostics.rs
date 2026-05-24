//! Diagnostics HUD ownership behind the backend facade.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::config::RendererSettingsHandle;
use crate::diagnostics::{
    DebugHudEncodeError, DebugHudInput, FrameDiagnosticsSnapshot, FrameTimingHudSnapshot,
    RendererInfoSnapshot, SceneTransformsSnapshot, TextureDebugSnapshot,
};
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

use super::super::debug_hud_bundle::DebugHudBundle;

/// Dear ImGui overlay state and diagnostics snapshots owned by the backend.
pub(super) struct BackendDiagnostics {
    debug_hud: DebugHudBundle,
}

impl BackendDiagnostics {
    /// Creates an inert diagnostics owner before GPU attach.
    pub(super) fn new() -> Self {
        Self {
            debug_hud: DebugHudBundle::new(),
        }
    }

    /// Creates GPU resources for the diagnostics overlay.
    pub(super) fn attach(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        renderer_settings: RendererSettingsHandle,
        config_save_path: PathBuf,
        suppress_renderer_config_disk_writes: bool,
    ) {
        self.debug_hud.attach(
            device,
            queue,
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        );
    }

    /// Mutable HUD bundle for graph execution access packets.
    pub(super) fn bundle_mut(&mut self) -> &mut DebugHudBundle {
        &mut self.debug_hud
    }

    /// Updates whether main HUD diagnostics run.
    pub(super) fn set_main_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_main_enabled(enabled);
    }

    /// Updates whether texture HUD diagnostics run.
    pub(super) fn set_textures_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_textures_enabled(enabled);
    }

    /// Clears the current-view Texture2D set.
    pub(super) fn clear_current_view_texture_2d_asset_ids(&mut self) {
        self.debug_hud.clear_current_view_texture_2d_asset_ids();
    }

    /// Texture2D ids used by submitted world draws for the current view.
    pub(super) fn current_view_texture_2d_asset_ids(&self) -> &BTreeSet<i32> {
        self.debug_hud.current_view_texture_2d_asset_ids()
    }

    /// Updates pointer state for the ImGui overlay.
    pub(super) fn set_input(&mut self, input: DebugHudInput) {
        self.debug_hud.set_input(input);
    }

    /// Updates the wall-clock roundtrip in milliseconds.
    pub(super) fn set_wall_frame_time_ms(&mut self, frame_time_ms: f64) {
        self.debug_hud.set_wall_frame_time_ms(frame_time_ms);
    }

    /// Last inter-frame time in milliseconds supplied by the app.
    pub(super) fn frame_time_ms(&self) -> f64 {
        self.debug_hud.frame_time_ms()
    }

    /// Last ImGui mouse capture flag.
    pub(super) fn last_want_capture_mouse(&self) -> bool {
        self.debug_hud.last_want_capture_mouse()
    }

    /// Last ImGui keyboard capture flag.
    pub(super) fn last_want_capture_keyboard(&self) -> bool {
        self.debug_hud.last_want_capture_keyboard()
    }

    /// Whether the HUD will draw visible content this frame.
    pub(super) fn has_visible_content(&self) -> bool {
        self.debug_hud.has_visible_content()
    }

    /// Clears cached input-capture state when HUD encoding is skipped.
    pub(super) fn clear_input_capture(&mut self) {
        self.debug_hud.clear_input_capture();
    }

    /// Stores renderer info for the next HUD frame.
    pub(super) fn set_snapshot(&mut self, snapshot: RendererInfoSnapshot) {
        self.debug_hud.set_snapshot(snapshot);
    }

    /// Stores frame diagnostics for the next HUD frame.
    pub(super) fn set_frame_diagnostics(&mut self, snapshot: FrameDiagnosticsSnapshot) {
        self.debug_hud.set_frame_diagnostics(snapshot);
    }

    /// Stores frame timing for the next HUD frame.
    pub(super) fn set_frame_timing(&mut self, snapshot: FrameTimingHudSnapshot) {
        self.debug_hud.set_frame_timing(snapshot);
    }

    /// Stores GPU pass timing rows and query stats for the next HUD frame.
    pub(super) fn set_gpu_profiler_snapshot(
        &mut self,
        snapshot: crate::profiling::GpuProfilerSnapshot,
    ) {
        self.debug_hud.set_gpu_profiler_snapshot(snapshot);
    }

    /// Clears stats and shader-route payloads.
    pub(super) fn clear_stats_snapshots(&mut self) {
        self.debug_hud.clear_stats_snapshots();
    }

    /// Last world-mesh draw stats captured by the HUD.
    pub(super) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.debug_hud.last_world_mesh_draw_stats()
    }

    /// Last world-mesh draw state rows captured by the HUD.
    pub(super) fn last_world_mesh_draw_state_rows(&self) -> Vec<WorldMeshDrawStateRow> {
        self.debug_hud.last_world_mesh_draw_state_rows()
    }

    /// Stores scene transforms for the next HUD frame.
    pub(super) fn set_scene_transforms_snapshot(&mut self, snapshot: SceneTransformsSnapshot) {
        self.debug_hud.set_scene_transforms_snapshot(snapshot);
    }

    /// Clears scene transform diagnostics.
    pub(super) fn clear_scene_transforms_snapshot(&mut self) {
        self.debug_hud.clear_scene_transforms_snapshot();
    }

    /// Stores texture diagnostics for the next HUD frame.
    pub(super) fn set_texture_debug_snapshot(&mut self, snapshot: TextureDebugSnapshot) {
        self.debug_hud.set_texture_debug_snapshot(snapshot);
    }

    /// Clears texture diagnostics.
    pub(super) fn clear_texture_debug_snapshot(&mut self) {
        self.debug_hud.clear_texture_debug_snapshot();
    }

    /// Composites the debug HUD with `LoadOp::Load`.
    pub(super) fn encode_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), DebugHudEncodeError> {
        profiling::scope!("hud::encode");
        self.debug_hud
            .encode_overlay(device, queue, encoder, backbuffer, extent, profiler)
    }
}
