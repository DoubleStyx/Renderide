//! [`super::super::RenderBackend`] methods that forward to the Dear ImGui debug HUD diagnostics.
//!
//! These are thin delegates to [`super::diagnostics::BackendDiagnostics`]; they were split out
//! of `facade.rs` so the core facade only carries struct definition, attach, and render-graph
//! orchestration.

use crate::diagnostics::{DebugHudInput, PerViewHudConfig, SceneTransformsSnapshot};
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

use super::super::RenderBackend;

impl RenderBackend {
    /// Updates per-view HUD diagnostics capture interests for the next render graph.
    pub(crate) fn set_debug_hud_per_view_config(&mut self, config: PerViewHudConfig) {
        self.diagnostics.set_per_view_config(config);
    }

    /// Clears the current-view Texture2D set before collecting this frame's submitted draws.
    pub(crate) fn clear_debug_hud_current_view_texture_2d_asset_ids(&mut self) {
        self.diagnostics.clear_current_view_texture_2d_asset_ids();
    }

    /// Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn debug_hud_current_view_texture_2d_asset_ids(
        &self,
    ) -> &std::collections::BTreeSet<i32> {
        self.diagnostics.current_view_texture_2d_asset_ids()
    }

    /// Updates pointer state for the ImGui overlay (called once per render_views).
    pub fn set_debug_hud_input(&mut self, input: DebugHudInput) {
        self.diagnostics.set_input(input);
    }

    /// Updates the wall-clock roundtrip (ms) for the HUD's FPS / Frame readout.
    pub fn set_debug_hud_wall_frame_time_ms(&mut self, frame_time_ms: f64) {
        self.diagnostics.set_wall_frame_time_ms(frame_time_ms);
    }

    /// Last inter-frame time in milliseconds supplied by the app for HUD FPS.
    pub(crate) fn debug_frame_time_ms(&self) -> f64 {
        self.diagnostics.frame_time_ms()
    }

    /// [`imgui::Io::want_capture_mouse`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.diagnostics.last_want_capture_mouse()
    }

    /// [`imgui::Io::want_capture_keyboard`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.diagnostics.last_want_capture_keyboard()
    }

    /// Whether the HUD will draw visible content this frame.
    pub(crate) fn debug_hud_has_visible_content(&self) -> bool {
        self.diagnostics.has_visible_content()
    }

    /// Clears cached input-capture state when HUD encoding is skipped.
    pub(crate) fn clear_debug_hud_input_capture(&mut self) {
        self.diagnostics.clear_input_capture();
    }

    /// Stores [`crate::diagnostics::RendererInfoSnapshot`] for the next HUD frame.
    pub(crate) fn set_debug_hud_snapshot(
        &mut self,
        snapshot: crate::diagnostics::RendererInfoSnapshot,
    ) {
        self.diagnostics.set_snapshot(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_diagnostics(
        &mut self,
        snapshot: crate::diagnostics::FrameDiagnosticsSnapshot,
    ) {
        self.diagnostics.set_frame_diagnostics(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_timing(
        &mut self,
        snapshot: crate::diagnostics::FrameTimingHudSnapshot,
    ) {
        self.diagnostics.set_frame_timing(snapshot);
    }

    /// Clears the **Frame timing** HUD payload.
    pub(crate) fn clear_debug_hud_frame_timing(&mut self) {
        self.diagnostics.clear_frame_timing();
    }

    /// Pushes the latest GPU profiler snapshot into the debug HUD's **GPU passes** tab.
    pub(crate) fn set_debug_hud_gpu_profiler_snapshot(
        &mut self,
        snapshot: crate::profiling::GpuProfilerSnapshot,
    ) {
        self.diagnostics.set_gpu_profiler_snapshot(snapshot);
    }

    /// Clears the **GPU passes** HUD payload.
    pub(crate) fn clear_debug_hud_gpu_profiler_snapshot(&mut self) {
        self.diagnostics.clear_gpu_profiler_snapshot();
    }

    /// Clears Stats / Shader routes payloads only (not frame timing or scene transforms).
    pub(crate) fn clear_debug_hud_stats_snapshots(&mut self) {
        self.diagnostics.clear_stats_snapshots();
    }

    pub(crate) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.diagnostics.last_world_mesh_draw_stats()
    }

    pub(crate) fn last_world_mesh_draw_state_rows(&self) -> Vec<WorldMeshDrawStateRow> {
        self.diagnostics.last_world_mesh_draw_state_rows()
    }

    /// Updates the **Scene transforms** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_scene_transforms_snapshot(
        &mut self,
        snapshot: SceneTransformsSnapshot,
    ) {
        self.diagnostics.set_scene_transforms_snapshot(snapshot);
    }

    /// Clears the **Scene transforms** HUD payload.
    pub(crate) fn clear_debug_hud_scene_transforms_snapshot(&mut self) {
        self.diagnostics.clear_scene_transforms_snapshot();
    }

    /// Updates the **Textures** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_texture_debug_snapshot(
        &mut self,
        snapshot: crate::diagnostics::TextureDebugSnapshot,
    ) {
        self.diagnostics.set_texture_debug_snapshot(snapshot);
    }

    /// Clears the **Textures** HUD payload.
    pub(crate) fn clear_debug_hud_texture_debug_snapshot(&mut self) {
        self.diagnostics.clear_texture_debug_snapshot();
    }

    /// Composites the debug HUD with `LoadOp::Load` onto the swapchain in `encoder`.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) -> Result<(), crate::diagnostics::DebugHudEncodeError> {
        self.diagnostics
            .encode_overlay(device, queue, encoder, backbuffer, extent, profiler)
    }
}
