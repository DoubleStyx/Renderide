//! Dear ImGui overlay state and per-frame capture flags.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::config::RendererSettingsHandle;
use crate::diagnostics::{
    DebugHud, DebugHudEncodeError, DebugHudInput, FrameDiagnosticsSnapshot, FrameTimingHudSnapshot,
    PerViewHudOutputs, RendererInfoSnapshot, SceneTransformsSnapshot, TextureDebugSnapshot,
};
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

/// ImGui overlay, input/timing state, and mesh-draw stats for the diagnostics HUD.
pub struct DebugHudBundle {
    hud: Option<DebugHud>,
    input: DebugHudInput,
    frame_time_ms: f64,
    want_capture_mouse: bool,
    want_capture_keyboard: bool,
    last_world_mesh_draw_stats: WorldMeshDrawStats,
    last_world_mesh_draw_state_rows: Vec<WorldMeshDrawStateRow>,
    main_enabled: bool,
    textures_enabled: bool,
    current_view_texture_2d_asset_ids: BTreeSet<i32>,
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
            last_world_mesh_draw_state_rows: Vec::new(),
            main_enabled: false,
            textures_enabled: false,
            current_view_texture_2d_asset_ids: BTreeSet::new(),
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
        suppress_renderer_config_disk_writes: bool,
    ) {
        self.hud = Some(DebugHud::new(
            device,
            queue,
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        ));
    }

    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_main_enabled(&mut self, enabled: bool) {
        self.main_enabled = enabled;
    }

    /// Whether main debug HUD is on (mesh-draw stats for [`crate::passes::WorldMeshForwardOpaquePass`]).
    pub(crate) fn main_enabled(&self) -> bool {
        self.main_enabled
    }

    /// Updates whether texture HUD diagnostics run.
    pub fn set_textures_enabled(&mut self, enabled: bool) {
        self.textures_enabled = enabled;
    }

    /// Whether texture debug HUD capture is on.
    pub(crate) fn textures_enabled(&self) -> bool {
        self.textures_enabled
    }

    /// Clears the current-view Texture2D id set before collecting this frame's submitted draws.
    pub(crate) fn clear_current_view_texture_2d_asset_ids(&mut self) {
        self.current_view_texture_2d_asset_ids.clear();
    }

    /// Adds Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn note_current_view_texture_2d_asset_ids(
        &mut self,
        asset_ids: impl IntoIterator<Item = i32>,
    ) {
        self.current_view_texture_2d_asset_ids
            .extend(asset_ids.into_iter().filter(|id| *id >= 0));
    }

    /// Merges one view's deferred HUD payload into the bundle on the main thread.
    pub(crate) fn apply_per_view_outputs(&mut self, outputs: &PerViewHudOutputs) {
        if let Some(stats) = outputs.world_mesh_draw_stats {
            self.set_last_world_mesh_draw_stats(stats);
        }
        if let Some(rows) = outputs.world_mesh_draw_state_rows.clone() {
            self.set_last_world_mesh_draw_state_rows(rows);
        }
        if !outputs.current_view_texture_2d_asset_ids.is_empty() {
            self.note_current_view_texture_2d_asset_ids(
                outputs.current_view_texture_2d_asset_ids.iter().copied(),
            );
        }
    }

    /// Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn current_view_texture_2d_asset_ids(&self) -> &BTreeSet<i32> {
        &self.current_view_texture_2d_asset_ids
    }

    /// Updates pointer state for the optional ImGui overlay (called once per render_views).
    pub fn set_input(&mut self, input: DebugHudInput) {
        self.input = input;
    }

    /// Updates the wall-clock roundtrip (ms) for the HUD's FPS / Frame readout.
    ///
    /// Set in the tick prologue from the delta between consecutive `tick_frame` starts so the
    /// value cleanly reflects the roundtrip period rather than a sub-tick window.
    pub fn set_wall_frame_time_ms(&mut self, frame_time_ms: f64) {
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

    /// Forwards the latest flattened GPU pass timings to the wrapped HUD.
    pub(crate) fn set_gpu_pass_timings(&mut self, timings: Vec<crate::profiling::GpuPassEntry>) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_gpu_pass_timings(timings);
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

    pub(crate) fn set_last_world_mesh_draw_state_rows(&mut self, rows: Vec<WorldMeshDrawStateRow>) {
        self.last_world_mesh_draw_state_rows = rows;
    }

    pub(crate) fn last_world_mesh_draw_state_rows(&self) -> Vec<WorldMeshDrawStateRow> {
        self.last_world_mesh_draw_state_rows.clone()
    }

    pub(crate) fn set_scene_transforms_snapshot(&mut self, snapshot: SceneTransformsSnapshot) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_scene_transforms_snapshot(snapshot);
        }
    }

    pub(crate) fn set_texture_debug_snapshot(&mut self, snapshot: TextureDebugSnapshot) {
        if let Some(hud) = self.hud.as_mut() {
            hud.set_texture_debug_snapshot(snapshot);
        }
    }

    /// Clears the **Textures** HUD payload.
    pub(crate) fn clear_texture_debug_snapshot(&mut self) {
        if let Some(hud) = self.hud.as_mut() {
            hud.clear_texture_debug_snapshot();
        }
    }

    /// Returns `true` when the HUD is mounted and will draw at least one window this frame.
    ///
    /// Used by the render-graph executor to short-circuit HUD encoder creation entirely (no
    /// command encoder, no GPU profiler scope, no submitted command buffer) when nothing visible
    /// would render. When this returns `false`, callers must clear input-capture state so stale
    /// `want_capture_*` flags from a previously visible HUD do not leak into input dispatch.
    pub(crate) fn has_visible_content(&self) -> bool {
        self.hud
            .as_ref()
            .is_some_and(|hud| hud.has_visible_content())
    }

    /// Forces input-capture flags to `false`; called when the HUD encoder is skipped so the rest
    /// of the runtime correctly routes input to the world while no HUD window is visible.
    pub(crate) fn clear_input_capture(&mut self) {
        self.want_capture_mouse = false;
        self.want_capture_keyboard = false;
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
