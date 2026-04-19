//! ImGui window bodies for [`super::DebugHud`].

mod draw_state_tab;
mod frame_timing;
mod gpu_memory_tab;
mod labels;
mod renderer_config;
mod renderer_config_window;
mod scene_transforms;
mod shader_routes_tab;
mod stats_tab;
mod texture_debug;

use imgui::{Io, MouseButton as ImGuiMouseButton};

use super::DebugHud;

/// Feeds winit-derived [`crate::diagnostics::DebugHudInput`] into ImGui `io` before each frame.
pub(super) fn apply_input(io: &mut Io, input: &crate::diagnostics::DebugHudInput) {
    if input.mouse_active && input.window_focused {
        io.add_mouse_pos_event(input.cursor_px);
    } else {
        io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
    }
    io.add_mouse_button_event(ImGuiMouseButton::Left, input.left);
    io.add_mouse_button_event(ImGuiMouseButton::Right, input.right);
    io.add_mouse_button_event(ImGuiMouseButton::Middle, input.middle);
    io.add_mouse_button_event(ImGuiMouseButton::Extra1, input.extra1);
    io.add_mouse_button_event(ImGuiMouseButton::Extra2, input.extra2);
    const WHEEL_UNIT: f32 = 120.0;
    io.add_mouse_wheel_event([
        input.mouse_wheel_delta.x / WHEEL_UNIT,
        input.mouse_wheel_delta.y / WHEEL_UNIT,
    ]);
}

impl DebugHud {
    /// Wall-clock **FPS**, frame interval, and CPU/GPU submit splits use the same definitions as
    /// [`FrameDiagnosticsSnapshot`] (see **Frame timing** window).
    pub(super) fn frame_timing_window(
        ui: &imgui::Ui,
        timing: Option<&crate::diagnostics::FrameTimingHudSnapshot>,
    ) {
        frame_timing::frame_timing_window(ui, timing);
    }

    /// Unified IPC, adapter, scene, draws, and resources (FPS / submit timing: **Frame timing** window).
    pub(super) fn main_debug_panel(
        ui: &imgui::Ui,
        renderer: Option<&crate::diagnostics::RendererInfoSnapshot>,
        frame: Option<&crate::diagnostics::FrameDiagnosticsSnapshot>,
    ) {
        stats_tab::main_debug_panel(ui, renderer, frame);
    }

    /// Host shader asset id, logical name (or `<none>`), and material family per line (see **Shader routes** tab).
    pub(super) fn shader_mappings_tab(
        ui: &imgui::Ui,
        frame: Option<&crate::diagnostics::FrameDiagnosticsSnapshot>,
        only_fallback: &mut bool,
    ) {
        shader_routes_tab::shader_mappings_tab(ui, frame, only_fallback);
    }

    /// Sorted draw rows with runtime material state.
    pub(super) fn draw_state_tab(
        ui: &imgui::Ui,
        frame: Option<&crate::diagnostics::FrameDiagnosticsSnapshot>,
        ui_only: &mut bool,
        only_overrides: &mut bool,
    ) {
        draw_state_tab::draw_state_tab(ui, frame, ui_only, only_overrides);
    }

    /// Texture pool window with current-view filtering.
    pub(super) fn texture_debug_window(
        ui: &imgui::Ui,
        snapshot: &crate::diagnostics::TextureDebugSnapshot,
        open: &mut bool,
        current_view_only: &mut bool,
    ) {
        texture_debug::texture_debug_window(ui, snapshot, open, current_view_only);
    }

    /// Full [`wgpu::AllocatorReport`] from [`FrameDiagnosticsSnapshot::gpu_allocator_report`], refreshed on a timer.
    ///
    /// Row labels mirror wgpu `label` strings on buffers and textures (often empty).
    pub(super) fn gpu_memory_tab(
        ui: &imgui::Ui,
        frame: Option<&crate::diagnostics::FrameDiagnosticsSnapshot>,
    ) {
        gpu_memory_tab::gpu_memory_tab(ui, frame);
    }

    /// Third overlay window: editable [`crate::config::RendererSettings`] with immediate disk sync.
    pub(super) fn renderer_config_window(
        ui: &imgui::Ui,
        settings: &crate::config::RendererSettingsHandle,
        save_path: &std::path::Path,
        suppress_renderer_config_disk_writes: bool,
        open: &mut bool,
    ) {
        renderer_config_window::renderer_config_window(
            ui,
            settings,
            save_path,
            suppress_renderer_config_disk_writes,
            open,
        );
    }

    /// Second overlay window: one tab per render space and a clipped table of world TRS rows.
    pub(super) fn scene_transforms_window(
        ui: &imgui::Ui,
        snapshot: &crate::diagnostics::SceneTransformsSnapshot,
        open: &mut bool,
    ) {
        scene_transforms::scene_transforms_window(ui, snapshot, open);
    }
}
