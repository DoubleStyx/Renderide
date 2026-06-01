//! Debug, diagnostics, and watchdog renderer-config HUD controls.

use crate::config::{
    DebugHudSettings, PowerPreferenceSetting, RenderGraphValidationMode, RendererSettings,
    WatchdogAction,
};

use super::controls::{drag_f32_slider_setting, drag_u32_slider_setting};
use super::log_folder::open_log_folder;

const MIN_WATCHDOG_POLL_INTERVAL_MS: u32 = 10;
const MAX_WATCHDOG_POLL_INTERVAL_MS: u32 = 10_000;
const MAX_WATCHDOG_THRESHOLD_MS: u32 = 600_000;

/// Debug HUD toggles, logging, validation layers, power preference, and watchdog settings.
pub(super) fn debug_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Debug");
    ui.indent();
    debug_hud_section(ui, g, dirty);
    ui.separator();
    debug_diagnostics_section(ui, g, dirty);
    ui.separator();
    watchdog_section(ui, g, dirty);
    ui.unindent();
}

/// Edits visibility and layout settings for diagnostic HUD windows.
fn debug_hud_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("HUD");
    ui.indent();
    if ui.checkbox("Show ImGui UI", &mut g.debug.hud.imgui_visible) {
        *dirty = true;
    }
    ui.text_disabled("F7 toggles this setting even when the ImGui UI is hidden.");
    if ui.checkbox("Frame timing HUD", &mut g.debug.debug_hud_frame_timing) {
        *dirty = true;
    }
    ui.text_disabled("FPS and CPU/GPU frame intervals; snapshot is cheap.");
    if ui.checkbox(
        "Debug HUD (Stats / Shader routes / Draw state / GPU memory)",
        &mut g.debug.debug_hud_enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled("Main debug panels and per-frame diagnostics capture when enabled.");
    if ui.checkbox("Scene transforms HUD", &mut g.debug.debug_hud_transforms) {
        *dirty = true;
    }
    ui.text_disabled(
        "Per-space world transform table; separate from main HUD (can be expensive on large scenes).",
    );
    if ui.checkbox("Textures HUD", &mut g.debug.debug_hud_textures) {
        *dirty = true;
    }
    ui.text_disabled("Texture pool rows and current-view usage; can be noisy in large scenes.");
    if ui.checkbox("Links HUD", &mut g.debug.debug_hud_links) {
        *dirty = true;
    }
    ui.text_disabled("Feedback, bug report, and discussion links.");
    if ui.checkbox("Persist HUD layout", &mut g.debug.hud.persist_layout) {
        *dirty = true;
    }
    ui.text_disabled("Saves ImGui window placement to renderide-imgui.ini next to config.toml.");
    let mut ui_scale = g.debug.hud.resolved_ui_scale();
    if drag_f32_slider_setting(
        ui,
        "HUD UI scale",
        &mut ui_scale,
        DebugHudSettings::MIN_UI_SCALE,
        DebugHudSettings::MAX_UI_SCALE,
        None,
    ) {
        g.debug.hud.ui_scale = ui_scale;
        *dirty = true;
    }
    ui.unindent();
}

/// Edits logging, validation, and adapter preference diagnostics settings.
fn debug_diagnostics_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Diagnostics");
    ui.indent();
    if ui.checkbox("Log verbose", &mut g.debug.log_verbose) {
        *dirty = true;
    }
    let logs_root = logger::logs_root();
    ui.text_wrapped(format!("Log folder: {}", logs_root.display()));
    if ui.small_button("Open log folder")
        && let Err(e) = open_log_folder(&logs_root)
    {
        logger::warn!("Failed to open log folder: {e}");
    }
    if ui.checkbox("GPU validation layers", &mut g.debug.gpu_validation_layers) {
        *dirty = true;
    }
    ui.text_disabled(
        "Vulkan validation layers significantly reduce performance; enable only when debugging. Restart required to apply (desktop and OpenXR).",
    );
    ui.text_disabled("Render graph validation");
    for (i, &mode) in RenderGraphValidationMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(600 + i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.debug.render_graph_validation == mode)
            .build()
        {
            g.debug.render_graph_validation = mode;
            *dirty = true;
        }
    }
    ui.text_disabled("Warn logs declaration/runtime issues; Strict turns them into graph errors.");
    ui.text_disabled("Power preference (applies at next renderer launch)");
    for (i, &pref) in PowerPreferenceSetting::ALL.iter().enumerate() {
        let _id = ui.push_id_int(i as i32);
        if ui
            .selectable_config(pref.label())
            .selected(g.debug.power_preference == pref)
            .build()
        {
            g.debug.power_preference = pref;
            *dirty = true;
        }
    }
    ui.unindent();
}

/// Edits cooperative hitch and hang watchdog settings.
fn watchdog_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Watchdog");
    ui.indent();
    ui.text_disabled("Cooperative hitch/hang detector; restart required to apply changes.");
    if ui.checkbox("Enable watchdog", &mut g.watchdog.enabled) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Poll interval (ms)",
        &mut g.watchdog.poll_interval_ms,
        MIN_WATCHDOG_POLL_INTERVAL_MS,
        MAX_WATCHDOG_POLL_INTERVAL_MS,
    ) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Hitch threshold (ms, 0 = disabled)",
        &mut g.watchdog.hitch_threshold_ms,
        0,
        MAX_WATCHDOG_THRESHOLD_MS,
    ) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Hang threshold (ms)",
        &mut g.watchdog.hang_threshold_ms,
        1,
        MAX_WATCHDOG_THRESHOLD_MS,
    ) {
        *dirty = true;
    }
    ui.text("Hang action");
    for (i, &action) in WatchdogAction::ALL.iter().enumerate() {
        let _id = ui.push_id_int(600 + i as i32);
        if ui
            .selectable_config(action.label())
            .selected(g.watchdog.action == action)
            .build()
        {
            g.watchdog.action = action;
            *dirty = true;
        }
    }
    ui.unindent();
}
