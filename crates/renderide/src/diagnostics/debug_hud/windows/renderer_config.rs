//! Renderer config window: display, rendering, and debug settings with immediate disk sync.

use crate::config::{
    save_renderer_settings, MsaaSampleCount, PowerPreferenceSetting, SceneColorFormat, VsyncMode,
};

use imgui::Drag;

/// Focused / unfocused FPS caps (Renderer config window).
fn renderer_config_display_section(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Display");
    ui.indent();
    let mut ff = g.display.focused_fps_cap as f32;
    if Drag::new("Focused FPS cap (0 = uncapped)")
        .range(0.0, 2000.0)
        .speed(1.0)
        .build(ui, &mut ff)
    {
        g.display.focused_fps_cap = ff.round().clamp(0.0, u32::MAX as f32) as u32;
        *dirty = true;
    }
    let mut uf = g.display.unfocused_fps_cap as f32;
    if Drag::new("Unfocused FPS cap (0 = uncapped)")
        .range(0.0, 2000.0)
        .speed(1.0)
        .build(ui, &mut uf)
    {
        g.display.unfocused_fps_cap = uf.round().clamp(0.0, u32::MAX as f32) as u32;
        *dirty = true;
    }
    ui.unindent();
}

/// VSync toggle (Renderer config window).
fn renderer_config_rendering_section(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Rendering");
    ui.indent();
    ui.text_disabled("VSync (swapchain present mode; applies immediately, no restart).");
    for (i, &mode) in VsyncMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(200 + i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.rendering.vsync == mode)
            .build()
        {
            g.rendering.vsync = mode;
            *dirty = true;
        }
    }
    ui.text_disabled("MSAA (main window forward path; clamped to GPU max).");
    for (i, &msaa) in MsaaSampleCount::ALL.iter().enumerate() {
        let _id = ui.push_id_int(i as i32);
        if ui
            .selectable_config(msaa.label())
            .selected(g.rendering.msaa == msaa)
            .build()
        {
            g.rendering.msaa = msaa;
            *dirty = true;
        }
    }
    ui.text_disabled(
        "Scene color format (forward HDR target; compose writes swapchain / XR / RT).",
    );
    for (i, &fmt) in SceneColorFormat::ALL.iter().enumerate() {
        let _id = ui.push_id_int(100 + i as i32);
        if ui
            .selectable_config(fmt.label())
            .selected(g.rendering.scene_color_format == fmt)
            .build()
        {
            g.rendering.scene_color_format = fmt;
            *dirty = true;
        }
    }
    ui.unindent();
}

/// Debug HUD toggles, logging, validation layers, power preference (Renderer config window).
fn renderer_config_debug_section(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Debug");
    ui.indent();
    if ui.checkbox("Frame timing HUD", &mut g.debug.debug_hud_frame_timing) {
        *dirty = true;
    }
    ui.text_disabled("FPS and CPU/GPU submit intervals; snapshot is cheap.");
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
    if ui.checkbox("Log verbose", &mut g.debug.log_verbose) {
        *dirty = true;
    }
    if ui.checkbox("GPU validation layers", &mut g.debug.gpu_validation_layers) {
        *dirty = true;
    }
    ui.text_disabled(
        "Vulkan validation layers significantly reduce performance; enable only when debugging. Restart required to apply (desktop and OpenXR).",
    );
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

/// Body of **Renderer config**: tabbed groups (Display / Rendering / Debug / Post-Processing) and
/// immediate disk save.
///
/// Each tab body marks a shared `dirty` flag; once any tab modifies a setting, the whole
/// [`crate::config::RendererSettings`] struct is serialised back to disk so newly added sub-tables
/// (e.g. `[post_processing]`, `[post_processing.tonemap]`) round-trip without separate plumbing.
pub(super) fn renderer_config_panel_body(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    save_path: &std::path::Path,
    suppress_renderer_config_disk_writes: bool,
) {
    let mut dirty = false;
    if let Some(_bar) = ui.tab_bar("renderer_config_tabs") {
        if let Some(_t) = ui.tab_item("Display") {
            renderer_config_display_section(ui, g, &mut dirty);
        }
        if let Some(_t) = ui.tab_item("Rendering") {
            renderer_config_rendering_section(ui, g, &mut dirty);
        }
        if let Some(_t) = ui.tab_item("Debug") {
            renderer_config_debug_section(ui, g, &mut dirty);
        }
        if let Some(_t) = ui.tab_item("Post-Processing") {
            super::renderer_config_post_processing::renderer_config_post_processing_tab(
                ui, g, &mut dirty,
            );
        }
    }

    if dirty {
        if suppress_renderer_config_disk_writes {
            logger::error!(
                "Refusing to save renderer config to {}: disk writes suppressed after startup extract failure",
                save_path.display()
            );
        } else if let Err(e) = save_renderer_settings(save_path, g) {
            logger::warn!(
                "Failed to save renderer config to {}: {e}",
                save_path.display()
            );
        }
    }

    ui.separator();
    ui.text_disabled(format!("Persist: {}", save_path.display()));
}
