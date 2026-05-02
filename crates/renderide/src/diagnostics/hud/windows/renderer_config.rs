//! **Renderer config** HUD window -- editable [`crate::config::RendererSettings`] with immediate
//! disk sync.
//!
//! Merges what used to live in three files (the window envelope, the four-tab body, and the
//! Post-Processing tab body) into one [`HudWindow`] impl with private section helpers per tab.

use std::path::Path;

use imgui::Drag;

use crate::config::{
    BloomCompositeMode, ClusterAssignmentMode, MsaaSampleCount, PowerPreferenceSetting,
    RendererSettings, RendererSettingsHandle, SceneColorFormat, TonemapMode, VsyncMode,
    save_renderer_settings,
};

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;

/// Inputs for [`RendererConfigWindow`]: live settings handle, disk save target, and the
/// startup-extract failure flag.
pub struct RendererConfigData<'a> {
    /// Live settings + persistence target.
    pub settings: &'a RendererSettingsHandle,
    /// Path the renderer writes `config.toml` back to on dirty changes.
    pub save_path: &'a Path,
    /// When `true`, the overlay refuses to write `config.toml` (startup Figment extract failed).
    pub suppress_renderer_config_disk_writes: bool,
}

/// **Renderer config** HUD window.
pub struct RendererConfigWindow;

impl HudWindow for RendererConfigWindow {
    type Data<'a> = RendererConfigData<'a>;
    type State = HudUiState;

    fn title(&self) -> &str {
        "Renderer config"
    }

    fn anchor(&self, _viewport: Viewport) -> WindowSlot {
        WindowSlot {
            position: [layout::MARGIN, layout::MARGIN],
            size_min: [layout::RENDERER_CONFIG_W, layout::RENDERER_CONFIG_H],
            size_max: [layout::RENDERER_CONFIG_W, layout::RENDERER_CONFIG_H],
        }
    }

    fn bg_alpha(&self) -> f32 {
        0.88
    }

    fn read_open_flag(&self, state: &Self::State) -> Option<bool> {
        Some(state.renderer_config_open)
    }

    fn write_open_flag(&self, state: &mut Self::State, value: bool) {
        state.renderer_config_open = value;
    }

    fn body(&self, ui: &imgui::Ui, data: Self::Data<'_>, _state: &mut Self::State) {
        let RendererConfigData {
            settings,
            save_path,
            suppress_renderer_config_disk_writes,
        } = data;

        ui.text_wrapped(
            "This file is owned by the renderer. Do not edit config.toml manually while \
             the process is running -- your changes may be overwritten or lost. Use these \
             controls instead.",
        );
        if suppress_renderer_config_disk_writes {
            ui.text_colored(
                [1.0, 0.35, 0.35, 1.0],
                "Disk save is disabled: startup Figment extract failed. Fix config.toml and restart.",
            );
        }
        ui.separator();

        let Ok(mut g) = settings.write() else {
            ui.text_colored([1.0, 0.4, 0.4, 1.0], "Settings store is unavailable.");
            return;
        };

        renderer_config_panel_body(ui, &mut g, save_path, suppress_renderer_config_disk_writes);
    }
}

/// Body of **Renderer config**: tabbed groups (Display / Rendering / Debug / Post-Processing) and
/// immediate disk save.
///
/// Each tab body marks a shared `dirty` flag; once any tab modifies a setting, the whole
/// [`RendererSettings`] struct is serialised back to disk so newly added sub-tables (e.g.
/// `[post_processing]`, `[post_processing.tonemap]`) round-trip without separate plumbing.
fn renderer_config_panel_body(
    ui: &imgui::Ui,
    g: &mut RendererSettings,
    save_path: &Path,
    suppress_renderer_config_disk_writes: bool,
) {
    let mut dirty = false;
    if let Some(_bar) = ui.tab_bar("renderer_config_tabs") {
        if let Some(_t) = ui.tab_item("Display") {
            display_section(ui, g, &mut dirty);
        }
        if let Some(_t) = ui.tab_item("Rendering") {
            rendering_section(ui, g, &mut dirty);
        }
        if let Some(_t) = ui.tab_item("Debug") {
            debug_section(ui, g, &mut dirty);
        }
        if let Some(_t) = ui.tab_item("Post-Processing") {
            post_processing_section(ui, g, &mut dirty);
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

/// Focused / unfocused FPS caps.
fn display_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
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

/// VSync, MSAA, scene color format, clustered light backend.
fn rendering_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
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
    ui.text_disabled("Clustered-light assignment backend.");
    for (i, &mode) in ClusterAssignmentMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(300 + i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.rendering.cluster_assignment == mode)
            .build()
        {
            g.rendering.cluster_assignment = mode;
            *dirty = true;
        }
    }
    ui.unindent();
}

/// Debug HUD toggles, logging, validation layers, power preference.
fn debug_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
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

/// Master toggle, GTAO, bloom, tonemap.
fn post_processing_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Post-Processing");
    ui.indent();
    post_processing_master(ui, g, dirty);
    ui.separator();
    post_processing_gtao(ui, g, dirty);
    ui.separator();
    post_processing_bloom(ui, g, dirty);
    ui.separator();
    post_processing_tonemap(ui, g, dirty);
    ui.unindent();
}

fn post_processing_master(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("master");
    if ui.checkbox(
        "Enable post-processing stack",
        &mut g.post_processing.enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled(
        "Master toggle for the post-processing chain (HDR scene color -> display target). \
         Applied on the next frame (the render graph is rebuilt automatically when the chain \
         topology changes).",
    );
}

fn post_processing_gtao(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("gtao");
    ui.text_disabled(
        "GTAO (Ground-Truth Ambient Occlusion): reconstructs view-space normals from depth \
         and modulates HDR scene color by a physical visibility factor. Runs pre-tonemap.",
    );
    if ui.checkbox("Enable GTAO", &mut g.post_processing.gtao.enabled) {
        *dirty = true;
    }
    let gtao = &mut g.post_processing.gtao;
    if ui
        .slider_config("Radius (m)", 0.05_f32, 2.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.radius_meters)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Intensity", 0.0_f32, 2.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.intensity)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Max pixel radius", 16.0_f32, 256.0_f32)
        .display_format("%.0f")
        .build(&mut gtao.max_pixel_radius)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Steps", 2_u32, 16_u32)
        .build(&mut gtao.step_count)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Falloff range", 0.05_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.falloff_range)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Multi-bounce albedo", 0.0_f32, 0.9_f32)
        .display_format("%.2f")
        .build(&mut gtao.albedo_multibounce)
    {
        *dirty = true;
    }
}

fn post_processing_bloom(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("bloom");
    ui.text_disabled(
        "Bloom (dual-filter, COD: Advanced Warfare / Bevy port): HDR-linear scatter via a \
         mip-chain downsample/upsample pyramid with Karis firefly reduction on mip 0. Runs \
         pre-tonemap. Changing `max mip dimension` rebuilds the render graph; other knobs take \
         effect next frame via the shared params UBO / per-mip blend constant.",
    );
    if ui.checkbox("Enable bloom", &mut g.post_processing.bloom.enabled) {
        *dirty = true;
    }
    let bloom = &mut g.post_processing.bloom;
    if ui
        .slider_config("Intensity", 0.0_f32, 1.0_f32)
        .display_format("%.3f")
        .build(&mut bloom.intensity)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Low-frequency boost", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.low_frequency_boost)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Low-frequency boost curvature", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.low_frequency_boost_curvature)
    {
        *dirty = true;
    }
    if ui
        .slider_config("High-pass frequency", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.high_pass_frequency)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Prefilter threshold (HDR)", 0.0_f32, 8.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.prefilter_threshold)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Prefilter threshold softness", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut bloom.prefilter_threshold_softness)
    {
        *dirty = true;
    }
    ui.text("Composite mode");
    for (i, &mode) in BloomCompositeMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(0x1000 + i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(bloom.composite_mode == mode)
            .build()
        {
            bloom.composite_mode = mode;
            *dirty = true;
        }
    }
    if ui
        .slider_config("Max mip dimension (px)", 64_u32, 2048_u32)
        .build(&mut bloom.max_mip_dimension)
    {
        *dirty = true;
    }
    let effective_max_mip_dimension = bloom.effective_max_mip_dimension();
    if effective_max_mip_dimension != bloom.max_mip_dimension {
        ui.text_disabled(format!(
            "Effective max mip dimension: {effective_max_mip_dimension} px (rounded down to power of two)."
        ));
    }
}

fn post_processing_tonemap(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("tonemap");
    ui.text_disabled("Tonemap (HDR linear -> display-referred 0..1 linear).");
    for (i, &mode) in TonemapMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.post_processing.tonemap.mode == mode)
            .build()
        {
            g.post_processing.tonemap.mode = mode;
            *dirty = true;
        }
    }
    ui.text_disabled(
        "ACES Fitted is the high-quality reference curve used by AAA pipelines. \
         `None` skips tonemapping (HDR pass-through; values >1 will clip in the swapchain).",
    );
}
