//! **Renderer config** HUD window -- editable [`crate::config::RendererSettings`] with immediate
//! disk sync.
//!
//! Merges what used to live in three files (the window envelope, the five-tab body, and the
//! Post-Processing tab body) into one [`HudWindow`] impl with private section helpers per tab.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use imgui::{Drag, TabItem, TabItemFlags};

use crate::config::{
    AutoExposureSettings, BloomCompositeMode, DebugHudRendererConfigTab, DebugHudSettings,
    GraphicsApiSetting, GtaoSettings, MotionBlurSettings, MsaaSampleCount, PowerPreferenceSetting,
    RendererSettings, RendererSettingsHandle, SceneColorFormat, TonemapMode, VsyncMode,
    WatchdogAction, save_renderer_settings, save_renderer_settings_pruned,
};

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;

const MAX_ASSET_INTEGRATION_BUDGET_MS: u32 = 100;
const MIN_WATCHDOG_POLL_INTERVAL_MS: u32 = 10;
const MAX_WATCHDOG_POLL_INTERVAL_MS: u32 = 10_000;
const MAX_WATCHDOG_THRESHOLD_MS: u32 = 600_000;

#[derive(Debug, thiserror::Error)]
enum OpenLogFolderError {
    #[error("failed to spawn {program} for {path}: {source}")]
    Spawn {
        program: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{program} failed for {path} with status {status}")]
    ExitStatus {
        program: &'static str,
        path: PathBuf,
        status: ExitStatus,
    },
}

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
            size: [layout::RENDERER_CONFIG_W, layout::RENDERER_CONFIG_H],
            size_min: [360.0, 260.0],
            size_max: [f32::INFINITY, f32::INFINITY],
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

    fn body(&self, ui: &imgui::Ui, data: Self::Data<'_>, state: &mut Self::State) {
        let RendererConfigData {
            settings,
            save_path,
            suppress_renderer_config_disk_writes,
        } = data;

        ui.text_disabled("Press F7 to hide/show the ImGui UI.");
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

        renderer_config_panel_body(
            ui,
            &mut g,
            save_path,
            suppress_renderer_config_disk_writes,
            state,
        );
    }
}

/// Body of **Renderer config**: tabbed groups (Display / Rendering / Debug / Post-Processing / Experimental) and
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
    state: &mut HudUiState,
) {
    let mut dirty = false;
    ui.text_disabled(format!("Config version: {}", g.config_version));

    if !state.renderer_config_tabs.all_open() && ui.small_button("Show all config tabs") {
        state.renderer_config_tabs = Default::default();
        state.renderer_config_tab_restore_pending = true;
    }

    if let Some(_bar) = ui.tab_bar("renderer_config_tabs") {
        for &tab in DebugHudRendererConfigTab::ALL {
            let mut tab_open = state.renderer_config_tabs.is_open(tab);
            if !tab_open {
                continue;
            }
            let flags =
                if state.renderer_config_tab_restore_pending && state.renderer_config_tab == tab {
                    TabItemFlags::SET_SELECTED
                } else {
                    TabItemFlags::empty()
                };
            if let Some(_t) = TabItem::new(tab.label())
                .opened(&mut tab_open)
                .flags(flags)
                .begin(ui)
            {
                state.renderer_config_tab = tab;
                state.renderer_config_tab_restore_pending = false;
                match tab {
                    DebugHudRendererConfigTab::Display => display_section(ui, g, &mut dirty),
                    DebugHudRendererConfigTab::Rendering => rendering_section(ui, g, &mut dirty),
                    DebugHudRendererConfigTab::Debug => debug_section(ui, g, &mut dirty),
                    DebugHudRendererConfigTab::PostProcessing => {
                        post_processing_section(ui, g, &mut dirty);
                    }
                    DebugHudRendererConfigTab::Experimental => {
                        experimental_section(ui, g, &mut dirty);
                    }
                }
            }
            state.renderer_config_tabs.set_open(tab, tab_open);
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
    if ui.small_button("Clean up config file") {
        if suppress_renderer_config_disk_writes {
            logger::error!(
                "Refusing to clean up renderer config at {}: disk writes suppressed after startup extract failure",
                save_path.display()
            );
        } else if let Err(e) = save_renderer_settings_pruned(save_path, g) {
            logger::warn!(
                "Failed to clean up renderer config at {}: {e}",
                save_path.display()
            );
        }
    }
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

fn drag_u32_setting(
    ui: &imgui::Ui,
    label: &str,
    value: &mut u32,
    min: u32,
    max: u32,
    speed: f32,
) -> bool {
    let mut edited = *value as f32;
    if Drag::new(label)
        .range(min as f32, max as f32)
        .speed(speed)
        .build(ui, &mut edited)
    {
        *value = edited.round().clamp(min as f32, max as f32) as u32;
        return true;
    }
    false
}

/// VSync, graphics API, MSAA, scene color format, and asset integration budget.
fn rendering_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Rendering");
    ui.indent();
    rendering_presentation_section(ui, g, dirty);
    ui.separator();
    rendering_graph_section(ui, g, dirty);
    ui.separator();
    rendering_asset_section(ui, g, dirty);
    ui.unindent();
}

fn rendering_presentation_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Presentation");
    ui.indent();
    ui.text_disabled("VSync (On = adaptive FifoRelaxed/Fifo; applies immediately, no restart).");
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
    ui.text_disabled(
        "Graphics API (startup only; restart required, falls back to Auto if unavailable).",
    );
    for (i, &api) in GraphicsApiSetting::ALL.iter().enumerate() {
        let _id = ui.push_id_int(400 + i as i32);
        if ui
            .selectable_config(api.label())
            .selected(g.rendering.graphics_api == api)
            .build()
        {
            g.rendering.graphics_api = api;
            *dirty = true;
        }
    }
    ui.unindent();
}

fn rendering_graph_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Graph");
    ui.indent();
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
        "Scene color format (unsigned formats promote to RGBA16Float while negative lights are active).",
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

fn rendering_asset_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Assets");
    ui.indent();
    if drag_u32_setting(
        ui,
        "Asset integration budget (ms)",
        &mut g.rendering.asset_integration_budget_ms,
        0,
        MAX_ASSET_INTEGRATION_BUDGET_MS,
        1.0,
    ) {
        *dirty = true;
    }
    if drag_u32_setting(
        ui,
        "Extra particle integration budget (ms)",
        &mut g.rendering.asset_particle_integration_budget_ms,
        0,
        MAX_ASSET_INTEGRATION_BUDGET_MS,
        1.0,
    ) {
        *dirty = true;
    }
    ui.text_disabled("Cooperative per-frame asset and extra dynamic-buffer integration budgets.");
    ui.unindent();
}

/// Debug HUD toggles, logging, validation layers, power preference.
fn debug_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Debug");
    ui.indent();
    debug_hud_section(ui, g, dirty);
    ui.separator();
    debug_diagnostics_section(ui, g, dirty);
    ui.separator();
    watchdog_section(ui, g, dirty);
    ui.unindent();
}

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
    if ui.checkbox("Persist HUD layout", &mut g.debug.hud.persist_layout) {
        *dirty = true;
    }
    ui.text_disabled("Saves ImGui window placement to renderide-imgui.ini next to config.toml.");
    let mut ui_scale = g.debug.hud.resolved_ui_scale();
    if Drag::new("HUD UI scale")
        .range(
            DebugHudSettings::MIN_UI_SCALE,
            DebugHudSettings::MAX_UI_SCALE,
        )
        .speed(0.01)
        .build(ui, &mut ui_scale)
    {
        g.debug.hud.ui_scale = ui_scale.clamp(
            DebugHudSettings::MIN_UI_SCALE,
            DebugHudSettings::MAX_UI_SCALE,
        );
        *dirty = true;
    }
    ui.unindent();
}

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

fn log_folder_opener_program() -> &'static str {
    if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    }
}

fn open_log_folder(path: &Path) -> Result<(), OpenLogFolderError> {
    let program = log_folder_opener_program();
    let status =
        Command::new(program)
            .arg(path)
            .status()
            .map_err(|source| OpenLogFolderError::Spawn {
                program,
                path: path.to_path_buf(),
                source,
            })?;
    if status.success() {
        Ok(())
    } else {
        Err(OpenLogFolderError::ExitStatus {
            program,
            path: path.to_path_buf(),
            status,
        })
    }
}

fn watchdog_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Watchdog");
    ui.indent();
    ui.text_disabled("Cooperative hitch/hang detector; restart required to apply changes.");
    if ui.checkbox("Enable watchdog", &mut g.watchdog.enabled) {
        *dirty = true;
    }
    if drag_u32_setting(
        ui,
        "Poll interval (ms)",
        &mut g.watchdog.poll_interval_ms,
        MIN_WATCHDOG_POLL_INTERVAL_MS,
        MAX_WATCHDOG_POLL_INTERVAL_MS,
        10.0,
    ) {
        *dirty = true;
    }
    if drag_u32_setting(
        ui,
        "Hitch threshold (ms, 0 = disabled)",
        &mut g.watchdog.hitch_threshold_ms,
        0,
        MAX_WATCHDOG_THRESHOLD_MS,
        50.0,
    ) {
        *dirty = true;
    }
    if drag_u32_setting(
        ui,
        "Hang threshold (ms)",
        &mut g.watchdog.hang_threshold_ms,
        1,
        MAX_WATCHDOG_THRESHOLD_MS,
        100.0,
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

/// Experimental feature flags.
fn experimental_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Experimental");
    ui.indent();
    if ui.checkbox(
        "Use reflection probe SH2",
        &mut g.experimental.reflection_probe_sh2_enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled(
        "When disabled, reflection probes contribute specular reflections only; diffuse SH2 comes from AmbientLightSH2.",
    );
    if ui.checkbox(
        "Dev WGSL material hot reload",
        &mut g.experimental.material_shader_hot_reload_enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled(
        "When enabled, local material WGSL target edits invalidate shader generations and requeue affected pipelines.",
    );
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
    post_processing_motion_blur(ui, g, dirty);
    ui.separator();
    post_processing_auto_exposure(ui, g, dirty);
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
        "GTAO (Ground-Truth Ambient Occlusion): samples the forward view-normal prepass \
         and modulates HDR scene color by a physical visibility factor. Runs pre-tonemap.",
    );
    if ui.checkbox("Enable GTAO", &mut g.post_processing.gtao.enabled) {
        *dirty = true;
    }
    let gtao = &mut g.post_processing.gtao;
    gtao_quality_controls(ui, gtao, dirty);
    gtao_sampling_controls(ui, gtao, dirty);
    gtao_denoise_controls(ui, gtao, dirty);
}

fn gtao_quality_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Quality");
    ui.indent();
    if ui
        .slider_config("Quality level", 0_u32, 3_u32)
        .build(&mut gtao.quality_level)
    {
        *dirty = true;
    }
    ui.text_disabled("0 = low, 1 = medium, 2 = high, 3 = ultra.");
    if ui
        .slider_config("Step floor", 1_u32, 8_u32)
        .build(&mut gtao.step_count)
    {
        *dirty = true;
    }
    ui.text_disabled("Advanced floor for steps per slice; quality preset remains primary.");
    ui.unindent();
}

fn gtao_sampling_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Sampling");
    ui.indent();
    if ui
        .slider_config("Radius (m)", 0.05_f32, 2.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.radius_meters)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Radius multiplier", 0.3_f32, 3.0_f32)
        .display_format("%.3f")
        .build(&mut gtao.radius_multiplier)
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
        .slider_config("Max pixel radius", 16.0_f32, 2048.0_f32)
        .display_format("%.0f")
        .build(&mut gtao.max_pixel_radius)
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
        .slider_config("Sample distribution power", 1.0_f32, 3.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.sample_distribution_power)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Thin occluder compensation", 0.0_f32, 0.7_f32)
        .display_format("%.2f")
        .build(&mut gtao.thin_occluder_compensation)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Final value power", 0.5_f32, 5.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.final_value_power)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Depth mip sampling offset", 0.0_f32, 30.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.depth_mip_sampling_offset)
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
    ui.unindent();
}

fn gtao_denoise_controls(ui: &imgui::Ui, gtao: &mut GtaoSettings, dirty: &mut bool) {
    ui.text("Denoise");
    ui.indent();
    if ui
        .slider_config("Denoise passes", 0_u32, 3_u32)
        .build(&mut gtao.denoise_passes)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Denoise blur beta", 0.0_f32, 8.0_f32)
        .display_format("%.2f")
        .build(&mut gtao.denoise_blur_beta)
    {
        *dirty = true;
    }
    ui.unindent();
}

fn post_processing_bloom(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("bloom");
    ui.text_disabled(
        "Bloom (dual-filter): HDR-linear scatter via a \
         mip-chain downsample/upsample pyramid with Karis firefly reduction on mip 0. Runs \
         pre-tonemap. Energy-conserving mode redistributes the bloom source term. Changing \
         `max mip dimension` rebuilds the render graph; other knobs take effect next frame via the \
         shared params UBO / per-mip blend constant.",
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

fn post_processing_motion_blur(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("motion_blur");
    ui.text_disabled(
        "Motion blur: derives screen-space velocity only while enabled, then blurs HDR scene \
         color after bloom and before tonemapping. VR views are opt-in.",
    );
    if ui.checkbox(
        "Enable motion blur",
        &mut g.post_processing.motion_blur.enabled,
    ) {
        *dirty = true;
    }
    if ui.checkbox("Allow in VR", &mut g.post_processing.motion_blur.allow_vr) {
        *dirty = true;
    }
    let motion_blur = &mut g.post_processing.motion_blur;
    if ui
        .slider_config("Shutter angle", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut motion_blur.shutter_angle)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Samples", 0_u32, MotionBlurSettings::MAX_SAMPLE_COUNT)
        .build(&mut motion_blur.sample_count)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Max velocity (px)", 0.0_f32, 512.0_f32)
        .display_format("%.0f")
        .build(&mut motion_blur.max_velocity_pixels)
    {
        *dirty = true;
    }
    if !motion_blur.is_effectively_enabled() {
        ui.text_disabled("Effective state: disabled by zero samples, shutter, or velocity clamp.");
    }
}

fn post_processing_auto_exposure(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    let _id = ui.push_id("auto_exposure");
    ui.text_disabled(
        "Auto-exposure: builds a scene-linear log-luminance histogram, targets middle gray, ignores dark/bright percentile tails, and adapts exposure before tonemapping.",
    );
    if ui.checkbox(
        "Enable auto-exposure",
        &mut g.post_processing.auto_exposure.enabled,
    ) {
        *dirty = true;
    }
    let auto = &mut g.post_processing.auto_exposure;
    if ui
        .slider_config("Min EV", -16.0_f32, 16.0_f32)
        .display_format("%.2f")
        .build(&mut auto.min_ev)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Max EV", -16.0_f32, 16.0_f32)
        .display_format("%.2f")
        .build(&mut auto.max_ev)
    {
        *dirty = true;
    }
    let (min_ev, max_ev) = auto.resolved_ev_range();
    if (min_ev, max_ev) != (auto.min_ev, auto.max_ev) {
        ui.text_disabled(format!("Effective EV range: {min_ev:.2} to {max_ev:.2}."));
    }
    if ui
        .slider_config("Low percentile", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut auto.low_percent)
    {
        *dirty = true;
    }
    if ui
        .slider_config("High percentile", 0.0_f32, 1.0_f32)
        .display_format("%.2f")
        .build(&mut auto.high_percent)
    {
        *dirty = true;
    }
    let (low, high) = auto.resolved_filter();
    if (low, high) != (auto.low_percent, auto.high_percent) {
        ui.text_disabled(format!(
            "Effective percentile filter: {low:.2} to {high:.2}."
        ));
    }
    if ui
        .slider_config("Brighten speed (EV/s)", 0.0_f32, 12.0_f32)
        .display_format("%.2f")
        .build(&mut auto.speed_brighten)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Darken speed (EV/s)", 0.0_f32, 12.0_f32)
        .display_format("%.2f")
        .build(&mut auto.speed_darken)
    {
        *dirty = true;
    }
    if ui
        .slider_config(
            "Transition distance (EV)",
            AutoExposureSettings::MIN_TRANSITION_DISTANCE,
            8.0_f32,
        )
        .display_format("%.2f")
        .build(&mut auto.exponential_transition_distance)
    {
        *dirty = true;
    }
    if ui
        .slider_config("Middle-gray compensation (EV)", -8.0_f32, 8.0_f32)
        .display_format("%.2f")
        .build(&mut auto.compensation_ev)
    {
        *dirty = true;
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
        "ACES Fitted is filmic with stronger hue shifts. AgX is more neutral. `None` skips \
         tonemapping (HDR pass-through; values >1 will clip in the swapchain).",
    );
}

#[cfg(test)]
mod tests {
    use super::log_folder_opener_program;

    #[test]
    fn log_folder_opener_program_matches_platform() {
        let expected = if cfg!(target_os = "windows") {
            "explorer"
        } else if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };

        assert_eq!(log_folder_opener_program(), expected);
    }
}
