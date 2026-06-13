//! **Renderer config** HUD window -- editable [`crate::config::RendererSettings`] with immediate
//! disk sync.

mod controls;
mod debug;
mod display;
mod experimental;
mod log_folder;
mod post_processing;
mod rendering;

use std::path::Path;

use imgui::{TabItem, TabItemFlags};

use crate::config::{
    DebugHudRendererConfigTab, RendererSettings, RendererSettingsHandle, save_renderer_settings,
    save_renderer_settings_pruned,
};

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;
use debug::debug_section;
use display::display_section;
use experimental::experimental_section;
use post_processing::post_processing_section;
use rendering::rendering_section;

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

/// Body of **Renderer config**: tabbed settings groups with immediate disk save.
///
/// Each tab body marks a shared `dirty` flag; once any tab modifies a setting, the whole
/// [`RendererSettings`] struct is serialised back to disk so newly added sub-tables round-trip
/// without separate plumbing.
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
                    DebugHudRendererConfigTab::Display => {
                        if display_section(ui, g, &mut dirty) {
                            reset_renderer_config_to_defaults(g, state);
                            dirty = true;
                        }
                    }
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

fn reset_renderer_config_to_defaults(g: &mut RendererSettings, state: &mut HudUiState) {
    *g = RendererSettings::from_defaults();
    *state = HudUiState::from_settings(&g.debug.hud);
}

#[cfg(test)]
mod tests {
    use crate::config::DebugHudRendererConfigTab;

    use super::*;

    #[test]
    fn reset_renderer_config_to_defaults_refreshes_settings_and_hud_state() {
        let mut settings = RendererSettings::from_defaults();
        settings.display.focused_fps_cap = 30;
        settings.debug.hud.draw_state_ui_only = true;
        settings.debug.hud.renderer_config_tab = DebugHudRendererConfigTab::PostProcessing;
        settings
            .debug
            .hud
            .renderer_config_tabs
            .set_open(DebugHudRendererConfigTab::Display, false);
        let mut state = HudUiState::from_settings(&settings.debug.hud);

        reset_renderer_config_to_defaults(&mut settings, &mut state);

        let defaults = RendererSettings::from_defaults();
        assert_eq!(settings, defaults);
        assert_eq!(state, HudUiState::from_settings(&defaults.debug.hud));
    }
}
