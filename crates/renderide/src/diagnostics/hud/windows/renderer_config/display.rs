//! Display-related renderer-config HUD controls.

use crate::config::RendererSettings;

use super::controls::drag_u32_slider_setting;

const RESET_TO_DEFAULTS_POPUP: &str = "Reset renderer config to defaults";

/// Foreground/background desktop FPS caps and the full-settings reset affordance.
pub(super) fn display_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) -> bool {
    let mut reset_to_defaults = false;
    ui.text("Display");
    ui.indent();
    if drag_u32_slider_setting(
        ui,
        "Foreground FPS cap (0 = uncapped)",
        &mut g.display.focused_fps_cap,
        0,
        2000,
    ) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Background FPS cap (0 = uncapped)",
        &mut g.display.unfocused_fps_cap,
        0,
        2000,
    ) {
        *dirty = true;
    }
    ui.unindent();
    ui.separator();
    if ui.small_button("Reset all settings to defaults") {
        ui.open_popup(RESET_TO_DEFAULTS_POPUP);
    }
    if let Some(_popup) = ui.begin_modal_popup(RESET_TO_DEFAULTS_POPUP) {
        ui.text_wrapped(
            "Reset every renderer setting to its built-in default? This updates the live settings \
             immediately and saves config.toml after confirmation.",
        );
        ui.spacing();
        if ui.button("Reset") {
            reset_to_defaults = true;
            ui.close_current_popup();
        }
        ui.same_line();
        if ui.button("Cancel") {
            ui.close_current_popup();
        }
    }
    reset_to_defaults
}
