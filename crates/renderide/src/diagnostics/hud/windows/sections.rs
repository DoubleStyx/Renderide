//! Shared collapsible section helpers for HUD windows and tabs.

use imgui::TreeNodeFlags;

/// Renders a labeled collapsible section with consistent indentation and spacing.
pub(super) fn collapsible_section(
    ui: &imgui::Ui,
    label: &str,
    default_open: bool,
    body: impl FnOnce(&imgui::Ui),
) {
    let flags = if default_open {
        TreeNodeFlags::DEFAULT_OPEN
    } else {
        TreeNodeFlags::empty()
    };
    if ui.collapsing_header(label, flags) {
        ui.indent_by(8.0);
        body(ui);
        ui.unindent_by(8.0);
        ui.spacing();
    }
}

/// Renders a labeled collapsible section and persists the current expanded state.
pub(super) fn collapsible_section_with_state(
    ui: &imgui::Ui,
    label: &str,
    open: &mut bool,
    body: impl FnOnce(&imgui::Ui),
) {
    let flags = if *open {
        TreeNodeFlags::DEFAULT_OPEN
    } else {
        TreeNodeFlags::empty()
    };
    let is_open = ui.collapsing_header(label, flags);
    *open = is_open;
    if is_open {
        ui.indent_by(8.0);
        body(ui);
        ui.unindent_by(8.0);
        ui.spacing();
    }
}
