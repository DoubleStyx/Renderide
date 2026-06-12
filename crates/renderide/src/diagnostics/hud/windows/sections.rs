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
