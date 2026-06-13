//! **Shader routes** tab -- host shader -> renderer pipeline routing list.

use crate::diagnostics::FrameDiagnosticsSnapshot;

use super::super::super::state::HudUiState;
use super::super::super::view::TabView;
use super::super::sections::collapsible_section;

/// **Shader routes** tab dispatched from [`super::MainDebugWindow`].
pub struct ShaderRoutesTab;

impl TabView for ShaderRoutesTab {
    type Data<'a> = Option<&'a FrameDiagnosticsSnapshot>;
    type State = HudUiState;

    fn render(&self, ui: &imgui::Ui, data: Self::Data<'_>, state: &mut Self::State) {
        let Some(d) = data else {
            ui.text("Waiting for frame diagnostics...");
            return;
        };
        collapsible_section(ui, "Filters", true, |ui| {
            ui.checkbox(
                "Only fallback routes",
                &mut state.shader_routes_only_fallback,
            );
        });
        collapsible_section(ui, "Routes", true, |ui| {
            if d.shader_routes.rows.is_empty() {
                ui.text("No shader route data");
            } else {
                for route in &d.shader_routes.rows {
                    if state.shader_routes_only_fallback && route.implemented {
                        continue;
                    }
                    ui.text_wrapped(format!(
                        "{}  {}  {}  {}  {}",
                        route.shader_asset_id,
                        route.shader_asset_name.as_deref().unwrap_or("<none>"),
                        route
                            .shader_variant_bits
                            .map(|bits| format!("0x{bits:08X}"))
                            .unwrap_or_else(|| "<no variant>".to_string()),
                        route.pipeline_label,
                        if route.implemented {
                            "implemented"
                        } else {
                            "fallback"
                        },
                    ));
                }
            }
        });
    }
}
