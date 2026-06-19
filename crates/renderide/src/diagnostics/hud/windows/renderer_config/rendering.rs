//! Rendering-related renderer-config HUD controls.

use crate::config::{
    GraphicsApiSetting, MsaaSampleCount, PresentationModeSetting, RendererSettings,
    SceneColorFormat,
};

use super::controls::drag_u32_slider_setting;

const MAX_ASSET_INTEGRATION_BUDGET_MS: u32 = 100;

/// Presentation mode, graphics API, MSAA, scene color format, and asset integration budget.
pub(super) fn rendering_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Rendering");
    ui.indent();
    rendering_presentation_section(ui, g, dirty);
    ui.separator();
    rendering_graph_section(ui, g, dirty);
    ui.separator();
    rendering_asset_section(ui, g, dirty);
    ui.unindent();
}

/// Edits presentation settings that affect swapchain mode or startup graphics API.
fn rendering_presentation_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Presentation");
    ui.indent();
    ui.text_disabled(
        "Presentation mode (FIFO is traditional VSync; Immediate is traditional no VSync).",
    );
    for (i, &mode) in PresentationModeSetting::ALL.iter().enumerate() {
        let _id = ui.push_id_int(200 + i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.rendering.presentation_mode == mode)
            .build()
        {
            g.rendering.presentation_mode = mode;
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

/// Edits render-graph-facing output format and sample-count settings.
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

/// Edits per-frame asset integration budgets.
fn rendering_asset_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Assets");
    ui.indent();
    if drag_u32_slider_setting(
        ui,
        "Asset integration budget (ms)",
        &mut g.rendering.asset_integration_budget_ms,
        0,
        MAX_ASSET_INTEGRATION_BUDGET_MS,
    ) {
        *dirty = true;
    }
    if drag_u32_slider_setting(
        ui,
        "Extra particle integration budget (ms)",
        &mut g.rendering.asset_particle_integration_budget_ms,
        0,
        MAX_ASSET_INTEGRATION_BUDGET_MS,
    ) {
        *dirty = true;
    }
    ui.text_disabled("Cooperative per-frame asset and extra dynamic-buffer integration budgets.");
    ui.unindent();
}
