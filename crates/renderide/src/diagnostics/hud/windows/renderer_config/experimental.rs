//! Experimental renderer-config HUD controls.

use crate::config::RendererSettings;
use crate::reflection_probes::specular::MAX_LOCAL_PROBES;

use super::controls::drag_u32_slider_setting;

/// Experimental feature flags.
pub(super) fn experimental_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Experimental");
    ui.indent();
    let mut mrp = g.experimental.effective_max_local_reflection_probes() as u32;
    if drag_u32_slider_setting(
        ui,
        "Maximum number of local reflection probes per mesh",
        &mut mrp,
        0,
        MAX_LOCAL_PROBES as u32,
    ) {
        g.experimental.max_local_reflection_probes = mrp as usize;
        *dirty = true;
    }
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
    if ui.checkbox(
        "360 degree main view",
        &mut g.experimental.main_view_360_enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled(
        "When enabled, the desktop/headless main view renders through a cubemap-backed equirectangular projection.",
    );
    ui.unindent();
}
