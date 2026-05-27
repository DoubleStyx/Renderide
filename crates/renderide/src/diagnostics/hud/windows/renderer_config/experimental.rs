//! Experimental renderer-config HUD controls.

use crate::config::RendererSettings;
use crate::reflection_probes::specular::MAX_LOCAL_PROBES;

/// Experimental feature flags.
pub(super) fn experimental_section(ui: &imgui::Ui, g: &mut RendererSettings, dirty: &mut bool) {
    ui.text("Experimental");
    ui.indent();
    let mut mrp = g.experimental.effective_max_local_reflection_probes();
    if ui.slider(
        "Maximum number of local reflection probes per mesh",
        0,
        MAX_LOCAL_PROBES,
        &mut mrp,
    ) {
        g.experimental.max_local_reflection_probes = mrp.clamp(0, MAX_LOCAL_PROBES);
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
    ui.unindent();
}
