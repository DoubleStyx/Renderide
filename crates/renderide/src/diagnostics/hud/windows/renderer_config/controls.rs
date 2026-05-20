//! Shared controls used by renderer-config HUD sections.

use imgui::Drag;

/// Edits a `u32` setting through an ImGui drag widget with clamped integer output.
pub(in crate::diagnostics::hud::windows::renderer_config) fn drag_u32_setting(
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
