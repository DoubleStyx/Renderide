//! Shared controls used by renderer-config HUD sections.

use imgui::{Drag, SliderFlags};

const SLIDER_DRAG_STEPS: f32 = 200.0;
const MIN_U32_SLIDER_SPEED: f32 = 0.001;
const MIN_F32_SLIDER_SPEED: f32 = 0.001;

/// Edits a `u32` setting through an ImGui drag widget with clamped integer output.
pub(in crate::diagnostics::hud::windows::renderer_config) fn drag_u32_setting(
    ui: &imgui::Ui,
    label: &str,
    value: &mut u32,
    min: u32,
    max: u32,
    speed: f32,
) -> bool {
    let original = *value;
    let mut edited = *value;
    if Drag::new(label)
        .range(min, max)
        .speed(speed)
        .flags(SliderFlags::ALWAYS_CLAMP)
        .build(ui, &mut edited)
    {
        *value = edited.clamp(min, max);
        return *value != original;
    }
    false
}

/// Edits a former slider-backed `u32` setting with a range-scaled drag speed.
pub(in crate::diagnostics::hud::windows::renderer_config) fn drag_u32_slider_setting(
    ui: &imgui::Ui,
    label: &str,
    value: &mut u32,
    min: u32,
    max: u32,
) -> bool {
    drag_u32_setting(ui, label, value, min, max, u32_slider_drag_speed(min, max))
}

/// Edits an `f32` setting through an ImGui drag widget with clamped output.
pub(in crate::diagnostics::hud::windows::renderer_config) fn drag_f32_setting(
    ui: &imgui::Ui,
    label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    speed: f32,
    display_format: Option<&str>,
) -> bool {
    let original = *value;
    let mut edited = *value;
    let changed = if let Some(display_format) = display_format {
        Drag::new(label)
            .range(min, max)
            .speed(speed)
            .display_format(display_format)
            .flags(SliderFlags::ALWAYS_CLAMP)
            .build(ui, &mut edited)
    } else {
        Drag::new(label)
            .range(min, max)
            .speed(speed)
            .flags(SliderFlags::ALWAYS_CLAMP)
            .build(ui, &mut edited)
    };

    if changed && edited.is_finite() {
        *value = edited.clamp(min, max);
        return *value != original;
    }
    false
}

/// Edits a former slider-backed `f32` setting with a range-scaled drag speed.
pub(in crate::diagnostics::hud::windows::renderer_config) fn drag_f32_slider_setting(
    ui: &imgui::Ui,
    label: &str,
    value: &mut f32,
    min: f32,
    max: f32,
    display_format: Option<&str>,
) -> bool {
    drag_f32_setting(
        ui,
        label,
        value,
        min,
        max,
        f32_slider_drag_speed(min, max),
        display_format,
    )
}

fn u32_slider_drag_speed(min: u32, max: u32) -> f32 {
    (max.saturating_sub(min) as f32 / SLIDER_DRAG_STEPS).max(MIN_U32_SLIDER_SPEED)
}

fn f32_slider_drag_speed(min: f32, max: f32) -> f32 {
    let span = (max - min).abs();
    if span.is_finite() {
        (span / SLIDER_DRAG_STEPS).max(MIN_F32_SLIDER_SPEED)
    } else {
        MIN_F32_SLIDER_SPEED
    }
}
