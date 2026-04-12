//! First-use positions for HUD windows so **Renderer config**, **Frame timing**, **Renderide debug**,
//! and **Scene transforms** do not share the same anchor (ImGui `FirstUseEver` only applies once).

/// Margin from the viewport edge for anchored HUD windows.
pub const MARGIN: f32 = 12.0;
/// Gap between stacked HUD windows on the left column.
pub const GAP: f32 = 16.0;
/// Matches the first-use width of the **Renderer config** window.
pub const RENDERER_CONFIG_W: f32 = 440.0;
/// Matches the first-use height of the **Renderer config** window.
pub const RENDERER_CONFIG_H: f32 = 400.0;
/// Reserved vertical space for the auto-sized **Frame timing** window so **Scene transforms**
/// can be placed below without overlapping on first use.
pub const FRAME_TIMING_RESERVE_H: f32 = 140.0;

/// First-use position for **Frame timing**: directly under **Renderer config** (same column).
pub fn frame_timing_xy() -> [f32; 2] {
    [MARGIN, MARGIN + RENDERER_CONFIG_H + GAP]
}

/// Minimum Y for **Scene transforms** so it stays below **Renderer config** + **Frame timing**.
pub fn scene_transforms_min_y() -> f32 {
    MARGIN + RENDERER_CONFIG_H + GAP + FRAME_TIMING_RESERVE_H + GAP
}

/// First-use Y for **Scene transforms**: prefers the bottom of the viewport minus the window
/// height, but not above [`scene_transforms_min_y`] (avoids covering the config / timing stack).
pub fn scene_transforms_y(viewport_h: f32, window_h: f32) -> f32 {
    let bottom_anchored = viewport_h - window_h - MARGIN;
    bottom_anchored.max(scene_transforms_min_y())
}
