//! First-use positions and size constraints for HUD windows.
//!
//! Layout has two layers:
//!
//! - Stacked-column constants (`MARGIN`, `GAP`, `RENDERER_CONFIG_*`, `FRAME_TIMING_RESERVE_H`)
//!   and first-use slot helpers drive the anchored windows so **Renderer config**,
//!   **Frame timing**, **Feedback / Bug Report**, **Renderide debug**, and **Scene transforms**
//!   do not share the same anchor (ImGui `FirstUseEver` only applies once).
//! - The structured [`Viewport`] / [`WindowSlot`] types describe a window's first-use placement
//!   declaratively. Slot helpers resolve against the current viewport into concrete values
//!   consumed by ImGui `position` and `size_constraints` calls.

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
pub const FRAME_TIMING_RESERVE_H: f32 = 225.0;
/// First-use width of the **Renderide debug** main panel (anchored to the viewport's top-right
/// corner). Pulled out of the panel render path so layout decisions live in one place.
pub const MAIN_DEBUG_PANEL_W: f32 = 760.0;
/// First-use height of the **Renderide debug** main panel.
pub const MAIN_DEBUG_PANEL_H: f32 = 460.0;
/// First-use width of the compact **Feedback / Bug Report** panel.
pub const FEEDBACK_PANEL_W: f32 = 292.0;
/// Reserved first-use height of the compact **Feedback / Bug Report** panel.
pub const FEEDBACK_PANEL_H: f32 = 54.0;

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

/// First-use slot for the compact **Feedback / Bug Report** panel.
pub fn feedback_panel_slot(viewport: Viewport) -> WindowSlot {
    top_right_slot(
        viewport,
        FEEDBACK_PANEL_W,
        MARGIN,
        [FEEDBACK_PANEL_W, FEEDBACK_PANEL_H],
        [0.0, 0.0],
    )
}

/// First-use slot for the **Renderide debug** panel below **Feedback / Bug Report**.
pub fn main_debug_panel_slot(viewport: Viewport) -> WindowSlot {
    top_right_slot(
        viewport,
        MAIN_DEBUG_PANEL_W,
        MARGIN + FEEDBACK_PANEL_H + GAP,
        [MAIN_DEBUG_PANEL_W, MAIN_DEBUG_PANEL_H],
        [420.0, 160.0],
    )
}

fn top_right_slot(
    viewport: Viewport,
    width: f32,
    y: f32,
    size: [f32; 2],
    size_min: [f32; 2],
) -> WindowSlot {
    let panel_x = (viewport.width as f32 - width - MARGIN).max(MARGIN);
    WindowSlot {
        position: [panel_x, y],
        size,
        size_min,
        size_max: [1.0e9, 1.0e9],
    }
}

/// Current viewport extent in physical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Viewport {
    /// Viewport width in physical pixels.
    pub width: u32,
    /// Viewport height in physical pixels.
    pub height: u32,
}

/// Concrete first-use position and size-constraint pair resolved from a [`Viewport`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowSlot {
    /// First-use top-left position in physical pixels.
    pub position: [f32; 2],
    /// First-use size in physical pixels.
    pub size: [f32; 2],
    /// First-use minimum size constraint.
    pub size_min: [f32; 2],
    /// First-use maximum size constraint.
    pub size_max: [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::{
        FEEDBACK_PANEL_H, FEEDBACK_PANEL_W, GAP, MAIN_DEBUG_PANEL_H, MAIN_DEBUG_PANEL_W, MARGIN,
        Viewport, feedback_panel_slot, main_debug_panel_slot,
    };

    #[test]
    fn feedback_panel_slot_pins_to_top_right() {
        let v = Viewport {
            width: 1920,
            height: 1080,
        };
        let slot = feedback_panel_slot(v);

        assert_eq!(slot.position[1], MARGIN);
        assert_eq!(slot.size, [FEEDBACK_PANEL_W, FEEDBACK_PANEL_H]);
        assert!((slot.position[0] - (1920.0 - FEEDBACK_PANEL_W - MARGIN)).abs() < 0.5);
    }

    #[test]
    fn main_debug_panel_slot_starts_below_feedback_panel() {
        let v = Viewport {
            width: 1920,
            height: 1080,
        };
        let slot = main_debug_panel_slot(v);

        assert_eq!(slot.position[1], MARGIN + FEEDBACK_PANEL_H + GAP);
        assert_eq!(slot.size, [MAIN_DEBUG_PANEL_W, MAIN_DEBUG_PANEL_H]);
        assert!((slot.position[0] - (1920.0 - MAIN_DEBUG_PANEL_W - MARGIN)).abs() < 0.5);
    }

    #[test]
    fn top_right_panel_slots_clamp_to_margin_on_narrow_viewports() {
        let v = Viewport {
            width: 240,
            height: 400,
        };

        assert_eq!(feedback_panel_slot(v).position[0], MARGIN);
        assert_eq!(main_debug_panel_slot(v).position[0], MARGIN);
    }
}
