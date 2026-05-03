//! First-use positions and size constraints for HUD windows.
//!
//! Layout has two layers:
//!
//! - Stacked-column constants (`MARGIN`, `GAP`, `RENDERER_CONFIG_*`, `FRAME_TIMING_RESERVE_H`)
//!   and the legacy first-use position helpers (`frame_timing_xy`, `scene_transforms_y`) drive
//!   the four anchored windows so **Renderer config**, **Frame timing**, **Renderide debug**,
//!   and **Scene transforms** do not share the same anchor (ImGui `FirstUseEver` only applies
//!   once).
//! - The structured [`Viewport`] / [`WindowAnchor`] / [`WindowSlot`] types describe a window's
//!   first-use placement declaratively. Anchors resolve against the current viewport into a
//!   concrete [`WindowSlot`] consumed by ImGui `position` and `size_constraints` calls.

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
/// First-use width of the **Renderide debug** main panel (anchored to the viewport's top-right
/// corner). Pulled out of the panel render path so layout decisions live in one place.
pub const MAIN_DEBUG_PANEL_W: f32 = 760.0;
/// First-use height of the **Renderide debug** main panel.
pub const MAIN_DEBUG_PANEL_H: f32 = 460.0;

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

/// Current viewport extent in physical pixels, used by [`WindowAnchor::resolve`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Viewport {
    /// Viewport width in physical pixels.
    pub width: u32,
    /// Viewport height in physical pixels.
    pub height: u32,
}

/// Declarative first-use placement of a HUD window.
///
/// Anchors are resolved against the current [`Viewport`] into a concrete [`WindowSlot`]; the
/// caller hands the slot to ImGui via `position` and `size_constraints`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WindowAnchor {
    /// Pin to the viewport top-right, offset by [`MARGIN`]. Width-only anchor -- ImGui's
    /// `ALWAYS_AUTO_RESIZE` is expected to drive the height.
    TopRight {
        /// First-use width.
        width: f32,
    },
}

/// Concrete first-use position and size-constraint pair resolved from a [`WindowAnchor`].
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

impl WindowAnchor {
    /// Resolve this anchor against the current viewport into a [`WindowSlot`].
    pub fn resolve(self, viewport: Viewport) -> WindowSlot {
        match self {
            WindowAnchor::TopRight { width } => {
                let panel_x = (viewport.width as f32 - width - MARGIN).max(MARGIN);
                WindowSlot {
                    position: [panel_x, MARGIN],
                    size: [width, MAIN_DEBUG_PANEL_H],
                    size_min: [420.0, 160.0],
                    size_max: [1.0e9, 1.0e9],
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAIN_DEBUG_PANEL_H, MARGIN, Viewport, WindowAnchor};

    #[test]
    fn top_right_anchor_pulls_window_in_by_margin_on_wide_viewport() {
        let v = Viewport {
            width: 1920,
            height: 1080,
        };
        let slot = WindowAnchor::TopRight { width: 760.0 }.resolve(v);
        assert_eq!(slot.position[1], MARGIN);
        assert_eq!(slot.size, [760.0, MAIN_DEBUG_PANEL_H]);
        // 1920 - 760 - 12 = 1148
        assert!((slot.position[0] - 1148.0).abs() < 0.5);
    }

    #[test]
    fn top_right_anchor_clamps_to_margin_when_viewport_narrower_than_panel() {
        let v = Viewport {
            width: 600,
            height: 400,
        };
        let slot = WindowAnchor::TopRight { width: 760.0 }.resolve(v);
        // 600 - 760 - 12 < 0; clamp to MARGIN.
        assert_eq!(slot.position, [MARGIN, MARGIN]);
    }
}
