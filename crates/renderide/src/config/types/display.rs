//! Display caps. Persisted as `[display]`.

use serde::{Deserialize, Serialize};

/// Desktop display frame-pacing caps. Persisted as `[display]`.
///
/// Non-zero values cap desktop redraw scheduling via winit (`ControlFlow::WaitUntil`) while
/// presentation mode is not presentation-paced. HMD compositor-paced frames ignore these caps so
/// headset frame pacing is unchanged, but desktop presentation still uses them when an OpenXR
/// session exists.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// Target max FPS while the renderer window is the foreground input window (0 = uncapped).
    #[serde(rename = "focused_fps")]
    pub focused_fps_cap: u32,
    /// Target max FPS while the renderer window is in the background (0 = uncapped).
    #[serde(rename = "unfocused_fps")]
    pub unfocused_fps_cap: u32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            focused_fps_cap: 240,
            unfocused_fps_cap: 60,
        }
    }
}
