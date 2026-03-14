//! Session and render configuration types.
//!
//! Engine-agnostic configuration structures used by the renderer framework.

use crate::shared::HeadOutputDevice;

/// Session configuration derived from host init data.
#[derive(Clone, Default)]
pub struct SessionConfig {
    /// Shared memory prefix for mmap-based IPC buffers.
    pub shared_memory_prefix: Option<String>,
    /// Output device (screen, VR, etc.).
    pub output_device: HeadOutputDevice,
}

/// Render configuration (clip planes, FOV, display settings).
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Near clip plane distance.
    pub near_clip: f32,
    /// Far clip plane distance.
    pub far_clip: f32,
    /// Desktop field of view in degrees.
    pub desktop_fov: f32,
    /// Whether vertical sync is enabled.
    pub vsync: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            near_clip: 0.01,
            far_clip: 1024.0,
            desktop_fov: 75.0,
            vsync: false,
        }
    }
}
