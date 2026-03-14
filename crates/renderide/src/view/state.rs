//! View state: primary view, clip planes, FOV.

use crate::core::View;

/// Holds current view configuration from the host.
pub struct ViewState {
    /// Primary view (from active render space or first camera task).
    pub primary_view: Option<View>,
    /// Near clip plane.
    pub near_clip: f32,
    /// Far clip plane.
    pub far_clip: f32,
    /// Desktop field of view in degrees.
    pub desktop_fov: f32,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            primary_view: None,
            near_clip: 0.01,
            far_clip: 1024.0,
            desktop_fov: 75.0,
        }
    }
}
