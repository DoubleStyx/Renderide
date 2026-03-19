//! Session state: init lifecycle, view configuration.

use crate::scene::View;

/// Init lifecycle state. Replaces `init_received`/`init_finalized` booleans to prevent invalid combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitState {
    /// No renderer_init_data received; only InitCommandHandler accepts commands.
    #[default]
    Uninitialized,
    /// renderer_init_data received; waiting for renderer_init_finalize_data.
    InitReceived,
    /// Init complete (finalize received or standalone mode). Normal operation.
    Finalized,
}

impl InitState {
    /// Whether only init commands are accepted.
    pub fn is_uninitialized(self) -> bool {
        matches!(self, InitState::Uninitialized)
    }

    /// Whether init is complete and normal operation can proceed.
    pub fn is_finalized(self) -> bool {
        matches!(self, InitState::Finalized)
    }
}

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
