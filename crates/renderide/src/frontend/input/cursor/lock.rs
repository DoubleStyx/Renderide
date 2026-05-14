//! Host [`crate::shared::OutputState`] cursor lock policy: grab/visibility transitions and
//! software cursor-position anchoring while relative mouse mode is active.

use glam::IVec2;
use glam::Vec2;
use winit::dpi::LogicalSize;
use winit::dpi::{LogicalPosition, Position};
use winit::window::{CursorGrabMode, ImeRequest, Window};

use super::super::accumulator::WindowInputAccumulator;
use super::ime::enable_ime_on_window;
use crate::shared::OutputState;

#[cfg(any(test, not(target_os = "macos")))]
const CONFINED_RECENTER_MARGIN_PX: f32 = 8.0;
#[cfg(any(test, not(target_os = "macos")))]
const CONFINED_RECENTER_DRIFT_PX: f32 = 96.0;

/// Tracks host [`OutputState`] cursor fields between frames so unchanged lock state avoids
/// redundant window-system calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorOutputTracking {
    last_lock_cursor: bool,
    last_lock_position: Option<IVec2>,
    active_grab_mode: Option<CursorGrabMode>,
}

fn warp_cursor_logical(window: &dyn Window, p: Vec2) -> Result<(), winit::error::RequestError> {
    let logical = LogicalPosition::new(f64::from(p.x), f64::from(p.y));
    window.set_cursor_position(Position::Logical(logical))
}

fn logical_window_size(window: &dyn Window) -> LogicalSize<f64> {
    window.surface_size().to_logical(window.scale_factor())
}

fn lock_cursor_position_or_center(
    lock_cursor_position: Option<IVec2>,
    window: &dyn Window,
) -> Vec2 {
    lock_cursor_position.map_or_else(
        || {
            let logical = logical_window_size(window);
            Vec2::new((logical.width / 2.0) as f32, (logical.height / 2.0) as f32)
        },
        |p| Vec2::new(p.x as f32, p.y as f32),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CursorGrabPreference {
    primary: CursorGrabMode,
    fallback: CursorGrabMode,
}

fn grab_preference(lock_cursor_position: Option<IVec2>) -> CursorGrabPreference {
    if lock_cursor_position.is_some() {
        CursorGrabPreference {
            primary: CursorGrabMode::Confined,
            fallback: CursorGrabMode::Locked,
        }
    } else {
        CursorGrabPreference {
            primary: CursorGrabMode::Locked,
            fallback: CursorGrabMode::Confined,
        }
    }
}

fn apply_cursor_grab(
    window: &dyn Window,
    preference: CursorGrabPreference,
) -> Result<CursorGrabMode, winit::error::RequestError> {
    if window.set_cursor_grab(preference.primary).is_ok() {
        return Ok(preference.primary);
    }
    window.set_cursor_grab(preference.fallback)?;
    Ok(preference.fallback)
}

fn lock_state_changed(state: &OutputState, track: &CursorOutputTracking) -> bool {
    state.lock_cursor != track.last_lock_cursor
        || state.lock_cursor_position != track.last_lock_position
}

#[cfg(any(test, not(target_os = "macos")))]
fn should_recenter_confined_cursor(observed: Vec2, target: Vec2, resolution: (u32, u32)) -> bool {
    if resolution.0 == 0 || resolution.1 == 0 {
        return false;
    }
    let max_x = resolution.0 as f32;
    let max_y = resolution.1 as f32;
    let near_edge = observed.x <= CONFINED_RECENTER_MARGIN_PX
        || observed.y <= CONFINED_RECENTER_MARGIN_PX
        || observed.x >= max_x - CONFINED_RECENTER_MARGIN_PX
        || observed.y >= max_y - CONFINED_RECENTER_MARGIN_PX;
    let large_drift = observed.distance_squared(target)
        >= CONFINED_RECENTER_DRIFT_PX * CONFINED_RECENTER_DRIFT_PX;
    near_edge || large_drift
}

/// Maintains the reported cursor position while the host requests cursor lock.
///
/// Call after [`apply_output_state_to_window`] when [`OutputState::lock_cursor`] is true so relative
/// look and IPC [`crate::shared::MouseState::window_position`] stay anchored without issuing OS
/// grab or warp calls every frame. When the platform fell back to `Confined`, the hidden OS cursor
/// is re-centered only after edge contact or large drift.
#[cfg(not(target_os = "macos"))]
pub fn apply_per_frame_cursor_lock_when_locked(
    window: &dyn Window,
    acc: &mut WindowInputAccumulator,
    lock_cursor_position: Option<IVec2>,
    track: &CursorOutputTracking,
) -> Result<(), winit::error::RequestError> {
    let sf = window.scale_factor();
    acc.sync_window_resolution_logical(window);
    let target = lock_cursor_position_or_center(lock_cursor_position, window);
    let observed = acc.window_position;
    let recenter_confined = track.active_grab_mode == Some(CursorGrabMode::Confined)
        && should_recenter_confined_cursor(observed, target, acc.window_resolution);
    acc.set_window_position_from_logical(target, sf);
    if recenter_confined {
        warp_cursor_logical(window, target)?;
        acc.set_window_position_from_logical(target, sf);
    }
    Ok(())
}

/// Maintains the reported cursor position while the host requests cursor lock.
///
/// On macOS this deliberately avoids OS cursor warps because reapplying center warps every frame
/// breaks relative mouse input with winit. Grab and visibility for [`OutputState::lock_cursor`] are
/// still applied from [`apply_output_state_to_window`].
#[cfg(target_os = "macos")]
pub fn apply_per_frame_cursor_lock_when_locked(
    window: &dyn Window,
    acc: &mut WindowInputAccumulator,
    lock_cursor_position: Option<IVec2>,
    _track: &CursorOutputTracking,
) -> Result<(), winit::error::RequestError> {
    let target = lock_cursor_position_or_center(lock_cursor_position, window);
    acc.sync_window_resolution_logical(window);
    acc.set_window_position_from_logical(target, window.scale_factor());
    Ok(())
}

/// Applies host [`OutputState`] to the winit window (IME, grab transitions, and transition warps).
/// Use [`apply_per_frame_cursor_lock_when_locked`] each frame while locked for software anchoring.
pub fn apply_output_state_to_window(
    window: &dyn Window,
    state: &OutputState,
    track: &mut CursorOutputTracking,
) -> Result<(), winit::error::RequestError> {
    if state.keyboard_input_active {
        if window.ime_capabilities().is_none() {
            enable_ime_on_window(window);
        }
    } else if window.ime_capabilities().is_some() {
        let _ = window.request_ime_update(ImeRequest::Disable);
    }

    if !lock_state_changed(state, track) {
        return Ok(());
    }

    let prev_lock_position_for_unlock = track.last_lock_position;

    if state.lock_cursor {
        let active_grab_mode = Some(apply_cursor_grab(
            window,
            grab_preference(state.lock_cursor_position),
        )?);
        window.set_cursor_visible(false);
        let target = lock_cursor_position_or_center(state.lock_cursor_position, window);
        let _ = warp_cursor_logical(window, target);
        track.last_lock_cursor = state.lock_cursor;
        track.last_lock_position = state.lock_cursor_position;
        track.active_grab_mode = active_grab_mode;
        return Ok(());
    }

    window.set_cursor_grab(CursorGrabMode::None)?;
    window.set_cursor_visible(true);
    if let Some(p) = prev_lock_position_for_unlock {
        let _ = warp_cursor_logical(window, Vec2::new(p.x as f32, p.y as f32));
    }
    track.last_lock_cursor = state.lock_cursor;
    track.last_lock_position = state.lock_cursor_position;
    track.active_grab_mode = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use glam::{IVec2, Vec2};
    use winit::window::CursorGrabMode;

    use super::{
        CONFINED_RECENTER_DRIFT_PX, CONFINED_RECENTER_MARGIN_PX, CursorGrabPreference,
        CursorOutputTracking, grab_preference, lock_state_changed, should_recenter_confined_cursor,
    };
    use crate::shared::OutputState;

    #[test]
    fn grab_preference_uses_locked_for_relative_mode() {
        assert_eq!(
            grab_preference(None),
            CursorGrabPreference {
                primary: CursorGrabMode::Locked,
                fallback: CursorGrabMode::Confined,
            }
        );
    }

    #[test]
    fn grab_preference_uses_confined_for_explicit_position() {
        assert_eq!(
            grab_preference(Some(IVec2::new(12, 34))),
            CursorGrabPreference {
                primary: CursorGrabMode::Confined,
                fallback: CursorGrabMode::Locked,
            }
        );
    }

    #[test]
    fn unchanged_lock_state_skips_window_policy() {
        let track = CursorOutputTracking {
            last_lock_cursor: true,
            last_lock_position: Some(IVec2::new(100, 200)),
            active_grab_mode: Some(CursorGrabMode::Locked),
        };
        let state = OutputState {
            lock_cursor: true,
            lock_cursor_position: Some(IVec2::new(100, 200)),
            ..Default::default()
        };

        assert!(!lock_state_changed(&state, &track));
    }

    #[test]
    fn lock_position_change_reapplies_window_policy() {
        let track = CursorOutputTracking {
            last_lock_cursor: true,
            last_lock_position: Some(IVec2::new(100, 200)),
            active_grab_mode: Some(CursorGrabMode::Locked),
        };
        let state = OutputState {
            lock_cursor: true,
            lock_cursor_position: Some(IVec2::new(120, 200)),
            ..Default::default()
        };

        assert!(lock_state_changed(&state, &track));
    }

    #[test]
    fn confined_recenter_ignores_stable_cursor_near_target() {
        assert!(!should_recenter_confined_cursor(
            Vec2::new(400.0, 300.0),
            Vec2::new(402.0, 301.0),
            (800, 600),
        ));
    }

    #[test]
    fn confined_recenter_triggers_near_edge() {
        assert!(should_recenter_confined_cursor(
            Vec2::new(CONFINED_RECENTER_MARGIN_PX - 1.0, 300.0),
            Vec2::new(400.0, 300.0),
            (800, 600),
        ));
    }

    #[test]
    fn confined_recenter_triggers_on_large_drift() {
        assert!(should_recenter_confined_cursor(
            Vec2::new(400.0 + CONFINED_RECENTER_DRIFT_PX, 300.0),
            Vec2::new(400.0, 300.0),
            (800, 600),
        ));
    }

    #[test]
    fn confined_recenter_ignores_unknown_resolution() {
        assert!(!should_recenter_confined_cursor(
            Vec2::ZERO,
            Vec2::new(400.0, 300.0),
            (0, 600),
        ));
    }
}
