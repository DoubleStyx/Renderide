//! Host [`crate::shared::OutputState`] cursor policy and winit grab/warp helpers.

use glam::IVec2;
#[cfg(not(target_os = "macos"))]
use glam::Vec2;
use winit::dpi::LogicalPosition;
#[cfg(not(target_os = "macos"))]
use winit::dpi::LogicalSize;
use winit::window::{CursorGrabMode, Window};

use super::accumulator::WindowInputAccumulator;
use crate::shared::OutputState;

/// Tracks host [`OutputState`] cursor fields between frames for parity with the Unity renderer
/// mouse driver (early exit when unchanged, unlock warp to the previous confined position).
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorOutputTracking {
    last_lock_cursor: bool,
    last_lock_position: Option<IVec2>,
}

fn warp_cursor_logical(window: &Window, p: &IVec2) -> Result<(), winit::error::ExternalError> {
    let logical = LogicalPosition::new(p.x as f64, p.y as f64);
    let physical = logical.to_physical::<f64>(window.scale_factor());
    window.set_cursor_position(physical)
}

/// Reapplies grab and warp **every frame** while the host requests cursor lock (matches the legacy
/// renderer redraw path: center when no freeze position, else the host lock point).
///
/// Call after [`apply_output_state_to_window`] when [`OutputState::lock_cursor`] is true so relative
/// look and IPC [`crate::shared::MouseState::window_position`] stay aligned with the OS cursor.
#[cfg(not(target_os = "macos"))]
pub fn apply_per_frame_cursor_lock_when_locked(
    window: &Window,
    acc: &mut WindowInputAccumulator,
    lock_cursor_position: Option<IVec2>,
) -> Result<(), winit::error::ExternalError> {
    let sf = window.scale_factor();
    acc.sync_window_resolution_logical(window);

    if let Some(p) = lock_cursor_position {
        window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))?;
        window.set_cursor_visible(false);
        warp_cursor_logical(window, &p)?;
        acc.set_window_position_from_logical(Vec2::new(p.x as f32, p.y as f32), sf);
    } else {
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))?;
        window.set_cursor_visible(false);
        let physical = window.inner_size();
        let logical_sz: LogicalSize<f64> = physical.to_logical(sf);
        let cx = (logical_sz.width / 2.0) as f32;
        let cy = (logical_sz.height / 2.0) as f32;
        let logical_center = LogicalPosition::new(cx as f64, cy as f64);
        let phys_center = logical_center.to_physical::<f64>(sf);
        window.set_cursor_position(phys_center)?;
        acc.set_window_position_from_logical(Vec2::new(cx, cy), sf);
    }
    Ok(())
}

/// Host cursor lock without per-frame warping.
///
/// On macOS this is intentionally a no-op: reapplying center warps every frame breaks relative mouse
/// input with winit. Grab and visibility for [`OutputState::lock_cursor`] are still applied from
/// [`apply_output_state_to_window`]; only continuous re-centering is omitted.
#[cfg(target_os = "macos")]
pub fn apply_per_frame_cursor_lock_when_locked(
    _window: &Window,
    _acc: &mut WindowInputAccumulator,
    _lock_cursor_position: Option<IVec2>,
) -> Result<(), winit::error::ExternalError> {
    Ok(())
}

/// Applies host [`OutputState`] to the winit window (IME, grab transitions, warps). Use
/// [`apply_per_frame_cursor_lock_when_locked`] each frame while locked for continuous re-centering
/// (a no-op on macOS; see that function’s documentation).
pub fn apply_output_state_to_window(
    window: &Window,
    state: &OutputState,
    track: &mut CursorOutputTracking,
) -> Result<(), winit::error::ExternalError> {
    window.set_ime_allowed(state.keyboard_input_active);

    if let Some(ref p) = state.lock_cursor_position {
        let _ = warp_cursor_logical(window, p);
    }

    if state.lock_cursor == track.last_lock_cursor
        && state.lock_cursor_position == track.last_lock_position
    {
        return Ok(());
    }

    let prev_lock_position_for_unlock = track.last_lock_position;

    track.last_lock_cursor = state.lock_cursor;
    track.last_lock_position = state.lock_cursor_position;

    if state.lock_cursor {
        if state.lock_cursor_position.is_some() {
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))?;
        } else {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))?;
        }
        window.set_cursor_visible(false);
        return Ok(());
    }

    window.set_cursor_grab(CursorGrabMode::None)?;
    window.set_cursor_visible(true);
    if let Some(ref p) = prev_lock_position_for_unlock {
        let _ = warp_cursor_logical(window, p);
    }
    Ok(())
}
