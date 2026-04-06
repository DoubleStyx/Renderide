//! Adapts winit 0.30 events into [`WindowInputAccumulator`](super::WindowInputAccumulator).

use std::path::Path;

use nalgebra::Vector2;
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::event::{DeviceEvent, ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent};
use winit::window::{CursorGrabMode, Window};

use super::accumulator::WindowInputAccumulator;
use super::key_map::winit_key_to_renderite_key;
use crate::shared::OutputState;

/// Tracks host [`OutputState`] cursor fields between frames for parity with the Unity renderer
/// mouse driver (early exit when unchanged, unlock warp to the previous confined position).
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorOutputTracking {
    last_lock_cursor: bool,
    last_lock_position: Option<Vector2<i32>>,
}

/// Applies a [`WindowEvent`] from winit to the accumulator.
///
/// [`WindowEvent::Resized`], [`WindowEvent::ScaleFactorChanged`], and cursor move use the same
/// **logical** pixel space as [`WindowInputAccumulator::window_position`].
pub fn apply_window_event(acc: &mut WindowInputAccumulator, window: &Window, event: &WindowEvent) {
    match event {
        WindowEvent::Resized(size) => {
            let logical: LogicalSize<f64> = size.to_logical(window.scale_factor());
            acc.window_resolution = (
                logical.width.round() as u32,
                logical.height.round() as u32,
            );
        }
        WindowEvent::ScaleFactorChanged { .. } => {
            acc.sync_window_resolution_logical(window);
        }
        WindowEvent::CursorMoved { position, .. } => {
            acc.set_cursor_from_physical(*position, window.scale_factor());
        }
        WindowEvent::CursorEntered { .. } => acc.mouse_active = true,
        WindowEvent::CursorLeft { .. } => acc.mouse_active = false,
        WindowEvent::Focused(focused) => acc.window_focused = *focused,
        WindowEvent::MouseInput { state, button, .. } => {
            let pressed = *state == ElementState::Pressed;
            match button {
                MouseButton::Left => acc.left_held = pressed,
                MouseButton::Right => acc.right_held = pressed,
                MouseButton::Middle => acc.middle_held = pressed,
                MouseButton::Back => acc.button4_held = pressed,
                MouseButton::Forward => acc.button5_held = pressed,
                MouseButton::Other(_) => {}
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            const SCROLL_SCALE: f32 = 120.0;
            match delta {
                MouseScrollDelta::LineDelta(x, y) => {
                    acc.scroll_delta.x += *x * SCROLL_SCALE;
                    acc.scroll_delta.y += *y * SCROLL_SCALE;
                }
                MouseScrollDelta::PixelDelta(p) => {
                    acc.scroll_delta.x += p.x as f32;
                    acc.scroll_delta.y += p.y as f32;
                }
            }
        }
        WindowEvent::KeyboardInput {
            event,
            is_synthetic,
            ..
        } => {
            if *is_synthetic {
                return;
            }
            if event.repeat {
                return;
            }
            if let Some(key) = winit_key_to_renderite_key(event.physical_key) {
                match event.state {
                    ElementState::Pressed => {
                        if !acc.held_keys.contains(&key) {
                            acc.held_keys.push(key);
                        }
                        if let Some(text) = event.text.as_ref() {
                            if !text.is_empty() {
                                acc.push_key_text(text.as_str());
                            }
                        }
                    }
                    ElementState::Released => {
                        acc.held_keys.retain(|held| *held != key);
                    }
                }
            } else if event.state == ElementState::Pressed {
                if let Some(text) = event.text.as_ref() {
                    if !text.is_empty() {
                        acc.push_key_text(text.as_str());
                    }
                }
            }
        }
        WindowEvent::Ime(ime) => match ime {
            Ime::Commit(s) => acc.push_ime_commit(s.as_str()),
            Ime::Enabled | Ime::Disabled | Ime::Preedit(_, _) => {}
        },
        WindowEvent::DroppedFile(path) => {
            acc.push_dropped_file_path(path_to_string_lossy(path));
        }
        _ => {}
    }
}

fn path_to_string_lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Applies relative pointer motion when the cursor is captured (locked / confined).
pub fn apply_device_event(acc: &mut WindowInputAccumulator, event: &DeviceEvent) {
    if let DeviceEvent::MouseMotion { delta } = event {
        acc.mouse_delta.x += delta.0 as f32;
        acc.mouse_delta.y -= delta.1 as f32;
    }
}

fn warp_cursor_logical(window: &Window, p: &Vector2<i32>) -> Result<(), winit::error::ExternalError> {
    let logical = LogicalPosition::new(p.x as f64, p.y as f64);
    let physical = logical.to_physical::<f64>(window.scale_factor());
    window.set_cursor_position(physical)
}

/// Reapplies grab and warp **every frame** while the host requests cursor lock (matches the legacy
/// renderer redraw path: center when no freeze position, else the host lock point).
///
/// Call after [`apply_output_state_to_window`] when [`OutputState::lock_cursor`] is true so relative
/// look and IPC [`MouseState::window_position`] stay aligned with the OS cursor.
pub fn apply_per_frame_cursor_lock_when_locked(
    window: &Window,
    acc: &mut WindowInputAccumulator,
    lock_cursor_position: Option<Vector2<i32>>,
) -> Result<(), winit::error::ExternalError> {
    let sf = window.scale_factor();
    acc.sync_window_resolution_logical(window);

    if let Some(p) = lock_cursor_position {
        window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))?;
        window.set_cursor_visible(false);
        warp_cursor_logical(window, &p)?;
        acc.set_window_position_from_logical(Vector2::new(p.x as f32, p.y as f32), sf);
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
        acc.set_window_position_from_logical(Vector2::new(cx, cy), sf);
    }
    Ok(())
}

/// Applies host [`OutputState`] to the winit window (IME, grab transitions, warps). Use
/// [`apply_per_frame_cursor_lock_when_locked`] each frame while locked for continuous re-centering.
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
