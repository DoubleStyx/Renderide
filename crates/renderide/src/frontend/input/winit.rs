//! Adapts winit 0.30 events into [`WindowInputAccumulator`](super::WindowInputAccumulator).

use std::path::Path;

use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent};
use winit::window::Window;

use super::accumulator::WindowInputAccumulator;
use super::key_map::winit_key_to_renderite_key;

/// Applies a [`WindowEvent`] from winit to the accumulator.
///
/// [`WindowEvent::Resized`], [`WindowEvent::ScaleFactorChanged`], and cursor move use the same
/// **logical** pixel space as [`WindowInputAccumulator::window_position`].
pub fn apply_window_event(acc: &mut WindowInputAccumulator, window: &Window, event: &WindowEvent) {
    match event {
        WindowEvent::Resized(size) => {
            profiling::scope!("frontend::window_event", "resize");
            let logical: LogicalSize<f64> = size.to_logical(window.scale_factor());
            acc.window_resolution = (logical.width.round() as u32, logical.height.round() as u32);
        }
        WindowEvent::ScaleFactorChanged { .. } => {
            profiling::scope!("frontend::window_event", "scale_factor");
            acc.sync_window_resolution_logical(window);
        }
        WindowEvent::CursorMoved { position, .. } => {
            profiling::scope!("frontend::window_event", "cursor_moved");
            acc.set_cursor_from_physical(*position, window.scale_factor());
        }
        WindowEvent::CursorEntered { .. } => {
            profiling::scope!("frontend::window_event", "cursor_entered");
            acc.mouse_active = true;
        }
        WindowEvent::CursorLeft { .. } => {
            profiling::scope!("frontend::window_event", "cursor_left");
            acc.mouse_active = false;
        }
        WindowEvent::Focused(focused) => {
            profiling::scope!("frontend::window_event", "focus");
            acc.window_focused = *focused;
            if !*focused {
                acc.clear_stuck_keyboard_on_focus_lost();
            }
        }
        WindowEvent::MouseInput { state, button, .. } => {
            profiling::scope!("frontend::window_event", "mouse_button");
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
            profiling::scope!("frontend::window_event", "scroll");
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
            profiling::scope!("frontend::window_event", "key");
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
        WindowEvent::Ime(ime) => {
            profiling::scope!("frontend::window_event", "ime");
            match ime {
                Ime::Commit(s) => acc.push_ime_commit(s.as_str()),
                Ime::Enabled | Ime::Disabled | Ime::Preedit(_, _) => {}
            }
        }
        WindowEvent::DroppedFile(path) => {
            profiling::scope!("frontend::window_event", "dropped_file");
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
        profiling::scope!("frontend::device_event", "mouse_motion");
        acc.mouse_delta.x += delta.0 as f32;
        acc.mouse_delta.y -= delta.1 as f32;
    }
}
