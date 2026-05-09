//! Adapts winit 0.30 events into [`WindowInputAccumulator`](super::WindowInputAccumulator).

use std::path::Path;

use winit::dpi::LogicalSize;
use winit::event::{
    ButtonSource, DeviceEvent, ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta,
    WindowEvent,
};
use winit::window::Window;

use super::accumulator::WindowInputAccumulator;
use super::event_transition::{
    HeldKeyTransition, KeyboardEventTransition, MouseButtonSlot, keyboard_event_transition,
    mouse_button_transition, scroll_delta_from_wheel,
};

/// Applies a [`WindowEvent`] from winit to the accumulator.
///
/// [`WindowEvent::Resized`], [`WindowEvent::ScaleFactorChanged`], and cursor move use the same
/// **logical** pixel space as [`WindowInputAccumulator::window_position`].
pub fn apply_window_event(
    acc: &mut WindowInputAccumulator,
    window: &dyn Window,
    event: &WindowEvent,
) {
    match event {
        WindowEvent::SurfaceResized(size) => {
            profiling::scope!("frontend::window_event", "resize");
            let logical: LogicalSize<f64> = size.to_logical(window.scale_factor());
            acc.window_resolution = (logical.width.round() as u32, logical.height.round() as u32);
        }
        WindowEvent::ScaleFactorChanged { .. } => {
            profiling::scope!("frontend::window_event", "scale_factor");
            acc.sync_window_resolution_logical(window);
        }
        WindowEvent::PointerMoved { position, .. } => {
            profiling::scope!("frontend::window_event", "cursor_moved");
            acc.set_cursor_from_physical(*position, window.scale_factor());
        }
        WindowEvent::PointerEntered { .. } => {
            profiling::scope!("frontend::window_event", "cursor_entered");
            acc.mouse_active = true;
        }
        WindowEvent::PointerLeft { .. } => {
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
        WindowEvent::ModifiersChanged(modifiers) => {
            profiling::scope!("frontend::window_event", "modifiers");
            acc.set_keyboard_modifiers(modifiers.state());
        }
        WindowEvent::PointerButton { state, button, .. } => {
            profiling::scope!("frontend::window_event", "mouse_button");
            if let ButtonSource::Mouse(mouse_button) = button {
                apply_mouse_button(acc, *state, *mouse_button);
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            profiling::scope!("frontend::window_event", "scroll");
            apply_mouse_wheel(acc, delta);
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
            apply_keyboard_event(acc, event);
        }
        WindowEvent::Ime(ime) => {
            profiling::scope!("frontend::window_event", "ime");
            match ime {
                Ime::Commit(s) => acc.push_ime_commit(s.as_str()),
                Ime::Enabled
                | Ime::Disabled
                | Ime::Preedit(_, _)
                | Ime::DeleteSurrounding { .. } => {}
            }
        }
        WindowEvent::DragDropped { paths, .. } => {
            profiling::scope!("frontend::window_event", "dropped_file");
            for path in paths {
                acc.push_dropped_file_path(path_to_string_lossy(path));
            }
        }
        _ => {}
    }
}

/// Updates per-button held flags for a [`WindowEvent::MouseInput`].
fn apply_mouse_button(acc: &mut WindowInputAccumulator, state: ElementState, button: MouseButton) {
    let Some(transition) = mouse_button_transition(state, button) else {
        return;
    };
    match transition.slot {
        MouseButtonSlot::Left => acc.left_held = transition.pressed,
        MouseButtonSlot::Right => acc.right_held = transition.pressed,
        MouseButtonSlot::Middle => acc.middle_held = transition.pressed,
        MouseButtonSlot::Button4 => acc.button4_held = transition.pressed,
        MouseButtonSlot::Button5 => acc.button5_held = transition.pressed,
    }
}

/// Accumulates scroll delta for a [`WindowEvent::MouseWheel`] (line-scale normalised to pixels).
fn apply_mouse_wheel(acc: &mut WindowInputAccumulator, delta: &MouseScrollDelta) {
    acc.scroll_delta += scroll_delta_from_wheel(delta);
}

/// Updates held-key list and queued text-input strings for a non-synthetic [`KeyEvent`].
fn apply_keyboard_event(acc: &mut WindowInputAccumulator, event: &KeyEvent) {
    let transition = keyboard_event_transition(event);
    apply_keyboard_transition(acc, transition);
}

fn apply_keyboard_transition(
    acc: &mut WindowInputAccumulator,
    transition: KeyboardEventTransition,
) {
    match transition.held_key {
        Some(HeldKeyTransition::Press(key)) if !acc.held_keys.contains(&key) => {
            acc.held_keys.push(key);
        }
        Some(HeldKeyTransition::Release(key)) => {
            acc.held_keys.retain(|held| *held != key);
        }
        Some(HeldKeyTransition::Press(_)) | None => {}
    }
    if let Some(text) = transition.text {
        acc.push_key_text(text.as_str());
    }
}

fn path_to_string_lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Applies relative pointer motion when the cursor is captured (locked / confined).
pub fn apply_device_event(acc: &mut WindowInputAccumulator, event: &DeviceEvent) {
    if let DeviceEvent::PointerMotion { delta } = event {
        profiling::scope!("frontend::device_event", "mouse_motion");
        acc.mouse_delta.x += delta.0 as f32;
        acc.mouse_delta.y -= delta.1 as f32;
    }
}
