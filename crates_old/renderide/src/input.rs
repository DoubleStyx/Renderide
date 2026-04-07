//! Window input state and key mapping for host IPC.

use nalgebra::Vector2;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::shared::{InputState, Key, KeyboardState, MouseState, WindowState};

/// Accumulated window input for gaze/mouse (packed into host IPC via [`FrameStartData`](crate::shared::FrameStartData)).
///
/// Mouse and scroll deltas accumulate across winit events. [`Session::update`](crate::session::Session::update)
/// calls [`WindowInputState::take_input_state`] only when it actually sends `frame_start_data`, so movement
/// is not cleared on redraws that wait for the host.
pub struct WindowInputState {
    pub mouse_delta: Vector2<f32>,
    pub scroll_delta: Vector2<f32>,
    pub window_position: Vector2<f32>,
    pub window_resolution: (u32, u32),
    pub left_held: bool,
    pub right_held: bool,
    pub middle_held: bool,
    pub button4_held: bool,
    pub button5_held: bool,
    pub mouse_active: bool,
    pub window_focused: bool,
    pub held_keys: Vec<Key>,
}

impl Default for WindowInputState {
    fn default() -> Self {
        Self {
            mouse_delta: Vector2::zeros(),
            scroll_delta: Vector2::zeros(),
            window_position: Vector2::zeros(),
            window_resolution: (0, 0),
            left_held: false,
            right_held: false,
            middle_held: false,
            button4_held: false,
            button5_held: false,
            mouse_active: false,
            window_focused: true,
            held_keys: Vec::new(),
        }
    }
}

impl WindowInputState {
    /// Consumes accumulated mouse and scroll deltas and returns an [`InputState`] for the host.
    ///
    /// Clears `mouse_delta` and `scroll_delta`. The session invokes this only when it sends
    /// `frame_start_data`, not on every redraw.
    pub fn take_input_state(&mut self) -> InputState {
        let mouse = MouseState {
            is_active: self.mouse_active,
            left_button_state: self.left_held,
            right_button_state: self.right_held,
            middle_button_state: self.middle_held,
            button4_state: self.button4_held,
            button5_state: self.button5_held,
            desktop_position: self.window_position,
            window_position: self.window_position,
            direct_delta: std::mem::take(&mut self.mouse_delta),
            scroll_wheel_delta: std::mem::take(&mut self.scroll_delta),
        };
        let window = WindowState {
            is_window_focused: self.window_focused,
            is_fullscreen: false,
            window_resolution: Vector2::new(
                self.window_resolution.0 as i32,
                self.window_resolution.1 as i32,
            ),
            resolution_settings_applied: false,
            drag_and_drop_event: None,
        };
        let keyboard = Some(KeyboardState {
            type_delta: None,
            held_keys: self.held_keys.clone(),
        });
        InputState {
            mouse: Some(mouse),
            keyboard,
            window: Some(window),
            vr: None,
            gamepads: Vec::new(),
            touches: Vec::new(),
            displays: Vec::new(),
        }
    }
}

/// Maps winit PhysicalKey to Renderite Key.
pub fn winit_key_to_renderite_key(physical_key: PhysicalKey) -> Option<Key> {
    let code = match physical_key {
        PhysicalKey::Code(c) => c,
        PhysicalKey::Unidentified(_) => return None,
    };
    Some(match code {
        KeyCode::Backspace => Key::backspace,
        KeyCode::Tab => Key::tab,
        KeyCode::Enter => Key::r#return,
        KeyCode::Escape => Key::escape,
        KeyCode::Space => Key::space,
        KeyCode::Digit0 => Key::alpha0,
        KeyCode::Digit1 => Key::alpha1,
        KeyCode::Digit2 => Key::alpha2,
        KeyCode::Digit3 => Key::alpha3,
        KeyCode::Digit4 => Key::alpha4,
        KeyCode::Digit5 => Key::alpha5,
        KeyCode::Digit6 => Key::alpha6,
        KeyCode::Digit7 => Key::alpha7,
        KeyCode::Digit8 => Key::alpha8,
        KeyCode::Digit9 => Key::alpha9,
        KeyCode::KeyA => Key::a,
        KeyCode::KeyB => Key::b,
        KeyCode::KeyC => Key::c,
        KeyCode::KeyD => Key::d,
        KeyCode::KeyE => Key::e,
        KeyCode::KeyF => Key::f,
        KeyCode::KeyG => Key::g,
        KeyCode::KeyH => Key::h,
        KeyCode::KeyI => Key::i,
        KeyCode::KeyJ => Key::j,
        KeyCode::KeyK => Key::k,
        KeyCode::KeyL => Key::l,
        KeyCode::KeyM => Key::m,
        KeyCode::KeyN => Key::n,
        KeyCode::KeyO => Key::o,
        KeyCode::KeyP => Key::p,
        KeyCode::KeyQ => Key::q,
        KeyCode::KeyR => Key::r,
        KeyCode::KeyS => Key::s,
        KeyCode::KeyT => Key::t,
        KeyCode::KeyU => Key::u,
        KeyCode::KeyV => Key::v,
        KeyCode::KeyW => Key::w,
        KeyCode::KeyX => Key::x,
        KeyCode::KeyY => Key::y,
        KeyCode::KeyZ => Key::z,
        KeyCode::BracketLeft => Key::left_bracket,
        KeyCode::Backslash => Key::backslash,
        KeyCode::BracketRight => Key::right_bracket,
        KeyCode::Minus => Key::minus,
        KeyCode::Equal => Key::equals,
        KeyCode::Backquote => Key::back_quote,
        KeyCode::Semicolon => Key::semicolon,
        KeyCode::Quote => Key::quote,
        KeyCode::Comma => Key::comma,
        KeyCode::Period => Key::period,
        KeyCode::Slash => Key::slash,
        KeyCode::Numpad0 => Key::keypad0,
        KeyCode::Numpad1 => Key::keypad1,
        KeyCode::Numpad2 => Key::keypad2,
        KeyCode::Numpad3 => Key::keypad3,
        KeyCode::Numpad4 => Key::keypad4,
        KeyCode::Numpad5 => Key::keypad5,
        KeyCode::Numpad6 => Key::keypad6,
        KeyCode::Numpad7 => Key::keypad7,
        KeyCode::Numpad8 => Key::keypad8,
        KeyCode::Numpad9 => Key::keypad9,
        KeyCode::NumpadDecimal => Key::keypad_period,
        KeyCode::NumpadDivide => Key::keypad_divide,
        KeyCode::NumpadMultiply => Key::keypad_multiply,
        KeyCode::NumpadSubtract => Key::keypad_minus,
        KeyCode::NumpadAdd => Key::keypad_plus,
        KeyCode::NumpadEnter => Key::keypad_enter,
        KeyCode::NumpadEqual => Key::keypad_equals,
        KeyCode::ArrowUp => Key::up_arrow,
        KeyCode::ArrowDown => Key::down_arrow,
        KeyCode::ArrowLeft => Key::left_arrow,
        KeyCode::ArrowRight => Key::right_arrow,
        KeyCode::Insert => Key::insert,
        KeyCode::Home => Key::home,
        KeyCode::End => Key::end,
        KeyCode::PageUp => Key::page_up,
        KeyCode::PageDown => Key::page_down,
        KeyCode::F1 => Key::f1,
        KeyCode::F2 => Key::f2,
        KeyCode::F3 => Key::f3,
        KeyCode::F4 => Key::f4,
        KeyCode::F5 => Key::f5,
        KeyCode::F6 => Key::f6,
        KeyCode::F7 => Key::f7,
        KeyCode::F8 => Key::f8,
        KeyCode::F9 => Key::f9,
        KeyCode::F10 => Key::f10,
        KeyCode::F11 => Key::f11,
        KeyCode::F12 => Key::f12,
        KeyCode::F13 => Key::f13,
        KeyCode::F14 => Key::f14,
        KeyCode::F15 => Key::f15,
        KeyCode::NumLock => Key::numlock,
        KeyCode::CapsLock => Key::caps_lock,
        KeyCode::ScrollLock => Key::scroll_lock,
        KeyCode::ShiftLeft => Key::left_shift,
        KeyCode::ShiftRight => Key::right_shift,
        KeyCode::ControlLeft => Key::left_control,
        KeyCode::ControlRight => Key::right_control,
        KeyCode::AltLeft => Key::left_alt,
        KeyCode::AltRight => Key::right_alt,
        KeyCode::SuperLeft => Key::left_windows,
        KeyCode::SuperRight => Key::right_windows,
        KeyCode::Delete => Key::delete,
        KeyCode::PrintScreen => Key::print,
        KeyCode::Pause => Key::pause,
        KeyCode::ContextMenu => Key::menu,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::WindowInputState;
    use nalgebra::Vector2;

    /// Regression: mouse deltas must survive multiple logical "redraw" ticks until a single
    /// `take_input_state` (which happens when `frame_start_data` is sent).
    #[test]
    fn mouse_delta_accumulates_until_take_input_state() {
        let mut w = WindowInputState::default();
        w.mouse_delta += Vector2::new(1.0, 2.0);
        w.mouse_delta += Vector2::new(3.0, 4.0);
        let first = w.take_input_state();
        let mouse = first.mouse.expect("mouse state");
        assert_eq!(mouse.direct_delta.x, 4.0);
        assert_eq!(mouse.direct_delta.y, 6.0);
        let second = w.take_input_state();
        let mouse2 = second.mouse.expect("mouse state");
        assert_eq!(mouse2.direct_delta.x, 0.0);
        assert_eq!(mouse2.direct_delta.y, 0.0);
    }

    #[test]
    fn scroll_delta_accumulates_until_take_input_state() {
        let mut w = WindowInputState::default();
        w.scroll_delta += Vector2::new(0.0, 120.0);
        w.scroll_delta += Vector2::new(0.0, 60.0);
        let taken = w.take_input_state();
        let mouse = taken.mouse.expect("mouse state");
        assert_eq!(mouse.scroll_wheel_delta.y, 180.0);
    }
}
