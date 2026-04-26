//! Maps winit physical keys to host [`Key`](crate::shared::Key) values.
//!
//! Covers the alphanumeric, numpad, function-key, navigation, and modifier subset the host
//! understands; unknown [`KeyCode`](winit::keyboard::KeyCode) variants resolve to `None` so they
//! are not added to [`super::WindowInputAccumulator::held_keys`].

use winit::keyboard::{KeyCode, PhysicalKey};

use crate::shared::Key;

/// Alphanumeric block, punctuation row, bracket row, and bare modifier keys (not arrows / numpad / F-keys).
fn map_keycode_alphanumeric_span(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Tab => Key::Tab,
        KeyCode::Enter => Key::Return,
        KeyCode::Escape => Key::Escape,
        KeyCode::Space => Key::Space,
        KeyCode::Digit0 => Key::Alpha0,
        KeyCode::Digit1 => Key::Alpha1,
        KeyCode::Digit2 => Key::Alpha2,
        KeyCode::Digit3 => Key::Alpha3,
        KeyCode::Digit4 => Key::Alpha4,
        KeyCode::Digit5 => Key::Alpha5,
        KeyCode::Digit6 => Key::Alpha6,
        KeyCode::Digit7 => Key::Alpha7,
        KeyCode::Digit8 => Key::Alpha8,
        KeyCode::Digit9 => Key::Alpha9,
        KeyCode::KeyA => Key::A,
        KeyCode::KeyB => Key::B,
        KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J,
        KeyCode::KeyK => Key::K,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyN => Key::N,
        KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyT => Key::T,
        KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y,
        KeyCode::KeyZ => Key::Z,
        KeyCode::BracketLeft => Key::LeftBracket,
        KeyCode::Backslash => Key::Backslash,
        KeyCode::BracketRight => Key::RightBracket,
        KeyCode::Minus => Key::Minus,
        KeyCode::Equal => Key::Equals,
        KeyCode::Backquote => Key::BackQuote,
        KeyCode::Semicolon => Key::Semicolon,
        KeyCode::Quote => Key::Quote,
        KeyCode::Comma => Key::Comma,
        KeyCode::Period => Key::Period,
        KeyCode::Slash => Key::Slash,
        _ => return None,
    })
}

/// Keypad digits and keypad operators (includes keypad Enter and Equals).
fn map_keycode_numpad(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::Numpad0 => Key::Keypad0,
        KeyCode::Numpad1 => Key::Keypad1,
        KeyCode::Numpad2 => Key::Keypad2,
        KeyCode::Numpad3 => Key::Keypad3,
        KeyCode::Numpad4 => Key::Keypad4,
        KeyCode::Numpad5 => Key::Keypad5,
        KeyCode::Numpad6 => Key::Keypad6,
        KeyCode::Numpad7 => Key::Keypad7,
        KeyCode::Numpad8 => Key::Keypad8,
        KeyCode::Numpad9 => Key::Keypad9,
        KeyCode::NumpadDecimal => Key::KeypadPeriod,
        KeyCode::NumpadDivide => Key::KeypadDivide,
        KeyCode::NumpadMultiply => Key::KeypadMultiply,
        KeyCode::NumpadSubtract => Key::KeypadMinus,
        KeyCode::NumpadAdd => Key::KeypadPlus,
        KeyCode::NumpadEnter => Key::KeypadEnter,
        KeyCode::NumpadEqual => Key::KeypadEquals,
        _ => return None,
    })
}

/// Navigation cluster, function keys, and left/right shift/ctrl/alt/super.
fn map_keycode_nav_function_modifiers(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::ArrowUp => Key::UpArrow,
        KeyCode::ArrowDown => Key::DownArrow,
        KeyCode::ArrowLeft => Key::LeftArrow,
        KeyCode::ArrowRight => Key::RightArrow,
        KeyCode::Insert => Key::Insert,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::F1 => Key::F1,
        KeyCode::F2 => Key::F2,
        KeyCode::F3 => Key::F3,
        KeyCode::F4 => Key::F4,
        KeyCode::F5 => Key::F5,
        KeyCode::F6 => Key::F6,
        KeyCode::F7 => Key::F7,
        KeyCode::F8 => Key::F8,
        KeyCode::F9 => Key::F9,
        KeyCode::F10 => Key::F10,
        KeyCode::F11 => Key::F11,
        KeyCode::F12 => Key::F12,
        KeyCode::F13 => Key::F13,
        KeyCode::F14 => Key::F14,
        KeyCode::F15 => Key::F15,
        KeyCode::NumLock => Key::Numlock,
        KeyCode::CapsLock => Key::CapsLock,
        KeyCode::ScrollLock => Key::ScrollLock,
        KeyCode::ShiftLeft => Key::LeftShift,
        KeyCode::ShiftRight => Key::RightShift,
        KeyCode::ControlLeft => Key::LeftControl,
        KeyCode::ControlRight => Key::RightControl,
        KeyCode::AltLeft => Key::LeftAlt,
        KeyCode::AltRight => Key::RightAlt,
        KeyCode::SuperLeft => Key::LeftWindows,
        KeyCode::SuperRight => Key::RightWindows,
        KeyCode::Delete => Key::Delete,
        KeyCode::PrintScreen => Key::Print,
        KeyCode::Pause => Key::Pause,
        KeyCode::ContextMenu => Key::Menu,
        _ => return None,
    })
}

/// Maps winit [`PhysicalKey`] to the IPC [`Key`] enum, if the host defines a matching variant.
pub fn winit_key_to_renderite_key(physical_key: PhysicalKey) -> Option<Key> {
    let code = match physical_key {
        PhysicalKey::Code(c) => c,
        PhysicalKey::Unidentified(_) => return None,
    };
    map_keycode_alphanumeric_span(code)
        .or_else(|| map_keycode_numpad(code))
        .or_else(|| map_keycode_nav_function_modifiers(code))
}

#[cfg(test)]
mod tests {
    use winit::keyboard::{KeyCode, NativeKeyCode, PhysicalKey};

    use super::winit_key_to_renderite_key;
    use crate::shared::Key;

    #[test]
    fn maps_arrows_and_modifier_keys() {
        assert_eq!(
            winit_key_to_renderite_key(PhysicalKey::Code(KeyCode::ArrowUp)),
            Some(Key::UpArrow)
        );
        assert_eq!(
            winit_key_to_renderite_key(PhysicalKey::Code(KeyCode::ShiftLeft)),
            Some(Key::LeftShift)
        );
    }

    #[test]
    fn unidentified_physical_key_maps_to_none() {
        assert!(
            winit_key_to_renderite_key(PhysicalKey::Unidentified(NativeKeyCode::Unidentified))
                .is_none()
        );
    }
}
