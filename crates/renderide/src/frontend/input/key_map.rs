//! Maps winit physical keys to host [`Key`](crate::shared::Key) values.
//!
//! Covers the same subset as the legacy `crates_old` table; unknown [`KeyCode`](winit::keyboard::KeyCode)
//! variants resolve to `None` so they are not added to [`super::WindowInputAccumulator::held_keys`].

use winit::keyboard::{KeyCode, PhysicalKey};

use crate::shared::Key;

/// Maps winit [`PhysicalKey`] to the IPC [`Key`] enum, if the host defines a matching variant.
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
    use winit::keyboard::{KeyCode, NativeKeyCode, PhysicalKey};

    use super::winit_key_to_renderite_key;
    use crate::shared::Key;

    #[test]
    fn maps_arrows_and_modifier_keys() {
        assert_eq!(
            winit_key_to_renderite_key(PhysicalKey::Code(KeyCode::ArrowUp)),
            Some(Key::up_arrow)
        );
        assert_eq!(
            winit_key_to_renderite_key(PhysicalKey::Code(KeyCode::ShiftLeft)),
            Some(Key::left_shift)
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
