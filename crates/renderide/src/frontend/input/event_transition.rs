//! Pure input event transitions used by the winit adapter.

use glam::Vec2;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::PhysicalKey;

use super::key_map::winit_key_to_renderite_key;
use crate::shared::Key;

/// Mouse button storage slot in [`super::WindowInputAccumulator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MouseButtonSlot {
    /// Primary button.
    Left,
    /// Secondary button.
    Right,
    /// Middle button.
    Middle,
    /// Back button.
    Button4,
    /// Forward button.
    Button5,
}

/// Pure mouse-button state transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MouseButtonTransition {
    /// Accumulator slot to update.
    pub(crate) slot: MouseButtonSlot,
    /// New held state.
    pub(crate) pressed: bool,
}

/// Held-key list transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HeldKeyTransition {
    /// Add key to held list if absent.
    Press(Key),
    /// Remove key from held list if present.
    Release(Key),
}

/// Pure keyboard event transition.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KeyboardEventTransition {
    /// Held-key mutation, if the physical key maps to the host key enum.
    pub(crate) held_key: Option<HeldKeyTransition>,
    /// Printable text to append to the text buffer.
    pub(crate) text: Option<String>,
}

/// Converts a winit mouse button event into an accumulator transition.
pub(crate) fn mouse_button_transition(
    state: ElementState,
    button: MouseButton,
) -> Option<MouseButtonTransition> {
    let slot = match button {
        MouseButton::Left => MouseButtonSlot::Left,
        MouseButton::Right => MouseButtonSlot::Right,
        MouseButton::Middle => MouseButtonSlot::Middle,
        MouseButton::Back => MouseButtonSlot::Button4,
        MouseButton::Forward => MouseButtonSlot::Button5,
        _ => return None,
    };
    Some(MouseButtonTransition {
        slot,
        pressed: state == ElementState::Pressed,
    })
}

/// Converts winit scroll units into host scroll delta.
pub(crate) fn scroll_delta_from_wheel(delta: &MouseScrollDelta) -> Vec2 {
    const SCROLL_SCALE: f32 = 120.0;
    match delta {
        MouseScrollDelta::LineDelta(x, y) => Vec2::new(*x * SCROLL_SCALE, *y * SCROLL_SCALE),
        MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x as f32, p.y as f32),
    }
}

/// Converts a non-synthetic key event into held-key/text transitions.
pub(crate) fn keyboard_event_transition(event: &KeyEvent) -> KeyboardEventTransition {
    keyboard_transition_from_parts(
        event.physical_key,
        event.state,
        event.repeat,
        event.text.as_deref(),
    )
}

/// Converts key event fields into held-key/text transitions.
pub(crate) fn keyboard_transition_from_parts(
    physical_key: PhysicalKey,
    state: ElementState,
    repeat: bool,
    text: Option<&str>,
) -> KeyboardEventTransition {
    if repeat {
        return KeyboardEventTransition {
            held_key: None,
            text: None,
        };
    }
    let text = (state == ElementState::Pressed)
        .then_some(text)
        .flatten()
        .filter(|text| !text.is_empty())
        .map(|text| text.to_string());
    let held_key = winit_key_to_renderite_key(physical_key).map(|key| match state {
        ElementState::Pressed => HeldKeyTransition::Press(key),
        ElementState::Released => HeldKeyTransition::Release(key),
    });
    KeyboardEventTransition { held_key, text }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::{KeyCode, PhysicalKey};

    #[test]
    fn mouse_buttons_map_to_slots() {
        let transition =
            mouse_button_transition(ElementState::Pressed, MouseButton::Back).expect("mapped");
        assert_eq!(transition.slot, MouseButtonSlot::Button4);
        assert!(transition.pressed);
        assert!(mouse_button_transition(ElementState::Pressed, MouseButton::Button9).is_none());
    }

    #[test]
    fn line_scroll_scales_to_pixels() {
        let delta = scroll_delta_from_wheel(&MouseScrollDelta::LineDelta(1.0, -2.0));
        assert_eq!(delta, Vec2::new(120.0, -240.0));
    }

    #[test]
    fn keyboard_transition_maps_held_key_and_text() {
        let transition = keyboard_transition_from_parts(
            PhysicalKey::Code(KeyCode::KeyA),
            ElementState::Pressed,
            false,
            Some("a"),
        );
        assert_eq!(transition.held_key, Some(HeldKeyTransition::Press(Key::A)));
        assert_eq!(transition.text.as_deref(), Some("a"));
    }
}
