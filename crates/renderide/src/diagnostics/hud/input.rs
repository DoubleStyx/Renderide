//! HUD input transport and ImGui IO bridge.
//!
//! [`DebugHudInput`] is the per-frame snapshot built from winit's
//! [`crate::frontend::input::WindowInputAccumulator`]; [`apply_input`] feeds it into ImGui's
//! [`imgui::Io`] before each frame so the HUD's input matches the renderer's logical input.
//! [`sanitize_input_state_for_imgui_host`] strips pointer, scroll, drag/drop, and keyboard data
//! from the [`InputState`] sent to the host while ImGui reports it captured the previous frame.

use glam::Vec2;
use imgui::{Io, Key as ImGuiKey, MouseButton as ImGuiMouseButton};
use winit::keyboard::ModifiersState;

use crate::shared::{InputState, Key};

const SUPPORTED_KEY_EVENTS: &[(Key, ImGuiKey)] = &[
    (Key::Tab, ImGuiKey::Tab),
    (Key::LeftArrow, ImGuiKey::LeftArrow),
    (Key::RightArrow, ImGuiKey::RightArrow),
    (Key::UpArrow, ImGuiKey::UpArrow),
    (Key::DownArrow, ImGuiKey::DownArrow),
    (Key::PageUp, ImGuiKey::PageUp),
    (Key::PageDown, ImGuiKey::PageDown),
    (Key::Home, ImGuiKey::Home),
    (Key::End, ImGuiKey::End),
    (Key::Insert, ImGuiKey::Insert),
    (Key::Delete, ImGuiKey::Delete),
    (Key::Backspace, ImGuiKey::Backspace),
    (Key::Space, ImGuiKey::Space),
    (Key::Return, ImGuiKey::Enter),
    (Key::Escape, ImGuiKey::Escape),
    (Key::LeftControl, ImGuiKey::LeftCtrl),
    (Key::LeftShift, ImGuiKey::LeftShift),
    (Key::LeftAlt, ImGuiKey::LeftAlt),
    (Key::LeftWindows, ImGuiKey::LeftSuper),
    (Key::RightControl, ImGuiKey::RightCtrl),
    (Key::RightShift, ImGuiKey::RightShift),
    (Key::RightAlt, ImGuiKey::RightAlt),
    (Key::RightWindows, ImGuiKey::RightSuper),
    (Key::Menu, ImGuiKey::Menu),
    (Key::Alpha0, ImGuiKey::Alpha0),
    (Key::Alpha1, ImGuiKey::Alpha1),
    (Key::Alpha2, ImGuiKey::Alpha2),
    (Key::Alpha3, ImGuiKey::Alpha3),
    (Key::Alpha4, ImGuiKey::Alpha4),
    (Key::Alpha5, ImGuiKey::Alpha5),
    (Key::Alpha6, ImGuiKey::Alpha6),
    (Key::Alpha7, ImGuiKey::Alpha7),
    (Key::Alpha8, ImGuiKey::Alpha8),
    (Key::Alpha9, ImGuiKey::Alpha9),
    (Key::A, ImGuiKey::A),
    (Key::B, ImGuiKey::B),
    (Key::C, ImGuiKey::C),
    (Key::D, ImGuiKey::D),
    (Key::E, ImGuiKey::E),
    (Key::F, ImGuiKey::F),
    (Key::G, ImGuiKey::G),
    (Key::H, ImGuiKey::H),
    (Key::I, ImGuiKey::I),
    (Key::J, ImGuiKey::J),
    (Key::K, ImGuiKey::K),
    (Key::L, ImGuiKey::L),
    (Key::M, ImGuiKey::M),
    (Key::N, ImGuiKey::N),
    (Key::O, ImGuiKey::O),
    (Key::P, ImGuiKey::P),
    (Key::Q, ImGuiKey::Q),
    (Key::R, ImGuiKey::R),
    (Key::S, ImGuiKey::S),
    (Key::T, ImGuiKey::T),
    (Key::U, ImGuiKey::U),
    (Key::V, ImGuiKey::V),
    (Key::W, ImGuiKey::W),
    (Key::X, ImGuiKey::X),
    (Key::Y, ImGuiKey::Y),
    (Key::Z, ImGuiKey::Z),
    (Key::F1, ImGuiKey::F1),
    (Key::F2, ImGuiKey::F2),
    (Key::F3, ImGuiKey::F3),
    (Key::F4, ImGuiKey::F4),
    (Key::F5, ImGuiKey::F5),
    (Key::F6, ImGuiKey::F6),
    (Key::F7, ImGuiKey::F7),
    (Key::F8, ImGuiKey::F8),
    (Key::F9, ImGuiKey::F9),
    (Key::F10, ImGuiKey::F10),
    (Key::F11, ImGuiKey::F11),
    (Key::F12, ImGuiKey::F12),
    (Key::Quote, ImGuiKey::Apostrophe),
    (Key::Comma, ImGuiKey::Comma),
    (Key::Minus, ImGuiKey::Minus),
    (Key::Period, ImGuiKey::Period),
    (Key::Slash, ImGuiKey::Slash),
    (Key::Semicolon, ImGuiKey::Semicolon),
    (Key::Equals, ImGuiKey::Equal),
    (Key::LeftBracket, ImGuiKey::LeftBracket),
    (Key::Backslash, ImGuiKey::Backslash),
    (Key::RightBracket, ImGuiKey::RightBracket),
    (Key::BackQuote, ImGuiKey::GraveAccent),
    (Key::CapsLock, ImGuiKey::CapsLock),
    (Key::ScrollLock, ImGuiKey::ScrollLock),
    (Key::Numlock, ImGuiKey::NumLock),
    (Key::Print, ImGuiKey::PrintScreen),
    (Key::Pause, ImGuiKey::Pause),
    (Key::Keypad0, ImGuiKey::Keypad0),
    (Key::Keypad1, ImGuiKey::Keypad1),
    (Key::Keypad2, ImGuiKey::Keypad2),
    (Key::Keypad3, ImGuiKey::Keypad3),
    (Key::Keypad4, ImGuiKey::Keypad4),
    (Key::Keypad5, ImGuiKey::Keypad5),
    (Key::Keypad6, ImGuiKey::Keypad6),
    (Key::Keypad7, ImGuiKey::Keypad7),
    (Key::Keypad8, ImGuiKey::Keypad8),
    (Key::Keypad9, ImGuiKey::Keypad9),
    (Key::KeypadPeriod, ImGuiKey::KeypadDecimal),
    (Key::KeypadDivide, ImGuiKey::KeypadDivide),
    (Key::KeypadMultiply, ImGuiKey::KeypadMultiply),
    (Key::KeypadMinus, ImGuiKey::KeypadSubtract),
    (Key::KeypadPlus, ImGuiKey::KeypadAdd),
    (Key::KeypadEnter, ImGuiKey::KeypadEnter),
    (Key::KeypadEquals, ImGuiKey::KeypadEqual),
];

/// Strips pointer, scroll, drag/drop, and keyboard data from `input` when ImGui reported capture on the previous frame.
///
/// The renderer feeds ImGui from the same winit accumulator as the host; this keeps [`InputState`]
/// sent in [`crate::shared::FrameStartData`] from receiving clicks and keys that belong to the HUD.
/// Cursor positions are preserved so normalized UVs stay coherent.
pub fn sanitize_input_state_for_imgui_host(
    input: &mut InputState,
    want_capture_mouse: bool,
    want_capture_keyboard: bool,
) {
    if want_capture_mouse {
        if let Some(ref mut mouse) = input.mouse {
            mouse.left_button_state = false;
            mouse.right_button_state = false;
            mouse.middle_button_state = false;
            mouse.button4_state = false;
            mouse.button5_state = false;
            mouse.direct_delta = Vec2::ZERO;
            mouse.scroll_wheel_delta = Vec2::ZERO;
        }
        if let Some(ref mut window) = input.window {
            window.drag_and_drop_event = None;
        }
    }
    if want_capture_keyboard && let Some(ref mut keyboard) = input.keyboard {
        keyboard.type_delta = None;
        keyboard.held_keys.clear();
    }
}

/// Pointer and window hints for ImGui, in **physical** pixels where noted.
#[derive(Clone, Debug, Default)]
pub struct DebugHudInput {
    /// Cursor position in physical pixels (or `[-inf, -inf]` when unavailable).
    pub cursor_px: [f32; 2],
    /// Whether the window currently has keyboard focus.
    pub window_focused: bool,
    /// Whether the cursor is over the client area (from winit accumulator).
    pub mouse_active: bool,
    /// Scroll wheel delta in Dear ImGui wheel units.
    pub mouse_wheel_delta: Vec2,
    /// Left mouse button held.
    pub left: bool,
    /// Right mouse button held.
    pub right: bool,
    /// Middle mouse button held.
    pub middle: bool,
    /// Fourth mouse button held (e.g. side back).
    pub extra1: bool,
    /// Fifth mouse button held (e.g. side forward).
    pub extra2: bool,
    /// Current keyboard modifiers from winit.
    pub keyboard_modifiers: ModifiersState,
    /// Keys currently held, in host [`Key`] form.
    pub held_keys: Vec<Key>,
    /// Text committed since the previous HUD input snapshot.
    pub text: String,
}

impl DebugHudInput {
    /// Builds input for the HUD from winit and the accumulated window/input state.
    ///
    /// Cursor is **`WindowInputAccumulator::window_position` (logical) x scale factor**, matching the
    /// swapchain / ImGui framebuffer in **physical** pixels.
    pub fn from_winit(
        window: &dyn winit::window::Window,
        acc: &mut crate::frontend::input::WindowInputAccumulator,
    ) -> Self {
        let sf = window.scale_factor() as f32;
        let cursor_px = if acc.mouse_active && acc.window_focused {
            [acc.window_position.x * sf, acc.window_position.y * sf]
        } else {
            [-f32::MAX, -f32::MAX]
        };
        Self {
            cursor_px,
            window_focused: acc.window_focused,
            mouse_active: acc.mouse_active,
            mouse_wheel_delta: acc.take_hud_scroll_delta(),
            left: acc.left_held,
            right: acc.right_held,
            middle: acc.middle_held,
            extra1: acc.button4_held,
            extra2: acc.button5_held,
            keyboard_modifiers: acc.keyboard_modifiers(),
            held_keys: acc.held_keys.clone(),
            text: acc.take_hud_text(),
        }
    }
}

/// Feeds winit-derived [`DebugHudInput`] into ImGui `io` before each frame.
///
/// Cursor position is parked off-screen when the host reports the mouse inactive or the window
/// unfocused so that ImGui does not treat stale positions as hovered events.
pub(crate) fn apply_input(io: &mut Io, input: &DebugHudInput) {
    if input.mouse_active && input.window_focused {
        io.add_mouse_pos_event(input.cursor_px);
    } else {
        io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
    }
    io.add_mouse_button_event(ImGuiMouseButton::Left, input.left);
    io.add_mouse_button_event(ImGuiMouseButton::Right, input.right);
    io.add_mouse_button_event(ImGuiMouseButton::Middle, input.middle);
    io.add_mouse_button_event(ImGuiMouseButton::Extra1, input.extra1);
    io.add_mouse_button_event(ImGuiMouseButton::Extra2, input.extra2);
    io.add_mouse_wheel_event([input.mouse_wheel_delta.x, input.mouse_wheel_delta.y]);
    apply_keyboard_input(io, input);
}

fn apply_keyboard_input(io: &mut Io, input: &DebugHudInput) {
    for (key, down) in modifier_key_states(input.keyboard_modifiers) {
        io.add_key_event(key, down);
    }
    for &(host_key, imgui_key) in SUPPORTED_KEY_EVENTS {
        io.add_key_event(imgui_key, input.held_keys.contains(&host_key));
    }
    for character in input.text.chars() {
        io.add_input_character(character);
    }
}

fn modifier_key_states(modifiers: ModifiersState) -> [(ImGuiKey, bool); 4] {
    [
        (ImGuiKey::ModCtrl, modifiers.control_key()),
        (ImGuiKey::ModShift, modifiers.shift_key()),
        (ImGuiKey::ModAlt, modifiers.alt_key()),
        (ImGuiKey::ModSuper, modifiers.meta_key()),
    ]
}

#[cfg(test)]
fn host_key_to_imgui_key(key: Key) -> Option<ImGuiKey> {
    SUPPORTED_KEY_EVENTS
        .iter()
        .find_map(|&(host_key, imgui_key)| (host_key == key).then_some(imgui_key))
}

#[cfg(test)]
mod input_bridge_tests {
    use imgui::Context;
    use winit::keyboard::ModifiersState;

    use super::{
        DebugHudInput, ImGuiKey, Key, apply_input, host_key_to_imgui_key, modifier_key_states,
    };

    #[test]
    fn host_keys_map_to_imgui_keys_needed_by_text_entry() {
        for (host_key, imgui_key) in [
            (Key::Alpha1, ImGuiKey::Alpha1),
            (Key::Backspace, ImGuiKey::Backspace),
            (Key::Return, ImGuiKey::Enter),
            (Key::LeftArrow, ImGuiKey::LeftArrow),
            (Key::KeypadPeriod, ImGuiKey::KeypadDecimal),
        ] {
            assert_eq!(host_key_to_imgui_key(host_key), Some(imgui_key));
        }
        assert_eq!(host_key_to_imgui_key(Key::F13), None);
    }

    #[test]
    fn modifier_states_include_ctrl_for_drag_input_activation() {
        let states = modifier_key_states(ModifiersState::CONTROL | ModifiersState::SHIFT);

        assert!(states.contains(&(ImGuiKey::ModCtrl, true)));
        assert!(states.contains(&(ImGuiKey::ModShift, true)));
        assert!(states.contains(&(ImGuiKey::ModAlt, false)));
        assert!(states.contains(&(ImGuiKey::ModSuper, false)));
    }

    #[test]
    fn apply_input_feeds_modifiers_and_held_keys_to_imgui() {
        let mut context = Context::create();
        context.fonts().build_rgba32_texture();
        let input = DebugHudInput {
            window_focused: true,
            keyboard_modifiers: ModifiersState::CONTROL,
            held_keys: vec![Key::Backspace, Key::Alpha1],
            text: "12.5".into(),
            ..Default::default()
        };

        {
            let io = context.io_mut();
            io.display_size = [100.0, 100.0];
            apply_input(io, &input);
        }

        let ui = context.frame();
        assert!(ui.io().key_ctrl);
        assert!(ui.is_key_down(ImGuiKey::Backspace));
        assert!(ui.is_key_down(ImGuiKey::Alpha1));
    }
}

#[cfg(test)]
mod sanitize_input_state_tests {
    use super::sanitize_input_state_for_imgui_host;
    use crate::shared::{
        DragAndDropEvent, InputState, Key, KeyboardState, MouseState, WindowState,
    };
    use glam::{IVec2, Vec2};

    #[test]
    fn sanitize_clears_mouse_buttons_and_deltas_when_want_mouse() {
        let mut input = InputState {
            mouse: Some(MouseState {
                is_active: true,
                left_button_state: true,
                right_button_state: true,
                middle_button_state: false,
                button4_state: true,
                button5_state: false,
                desktop_position: Vec2::new(10.0, 20.0),
                window_position: Vec2::new(10.0, 20.0),
                direct_delta: Vec2::new(1.0, 2.0),
                scroll_wheel_delta: Vec2::new(0.0, 120.0),
            }),
            keyboard: None,
            window: Some(WindowState {
                is_window_focused: true,
                is_fullscreen: false,
                window_resolution: IVec2::new(800, 600),
                resolution_settings_applied: false,
                drag_and_drop_event: Some(DragAndDropEvent {
                    paths: vec![Some("x".into())],
                    drop_point: IVec2::ZERO,
                }),
            }),
            vr: None,
            gamepads: vec![],
            touches: vec![],
            displays: vec![],
        };

        sanitize_input_state_for_imgui_host(&mut input, true, false);

        let m = input.mouse.expect("mouse");
        assert!(!m.left_button_state);
        assert!(!m.right_button_state);
        assert!(!m.button4_state);
        assert_eq!(m.direct_delta, Vec2::ZERO);
        assert_eq!(m.scroll_wheel_delta, Vec2::ZERO);
        assert_eq!(m.desktop_position, Vec2::new(10.0, 20.0));

        assert!(input.window.expect("window").drag_and_drop_event.is_none());
    }

    #[test]
    fn sanitize_clears_keyboard_when_want_keyboard() {
        let mut input = InputState {
            mouse: None,
            keyboard: Some(KeyboardState {
                type_delta: Some("hi".into()),
                held_keys: vec![Key::A],
            }),
            window: None,
            vr: None,
            gamepads: vec![],
            touches: vec![],
            displays: vec![],
        };

        sanitize_input_state_for_imgui_host(&mut input, false, true);

        let k = input.keyboard.expect("keyboard");
        assert!(k.type_delta.is_none());
        assert!(k.held_keys.is_empty());
    }

    #[test]
    fn sanitize_noop_when_flags_false() {
        let mut input = InputState {
            mouse: Some(MouseState {
                is_active: true,
                left_button_state: true,
                right_button_state: false,
                middle_button_state: false,
                button4_state: false,
                button5_state: false,
                desktop_position: Vec2::ZERO,
                window_position: Vec2::ZERO,
                direct_delta: Vec2::new(3.0, 4.0),
                scroll_wheel_delta: Vec2::ZERO,
            }),
            keyboard: Some(KeyboardState {
                type_delta: Some("x".into()),
                held_keys: vec![],
            }),
            window: None,
            vr: None,
            gamepads: vec![],
            touches: vec![],
            displays: vec![],
        };

        sanitize_input_state_for_imgui_host(&mut input, false, false);

        assert!(input.mouse.expect("mouse").left_button_state);
        assert_eq!(
            input.keyboard.expect("keyboard").type_delta.as_deref(),
            Some("x")
        );
    }
}
