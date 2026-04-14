//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]), **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]).

mod debug_hud;
mod debug_hud_fmt;
mod debug_hud_layout;
mod frame_diagnostics_snapshot;
mod frame_timing_hud_snapshot;
mod host_hud;
mod renderer_info_snapshot;
mod scene_transforms_snapshot;

pub use debug_hud::DebugHud;
pub use frame_diagnostics_snapshot::FrameDiagnosticsSnapshot;
pub use frame_timing_hud_snapshot::FrameTimingHudSnapshot;
pub use host_hud::HostHudGatherer;
pub use renderer_info_snapshot::RendererInfoSnapshot;
pub use scene_transforms_snapshot::{
    RenderSpaceTransformsSnapshot, SceneTransformsSnapshot, TransformRow, WorldTransformSample,
};

use glam::Vec2;

use crate::shared::InputState;

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
    if want_capture_keyboard {
        if let Some(ref mut keyboard) = input.keyboard {
            keyboard.type_delta = None;
            keyboard.held_keys.clear();
        }
    }
}

/// Pointer and window hints for ImGui, in **physical** pixels where noted.
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugHudInput {
    /// Cursor position in physical pixels (or `[-∞, -∞]` when unavailable).
    pub cursor_px: [f32; 2],
    /// Drawable size in physical pixels.
    pub window_px: (u32, u32),
    /// Whether the window currently has keyboard focus.
    pub window_focused: bool,
    /// Whether the cursor is over the client area (from winit accumulator).
    pub mouse_active: bool,
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
}

impl DebugHudInput {
    /// Builds input for the HUD from winit and the accumulated window/input state.
    ///
    /// Cursor is **`WindowInputAccumulator::window_position` (logical) × scale factor**, matching the
    /// swapchain / ImGui framebuffer in **physical** pixels.
    pub fn from_winit(
        window: &winit::window::Window,
        acc: &crate::frontend::input::WindowInputAccumulator,
    ) -> Self {
        let sf = window.scale_factor() as f32;
        let cursor_px = if acc.mouse_active && acc.window_focused {
            [acc.window_position.x * sf, acc.window_position.y * sf]
        } else {
            [-f32::MAX, -f32::MAX]
        };
        let s = window.inner_size();
        let window_px = (s.width, s.height);
        Self {
            cursor_px,
            window_px,
            window_focused: acc.window_focused,
            mouse_active: acc.mouse_active,
            left: acc.left_held,
            right: acc.right_held,
            middle: acc.middle_held,
            extra1: acc.button4_held,
            extra2: acc.button5_held,
        }
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
