//! Per-frame [`InputState`] snapshot construction for [`WindowInputAccumulator`].
//!
//! The snapshot is built lazily when the lock-step send needs it; deltas are
//! drained here so they don't double-count across consecutive frame starts.

use glam::IVec2;

use crate::shared::{DragAndDropEvent, InputState, KeyboardState, MouseState, WindowState};

use super::WindowInputAccumulator;

impl WindowInputAccumulator {
    /// Consumes accumulated deltas and returns an [`InputState`] for the host.
    ///
    /// `host_requests_cursor_lock`: merged into [`MouseState::is_active`] (Unity / old session parity).
    pub fn take_input_state(&mut self, host_requests_cursor_lock: bool) -> InputState {
        profiling::scope!("frontend::build_input_state");
        let type_delta = {
            let mut out = String::new();
            out.push_str(&std::mem::take(&mut self.ime_commit_buffer));
            out.push_str(&std::mem::take(&mut self.text_typing_buffer));
            if out.is_empty() { None } else { Some(out) }
        };
        let drag_and_drop_event = self.take_drag_and_drop_if_any();

        let mouse = MouseState {
            is_active: self.mouse_active || host_requests_cursor_lock,
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
            is_fullscreen: self.fullscreen,
            window_resolution: IVec2::new(
                self.window_resolution.0 as i32,
                self.window_resolution.1 as i32,
            ),
            resolution_settings_applied: false,
            drag_and_drop_event,
        };
        let keyboard = Some(KeyboardState {
            type_delta,
            held_keys: self.held_keys.clone(),
            composition_active: false,
            composition_text: None
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

    fn take_drag_and_drop_if_any(&mut self) -> Option<DragAndDropEvent> {
        if self.pending_drop_paths.is_empty() {
            return None;
        }
        let paths = std::mem::take(&mut self.pending_drop_paths)
            .into_iter()
            .map(Some)
            .collect();
        Some(DragAndDropEvent {
            paths,
            drop_point: self.last_cursor_pixel,
        })
    }
}
