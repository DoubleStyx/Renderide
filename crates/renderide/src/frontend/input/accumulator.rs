//! Platform-neutral input accumulation for [`FrameStartData`](crate::shared::FrameStartData).

use glam::{IVec2, Vec2};
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::window::Window;

use crate::shared::{DragAndDropEvent, InputState, Key, KeyboardState, MouseState, WindowState};

/// Holds window / pointer / keyboard state between winit events and host lock-step [`InputState`] snapshots.
///
/// Mouse and scroll deltas accumulate until [`Self::take_input_state`] (called when sending
/// `frame_start_data`), matching the historical Unity renderer begin-frame timing.
///
/// **Coordinate contract:** [`Self::window_position`] and [`Self::window_resolution`] are both in
/// **logical** pixels (DPI-aware). The host computes normalized window position as position divided
/// by resolution; both must use the same space.
pub struct WindowInputAccumulator {
    /// Accumulated relative motion (including [`DeviceEvent::MouseMotion`](winit::event::DeviceEvent)).
    pub mouse_delta: Vec2,
    /// Accumulated scroll wheel / trackpad scroll since the last [`Self::take_input_state`].
    pub scroll_delta: Vec2,
    /// Last scroll sample consumed by the HUD so repeated HUD frames don't see the same delta.
    hud_scroll_sample: Vec2,
    /// Pointer position in window space (logical pixels) for [`crate::shared::MouseState`].
    pub window_position: Vec2,
    /// Inner drawable size in **logical** pixels (matches [`Self::window_position`] for host UVs).
    pub window_resolution: (u32, u32),
    /// Left mouse button held.
    pub left_held: bool,
    /// Right mouse button held.
    pub right_held: bool,
    /// Middle mouse button held.
    pub middle_held: bool,
    /// Fourth mouse button (back) held.
    pub button4_held: bool,
    /// Fifth mouse button (forward) held.
    pub button5_held: bool,
    /// Whether the cursor is inside the client area.
    pub mouse_active: bool,
    /// Whether the window is focused.
    pub window_focused: bool,
    /// Keys currently held, in host [`Key`] form.
    pub held_keys: Vec<Key>,
    /// Text committed by IME since the last IPC snapshot (`KeyboardState.type_delta`).
    ime_commit_buffer: String,
    /// Single-character text from key events (supplements IME for simple typing).
    text_typing_buffer: String,
    /// Paths from [`WindowEvent::DroppedFile`](winit::event::WindowEvent) coalesced until take.
    pending_drop_paths: Vec<String>,
    /// Last cursor position in physical pixels (for drop-point reporting).
    last_cursor_pixel: IVec2,
}

impl Default for WindowInputAccumulator {
    fn default() -> Self {
        Self {
            mouse_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
            hud_scroll_sample: Vec2::ZERO,
            window_position: Vec2::ZERO,
            window_resolution: (0, 0),
            left_held: false,
            right_held: false,
            middle_held: false,
            button4_held: false,
            button5_held: false,
            mouse_active: false,
            window_focused: true,
            held_keys: Vec::new(),
            ime_commit_buffer: String::new(),
            text_typing_buffer: String::new(),
            pending_drop_paths: Vec::new(),
            last_cursor_pixel: IVec2::ZERO,
        }
    }
}

impl WindowInputAccumulator {
    /// Records IME-composed text committed by the platform.
    pub fn push_ime_commit(&mut self, text: &str) {
        self.ime_commit_buffer.push_str(text);
    }

    /// Records printable text associated with a key press (not repeats).
    pub fn push_key_text(&mut self, text: &str) {
        self.text_typing_buffer.push_str(text);
    }

    /// Records a file dropped onto the window; paths are batched into the next [`InputState`].
    pub fn push_dropped_file_path(&mut self, path_str: String) {
        self.pending_drop_paths.push(path_str);
    }

    /// Updates cursor position from winit [`WindowEvent::CursorMoved`](winit::event::WindowEvent::CursorMoved).
    ///
    /// `position` is in **physical** pixels; `window_position` stores **logical** pixels for host
    /// [`MouseState`]. `last_cursor_pixel` keeps the last **physical** position for drag/drop.
    pub fn set_cursor_from_physical(&mut self, position: PhysicalPosition<f64>, scale_factor: f64) {
        let logical: LogicalPosition<f64> = position.to_logical(scale_factor);
        self.window_position.x = logical.x as f32;
        self.window_position.y = logical.y as f32;
        self.last_cursor_pixel.x = position.x.round() as i32;
        self.last_cursor_pixel.y = position.y.round() as i32;
    }

    /// Refreshes [`Self::window_resolution`] from [`Window::inner_size`] in **logical** pixels.
    pub fn sync_window_resolution_logical(&mut self, window: &Window) {
        let physical = window.inner_size();
        let logical: LogicalSize<f64> = physical.to_logical(window.scale_factor());
        self.window_resolution = (logical.width.round() as u32, logical.height.round() as u32);
    }

    /// Sets [`Self::window_position`] from **logical** coordinates and updates [`Self::last_cursor_pixel`]
    /// to the corresponding physical point (for drag/drop parity).
    pub fn set_window_position_from_logical(&mut self, logical: Vec2, scale_factor: f64) {
        self.window_position = logical;
        let logical_pos = LogicalPosition::new(logical.x as f64, logical.y as f64);
        let physical = logical_pos.to_physical::<f64>(scale_factor);
        self.last_cursor_pixel.x = physical.x.round() as i32;
        self.last_cursor_pixel.y = physical.y.round() as i32;
    }

    /// Clears [`Self::held_keys`] when the window loses focus.
    ///
    /// After Alt+Tab or similar, the platform may not deliver key release events to this window,
    /// which would otherwise leave keys stuck in [`Self::held_keys`]. Mouse buttons are **not**
    /// cleared here: clearing them can regress click/drag-to-look when the OS emits brief or
    /// duplicate focus transitions, and relative motion ([`Self::mouse_delta`]) is unaffected by
    /// this helper in any case.
    pub fn clear_stuck_keyboard_on_focus_lost(&mut self) {
        self.held_keys.clear();
    }

    /// Consumes accumulated deltas and returns an [`InputState`] for the host.
    ///
    /// `host_requests_cursor_lock`: merged into [`MouseState::is_active`] (Unity / old session parity).
    pub fn take_input_state(&mut self, host_requests_cursor_lock: bool) -> InputState {
        profiling::scope!("frontend::build_input_state");
        let type_delta = {
            let mut out = String::new();
            out.push_str(&std::mem::take(&mut self.ime_commit_buffer));
            out.push_str(&std::mem::take(&mut self.text_typing_buffer));
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
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

        self.hud_scroll_sample = Vec2::ZERO;
        let window = WindowState {
            is_window_focused: self.window_focused,
            is_fullscreen: false,
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

    /// Returns the scroll delta observed since the HUD last read it and advances the HUD marker.
    ///
    /// The accumulator keeps a running `scroll_delta` for the host. The HUD should call this
    /// method to obtain the incremental scroll delta since its last read; the method returns
    /// `scroll_delta - hud_scroll_sample` and advances `hud_scroll_sample` so subsequent HUD
    /// frames do not reapply the same input. The host still consumes the full accumulated
    /// `scroll_delta` when `take_input_state()` is called.
    pub fn take_hud_scroll_delta(&mut self) -> Vec2 {
        let delta = self.scroll_delta - self.hud_scroll_sample;
        self.hud_scroll_sample = self.scroll_delta;
        delta
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

#[cfg(test)]
mod tests {
    use super::WindowInputAccumulator;
    use crate::shared::Key;
    use glam::Vec2;

    #[test]
    fn mouse_delta_accumulates_until_take_input_state() {
        let mut w = WindowInputAccumulator::default();
        w.mouse_delta += Vec2::new(1.0, 2.0);
        w.mouse_delta += Vec2::new(3.0, 4.0);
        let first = w.take_input_state(false);
        let mouse = first.mouse.expect("mouse state");
        assert_eq!(mouse.direct_delta.x, 4.0);
        assert_eq!(mouse.direct_delta.y, 6.0);
        let second = w.take_input_state(false);
        let mouse2 = second.mouse.expect("mouse state");
        assert_eq!(mouse2.direct_delta.x, 0.0);
        assert_eq!(mouse2.direct_delta.y, 0.0);
    }

    #[test]
    fn scroll_delta_accumulates_until_take_input_state() {
        let mut w = WindowInputAccumulator::default();
        w.scroll_delta += Vec2::new(0.0, 120.0);
        w.scroll_delta += Vec2::new(0.0, 60.0);
        let taken = w.take_input_state(false);
        let mouse = taken.mouse.expect("mouse state");
        assert_eq!(mouse.scroll_wheel_delta.y, 180.0);
    }

    #[test]
    fn cursor_lock_merges_into_mouse_active() {
        let mut w = WindowInputAccumulator {
            mouse_active: false,
            ..Default::default()
        };
        let s = w.take_input_state(true);
        assert!(s.mouse.expect("mouse").is_active);
    }

    #[test]
    fn focus_loss_clears_held_keys_but_preserves_mouse_buttons() {
        let mut w = WindowInputAccumulator::default();
        w.held_keys.push(Key::W);
        w.held_keys.push(Key::A);
        w.left_held = true;
        w.right_held = true;
        w.clear_stuck_keyboard_on_focus_lost();
        assert!(w.held_keys.is_empty());
        assert!(w.left_held);
        assert!(w.right_held);
    }

    #[test]
    fn ime_and_text_merge_into_type_delta() {
        let mut w = WindowInputAccumulator::default();
        w.push_ime_commit("hello");
        w.push_key_text("!");
        let s = w.take_input_state(false);
        assert_eq!(
            s.keyboard.expect("kb").type_delta.as_deref(),
            Some("hello!")
        );
        let s2 = w.take_input_state(false);
        assert!(s2.keyboard.expect("kb").type_delta.is_none());
    }

    /// Normalized UV at logical center when resolution and position share logical space.
    #[test]
    fn normalized_center_at_logical_half_resolution() {
        let mut w = WindowInputAccumulator {
            window_resolution: (800, 600),
            window_position: Vec2::new(400.0, 300.0),
            ..Default::default()
        };
        let inp = w.take_input_state(false);
        let mouse = inp.mouse.expect("mouse");
        let win = inp.window.expect("window");
        let nx = mouse.window_position.x / win.window_resolution.x as f32;
        let ny = mouse.window_position.y / win.window_resolution.y as f32;
        assert!((nx - 0.5).abs() < 1e-5);
        assert!((ny - 0.5).abs() < 1e-5);
    }

    #[test]
    fn set_window_position_from_logical_updates_physical_pixel() {
        let mut w = WindowInputAccumulator::default();
        w.set_window_position_from_logical(Vec2::new(100.0, 200.0), 2.0);
        assert_eq!(w.last_cursor_pixel.x, 200);
        assert_eq!(w.last_cursor_pixel.y, 400);
    }

    #[test]
    fn physical_inner_matches_logical_resolution_at_scale_factor() {
        use winit::dpi::{LogicalSize, PhysicalSize};

        let physical = PhysicalSize::new(1920u32, 1080u32);
        let logical: LogicalSize<f64> = physical.to_logical(2.0);
        assert_eq!(logical.width.round() as u32, 960);
        assert_eq!(logical.height.round() as u32, 540);
    }
}
