//! Platform-neutral input accumulation for [`FrameStartData`](crate::shared::FrameStartData).
//!
//! The accumulator holds raw deltas and held-key state between winit events; the per-frame
//! [`InputState`](crate::shared::InputState) snapshot lives in [`state_snapshot`] so the mutators
//! and the consumption path stay textually separated.

mod state_snapshot;

use glam::{IVec2, Vec2};
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::keyboard::ModifiersState;
use winit::window::Window;

use crate::shared::Key;

/// Holds window / pointer / keyboard state between winit events and host lock-step [`InputState`] snapshots.
///
/// Mouse and scroll deltas accumulate until [`Self::take_input_state`] (called when sending
/// `frame_start_data`), matching the historical Unity renderer begin-frame timing.
///
/// **Coordinate contract:** [`Self::window_position`] and [`Self::window_resolution`] are both in
/// **logical** pixels (DPI-aware). The host computes normalized window position as position divided
/// by resolution; both must use the same space.
pub struct WindowInputAccumulator {
    /// Accumulated relative motion (including [`DeviceEvent::PointerMotion`](winit::event::DeviceEvent::PointerMotion)).
    pub mouse_delta: Vec2,
    /// Accumulated scroll wheel / trackpad scroll since the last [`Self::take_input_state`].
    pub scroll_delta: Vec2,
    /// Accumulated ImGui wheel units since the last HUD input snapshot.
    hud_scroll_delta: Vec2,
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
    /// Whether winit reports keyboard focus for the renderer window.
    pub window_focused: bool,
    /// Keys currently held, in host [`Key`] form.
    pub held_keys: Vec<Key>,
    /// Text committed by IME since the last IPC snapshot (`KeyboardState.type_delta`).
    ime_commit_buffer: String,
    /// Single-character text from key events (supplements IME for simple typing).
    text_typing_buffer: String,
    /// Text committed since the last HUD input snapshot.
    hud_text_buffer: String,
    /// Paths from [`WindowEvent::DragDropped`](winit::event::WindowEvent::DragDropped) coalesced until take.
    pending_drop_paths: Vec<String>,
    /// Last cursor position in physical pixels (for drop-point reporting).
    last_cursor_pixel: IVec2,
    keyboard_modifiers: ModifiersState,
    fullscreen: bool,
}

impl Default for WindowInputAccumulator {
    fn default() -> Self {
        Self {
            mouse_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
            hud_scroll_delta: Vec2::ZERO,
            window_position: Vec2::ZERO,
            window_resolution: (0, 0),
            left_held: false,
            right_held: false,
            middle_held: false,
            button4_held: false,
            button5_held: false,
            mouse_active: false,
            window_focused: false,
            held_keys: Vec::new(),
            ime_commit_buffer: String::new(),
            text_typing_buffer: String::new(),
            hud_text_buffer: String::new(),
            pending_drop_paths: Vec::new(),
            last_cursor_pixel: IVec2::ZERO,
            keyboard_modifiers: ModifiersState::empty(),
            fullscreen: false,
        }
    }
}

impl WindowInputAccumulator {
    /// Records IME-composed text committed by the platform.
    pub fn push_ime_commit(&mut self, text: &str) {
        self.ime_commit_buffer.push_str(text);
        self.hud_text_buffer.push_str(text);
    }

    /// Records printable text associated with a key press (not repeats).
    pub fn push_key_text(&mut self, text: &str) {
        self.text_typing_buffer.push_str(text);
        self.hud_text_buffer.push_str(text);
    }

    /// Records a file dropped onto the window; paths are batched into the next [`InputState`].
    pub fn push_dropped_file_path(&mut self, path_str: String) {
        self.pending_drop_paths.push(path_str);
    }

    /// Records the current winit keyboard modifier state.
    pub fn set_keyboard_modifiers(&mut self, modifiers: ModifiersState) {
        self.keyboard_modifiers = modifiers;
    }

    /// Returns the current winit keyboard modifier state.
    pub fn keyboard_modifiers(&self) -> ModifiersState {
        self.keyboard_modifiers
    }

    /// Records whether the renderer window is currently fullscreen.
    pub fn set_fullscreen(&mut self, fullscreen: bool) {
        self.fullscreen = fullscreen;
    }

    /// Records whether winit reports keyboard focus for the renderer window.
    pub fn set_window_focused(&mut self, focused: bool) {
        self.window_focused = focused;
        if !focused {
            self.clear_stuck_keyboard_on_focus_lost();
        }
    }

    /// Updates cursor position from winit [`WindowEvent::PointerMoved`](winit::event::WindowEvent::PointerMoved).
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

    /// Refreshes [`Self::window_resolution`] from [`Window::surface_size`] in **logical** pixels.
    pub fn sync_window_resolution_logical(&mut self, window: &dyn Window) {
        let physical = window.surface_size();
        let logical: LogicalSize<f64> = physical.to_logical(window.scale_factor());
        self.window_resolution = (logical.width.round() as u32, logical.height.round() as u32);
    }

    /// Sets [`Self::window_position`] from **logical** coordinates and updates [`Self::last_cursor_pixel`]
    /// to the corresponding physical point (for drag/drop parity).
    pub fn set_window_position_from_logical(&mut self, logical: Vec2, scale_factor: f64) {
        self.window_position = logical;
        let logical_pos = LogicalPosition::new(f64::from(logical.x), f64::from(logical.y));
        let physical = logical_pos.to_physical::<f64>(scale_factor);
        self.last_cursor_pixel.x = physical.x.round() as i32;
        self.last_cursor_pixel.y = physical.y.round() as i32;
    }

    /// Clears [`Self::held_keys`] and modifier state when the window loses keyboard focus.
    ///
    /// After Alt+Tab or similar, the platform may not deliver key release events to this window,
    /// which would otherwise leave keys stuck in [`Self::held_keys`]. Mouse buttons are **not**
    /// cleared here: clearing them can regress click/drag-to-look when the OS emits brief or
    /// duplicate focus transitions, and relative motion ([`Self::mouse_delta`]) is unaffected by
    /// this helper in any case.
    pub fn clear_stuck_keyboard_on_focus_lost(&mut self) {
        self.held_keys.clear();
        self.keyboard_modifiers = ModifiersState::empty();
    }

    /// Accumulates a scroll event for both the host wire contract and the ImGui overlay.
    ///
    /// The two accumulators deliberately use different units: `host_delta` matches the IPC
    /// `MouseState` contract, while `hud_delta` is already in Dear ImGui wheel units.
    pub(in crate::frontend::input) fn push_scroll_delta(
        &mut self,
        host_delta: Vec2,
        hud_delta: Vec2,
    ) {
        self.scroll_delta += host_delta;
        self.hud_scroll_delta += hud_delta;
    }

    /// Returns the ImGui wheel delta observed since the HUD last read it.
    ///
    /// This is independent from [`Self::scroll_delta`] so lock-step host snapshots can drain their
    /// scroll payload before the render/HUD phase without losing scroll events for ImGui.
    pub fn take_hud_scroll_delta(&mut self) -> Vec2 {
        std::mem::take(&mut self.hud_scroll_delta)
    }

    /// Returns text committed since the HUD last read it.
    pub fn take_hud_text(&mut self) -> String {
        std::mem::take(&mut self.hud_text_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::WindowInputAccumulator;
    use crate::shared::Key;
    use glam::Vec2;
    use winit::keyboard::ModifiersState;

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
    fn hud_scroll_delta_survives_host_snapshot_drain() {
        let mut w = WindowInputAccumulator::default();
        w.push_scroll_delta(Vec2::new(0.0, 120.0), Vec2::new(0.0, 1.0));

        let taken = w.take_input_state(false);
        let mouse = taken.mouse.expect("mouse state");
        assert_eq!(mouse.scroll_wheel_delta, Vec2::new(0.0, 120.0));

        assert_eq!(w.take_hud_scroll_delta(), Vec2::new(0.0, 1.0));
        assert_eq!(w.take_hud_scroll_delta(), Vec2::ZERO);
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
        w.set_keyboard_modifiers(ModifiersState::ALT);
        w.left_held = true;
        w.right_held = true;
        w.clear_stuck_keyboard_on_focus_lost();
        assert!(w.held_keys.is_empty());
        assert_eq!(w.keyboard_modifiers(), ModifiersState::empty());
        assert!(w.left_held);
        assert!(w.right_held);
    }

    #[test]
    fn set_window_focused_tracks_winit_keyboard_focus_and_clears_keys_on_loss() {
        let mut w = WindowInputAccumulator::default();
        assert!(!w.window_focused);

        w.set_window_focused(true);
        w.held_keys.push(Key::W);
        w.set_keyboard_modifiers(ModifiersState::SHIFT);
        assert!(w.window_focused);

        w.set_window_focused(false);

        assert!(!w.window_focused);
        assert!(w.held_keys.is_empty());
        assert_eq!(w.keyboard_modifiers(), ModifiersState::empty());
    }

    #[test]
    fn fullscreen_state_flows_to_window_state() {
        let mut w = WindowInputAccumulator::default();

        w.set_fullscreen(true);
        let s = w.take_input_state(false);

        assert!(s.window.expect("window").is_fullscreen);
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

    #[test]
    fn hud_text_survives_host_snapshot_drain() {
        let mut w = WindowInputAccumulator::default();
        w.push_ime_commit("12");
        w.push_key_text(".");

        let s = w.take_input_state(false);
        assert_eq!(s.keyboard.expect("kb").type_delta.as_deref(), Some("12."));
        assert_eq!(w.take_hud_text(), "12.");
        assert!(w.take_hud_text().is_empty());
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
