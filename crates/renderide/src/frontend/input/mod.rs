//! Window input: accumulate winit events and pack [`InputState`](crate::shared::InputState) for IPC.

mod accumulator;
mod key_map;
mod vr_session;
mod winit;

pub use accumulator::WindowInputAccumulator;
pub use key_map::winit_key_to_renderite_key;
pub use vr_session::vr_inputs_for_session;
pub use winit::{
    apply_device_event, apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
    apply_window_event, CursorOutputTracking,
};
