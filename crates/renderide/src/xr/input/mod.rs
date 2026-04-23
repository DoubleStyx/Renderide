//! OpenXR VR controller input: action set, interaction profile bindings, pose resolution, and IPC state.

mod bindings;
mod frame;
mod openxr_action_paths;
mod openxr_actions;
mod openxr_input;
mod pose;
mod profile;
mod state;

pub use bindings::ProfileExtensionGates;
pub use openxr_input::OpenxrInput;
