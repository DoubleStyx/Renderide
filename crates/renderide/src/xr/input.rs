//! OpenXR VR controller input: action set, interaction profile bindings, pose resolution, and IPC state.

mod bindings;
mod frame;
mod hand_synth;
mod hand_tracking;
mod haptics;
mod latch;
mod manifest;
mod openxr_actions;
mod openxr_input;
mod pose;
mod profile;
mod state;

pub use bindings::ProfileExtensionGates;
pub use hand_synth::synthesize_hand_states;
pub(crate) use haptics::OpenxrHaptics;
pub use manifest::{ManifestError, load_manifest};
pub use openxr_input::OpenxrInput;
