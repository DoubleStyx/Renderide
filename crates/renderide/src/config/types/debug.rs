//! Debug, diagnostics, and adapter-selection settings. Persisted as `[debug]`.
//!
//! Three sibling concerns:
//! - [`power_preference`]: the GPU adapter power-mode enum.
//! - [`hud`]: persisted Dear ImGui HUD state (tabs, visibility, presentation flags).
//! - [`settings`]: the `[debug]` table struct that aggregates the master toggles plus the HUD
//!   state.

mod command_recording;
mod hud;
mod power_preference;
mod settings;

pub use command_recording::CommandRecordingMode;
pub use hud::{
    DebugHudMainTab, DebugHudMainTabVisibility, DebugHudRendererConfigTab,
    DebugHudRendererConfigTabVisibility, DebugHudSettings,
};
pub use power_preference::PowerPreferenceSetting;
pub use settings::DebugSettings;
