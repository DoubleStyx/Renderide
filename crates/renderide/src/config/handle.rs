//! Shared in-memory handle for [`super::types::RendererSettings`].
//!
//! The frame loop reads through this handle every tick; the debug HUD writes through it when
//! the user edits values; saves to disk go through [`super::save`]. Wrapping the settings in
//! `Arc<RwLock<...>>` (rather than handing out clones) means the HUD's edits are immediately
//! visible to the next frame without a propagation step.

use std::sync::Arc;

use super::load::ConfigLoadResult;
use super::types::RendererSettings;

/// Shared handle for the process-wide settings store (read by the frame loop, written by the HUD).
pub type RendererSettingsHandle = Arc<std::sync::RwLock<RendererSettings>>;

/// Builds a [`RendererSettingsHandle`] from post-load settings.
pub fn settings_handle_from(load: &ConfigLoadResult) -> RendererSettingsHandle {
    Arc::new(std::sync::RwLock::new(load.settings.clone()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::config::{ConfigResolveOutcome, ConfigSource};

    use super::*;

    #[test]
    fn settings_handle_starts_with_loaded_settings_clone() {
        let load = ConfigLoadResult {
            settings: RendererSettings {
                debug: crate::config::DebugSettings {
                    log_verbose: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            resolve: ConfigResolveOutcome {
                attempted_paths: Vec::new(),
                loaded_path: None,
                source: ConfigSource::None,
            },
            save_path: PathBuf::from("config.toml"),
            suppress_config_disk_writes: false,
        };

        let handle = settings_handle_from(&load);

        assert!(handle.read().expect("settings read").debug.log_verbose);
    }
}
