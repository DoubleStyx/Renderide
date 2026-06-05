//! Previous-layout config-file migration.

use std::path::{Path, PathBuf};

use crate::config::types::RendererSettings;

use super::super::resolve::{
    ConfigResolveOutcome, LEGACY_MIGRATED_SUFFIX, apply_migrated_config, legacy_search_candidates,
    read_config_file, record_attempted_path, user_config_path,
};
use super::super::save::save_migrated_renderer_config;
use super::migrate::{log_compatibility_drops, run_pipeline_tolerating_toml};
use super::persist_migrated_toml;

// TODO(remove-after-1.0): One-shot previous-layout config migration. Earlier builds dropped
// `config.toml` next to the binary, at the workspace root, or in the cwd. On startup, after
// finding no file in the user config directory, we scan those locations, copy the first hit into
// the user config directory, and best-effort rename the original to `config.toml.migrated`.
pub(super) fn migrate_legacy_config_if_present(
    resolve: &mut ConfigResolveOutcome,
    settings: &mut RendererSettings,
    suppress_config_disk_writes: &mut bool,
) -> bool {
    let Some(target) = user_config_path() else {
        return false;
    };

    if target.is_file() {
        return false;
    }

    for legacy in legacy_search_candidates() {
        record_attempted_path(resolve, legacy.clone());
        if !legacy.is_file() {
            continue;
        }

        match migrate_one_legacy_config(&legacy, &target) {
            Ok(()) => {
                logger::info!(
                    "Migrated renderer config from {} to {}",
                    legacy.display(),
                    target.display()
                );
                apply_migrated_config(resolve, target.clone());
                reload_after_migration(&target, settings, suppress_config_disk_writes);
                return true;
            }
            Err(e) => {
                logger::warn!(
                    "Failed to migrate previous renderer config {} to {}: {e}",
                    legacy.display(),
                    target.display()
                );
            }
        }
    }

    false
}

fn migrate_one_legacy_config(legacy: &Path, target: &Path) -> std::io::Result<()> {
    let contents = std::fs::read_to_string(legacy)?;
    save_migrated_renderer_config(target, &contents)?;

    let mut tombstone = legacy.as_os_str().to_owned();
    tombstone.push(LEGACY_MIGRATED_SUFFIX);
    if let Err(e) = std::fs::rename(legacy, PathBuf::from(tombstone)) {
        logger::warn!(
            "Migrated renderer config from {}, but failed to tombstone the original: {e}",
            legacy.display()
        );
    }
    Ok(())
}

fn reload_after_migration(
    path: &Path,
    settings: &mut RendererSettings,
    suppress_config_disk_writes: &mut bool,
) {
    match read_config_file(path) {
        Ok(content) => match run_pipeline_tolerating_toml(&content) {
            Ok(load) => {
                log_compatibility_drops(path, &load.drops);
                persist_migrated_toml(path, load.migrated_toml.as_deref());
                *settings = load.settings;
            }
            Err(e) => {
                logger::error!(
                    "Figment extract failed for migrated {}: {e:#}",
                    path.display()
                );
                *suppress_config_disk_writes = true;
            }
        },
        Err(e) => {
            logger::warn!(
                "Failed to read migrated {}: {e}; using defaults",
                path.display()
            );
        }
    }
}
