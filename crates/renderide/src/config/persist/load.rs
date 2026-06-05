//! File-aware entry for the layered config loader.
//!
//! Resolves the on-disk config path, dispatches to the canonical pipeline
//! (defaults -> TOML -> `RENDERIDE_*` env -> post-extract overrides), and tolerates unknown
//! keys plus versioned migrations so renderer up/downgrades survive disk state.
//!
//! Internals split into [`pipeline`] (the pure layering machinery) and [`migrate`] (TOML
//! compatibility and one-shot schema migrations); this module owns the public entry
//! [`load_renderer_settings`] plus the file-IO bookkeeping around it.

use std::path::{Path, PathBuf};

use super::resolve::{
    ConfigResolveOutcome, ConfigSource, apply_generated_config, is_dir_writable, read_config_file,
    renderide_config_env_nonempty, resolve_config_path, resolve_save_path,
};
use super::save::{save_migrated_renderer_config, save_renderer_settings_pruned};
use crate::config::types::RendererSettings;

mod migrate;
mod pipeline;
mod previous_layout;

use migrate::{log_compatibility_drops, run_pipeline_tolerating_toml};
use pipeline::run_pipeline;
use previous_layout::migrate_legacy_config_if_present;

#[cfg(test)]
use pipeline::apply_renderide_gpu_validation_env;

/// Controls whether the TOML config file is consulted during startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConfigFilePolicy {
    /// Normal: discover, load, and (if absent) auto-create `config.toml`.
    #[default]
    Load,
    /// Skip all file I/O; use struct defaults plus `RENDERIDE_*` env vars only.
    /// Forces `suppress_config_disk_writes = true`.
    Ignore,
}

/// Full load result: resolved path and save path for persistence.
#[derive(Clone, Debug)]
pub struct ConfigLoadResult {
    /// Effective settings after merge.
    pub settings: RendererSettings,
    /// Path resolution diagnostics.
    pub resolve: ConfigResolveOutcome,
    /// Target file for [`super::save::save_renderer_settings`] and the ImGui config window.
    pub save_path: PathBuf,
    /// When `true`, disk persistence is disabled until restart because startup config extraction
    /// failed in a way that could not be repaired by ignoring an incompatible TOML key.
    pub suppress_config_disk_writes: bool,
}

/// Resolves `config.toml`, runs the canonical pipeline, and produces a [`ConfigLoadResult`].
///
/// Precedence (top wins): post-extract mutators (`RENDERIDE_GPU_VALIDATION`) -> `RENDERIDE_*`
/// env -> TOML file (skipped under [`ConfigFilePolicy::Ignore`]) -> struct defaults.
///
/// When no file exists and [`renderide_config_env_nonempty`] is false, writes defaults to the
/// save path (see [`super::resolve::resolve_save_path`]) and loads that file. This
/// auto-creation is skipped when `policy` is [`ConfigFilePolicy::Ignore`].
pub fn load_renderer_settings(policy: ConfigFilePolicy) -> ConfigLoadResult {
    if policy == ConfigFilePolicy::Ignore {
        return load_with_ignore_policy();
    }

    let mut resolve = resolve_config_path();
    let mut suppress_config_disk_writes = false;
    let mut settings = initial_settings_from_resolve(&mut suppress_config_disk_writes, &resolve);

    if resolve.loaded_path.is_none()
        && !renderide_config_env_nonempty()
        && !migrate_legacy_config_if_present(
            &mut resolve,
            &mut settings,
            &mut suppress_config_disk_writes,
        )
    {
        maybe_create_default_config_and_reload(
            &mut resolve,
            &mut settings,
            &mut suppress_config_disk_writes,
        );
    }

    let save_path = resolve_save_path(&resolve);
    logger::trace!("Renderer config will persist to {}", save_path.display());

    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes,
    }
}

/// Builds the [`ConfigFilePolicy::Ignore`] result: skip TOML, run defaults+env+overrides only,
/// and force `suppress_config_disk_writes`.
fn load_with_ignore_policy() -> ConfigLoadResult {
    if renderide_config_env_nonempty() {
        logger::warn!(
            "--ignore-config is active; RENDERIDE_CONFIG is also set but the file will be skipped"
        );
    }
    let settings = match run_pipeline(None) {
        Ok(s) => s,
        Err(e) => {
            logger::error!(
                "Renderer config Figment extract failed (--ignore-config, defaults+env): {e:#}"
            );
            RendererSettings::default()
        }
    };
    let resolve = ConfigResolveOutcome {
        attempted_paths: vec![],
        loaded_path: None,
        source: ConfigSource::None,
    };
    let save_path = resolve_save_path(&resolve);
    logger::info!("--ignore-config: skipping TOML file; using struct defaults + RENDERIDE_* env");
    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes: true,
    }
}

/// Loads settings from a resolved config path, or defaults plus env when the file is missing or
/// unreadable.
fn initial_settings_from_resolve(
    suppress_config_disk_writes: &mut bool,
    resolve: &ConfigResolveOutcome,
) -> RendererSettings {
    if let Some(path) = resolve.loaded_path.as_ref() {
        logger::info!("Loading renderer config from {}", path.display());
        match read_config_file(path) {
            Ok(content) => match run_pipeline_tolerating_toml(&content) {
                Ok(load) => {
                    log_compatibility_drops(path, &load.drops);
                    persist_migrated_toml(path, load.migrated_toml.as_deref());
                    load.settings
                }
                Err(e) => {
                    logger::error!(
                        "Renderer config Figment extract failed for {}: {e:#}",
                        path.display()
                    );
                    *suppress_config_disk_writes = true;
                    RendererSettings::default()
                }
            },
            Err(e) => {
                logger::warn!("Failed to read {}: {e}; using defaults", path.display());
                fallback_to_defaults_plus_env(suppress_config_disk_writes)
            }
        }
    } else {
        logger::info!("Renderer config file not found; using built-in defaults");
        logger::trace!(
            "config search tried {} path(s)",
            resolve.attempted_paths.len()
        );
        fallback_to_defaults_plus_env(suppress_config_disk_writes)
    }
}

/// Runs the pipeline without a TOML layer (defaults + env + post-extract overrides) and falls
/// back to [`RendererSettings::default`] on Figment failure.
fn fallback_to_defaults_plus_env(suppress_config_disk_writes: &mut bool) -> RendererSettings {
    match run_pipeline(None) {
        Ok(s) => s,
        Err(e) => {
            logger::error!("Renderer config Figment extract failed (defaults+env): {e:#}");
            *suppress_config_disk_writes = true;
            RendererSettings::default()
        }
    }
}

/// When no config was loaded and env overrides are empty, writes default `config.toml` and
/// reloads from disk.
fn maybe_create_default_config_and_reload(
    resolve: &mut ConfigResolveOutcome,
    settings: &mut RendererSettings,
    suppress_config_disk_writes: &mut bool,
) {
    let path = resolve_save_path(resolve);
    if path.exists() {
        return;
    }
    let Some(parent) = path.parent() else {
        return;
    };
    if !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        logger::warn!(
            "Not creating default config at {} ({}): {e}",
            path.display(),
            parent.display()
        );
        return;
    }
    if !is_dir_writable(parent) {
        logger::trace!(
            "Not creating default config at {} (directory not writable)",
            path.display()
        );
        return;
    }
    match save_renderer_settings_pruned(&path, &RendererSettings::from_defaults()) {
        Ok(()) => {
            logger::info!("Created default renderer config at {}", path.display());
            apply_generated_config(resolve, path.clone());
            match read_config_file(&path) {
                Ok(content) => match run_pipeline_tolerating_toml(&content) {
                    Ok(load) => {
                        log_compatibility_drops(&path, &load.drops);
                        persist_migrated_toml(&path, load.migrated_toml.as_deref());
                        *settings = load.settings;
                    }
                    Err(e) => {
                        logger::error!(
                            "Figment extract failed for newly created {}: {e:#}",
                            path.display()
                        );
                        *suppress_config_disk_writes = true;
                    }
                },
                Err(e) => {
                    logger::warn!(
                        "Failed to read newly created {}: {e}; using defaults",
                        path.display()
                    );
                }
            }
        }
        Err(e) => {
            logger::warn!("Failed to create default config at {}: {e}", path.display());
        }
    }
}

fn persist_migrated_toml(path: &Path, migrated_toml: Option<&str>) {
    let Some(contents) = migrated_toml else {
        return;
    };

    match save_migrated_renderer_config(path, contents) {
        Ok(()) => logger::info!(
            "Migrated renderer config {} to config_version {}",
            path.display(),
            RendererSettings::CURRENT_CONFIG_VERSION
        ),
        Err(e) => logger::warn!(
            "Failed to persist migrated renderer config {}: {e}",
            path.display()
        ),
    }
}

/// Logs [`ConfigLoadResult::resolve`] at trace level for troubleshooting.
pub fn log_config_resolve_trace(resolve: &ConfigResolveOutcome) {
    if resolve.source == ConfigSource::None && !resolve.attempted_paths.is_empty() {
        for p in &resolve.attempted_paths {
            let exists = p.as_path().is_file();
            logger::trace!("  config candidate {} [{}]", p.display(), exists);
        }
    }
}

#[cfg(test)]
mod tests;
