//! Load and save [`super::types::RendererSettings`]: Figment merge, `config.toml`, atomic TOML write.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;

use super::resolve::{
    apply_generated_config, is_dir_writable, read_config_file, renderide_config_env_nonempty,
    resolve_config_path, resolve_save_path, ConfigResolveOutcome, ConfigSource, FILE_NAME_TOML,
};
use super::types::RendererSettings;

fn renderer_settings_figment() -> Figment {
    Figment::new()
        .merge(Serialized::defaults(RendererSettings::default()))
        .merge(Env::prefixed("RENDERIDE_").split("__"))
}

#[expect(
    clippy::result_large_err,
    reason = "`figment::Error` is large; only used on startup paths"
)]
fn try_extract_settings(figment: Figment) -> Result<RendererSettings, figment::Error> {
    figment.extract::<RendererSettings>()
}

#[expect(
    clippy::result_large_err,
    reason = "`figment::Error` is large; only used on startup paths"
)]
fn load_settings_from_toml_str(content: &str) -> Result<RendererSettings, figment::Error> {
    let figment = Figment::new()
        .merge(Serialized::defaults(RendererSettings::default()))
        .merge(Toml::string(content))
        .merge(Env::prefixed("RENDERIDE_").split("__"));
    try_extract_settings(figment)
}

/// Overrides [`super::types::DebugSettings::gpu_validation_layers`] when `RENDERIDE_GPU_VALIDATION` is set.
///
/// Truthy values (`1`, `true`, `yes`) force validation on; falsey (`0`, `false`, `no`) force off.
/// If unset, the value from config or defaults is unchanged.
pub fn apply_renderide_gpu_validation_env(settings: &mut RendererSettings) {
    match std::env::var("RENDERIDE_GPU_VALIDATION").as_deref() {
        Ok("1" | "true" | "yes") => settings.debug.gpu_validation_layers = true,
        Ok("0" | "false" | "no") => settings.debug.gpu_validation_layers = false,
        _ => {}
    }
}

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
    /// Target file for [`save_renderer_settings`] and the ImGui config window.
    pub save_path: PathBuf,
    /// When `true`, disk persistence is disabled until restart (Figment extract failed on an existing file).
    pub suppress_config_disk_writes: bool,
}

/// Shared handle for the process-wide settings store (read by the frame loop, written by the HUD).
pub type RendererSettingsHandle = Arc<std::sync::RwLock<RendererSettings>>;

/// Resolves `config.toml`, merges with figment layers, and builds [`RendererSettings`].
///
/// Precedence: struct defaults, then TOML file (unless [`ConfigFilePolicy::Ignore`]), then
/// `RENDERIDE_*` environment variables (see module docs in `config/mod.rs`).
/// [`apply_renderide_gpu_validation_env`] runs after extraction.
///
/// When no file exists and [`super::resolve::renderide_config_env_nonempty`] is false, writes
/// defaults to the save path (see [`super::resolve::resolve_save_path`]) and loads that file.
/// This auto-creation is skipped when `policy` is [`ConfigFilePolicy::Ignore`].
pub fn load_renderer_settings(policy: ConfigFilePolicy) -> ConfigLoadResult {
    if policy == ConfigFilePolicy::Ignore {
        if renderide_config_env_nonempty() {
            logger::warn!(
                "--ignore-config is active; RENDERIDE_CONFIG is also set but the file will be skipped"
            );
        }
        let mut settings = match try_extract_settings(renderer_settings_figment()) {
            Ok(s) => s,
            Err(e) => {
                logger::error!(
                    "Renderer config Figment extract failed (--ignore-config, defaults+env): {e:#}"
                );
                RendererSettings::default()
            }
        };
        apply_renderide_gpu_validation_env(&mut settings);
        let resolve = ConfigResolveOutcome {
            attempted_paths: vec![],
            loaded_path: None,
            source: ConfigSource::None,
        };
        let save_path = resolve_save_path(&resolve);
        logger::info!(
            "--ignore-config: skipping TOML file; using struct defaults + RENDERIDE_* env"
        );
        return ConfigLoadResult {
            settings,
            resolve,
            save_path,
            suppress_config_disk_writes: true,
        };
    }

    let mut resolve = resolve_config_path();
    let mut suppress_config_disk_writes = false;
    let mut settings = initial_settings_from_resolve(&mut suppress_config_disk_writes, &resolve);

    if resolve.loaded_path.is_none() && !renderide_config_env_nonempty() {
        maybe_create_default_config_and_reload(
            &mut resolve,
            &mut settings,
            &mut suppress_config_disk_writes,
        );
    }

    let save_path = resolve_save_path(&resolve);

    logger::trace!("Renderer config will persist to {}", save_path.display());

    apply_renderide_gpu_validation_env(&mut settings);

    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes,
    }
}

/// Loads settings from a resolved config path, or defaults plus env when the file is missing or unreadable.
fn initial_settings_from_resolve(
    suppress_config_disk_writes: &mut bool,
    resolve: &ConfigResolveOutcome,
) -> RendererSettings {
    match resolve.loaded_path.as_ref() {
        Some(path) => {
            logger::info!("Loading renderer config from {}", path.display());
            match read_config_file(path) {
                Ok(content) => match load_settings_from_toml_str(&content) {
                    Ok(s) => s,
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
                    match try_extract_settings(renderer_settings_figment()) {
                        Ok(s) => s,
                        Err(e2) => {
                            logger::error!(
                                "Renderer config Figment extract failed (defaults+env only): {e2:#}"
                            );
                            *suppress_config_disk_writes = true;
                            RendererSettings::default()
                        }
                    }
                }
            }
        }
        None => {
            logger::info!("Renderer config file not found; using built-in defaults");
            logger::trace!(
                "config search tried {} path(s)",
                resolve.attempted_paths.len()
            );
            match try_extract_settings(renderer_settings_figment()) {
                Ok(s) => s,
                Err(e) => {
                    logger::error!("Renderer config Figment extract failed (defaults+env): {e:#}");
                    *suppress_config_disk_writes = true;
                    RendererSettings::default()
                }
            }
        }
    }
}

/// When no config was loaded and env overrides are empty, writes default `config.toml` and reloads from disk.
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
    if !is_dir_writable(parent) {
        logger::trace!(
            "Not creating default config at {} (directory not writable)",
            path.display()
        );
        return;
    }
    match save_renderer_settings(&path, &RendererSettings::from_defaults()) {
        Ok(()) => {
            logger::info!("Created default renderer config at {}", path.display());
            apply_generated_config(resolve, path.clone());
            match read_config_file(&path) {
                Ok(content) => match load_settings_from_toml_str(&content) {
                    Ok(s) => {
                        *settings = s;
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

/// Writes `settings` to `path` as TOML atomically (temp file in the same directory, then rename).
pub fn save_renderer_settings(path: &Path, settings: &RendererSettings) -> io::Result<()> {
    let contents = toml::to_string_pretty(settings).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("TOML serialization failed: {e}"),
        )
    })?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(FILE_NAME_TOML);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".{file_name}.tmp"));
    std::fs::write(&tmp, contents.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Persists using [`ConfigLoadResult::save_path`] and logs failures.
pub fn save_renderer_settings_from_load(load: &ConfigLoadResult, settings: &RendererSettings) {
    if load.suppress_config_disk_writes {
        logger::error!(
            "Refusing to save renderer config to {}: initial load had Figment extraction errors; fix the file and restart",
            load.save_path.display()
        );
        return;
    }
    if let Err(e) = save_renderer_settings(&load.save_path, settings) {
        logger::warn!(
            "Failed to save renderer config to {}: {e}",
            load.save_path.display()
        );
    } else {
        logger::trace!("Saved renderer config to {}", load.save_path.display());
    }
}

/// Builds a [`RendererSettingsHandle`] from post-load settings.
pub fn settings_handle_from(load: &ConfigLoadResult) -> RendererSettingsHandle {
    Arc::new(std::sync::RwLock::new(load.settings.clone()))
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
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Serializes tests that mutate `RENDERIDE_*` process environment.
    static CONFIG_ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn atomic_save_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let s = RendererSettings::from_defaults();
        save_renderer_settings(&path, &s).expect("save");
        let text = std::fs::read_to_string(&path).expect("read");
        let s2: RendererSettings = toml::from_str(&text).expect("toml");
        assert_eq!(s, s2);
    }

    #[test]
    fn toml_roundtrip_string() {
        let s = RendererSettings::from_defaults();
        let text = toml::to_string_pretty(&s).expect("ser");
        let s2: RendererSettings = toml::from_str(&text).expect("de");
        assert_eq!(s, s2);
    }

    #[test]
    fn save_path_prefers_loaded() {
        let resolve = ConfigResolveOutcome {
            attempted_paths: vec![],
            loaded_path: Some(PathBuf::from("/tmp/x/config.toml")),
            source: ConfigSource::Search,
        };
        assert_eq!(
            resolve_save_path(&resolve),
            PathBuf::from("/tmp/x/config.toml")
        );
    }

    #[test]
    fn apply_renderide_gpu_validation_env_overrides_flag() {
        let _guard = CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        let mut s = RendererSettings::from_defaults();
        s.debug.gpu_validation_layers = false;
        std::env::set_var("RENDERIDE_GPU_VALIDATION", "1");
        apply_renderide_gpu_validation_env(&mut s);
        assert!(s.debug.gpu_validation_layers);

        s.debug.gpu_validation_layers = true;
        std::env::set_var("RENDERIDE_GPU_VALIDATION", "no");
        apply_renderide_gpu_validation_env(&mut s);
        assert!(!s.debug.gpu_validation_layers);

        std::env::remove_var("RENDERIDE_GPU_VALIDATION");
    }

    #[test]
    fn load_settings_from_toml_merges_renderide_env_nested_key() {
        let _guard = CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        std::env::set_var("RENDERIDE_DISPLAY__FOCUSED_FPS", "137");
        let toml = r#"
[display]
focused_fps = 10
"#;
        let s = load_settings_from_toml_str(toml).expect("figment extract");
        assert_eq!(s.display.focused_fps_cap, 137);
        std::env::remove_var("RENDERIDE_DISPLAY__FOCUSED_FPS");
    }

    #[test]
    fn ignore_config_skips_file_and_suppresses_writes() {
        let result = load_renderer_settings(ConfigFilePolicy::Ignore);
        assert_eq!(result.resolve.source, ConfigSource::None);
        assert!(result.resolve.loaded_path.is_none());
        assert!(result.resolve.attempted_paths.is_empty());
        assert!(result.suppress_config_disk_writes);
    }

    #[test]
    fn ignore_config_env_override_still_applies() {
        let _guard = CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        std::env::set_var("RENDERIDE_DISPLAY__FOCUSED_FPS", "137");
        let result = load_renderer_settings(ConfigFilePolicy::Ignore);
        assert_eq!(result.settings.display.focused_fps_cap, 137);
        std::env::remove_var("RENDERIDE_DISPLAY__FOCUSED_FPS");
    }
}
