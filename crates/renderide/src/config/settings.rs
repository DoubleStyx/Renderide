//! Process-wide renderer settings merged from defaults, optional `config.toml`, and environment.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;
use serde::{Deserialize, Serialize};

use super::resolve::{
    apply_generated_config, is_dir_writable, read_config_file, renderide_config_env_nonempty,
    resolve_config_path, resolve_save_path, ConfigResolveOutcome, ConfigSource, FILE_NAME_TOML,
};

/// Display-related caps. Persisted as `[display]`.
///
/// Non-zero values cap desktop redraw scheduling via winit (`ControlFlow::WaitUntil`); OpenXR VR
/// sessions ignore these caps so headset frame pacing is unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// Target max FPS when the window is focused (0 = uncapped).
    #[serde(rename = "focused_fps")]
    pub focused_fps_cap: u32,
    /// Target max FPS when unfocused (0 = uncapped).
    #[serde(rename = "unfocused_fps")]
    pub unfocused_fps_cap: u32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            focused_fps_cap: 240,
            unfocused_fps_cap: 60,
        }
    }
}

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Vertical sync via swapchain present mode ([`wgpu::PresentMode::AutoVsync`]); applied live
    /// without restart (see [`crate::gpu::GpuContext::set_vsync`]).
    pub vsync: bool,
}

/// Preferred GPU power mode for future adapter selection (stored; changing at runtime may require
/// re-initialization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerPreferenceSetting {
    /// Maps to [`wgpu::PowerPreference::LowPower`].
    LowPower,
    /// Maps to [`wgpu::PowerPreference::HighPerformance`].
    #[default]
    HighPerformance,
}

impl PowerPreferenceSetting {
    /// All variants for ImGui combo / persistence.
    pub const ALL: [Self; 2] = [Self::LowPower, Self::HighPerformance];

    /// Stable string for TOML / UI (`low_power` / `high_performance`).
    pub fn as_persist_str(self) -> &'static str {
        match self {
            Self::LowPower => "low_power",
            Self::HighPerformance => "high_performance",
        }
    }

    /// Parses case-insensitive persisted or UI token.
    pub fn from_persist_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low_power" | "low" => Some(Self::LowPower),
            "high_performance" | "high" | "performance" => Some(Self::HighPerformance),
            _ => None,
        }
    }

    /// Label for developer UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::LowPower => "Low power",
            Self::HighPerformance => "High performance",
        }
    }
}

/// Debug and diagnostics flags. Persisted as `[debug]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugSettings {
    /// When the `-LogLevel` CLI argument is **not** present, selects [`logger::LogLevel::Trace`] if true or
    /// [`logger::LogLevel::Debug`] if false. If `-LogLevel` is present, it always overrides this flag.
    pub log_verbose: bool,
    /// GPU power preference hint for adapter selection (see [`PowerPreferenceSetting`]).
    pub power_preference: PowerPreferenceSetting,
    /// When true, request backend validation (e.g. Vulkan validation layers) via wgpu instance
    /// flags. Significantly slows rendering; use only when debugging GPU API misuse. Default false. Applies to both desktop
    /// wgpu init and the OpenXR Vulkan / wgpu-hal bootstrap. Native **stdout** and **stderr** are
    /// forwarded to the renderer log file after logging starts (see [`crate::app::run`]), so layer
    /// and spirv-val output is captured regardless of this flag.
    /// Applied when the GPU stack is first created, not on later config updates.
    /// [`apply_renderide_gpu_validation_env`] and `WGPU_*` environment variables can still adjust
    /// flags at process start.
    pub gpu_validation_layers: bool,
    /// When true, show the **Frame timing** ImGui window (FPS and CPU/GPU submit-interval metrics). Cheap snapshot;
    /// independent of [`Self::debug_hud_enabled`]. Default true.
    #[serde(default = "default_debug_hud_frame_timing")]
    pub debug_hud_frame_timing: bool,
    /// When true, show **Renderide debug** (Stats / Shader routes) and run mesh-draw stats, frame diagnostics, and
    /// renderer info capture. Default false (performance-first; **Renderer config** or `debug_hud_enabled` in config).
    pub debug_hud_enabled: bool,
    /// When true, capture [`crate::diagnostics::SceneTransformsSnapshot`] each frame and show the **Scene transforms**
    /// ImGui window (can be expensive on large scenes). Independent of [`Self::debug_hud_enabled`] so you can enable
    /// transforms inspection without the main debug panels. Default false.
    pub debug_hud_transforms: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            log_verbose: false,
            power_preference: PowerPreferenceSetting::default(),
            gpu_validation_layers: false,
            debug_hud_frame_timing: true,
            debug_hud_enabled: false,
            debug_hud_transforms: false,
        }
    }
}

fn default_debug_hud_frame_timing() -> bool {
    true
}

/// Runtime settings for the renderer process: defaults, merged from file, and edited via the debug UI.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RendererSettings {
    /// Display caps and related options.
    pub display: DisplaySettings,
    /// Rendering options (e.g. vsync).
    pub rendering: RenderingSettings,
    /// Debug-only flags.
    pub debug: DebugSettings,
}

impl RendererSettings {
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }
}

fn renderer_settings_figment() -> Figment {
    Figment::new()
        .merge(Serialized::defaults(RendererSettings::default()))
        .merge(Env::prefixed("RENDERIDE_").split("__"))
}

fn extract_settings(figment: Figment) -> RendererSettings {
    match figment.extract::<RendererSettings>() {
        Ok(s) => s,
        Err(e) => {
            logger::warn!("Renderer config extract failed: {e}; using built-in defaults");
            RendererSettings::default()
        }
    }
}

fn load_settings_from_toml_str(content: &str) -> RendererSettings {
    let figment = Figment::new()
        .merge(Serialized::defaults(RendererSettings::default()))
        .merge(Toml::string(content))
        .merge(Env::prefixed("RENDERIDE_").split("__"));
    extract_settings(figment)
}

/// Overrides [`DebugSettings::gpu_validation_layers`] when `RENDERIDE_GPU_VALIDATION` is set.
///
/// Truthy values (`1`, `true`, `yes`) force validation on; falsey (`0`, `false`, `no`) force off.
/// If unset, the value from config or defaults is unchanged.
pub fn apply_renderide_gpu_validation_env(settings: &mut RendererSettings) {
    match std::env::var("RENDERIDE_GPU_VALIDATION").as_deref() {
        Ok("1") | Ok("true") | Ok("yes") => settings.debug.gpu_validation_layers = true,
        Ok("0") | Ok("false") | Ok("no") => settings.debug.gpu_validation_layers = false,
        _ => {}
    }
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
}

/// Shared handle for the process-wide settings store (read by the frame loop, written by the HUD).
pub type RendererSettingsHandle = Arc<std::sync::RwLock<RendererSettings>>;

/// Resolves `config.toml`, merges with figment layers, and builds [`RendererSettings`].
///
/// Precedence: struct defaults, then TOML file, then `RENDERIDE_*` environment variables (see module
/// docs in `config/mod.rs`). [`apply_renderide_gpu_validation_env`] runs after extraction.
///
/// When no file exists and [`super::resolve::renderide_config_env_nonempty`] is false, writes
/// defaults to the save path (see [`super::resolve::resolve_save_path`]) and loads that file.
pub fn load_renderer_settings() -> ConfigLoadResult {
    let mut resolve = resolve_config_path();
    let mut settings = match resolve.loaded_path.as_ref() {
        Some(path) => {
            logger::info!("Loading renderer config from {}", path.display());
            match read_config_file(path) {
                Ok(content) => load_settings_from_toml_str(&content),
                Err(e) => {
                    logger::warn!("Failed to read {}: {e}; using defaults", path.display());
                    extract_settings(renderer_settings_figment())
                }
            }
        }
        None => {
            logger::info!("Renderer config file not found; using built-in defaults");
            logger::trace!(
                "config search tried {} path(s)",
                resolve.attempted_paths.len()
            );
            extract_settings(renderer_settings_figment())
        }
    };

    if resolve.loaded_path.is_none() && !renderide_config_env_nonempty() {
        let path = resolve_save_path(&resolve);
        if !path.exists() {
            if let Some(parent) = path.parent() {
                if is_dir_writable(parent) {
                    match save_renderer_settings(&path, &RendererSettings::from_defaults()) {
                        Ok(()) => {
                            logger::info!("Created default renderer config at {}", path.display());
                            apply_generated_config(&mut resolve, path.clone());
                            match read_config_file(&path) {
                                Ok(content) => {
                                    settings = load_settings_from_toml_str(&content);
                                }
                                Err(e) => {
                                    logger::warn!(
                                        "Failed to read newly created {}: {e}; using defaults",
                                        path.display()
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            logger::warn!(
                                "Failed to create default config at {}: {e}",
                                path.display()
                            );
                        }
                    }
                } else {
                    logger::trace!(
                        "Not creating default config at {} (directory not writable)",
                        path.display()
                    );
                }
            }
        }
    }

    let save_path = resolve_save_path(&resolve);

    logger::trace!("Renderer config will persist to {}", save_path.display());

    apply_renderide_gpu_validation_env(&mut settings);

    ConfigLoadResult {
        settings,
        resolve,
        save_path,
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
}
