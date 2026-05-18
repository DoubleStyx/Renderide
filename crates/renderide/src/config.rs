//! Renderer configuration from `config.toml`.
//!
//! ## Precedence
//!
//! 1. **Struct defaults** -- [`RendererSettings::default`].
//! 2. **File** -- first match from resolution (see below).
//! 3. **Environment** -- variables prefixed with `RENDERIDE_`, nested keys use `__` (for example
//!    `RENDERIDE_DEBUG__GPU_VALIDATION_LAYERS=true`). Applied via the figment crate.
//! 4. **`RENDERIDE_GPU_VALIDATION`** -- if set, overrides [`DebugSettings::gpu_validation_layers`]
//!    after the above (see [`apply_renderide_gpu_validation_env`]).
//!
//! ## Resolution order
//!
//! 1. **`RENDERIDE_CONFIG`** -- path to `config.toml`. If set and the path is missing, a warning is
//!    logged and resolution continues.
//! 2. **User config directory** -- the per-platform config base from the `directories` crate with
//!    `Renderide/config.toml` appended:
//!    - Linux: `$XDG_CONFIG_HOME/Renderide/config.toml`, or `~/.config/Renderide/config.toml`
//!    - macOS: `~/Library/Application Support/Renderide/config.toml`
//!    - Windows: `%APPDATA%\Renderide\config.toml`
//!
//! ## Auto-creation
//!
//! If no file is found and **`RENDERIDE_CONFIG` is not set to a non-empty value**, the renderer
//! writes default settings to the user config path and loads that file. If creation fails, built-in
//! defaults are used.
//!
//! ## One-shot migration
//!
//! On startup, if the user config file is missing and no `RENDERIDE_CONFIG` override is set, the
//! renderer scans previous config locations: next to the binary, at the discovered workspace root,
//! in the current working directory, and two levels above it. The first hit is copied into the user
//! config directory and the original is renamed to `config.toml.migrated` when possible.
//!
//! ## Persistence
//!
//! The renderer owns the on-disk file when using the **Renderer config** (ImGui) window: values are
//! saved immediately on change while preserving keys this renderer version does not understand.
//! Avoid hand-editing the config file while the process is running; the next save from the UI will
//! rewrite known settings. Manual edits are best done with the renderer stopped, or use
//! [`save_renderer_settings`] to apply programmatically.
//!
//! Config files include a top-level `config_version` matching the renderer crate version that wrote
//! the file. Load-time migrations use that field to distinguish old on-disk semantics from current
//! settings.

mod handle;
pub mod labeled_enum;
pub(crate) mod persist;
mod types;
pub mod value;

/// Serializes tests that mutate or depend on `RENDERIDE_*` process environment variables.
#[cfg(test)]
pub(crate) static CONFIG_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use handle::{RendererSettingsHandle, settings_handle_from};
pub use persist::{
    ConfigFilePolicy, ConfigLoadResult, find_renderide_workspace_root, load_renderer_settings,
    log_config_resolve_trace, save_renderer_settings, save_renderer_settings_pruned,
};
#[cfg(test)]
pub(crate) use persist::{ConfigResolveOutcome, ConfigSource};
pub use types::{
    AutoExposureSettings, BloomCompositeMode, BloomSettings, DebugHudMainTab,
    DebugHudMainTabVisibility, DebugHudRendererConfigTab, DebugHudRendererConfigTabVisibility,
    DebugHudSettings, ExperimentalSettings, GraphicsApiSetting, GtaoSettings, MotionBlurSettings,
    MsaaSampleCount, PostProcessingSettings, PowerPreferenceSetting, RendererSettings,
    SceneColorFormat, TonemapMode, VsyncMode, WatchdogAction, WatchdogSettings,
};
#[cfg(test)]
pub(crate) use types::{DebugSettings, TonemapSettings};
