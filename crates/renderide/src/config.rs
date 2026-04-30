//! Renderer configuration from `config.toml`.
//!
//! ## Precedence
//!
//! 1. **Struct defaults** — [`RendererSettings::default`].
//! 2. **File** — first match from resolution (see below).
//! 3. **Environment** — variables prefixed with `RENDERIDE_`, nested keys use `__` (for example
//!    `RENDERIDE_DEBUG__GPU_VALIDATION_LAYERS=true`). Applied via the figment crate.
//! 4. **`RENDERIDE_GPU_VALIDATION`** — if set, overrides [`DebugSettings::gpu_validation_layers`]
//!    after the above (see [`apply_renderide_gpu_validation_env`]).
//!
//! ## Resolution order
//!
//! 1. **`RENDERIDE_CONFIG`** — path to `config.toml`. If set and the path is missing, a warning is
//!    logged and resolution continues.
//! 2. **Search** — `config.toml` under:
//!    - next to the current executable and its parent (binary output directory, e.g. `target/debug/`),
//!    - a discovered workspace root (directory containing `Cargo.toml` and
//!      `crates/renderide/Cargo.toml`, from cwd and the executable path),
//!    - current working directory and two levels up (e.g. repo root when running from `crates/renderide`).
//!
//! ## Auto-creation
//!
//! If no file is found and **`RENDERIDE_CONFIG` is not set to a non-empty value**, the renderer
//! writes default settings to the preferred save path (writable directory next to the executable when
//! possible, then [`resolve_save_path`]) and loads that file. If creation fails, built-in defaults are used.
//!
//! ## Persistence
//!
//! The renderer owns the on-disk file when using the **Renderer config** (ImGui) window: values are
//! saved immediately on change. Avoid hand-editing the config file while the process is running; the
//! next save from the UI will overwrite it. Manual edits are best done with the renderer stopped, or
//! use [`save_renderer_settings`] to apply programmatically.

mod persist;
mod reload;
mod resolve;
mod types;

/// Serializes tests that mutate or depend on `RENDERIDE_*` process environment variables.
#[cfg(test)]
pub(crate) static CONFIG_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use persist::{
    apply_renderide_gpu_validation_env, load_renderer_settings, log_config_resolve_trace,
    save_renderer_settings, save_renderer_settings_from_load, settings_handle_from,
    ConfigFilePolicy, ConfigLoadResult, RendererSettingsHandle,
};
pub use reload::{renderer_settings_changed, ConfigFileWatcher};
pub use resolve::{
    apply_generated_config, find_renderide_workspace_root, renderide_config_env_nonempty,
    resolve_config_path, resolve_save_path, ConfigResolveOutcome, ConfigSource, FILE_NAME_TOML,
};
pub use types::{
    BloomCompositeMode, BloomSettings, ClusterAssignmentMode, DebugSettings, DisplaySettings,
    GtaoSettings, MsaaSampleCount, PostProcessingSettings, PowerPreferenceSetting,
    RecordParallelism, RendererSettings, RenderingSettings, SceneColorFormat, TonemapMode,
    TonemapSettings, VsyncMode, WatchdogAction, WatchdogSettings,
};
