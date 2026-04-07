//! Renderer configuration from `config.ini`.
//!
//! ## Precedence
//!
//! 1. **`RENDERIDE_CONFIG`** — path to an INI file. If set and the path is missing, a warning is
//!    logged and resolution continues.
//! 2. **Search** (first existing `config.ini`):
//!    - `config.ini` under a discovered workspace root (directory containing `Cargo.toml` and
//!      `crates/renderide/Cargo.toml`, found from the current working directory and the executable
//!      path),
//!    - next to the current executable,
//!    - parent of the executable directory,
//!    - current working directory,
//!    - two levels up from cwd (e.g. repo root when running from `crates/renderide`).
//! 3. **Defaults** — when no file is found or read fails, [`RendererSettings`] stays at
//!    [`Default::default`].
//!
//! ## Auto-creation
//!
//! If no file is found and **`RENDERIDE_CONFIG` is not set to a non-empty value**, the renderer
//! writes default settings to the preferred save path (workspace root `config.ini` when the
//! workspace is discovered and writable, otherwise the same rules as [`resolve_save_path`]) and loads
//! that file. If creation fails, built-in defaults are used.
//!
//! Comment lines (`#` or `;`) and omitted keys retain defaults. Inline `#` / `;` strip the rest of
//! the value (legacy parity).
//!
//! ## Persistence
//!
//! The renderer owns the on-disk file when using the **Renderer config** (ImGui) window: values are
//! saved immediately on change. Avoid hand-editing `config.ini` while the process is running; the
//! next save from the UI will overwrite the file. Manual edits are best done with the renderer
//! stopped, or use [`save_renderer_settings`] to apply programmatically.

mod parse;
mod resolve;
mod settings;

pub use parse::{parse_ini_document, IniDocument, ParseWarning};
pub use resolve::{
    apply_generated_config, find_renderide_workspace_root, renderide_config_env_nonempty,
    resolve_config_path, resolve_save_path, ConfigResolveOutcome, ConfigSource,
};
pub use settings::{
    load_renderer_settings, log_config_resolve_trace, save_renderer_settings,
    save_renderer_settings_from_load, settings_handle_from, ConfigLoadResult, DebugSettings,
    DisplaySettings, PowerPreferenceSetting, RendererSettings, RendererSettingsHandle,
    RenderingSettings,
};
