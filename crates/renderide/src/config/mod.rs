//! Renderer configuration from `config.ini`.
//!
//! ## Precedence
//!
//! 1. **`RENDERIDE_CONFIG`** — path to an INI file. If set and the path is missing, a warning is
//!    logged and resolution continues.
//! 2. **Search** (first existing `config.ini`):
//!    - next to the current executable,
//!    - parent of the executable directory,
//!    - current working directory,
//!    - two levels up from cwd (e.g. repo root when running from `crates/renderide`).
//! 3. **Defaults** — when no file is found or read fails, [`RendererSettings`] stays at
//!    [`Default::default`].
//!
//! Comment lines (`#` or `;`) and omitted keys retain defaults. Inline `#` / `;` strip the rest of
//! the value (legacy parity).

mod parse;
mod resolve;
mod settings;

pub use parse::{parse_ini_document, IniDocument, ParseWarning};
pub use resolve::{resolve_config_path, ConfigResolveOutcome, ConfigSource};
pub use settings::{
    load_renderer_settings, log_config_resolve_trace, ConfigLoadResult, RendererSettings,
};
