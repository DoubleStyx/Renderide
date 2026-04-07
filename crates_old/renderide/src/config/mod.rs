//! Render configuration types and application-level settings.
//!
//! ## Config loading precedence
//!
//! Overall order (later layers override earlier ones):
//!
//! 1. **Defaults** — hardcoded values in [`Default`] impls.
//! 2. **`configuration.ini`** — searched next to the executable, then in the current working
//!    directory (see [`find_config_ini`]).
//! 3. **Environment variables** — highest priority; override INI and defaults where applicable.
//!
//! [`AppConfig`] (FPS caps, HUD toggle) and [`RenderConfig`] (vsync, camera, RTAO, ray-traced PBR
//! shadows, culling, GPU validation, etc.) each load their own keys from that stack; see their
//! respective `load` docs and
//! the table in `app.rs` for `[display]` / `[hud]` / `[camera]` vs `[rendering]`.
//!
//! Two separate structs are exposed:
//! - [`AppConfig`]  — client-side settings (FPS caps, HUD toggle).  Read once
//!   at startup in `app.rs`; never touched by IPC session commands.
//! - [`RenderConfig`] — rendering parameters that *can* be overridden by host
//!   IPC commands (vsync, RTAO, clip planes, …).  Loaded via [`RenderConfig::load`].

mod app_config;
mod ini;
mod render_config;

pub use app_config::AppConfig;
pub use ini::find_config_ini;
pub use render_config::{RenderConfig, ShaderDebugOverride};
