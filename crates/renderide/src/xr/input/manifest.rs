//! Data-driven OpenXR action and interaction profile binding manifest.
//!
//! Parses `actions.toml` (action set + action list) and `bindings/*.toml` (per-profile binding
//! tables) at session init, so adding an interaction profile is a data change rather than a Rust
//! edit. The runtime pipeline feeds the parsed [`ActionManifest`] into
//! [`super::openxr_actions::create_openxr_input_parts`] to create typed [`openxr::Action`] handles,
//! and into [`super::bindings::apply_suggested_interaction_bindings`] to submit suggested bindings
//! per profile.
//!
//! Assets live in `crates/renderide/assets/xr/` at source, and are copied next to the binary by
//! [`crate::build`](../../../build.rs) so the same resolution pattern used for `config.toml`
//! locates them at runtime.
//!
//! Validation at parse time rejects manifests that would otherwise produce confusing runtime
//! failures -- unknown action references, duplicated action ids, misrouted haptic bindings, or
//! unknown extension gate names.
//!
//! Parse is pure (`parse_action_manifest` / `parse_binding_profile` take string slices) so unit
//! tests do not touch the filesystem. Disk loading is confined to [`load_manifest`].

mod loader;
mod parser;
mod types;

pub use loader::load_manifest;
pub use types::ManifestError;
pub(super) use types::{ActionManifest, ActionType, ExtensionGate, Manifest};

#[cfg(test)]
mod tests;
