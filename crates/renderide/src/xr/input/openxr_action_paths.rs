//! Resolves the two user-hand paths consumed by [`super::openxr_actions::create_openxr_input_parts`].
//!
//! All interaction profile paths and per-action input/output paths are now described in the TOML
//! manifest (see [`super::manifest`]) and resolved on demand by
//! [`super::bindings::apply_suggested_interaction_bindings`] and
//! [`super::openxr_actions::ResolvedProfilePaths`]. Only `/user/hand/left` and
//! `/user/hand/right` are pre-resolved here because they are used by the per-frame
//! [`openxr::Session::current_interaction_profile`] queries.

use openxr as xr;

/// Resolved `/user/hand/left` and `/user/hand/right` paths.
pub(super) struct UserPaths {
    /// `/user/hand/left`
    pub(super) left_user_path: xr::Path,
    /// `/user/hand/right`
    pub(super) right_user_path: xr::Path,
}

/// Interns the two top-level user-hand path strings via [`openxr::Instance::string_to_path`].
pub(super) fn resolve_user_paths(instance: &xr::Instance) -> Result<UserPaths, xr::sys::Result> {
    Ok(UserPaths {
        left_user_path: instance.string_to_path("/user/hand/left")?,
        right_user_path: instance.string_to_path("/user/hand/right")?,
    })
}
