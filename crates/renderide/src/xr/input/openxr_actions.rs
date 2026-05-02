//! Creates the OpenXR action set, actions, and controller pose spaces from the data-only manifest.
//!
//! Every action listed in `crates/renderide/assets/xr/actions.toml` becomes a typed
//! [`xr::Action`] field on [`OpenxrInputActions`]; every binding table under
//! `crates/renderide/assets/xr/bindings/` is forwarded to
//! [`super::bindings::apply_suggested_interaction_bindings`]. Interaction profile paths that appear
//! in the manifest are pre-resolved into [`ResolvedProfilePaths`] for the per-frame detection loop
//! in [`super::openxr_input`].
//!
//! The lifecycle remains the spec-required order: create action set -> create actions -> suggest
//! bindings (per profile) -> attach action sets -> create action spaces.

mod action_handles;
mod profile_paths;
mod spaces;

use std::sync::atomic::AtomicU8;

use openxr as xr;

use super::bindings::{
    ProfileExtensionGates, apply_suggested_interaction_bindings, build_action_handle_map,
};
use super::manifest::Manifest;
use super::openxr_action_paths::{UserPaths, resolve_user_paths};

pub(super) use action_handles::OpenxrInputActions;
use action_handles::build_actions;
pub(super) use profile_paths::ResolvedProfilePaths;
use spaces::create_grip_and_aim_spaces;

/// Container for everything [`super::openxr_input::OpenxrInput`] needs after setup.
pub(super) struct OpenxrInputParts {
    /// OpenXR action set, kept alive for the session.
    pub(super) action_set: xr::ActionSet,
    /// `/user/hand/left` path used to query active profile per hand.
    pub(super) left_user_path: xr::Path,
    /// `/user/hand/right` path.
    pub(super) right_user_path: xr::Path,
    /// All typed action handles.
    pub(super) actions: OpenxrInputActions,
    /// Resolved interaction profile paths, used by per-frame profile detection.
    pub(super) profile_paths: ResolvedProfilePaths,
    /// Encoded last-seen left-hand profile; see [`super::profile::profile_code`].
    pub(super) left_profile_cache: AtomicU8,
    /// Encoded last-seen right-hand profile.
    pub(super) right_profile_cache: AtomicU8,
    /// Left grip pose space.
    pub(super) left_space: xr::Space,
    /// Right grip pose space.
    pub(super) right_space: xr::Space,
    /// Left aim pose space.
    pub(super) left_aim_space: xr::Space,
    /// Right aim pose space.
    pub(super) right_aim_space: xr::Space,
}

/// Manifest-driven end-to-end OpenXR input setup: action set, actions, suggested bindings, attach, spaces.
///
/// `gates` describes which OpenXR extensions were enabled on the instance; profiles whose
/// extension is disabled are skipped to avoid suggesting bindings against paths the runtime does
/// not recognise.
///
/// `manifest` supplies every action id, localized label, binding profile, and binding path; no
/// input data is baked into this file.
pub(super) fn create_openxr_input_parts(
    instance: &xr::Instance,
    session: &xr::Session<xr::Vulkan>,
    gates: &ProfileExtensionGates,
    manifest: &Manifest,
) -> Result<OpenxrInputParts, xr::sys::Result> {
    let UserPaths {
        left_user_path,
        right_user_path,
    } = resolve_user_paths(instance)?;

    let action_set = instance.create_action_set(
        &manifest.actions.action_set.id,
        &manifest.actions.action_set.localized_name,
        manifest.actions.action_set.priority,
    )?;

    let actions = build_actions(&action_set, &manifest.actions)?;
    let profile_paths = ResolvedProfilePaths::from_manifest(instance, manifest)?;

    let action_handle_map = build_action_handle_map(&actions);
    apply_suggested_interaction_bindings(instance, manifest, &action_handle_map, gates)?;

    session.attach_action_sets(&[&action_set])?;

    let (left_space, right_space, left_aim_space, right_aim_space) =
        create_grip_and_aim_spaces(session, &actions)?;

    Ok(OpenxrInputParts {
        action_set,
        left_user_path,
        right_user_path,
        actions,
        profile_paths,
        left_profile_cache: AtomicU8::new(0),
        right_profile_cache: AtomicU8::new(0),
        left_space,
        right_space,
        left_aim_space,
        right_aim_space,
    })
}
