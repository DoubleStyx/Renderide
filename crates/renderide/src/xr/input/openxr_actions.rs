//! Creates the OpenXR action set, actions, and controller pose spaces from the data-only manifest.
//!
//! Every action bound by `crates/renderide/assets/xr/bindings/` becomes a typed [`xr::Action`]
//! field on [`OpenxrInputActions`]; every binding table under
//! [`super::bindings::apply_suggested_interaction_bindings`]. Interaction profile paths that appear
//! in the manifest are pre-resolved into [`ResolvedProfilePaths`] for the per-frame detection loop
//! in [`super::openxr_input`].
//!
//! The lifecycle remains the spec-required order: create action set -> create actions -> suggest
//! bindings (per profile) -> attach action sets -> create action spaces.

mod action_handles;
mod profile_paths;

use std::sync::atomic::AtomicU8;

use openxr as xr;

use super::bindings::{
    ProfileExtensionGates, apply_suggested_interaction_bindings, build_action_handle_map,
};
use super::manifest::Manifest;

pub(super) use action_handles::OpenxrInputActions;
use action_handles::build_actions;
pub(super) use profile_paths::ResolvedProfilePaths;

/// Resolved `/user/hand/left` and `/user/hand/right` paths.
struct UserPaths {
    /// `/user/hand/left`
    left_user_path: xr::Path,
    /// `/user/hand/right`
    right_user_path: xr::Path,
}

/// Interns the two top-level user-hand path strings via [`openxr::Instance::string_to_path`].
///
/// All interaction profile paths and per-action input/output paths are described in the TOML
/// manifest (see [`super::manifest`]) and resolved on demand by
/// [`super::bindings::apply_suggested_interaction_bindings`] and [`ResolvedProfilePaths`].
/// Only `/user/hand/left` and `/user/hand/right` are pre-resolved here because they are used by
/// the per-frame [`openxr::Session::current_interaction_profile`] queries.
fn resolve_user_paths(instance: &xr::Instance) -> Result<UserPaths, xr::sys::Result> {
    Ok(UserPaths {
        left_user_path: instance.string_to_path("/user/hand/left")?,
        right_user_path: instance.string_to_path("/user/hand/right")?,
    })
}

/// Creates the controller pose spaces (grip and palm) anchored at
/// [`openxr::Posef::IDENTITY`] for the lifetime of the session.
fn create_pose_spaces(
    session: &xr::Session<xr::Vulkan>,
    actions: &OpenxrInputActions,
) -> Result<(xr::Space, xr::Space, xr::Space, xr::Space), xr::sys::Result> {
    Ok((
        actions
            .left_grip_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .right_grip_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .left_palm_ext_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .right_palm_ext_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
    ))
}
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
    /// Left palm pose space.
    pub(super) left_palm_ext_space: xr::Space,
    /// Right palm pose space.
    pub(super) right_palm_ext_space: xr::Space,
    /// Left hand OpenXR hand tracker.
    pub(super) left_hand_tracker: Option<xr::HandTracker>,
    /// Right hand OpenXR hand tracker.
    pub(super) right_hand_tracker: Option<xr::HandTracker>,
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

    {
        let action_handle_map = build_action_handle_map(&actions);
        apply_suggested_interaction_bindings(instance, manifest, &action_handle_map, gates)?;
    }

    session.attach_action_sets(&[&action_set])?;

    let (left_space, right_space, left_palm_space, right_palm_space) =
        create_pose_spaces(session, &actions)?;

    let (left_hand_tracker, right_hand_tracker) = if gates.hand_tracking_ext {
        (
            session.create_hand_tracker(xr::Hand::LEFT).ok(),
            session.create_hand_tracker(xr::Hand::RIGHT).ok(),
        )
    } else {
        (None, None)
    };

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
        left_palm_ext_space: left_palm_space,
        right_palm_ext_space: right_palm_space,
        left_hand_tracker,
        right_hand_tracker,
    })
}
