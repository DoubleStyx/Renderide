//! Creates the OpenXR action set, actions, and controller pose spaces from the data-only manifest.
//!
//! Every action listed in `crates/renderide/assets/xr/actions.toml` becomes a typed
//! [`xr::Action`] field on [`OpenxrInputActions`]; every binding table under
//! `crates/renderide/assets/xr/bindings/` is forwarded to
//! [`super::bindings::apply_suggested_interaction_bindings`]. Interaction profile paths that appear
//! in the manifest are pre-resolved into [`ResolvedProfilePaths`] for the per-frame detection loop
//! in [`super::openxr_input`].
//!
//! The lifecycle remains the spec-required order: create action set → create actions → suggest
//! bindings (per profile) → attach action sets → create action spaces.

use std::sync::atomic::AtomicU8;

use openxr as xr;

use super::bindings::{
    apply_suggested_interaction_bindings, build_action_handle_map, ProfileExtensionGates,
};
use super::manifest::{ActionManifest, ActionType, Manifest};
use super::openxr_action_paths::{resolve_user_paths, UserPaths};

/// Typed [`xr::Action`] handles for every action declared in the manifest.
///
/// Field names mirror the action ids in `actions.toml`; adding a new action requires adding a
/// matching field here so the binding loop can look it up by id. This struct is flat on purpose
/// to keep per-frame state read sites in [`super::openxr_input`] terse.
pub(super) struct OpenxrInputActions {
    /// Left-hand tracked grip pose. See `/user/hand/left/input/grip/pose`.
    pub(super) left_grip_pose: xr::Action<xr::Posef>,
    /// Right-hand tracked grip pose.
    pub(super) right_grip_pose: xr::Action<xr::Posef>,
    /// Left-hand aim pose (pointing ray origin).
    pub(super) left_aim_pose: xr::Action<xr::Posef>,
    /// Right-hand aim pose.
    pub(super) right_aim_pose: xr::Action<xr::Posef>,

    /// Left trigger analog value, 0..=1.
    pub(super) left_trigger: xr::Action<f32>,
    /// Right trigger analog value, 0..=1.
    pub(super) right_trigger: xr::Action<f32>,
    /// Left trigger finger-touch digital state.
    pub(super) left_trigger_touch: xr::Action<bool>,
    /// Right trigger finger-touch digital state.
    pub(super) right_trigger_touch: xr::Action<bool>,
    /// Left trigger fully-pressed click digital state.
    pub(super) left_trigger_click: xr::Action<bool>,
    /// Right trigger fully-pressed click digital state.
    pub(super) right_trigger_click: xr::Action<bool>,

    /// Left grip/squeeze analog value, 0..=1.
    pub(super) left_squeeze: xr::Action<f32>,
    /// Right grip/squeeze analog value, 0..=1.
    pub(super) right_squeeze: xr::Action<f32>,
    /// Left grip/squeeze fully-pressed click digital state.
    pub(super) left_squeeze_click: xr::Action<bool>,
    /// Right grip/squeeze fully-pressed click digital state.
    pub(super) right_squeeze_click: xr::Action<bool>,

    /// Left thumbstick 2D deflection, each axis -1..=1.
    pub(super) left_thumbstick: xr::Action<xr::Vector2f>,
    /// Right thumbstick 2D deflection.
    pub(super) right_thumbstick: xr::Action<xr::Vector2f>,
    /// Left thumbstick touch digital state.
    pub(super) left_thumbstick_touch: xr::Action<bool>,
    /// Right thumbstick touch digital state.
    pub(super) right_thumbstick_touch: xr::Action<bool>,
    /// Left thumbstick click digital state.
    pub(super) left_thumbstick_click: xr::Action<bool>,
    /// Right thumbstick click digital state.
    pub(super) right_thumbstick_click: xr::Action<bool>,

    /// Left trackpad 2D position.
    pub(super) left_trackpad: xr::Action<xr::Vector2f>,
    /// Right trackpad 2D position.
    pub(super) right_trackpad: xr::Action<xr::Vector2f>,
    /// Left trackpad touch digital state.
    pub(super) left_trackpad_touch: xr::Action<bool>,
    /// Right trackpad touch digital state.
    pub(super) right_trackpad_touch: xr::Action<bool>,
    /// Left trackpad click digital state.
    pub(super) left_trackpad_click: xr::Action<bool>,
    /// Right trackpad click digital state.
    pub(super) right_trackpad_click: xr::Action<bool>,
    /// Left trackpad press-force analog (Index).
    pub(super) left_trackpad_force: xr::Action<f32>,
    /// Right trackpad press-force analog.
    pub(super) right_trackpad_force: xr::Action<f32>,

    /// Left primary face button (e.g. X on Touch, A on Index).
    pub(super) left_primary: xr::Action<bool>,
    /// Right primary face button (A on Touch/Index).
    pub(super) right_primary: xr::Action<bool>,
    /// Left secondary face button (Y on Touch, B on Index).
    pub(super) left_secondary: xr::Action<bool>,
    /// Right secondary face button (B on Touch/Index).
    pub(super) right_secondary: xr::Action<bool>,
    /// Left primary face button capacitive touch.
    pub(super) left_primary_touch: xr::Action<bool>,
    /// Right primary face button capacitive touch.
    pub(super) right_primary_touch: xr::Action<bool>,
    /// Left secondary face button capacitive touch.
    pub(super) left_secondary_touch: xr::Action<bool>,
    /// Right secondary face button capacitive touch.
    pub(super) right_secondary_touch: xr::Action<bool>,

    /// Left menu/application button.
    pub(super) left_menu: xr::Action<bool>,
    /// Right menu/application button (not all profiles expose this).
    pub(super) right_menu: xr::Action<bool>,

    /// Left thumbrest capacitive touch.
    pub(super) left_thumbrest_touch: xr::Action<bool>,
    /// Right thumbrest capacitive touch.
    pub(super) right_thumbrest_touch: xr::Action<bool>,

    /// Left select action (simple/generic profiles).
    pub(super) left_select: xr::Action<bool>,
    /// Right select action (simple/generic profiles).
    pub(super) right_select: xr::Action<bool>,

    /// Left hand haptic output.
    pub(super) left_haptic: xr::Action<xr::Haptic>,
    /// Right hand haptic output.
    pub(super) right_haptic: xr::Action<xr::Haptic>,
}

/// Interaction profile paths resolved from the manifest for per-frame profile detection.
///
/// Missing entries remain `xr::Path::NULL` so path comparisons in
/// [`super::openxr_input::OpenxrInput::detect_profile`] safely return a non-match when a specific
/// profile is not in the manifest.
#[derive(Default)]
pub(super) struct ResolvedProfilePaths {
    /// `/interaction_profiles/oculus/touch_controller`
    pub(super) oculus_touch: xr::Path,
    /// `/interaction_profiles/valve/index_controller`
    pub(super) valve_index: xr::Path,
    /// `/interaction_profiles/htc/vive_controller`
    pub(super) htc_vive: xr::Path,
    /// `/interaction_profiles/microsoft/motion_controller`
    pub(super) microsoft_motion: xr::Path,
    /// `/interaction_profiles/khr/generic_controller`
    pub(super) generic_controller: xr::Path,
    /// `/interaction_profiles/khr/simple_controller`
    pub(super) simple_controller: xr::Path,
    /// `/interaction_profiles/bytedance/pico4_controller`
    pub(super) pico4_controller: xr::Path,
    /// `/interaction_profiles/bytedance/pico_neo3_controller`
    pub(super) pico_neo3_controller: xr::Path,
    /// `/interaction_profiles/hp/mixed_reality_controller`
    pub(super) hp_reverb_g2: xr::Path,
    /// `/interaction_profiles/samsung/odyssey_controller`
    pub(super) samsung_odyssey: xr::Path,
    /// `/interaction_profiles/htc/vive_cosmos_controller`
    pub(super) htc_vive_cosmos: xr::Path,
    /// `/interaction_profiles/htc/vive_focus3_controller`
    pub(super) htc_vive_focus3: xr::Path,
    /// `/interaction_profiles/facebook/touch_controller_pro`
    pub(super) meta_touch_pro: xr::Path,
    /// `/interaction_profiles/meta/touch_controller_plus`
    pub(super) meta_touch_plus: xr::Path,
}

impl ResolvedProfilePaths {
    /// Resolves every profile path listed in the manifest through the OpenXR instance and stores
    /// the handles in the corresponding well-known field. Unknown profile paths are ignored; new
    /// profiles that need runtime detection must also be wired here plus in `detect_profile`.
    fn from_manifest(
        instance: &xr::Instance,
        manifest: &Manifest,
    ) -> Result<Self, xr::sys::Result> {
        let mut out = Self::default();
        for profile in &manifest.profiles {
            let path = instance.string_to_path(&profile.profile)?;
            match profile.profile.as_str() {
                "/interaction_profiles/oculus/touch_controller" => out.oculus_touch = path,
                "/interaction_profiles/valve/index_controller" => out.valve_index = path,
                "/interaction_profiles/htc/vive_controller" => out.htc_vive = path,
                "/interaction_profiles/microsoft/motion_controller" => {
                    out.microsoft_motion = path;
                }
                "/interaction_profiles/khr/generic_controller" => out.generic_controller = path,
                "/interaction_profiles/khr/simple_controller" => out.simple_controller = path,
                "/interaction_profiles/bytedance/pico4_controller" => out.pico4_controller = path,
                "/interaction_profiles/bytedance/pico_neo3_controller" => {
                    out.pico_neo3_controller = path;
                }
                "/interaction_profiles/hp/mixed_reality_controller" => out.hp_reverb_g2 = path,
                "/interaction_profiles/samsung/odyssey_controller" => {
                    out.samsung_odyssey = path;
                }
                "/interaction_profiles/htc/vive_cosmos_controller" => {
                    out.htc_vive_cosmos = path;
                }
                "/interaction_profiles/htc/vive_focus3_controller" => {
                    out.htc_vive_focus3 = path;
                }
                "/interaction_profiles/facebook/touch_controller_pro" => {
                    out.meta_touch_pro = path;
                }
                "/interaction_profiles/meta/touch_controller_plus" => {
                    out.meta_touch_plus = path;
                }
                _ => {}
            }
        }
        Ok(out)
    }
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
    /// Left aim pose space.
    pub(super) left_aim_space: xr::Space,
    /// Right aim pose space.
    pub(super) right_aim_space: xr::Space,
}

/// Creates a typed action, verifying the declared manifest type matches `expected`.
///
/// Returns [`xr::sys::Result::ERROR_VALIDATION_FAILURE`] when the manifest omits the action id or
/// declares a different type; both cases are fatal setup failures with diagnostic logging.
fn create_typed_action<T: xr::ActionTy>(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
    id: &str,
    expected: ActionType,
) -> Result<xr::Action<T>, xr::sys::Result> {
    let Some(def) = manifest.get(id) else {
        logger::error!(
            "OpenXR action manifest is missing required action id '{id}'; check assets/xr/actions.toml"
        );
        return Err(xr::sys::Result::ERROR_VALIDATION_FAILURE);
    };
    if def.ty != expected {
        logger::error!(
            "OpenXR action '{id}' declared as {:?} in manifest but Rust expects {expected:?}",
            def.ty
        );
        return Err(xr::sys::Result::ERROR_VALIDATION_FAILURE);
    }
    action_set.create_action::<T>(id, &def.localized_name, &[])
}

/// Convenience: boolean digital action.
fn create_bool(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
    id: &str,
) -> Result<xr::Action<bool>, xr::sys::Result> {
    create_typed_action(action_set, manifest, id, ActionType::Bool)
}

/// Convenience: analog float action.
fn create_float(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
    id: &str,
) -> Result<xr::Action<f32>, xr::sys::Result> {
    create_typed_action(action_set, manifest, id, ActionType::Float)
}

/// Convenience: 2D vector action.
fn create_vec2(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
    id: &str,
) -> Result<xr::Action<xr::Vector2f>, xr::sys::Result> {
    create_typed_action(action_set, manifest, id, ActionType::Vector2f)
}

/// Convenience: tracked pose action.
fn create_pose(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
    id: &str,
) -> Result<xr::Action<xr::Posef>, xr::sys::Result> {
    create_typed_action(action_set, manifest, id, ActionType::Pose)
}

/// Convenience: haptic output action.
fn create_haptic(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
    id: &str,
) -> Result<xr::Action<xr::Haptic>, xr::sys::Result> {
    create_typed_action(action_set, manifest, id, ActionType::Haptic)
}

/// Materialises every action listed in [`ActionManifest`] into a typed [`OpenxrInputActions`].
fn build_actions(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
) -> Result<OpenxrInputActions, xr::sys::Result> {
    Ok(OpenxrInputActions {
        left_grip_pose: create_pose(action_set, manifest, "left_grip_pose")?,
        right_grip_pose: create_pose(action_set, manifest, "right_grip_pose")?,
        left_aim_pose: create_pose(action_set, manifest, "left_aim_pose")?,
        right_aim_pose: create_pose(action_set, manifest, "right_aim_pose")?,

        left_trigger: create_float(action_set, manifest, "left_trigger")?,
        right_trigger: create_float(action_set, manifest, "right_trigger")?,
        left_trigger_touch: create_bool(action_set, manifest, "left_trigger_touch")?,
        right_trigger_touch: create_bool(action_set, manifest, "right_trigger_touch")?,
        left_trigger_click: create_bool(action_set, manifest, "left_trigger_click")?,
        right_trigger_click: create_bool(action_set, manifest, "right_trigger_click")?,

        left_squeeze: create_float(action_set, manifest, "left_squeeze")?,
        right_squeeze: create_float(action_set, manifest, "right_squeeze")?,
        left_squeeze_click: create_bool(action_set, manifest, "left_squeeze_click")?,
        right_squeeze_click: create_bool(action_set, manifest, "right_squeeze_click")?,

        left_thumbstick: create_vec2(action_set, manifest, "left_thumbstick")?,
        right_thumbstick: create_vec2(action_set, manifest, "right_thumbstick")?,
        left_thumbstick_touch: create_bool(action_set, manifest, "left_thumbstick_touch")?,
        right_thumbstick_touch: create_bool(action_set, manifest, "right_thumbstick_touch")?,
        left_thumbstick_click: create_bool(action_set, manifest, "left_thumbstick_click")?,
        right_thumbstick_click: create_bool(action_set, manifest, "right_thumbstick_click")?,

        left_trackpad: create_vec2(action_set, manifest, "left_trackpad")?,
        right_trackpad: create_vec2(action_set, manifest, "right_trackpad")?,
        left_trackpad_touch: create_bool(action_set, manifest, "left_trackpad_touch")?,
        right_trackpad_touch: create_bool(action_set, manifest, "right_trackpad_touch")?,
        left_trackpad_click: create_bool(action_set, manifest, "left_trackpad_click")?,
        right_trackpad_click: create_bool(action_set, manifest, "right_trackpad_click")?,
        left_trackpad_force: create_float(action_set, manifest, "left_trackpad_force")?,
        right_trackpad_force: create_float(action_set, manifest, "right_trackpad_force")?,

        left_primary: create_bool(action_set, manifest, "left_primary")?,
        right_primary: create_bool(action_set, manifest, "right_primary")?,
        left_secondary: create_bool(action_set, manifest, "left_secondary")?,
        right_secondary: create_bool(action_set, manifest, "right_secondary")?,
        left_primary_touch: create_bool(action_set, manifest, "left_primary_touch")?,
        right_primary_touch: create_bool(action_set, manifest, "right_primary_touch")?,
        left_secondary_touch: create_bool(action_set, manifest, "left_secondary_touch")?,
        right_secondary_touch: create_bool(action_set, manifest, "right_secondary_touch")?,

        left_menu: create_bool(action_set, manifest, "left_menu")?,
        right_menu: create_bool(action_set, manifest, "right_menu")?,

        left_thumbrest_touch: create_bool(action_set, manifest, "left_thumbrest_touch")?,
        right_thumbrest_touch: create_bool(action_set, manifest, "right_thumbrest_touch")?,

        left_select: create_bool(action_set, manifest, "left_select")?,
        right_select: create_bool(action_set, manifest, "right_select")?,

        left_haptic: create_haptic(action_set, manifest, "left_haptic")?,
        right_haptic: create_haptic(action_set, manifest, "right_haptic")?,
    })
}

/// Creates grip and aim pose spaces for both hands.
fn create_grip_and_aim_spaces(
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
            .left_aim_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        actions
            .right_aim_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
    ))
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
