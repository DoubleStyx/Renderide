//! Typed OpenXR action handles and manifest-driven action creation.

use openxr as xr;

use super::super::manifest::{ActionManifest, ActionType};

/// Typed [`xr::Action`] handles for every action created by the renderer.
///
/// Field names mirror the action ids in `actions.toml` that are bound at runtime. Adding a bound
/// action requires adding a matching field here so the binding loop can look it up by id. This
/// struct is flat on purpose to keep per-frame state read sites terse.
pub(in crate::xr::input) struct OpenxrInputActions {
    /// Left-hand tracked grip pose. See `/user/hand/left/input/grip/pose`.
    pub(in crate::xr::input) left_grip_pose: xr::Action<xr::Posef>,
    /// Right-hand tracked grip pose.
    pub(in crate::xr::input) right_grip_pose: xr::Action<xr::Posef>,
    /// Left-hand palm pose from `XR_EXT_palm_pose`.
    pub(in crate::xr::input) left_palm_ext_pose: xr::Action<xr::Posef>,
    /// Right-hand palm pose from `XR_EXT_palm_pose`.
    pub(in crate::xr::input) right_palm_ext_pose: xr::Action<xr::Posef>,

    /// Left trigger analog value, 0..=1.
    pub(in crate::xr::input) left_trigger: xr::Action<f32>,
    /// Right trigger analog value, 0..=1.
    pub(in crate::xr::input) right_trigger: xr::Action<f32>,
    /// Left trigger finger-touch digital state.
    pub(in crate::xr::input) left_trigger_touch: xr::Action<bool>,
    /// Right trigger finger-touch digital state.
    pub(in crate::xr::input) right_trigger_touch: xr::Action<bool>,
    /// Left trigger fully-pressed click digital state.
    pub(in crate::xr::input) left_trigger_click: xr::Action<bool>,
    /// Right trigger fully-pressed click digital state.
    pub(in crate::xr::input) right_trigger_click: xr::Action<bool>,

    /// Left grip/squeeze analog value, 0..=1.
    pub(in crate::xr::input) left_squeeze: xr::Action<f32>,
    /// Right grip/squeeze analog value, 0..=1.
    pub(in crate::xr::input) right_squeeze: xr::Action<f32>,
    /// Left grip/squeeze fully-pressed click digital state.
    pub(in crate::xr::input) left_squeeze_click: xr::Action<bool>,
    /// Right grip/squeeze fully-pressed click digital state.
    pub(in crate::xr::input) right_squeeze_click: xr::Action<bool>,

    /// Left thumbstick 2D deflection, each axis -1..=1.
    pub(in crate::xr::input) left_thumbstick: xr::Action<xr::Vector2f>,
    /// Right thumbstick 2D deflection.
    pub(in crate::xr::input) right_thumbstick: xr::Action<xr::Vector2f>,
    /// Left thumbstick touch digital state.
    pub(in crate::xr::input) left_thumbstick_touch: xr::Action<bool>,
    /// Right thumbstick touch digital state.
    pub(in crate::xr::input) right_thumbstick_touch: xr::Action<bool>,
    /// Left thumbstick click digital state.
    pub(in crate::xr::input) left_thumbstick_click: xr::Action<bool>,
    /// Right thumbstick click digital state.
    pub(in crate::xr::input) right_thumbstick_click: xr::Action<bool>,

    /// Left trackpad 2D position.
    pub(in crate::xr::input) left_trackpad: xr::Action<xr::Vector2f>,
    /// Right trackpad 2D position.
    pub(in crate::xr::input) right_trackpad: xr::Action<xr::Vector2f>,
    /// Left trackpad touch digital state.
    pub(in crate::xr::input) left_trackpad_touch: xr::Action<bool>,
    /// Right trackpad touch digital state.
    pub(in crate::xr::input) right_trackpad_touch: xr::Action<bool>,
    /// Left trackpad click digital state.
    pub(in crate::xr::input) left_trackpad_click: xr::Action<bool>,
    /// Right trackpad click digital state.
    pub(in crate::xr::input) right_trackpad_click: xr::Action<bool>,
    /// Left trackpad press-force analog (Index).
    pub(in crate::xr::input) left_trackpad_force: xr::Action<f32>,
    /// Right trackpad press-force analog.
    pub(in crate::xr::input) right_trackpad_force: xr::Action<f32>,

    /// Left primary face button (e.g. X on Touch, A on Index).
    pub(in crate::xr::input) left_primary: xr::Action<bool>,
    /// Right primary face button (A on Touch/Index).
    pub(in crate::xr::input) right_primary: xr::Action<bool>,
    /// Left secondary face button (Y on Touch, B on Index).
    pub(in crate::xr::input) left_secondary: xr::Action<bool>,
    /// Right secondary face button (B on Touch/Index).
    pub(in crate::xr::input) right_secondary: xr::Action<bool>,
    /// Left primary face button capacitive touch.
    pub(in crate::xr::input) left_primary_touch: xr::Action<bool>,
    /// Right primary face button capacitive touch.
    pub(in crate::xr::input) right_primary_touch: xr::Action<bool>,
    /// Left secondary face button capacitive touch.
    pub(in crate::xr::input) left_secondary_touch: xr::Action<bool>,
    /// Right secondary face button capacitive touch.
    pub(in crate::xr::input) right_secondary_touch: xr::Action<bool>,

    /// Left menu/application button.
    pub(in crate::xr::input) left_menu: xr::Action<bool>,
    /// Right menu/application button (not all profiles expose this).
    pub(in crate::xr::input) right_menu: xr::Action<bool>,

    /// Left thumbrest capacitive touch.
    pub(in crate::xr::input) left_thumbrest_touch: xr::Action<bool>,
    /// Right thumbrest capacitive touch.
    pub(in crate::xr::input) right_thumbrest_touch: xr::Action<bool>,

    /// Left select action (simple/generic profiles).
    pub(in crate::xr::input) left_select: xr::Action<bool>,
    /// Right select action (simple/generic profiles).
    pub(in crate::xr::input) right_select: xr::Action<bool>,

    /// Left hand haptic output.
    pub(in crate::xr::input) left_haptic: xr::Action<xr::Haptic>,
    /// Right hand haptic output.
    pub(in crate::xr::input) right_haptic: xr::Action<xr::Haptic>,
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
pub(in crate::xr::input) fn build_actions(
    action_set: &xr::ActionSet,
    manifest: &ActionManifest,
) -> Result<OpenxrInputActions, xr::sys::Result> {
    Ok(OpenxrInputActions {
        left_grip_pose: create_pose(action_set, manifest, "left_grip_pose")?,
        right_grip_pose: create_pose(action_set, manifest, "right_grip_pose")?,
        left_palm_ext_pose: create_pose(action_set, manifest, "left_palm_ext_pose")?,
        right_palm_ext_pose: create_pose(action_set, manifest, "right_palm_ext_pose")?,

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
