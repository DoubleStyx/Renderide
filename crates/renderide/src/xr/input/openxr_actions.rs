//! OpenXR action creation, interaction profile paths, and grip/aim space setup.
//!
//! Extracted from [`super::OpenxrInput::new`] to keep the main input type focused on per-frame sampling.

use std::sync::atomic::AtomicU8;

use openxr as xr;

use super::bindings::{
    apply_suggested_interaction_bindings, ActionRefs, InteractionProfilePaths,
    ProfileExtensionGates,
};
use super::openxr_action_paths::{
    resolve_binding_subpaths, resolve_user_and_profile_paths, UserAndProfilePaths,
};

/// Intermediate container for all actions and spaces produced during [`create_openxr_input_parts`].
pub(super) struct OpenxrInputParts {
    pub action_set: xr::ActionSet,
    pub left_user_path: xr::Path,
    pub right_user_path: xr::Path,
    pub oculus_touch_profile: xr::Path,
    pub valve_index_profile: xr::Path,
    pub htc_vive_profile: xr::Path,
    pub microsoft_motion_profile: xr::Path,
    pub generic_controller_profile: xr::Path,
    pub simple_controller_profile: xr::Path,
    pub pico4_controller_profile: xr::Path,
    pub pico_neo3_controller_profile: xr::Path,
    pub hp_reverb_g2_profile: xr::Path,
    pub samsung_odyssey_profile: xr::Path,
    pub htc_vive_cosmos_profile: xr::Path,
    pub htc_vive_focus3_profile: xr::Path,
    pub meta_touch_pro_profile: xr::Path,
    pub meta_touch_plus_profile: xr::Path,
    pub left_profile_cache: AtomicU8,
    pub right_profile_cache: AtomicU8,
    pub left_grip_pose: xr::Action<xr::Posef>,
    pub right_grip_pose: xr::Action<xr::Posef>,
    pub left_trigger: xr::Action<f32>,
    pub right_trigger: xr::Action<f32>,
    pub left_trigger_touch: xr::Action<bool>,
    pub right_trigger_touch: xr::Action<bool>,
    pub left_trigger_click: xr::Action<bool>,
    pub right_trigger_click: xr::Action<bool>,
    pub left_squeeze: xr::Action<f32>,
    pub right_squeeze: xr::Action<f32>,
    pub left_squeeze_click: xr::Action<bool>,
    pub right_squeeze_click: xr::Action<bool>,
    pub left_thumbstick: xr::Action<xr::Vector2f>,
    pub right_thumbstick: xr::Action<xr::Vector2f>,
    pub left_thumbstick_touch: xr::Action<bool>,
    pub right_thumbstick_touch: xr::Action<bool>,
    pub left_thumbstick_click: xr::Action<bool>,
    pub right_thumbstick_click: xr::Action<bool>,
    pub left_trackpad: xr::Action<xr::Vector2f>,
    pub right_trackpad: xr::Action<xr::Vector2f>,
    pub left_trackpad_touch: xr::Action<bool>,
    pub right_trackpad_touch: xr::Action<bool>,
    pub left_trackpad_click: xr::Action<bool>,
    pub right_trackpad_click: xr::Action<bool>,
    pub left_trackpad_force: xr::Action<f32>,
    pub right_trackpad_force: xr::Action<f32>,
    pub left_primary: xr::Action<bool>,
    pub right_primary: xr::Action<bool>,
    pub left_secondary: xr::Action<bool>,
    pub right_secondary: xr::Action<bool>,
    pub left_primary_touch: xr::Action<bool>,
    pub right_primary_touch: xr::Action<bool>,
    pub left_secondary_touch: xr::Action<bool>,
    pub right_secondary_touch: xr::Action<bool>,
    pub left_menu: xr::Action<bool>,
    pub right_menu: xr::Action<bool>,
    pub left_thumbrest_touch: xr::Action<bool>,
    pub right_thumbrest_touch: xr::Action<bool>,
    pub left_select: xr::Action<bool>,
    pub right_select: xr::Action<bool>,
    pub left_space: xr::Space,
    pub right_space: xr::Space,
    pub left_aim_space: xr::Space,
    pub right_aim_space: xr::Space,
}

/// Grip and aim pose [`xr::Action`] pair for both hands (plus aim-only handles used for bindings).
struct GripAimPoseActions {
    left_grip_pose: xr::Action<xr::Posef>,
    right_grip_pose: xr::Action<xr::Posef>,
    left_aim_pose: xr::Action<xr::Posef>,
    right_aim_pose: xr::Action<xr::Posef>,
}

/// Registers left/right grip pose and left/right aim pose actions on the set.
fn register_grip_aim_pose_actions(
    action_set: &xr::ActionSet,
) -> Result<GripAimPoseActions, xr::sys::Result> {
    Ok(GripAimPoseActions {
        left_grip_pose: action_set.create_action::<xr::Posef>(
            "left_grip_pose",
            "Left grip pose",
            &[],
        )?,
        right_grip_pose: action_set.create_action::<xr::Posef>(
            "right_grip_pose",
            "Right grip pose",
            &[],
        )?,
        left_aim_pose: action_set.create_action::<xr::Posef>(
            "left_aim_pose",
            "Left aim pose",
            &[],
        )?,
        right_aim_pose: action_set.create_action::<xr::Posef>(
            "right_aim_pose",
            "Right aim pose",
            &[],
        )?,
    })
}

/// Trigger analog, touch, and click actions for both hands.
struct TriggerActions {
    left_trigger: xr::Action<f32>,
    right_trigger: xr::Action<f32>,
    left_trigger_touch: xr::Action<bool>,
    right_trigger_touch: xr::Action<bool>,
    left_trigger_click: xr::Action<bool>,
    right_trigger_click: xr::Action<bool>,
}

fn register_trigger_actions(action_set: &xr::ActionSet) -> Result<TriggerActions, xr::sys::Result> {
    Ok(TriggerActions {
        left_trigger: action_set.create_action::<f32>("left_trigger", "Left trigger", &[])?,
        right_trigger: action_set.create_action::<f32>("right_trigger", "Right trigger", &[])?,
        left_trigger_touch: action_set.create_action::<bool>(
            "left_trigger_touch",
            "Left trigger touch",
            &[],
        )?,
        right_trigger_touch: action_set.create_action::<bool>(
            "right_trigger_touch",
            "Right trigger touch",
            &[],
        )?,
        left_trigger_click: action_set.create_action::<bool>(
            "left_trigger_click",
            "Left trigger click",
            &[],
        )?,
        right_trigger_click: action_set.create_action::<bool>(
            "right_trigger_click",
            "Right trigger click",
            &[],
        )?,
    })
}

/// Squeeze analog and digital click actions for both hands.
struct SqueezeActions {
    left_squeeze: xr::Action<f32>,
    right_squeeze: xr::Action<f32>,
    left_squeeze_click: xr::Action<bool>,
    right_squeeze_click: xr::Action<bool>,
}

fn register_squeeze_actions(action_set: &xr::ActionSet) -> Result<SqueezeActions, xr::sys::Result> {
    Ok(SqueezeActions {
        left_squeeze: action_set.create_action::<f32>("left_squeeze", "Left squeeze", &[])?,
        right_squeeze: action_set.create_action::<f32>("right_squeeze", "Right squeeze", &[])?,
        left_squeeze_click: action_set.create_action::<bool>(
            "left_squeeze_click",
            "Left squeeze click",
            &[],
        )?,
        right_squeeze_click: action_set.create_action::<bool>(
            "right_squeeze_click",
            "Right squeeze click",
            &[],
        )?,
    })
}

/// Thumbstick vector2, touch, and click actions for both hands.
struct ThumbstickActions {
    left_thumbstick: xr::Action<xr::Vector2f>,
    right_thumbstick: xr::Action<xr::Vector2f>,
    left_thumbstick_touch: xr::Action<bool>,
    right_thumbstick_touch: xr::Action<bool>,
    left_thumbstick_click: xr::Action<bool>,
    right_thumbstick_click: xr::Action<bool>,
}

fn register_thumbstick_actions(
    action_set: &xr::ActionSet,
) -> Result<ThumbstickActions, xr::sys::Result> {
    Ok(ThumbstickActions {
        left_thumbstick: action_set.create_action::<xr::Vector2f>(
            "left_thumbstick",
            "Left thumbstick",
            &[],
        )?,
        right_thumbstick: action_set.create_action::<xr::Vector2f>(
            "right_thumbstick",
            "Right thumbstick",
            &[],
        )?,
        left_thumbstick_touch: action_set.create_action::<bool>(
            "left_thumbstick_touch",
            "Left thumbstick touch",
            &[],
        )?,
        right_thumbstick_touch: action_set.create_action::<bool>(
            "right_thumbstick_touch",
            "Right thumbstick touch",
            &[],
        )?,
        left_thumbstick_click: action_set.create_action::<bool>(
            "left_thumbstick_click",
            "Left thumbstick click",
            &[],
        )?,
        right_thumbstick_click: action_set.create_action::<bool>(
            "right_thumbstick_click",
            "Right thumbstick click",
            &[],
        )?,
    })
}

/// Trackpad vector2, touch, click, and force actions for both hands.
struct TrackpadActions {
    left_trackpad: xr::Action<xr::Vector2f>,
    right_trackpad: xr::Action<xr::Vector2f>,
    left_trackpad_touch: xr::Action<bool>,
    right_trackpad_touch: xr::Action<bool>,
    left_trackpad_click: xr::Action<bool>,
    right_trackpad_click: xr::Action<bool>,
    left_trackpad_force: xr::Action<f32>,
    right_trackpad_force: xr::Action<f32>,
}

fn register_trackpad_actions(
    action_set: &xr::ActionSet,
) -> Result<TrackpadActions, xr::sys::Result> {
    Ok(TrackpadActions {
        left_trackpad: action_set.create_action::<xr::Vector2f>(
            "left_trackpad",
            "Left trackpad",
            &[],
        )?,
        right_trackpad: action_set.create_action::<xr::Vector2f>(
            "right_trackpad",
            "Right trackpad",
            &[],
        )?,
        left_trackpad_touch: action_set.create_action::<bool>(
            "left_trackpad_touch",
            "Left trackpad touch",
            &[],
        )?,
        right_trackpad_touch: action_set.create_action::<bool>(
            "right_trackpad_touch",
            "Right trackpad touch",
            &[],
        )?,
        left_trackpad_click: action_set.create_action::<bool>(
            "left_trackpad_click",
            "Left trackpad click",
            &[],
        )?,
        right_trackpad_click: action_set.create_action::<bool>(
            "right_trackpad_click",
            "Right trackpad click",
            &[],
        )?,
        left_trackpad_force: action_set.create_action::<f32>(
            "left_trackpad_force",
            "Left trackpad force",
            &[],
        )?,
        right_trackpad_force: action_set.create_action::<f32>(
            "right_trackpad_force",
            "Right trackpad force",
            &[],
        )?,
    })
}

/// Face buttons (primary/secondary + touch), menu, thumbrest, and select actions for both hands.
struct FaceAndUtilityActions {
    left_primary: xr::Action<bool>,
    right_primary: xr::Action<bool>,
    left_secondary: xr::Action<bool>,
    right_secondary: xr::Action<bool>,
    left_primary_touch: xr::Action<bool>,
    right_primary_touch: xr::Action<bool>,
    left_secondary_touch: xr::Action<bool>,
    right_secondary_touch: xr::Action<bool>,
    left_menu: xr::Action<bool>,
    right_menu: xr::Action<bool>,
    left_thumbrest_touch: xr::Action<bool>,
    right_thumbrest_touch: xr::Action<bool>,
    left_select: xr::Action<bool>,
    right_select: xr::Action<bool>,
}

fn register_face_and_utility_actions(
    action_set: &xr::ActionSet,
) -> Result<FaceAndUtilityActions, xr::sys::Result> {
    Ok(FaceAndUtilityActions {
        left_primary: action_set.create_action::<bool>(
            "left_primary",
            "Left primary button",
            &[],
        )?,
        right_primary: action_set.create_action::<bool>(
            "right_primary",
            "Right primary button",
            &[],
        )?,
        left_secondary: action_set.create_action::<bool>(
            "left_secondary",
            "Left secondary button",
            &[],
        )?,
        right_secondary: action_set.create_action::<bool>(
            "right_secondary",
            "Right secondary button",
            &[],
        )?,
        left_primary_touch: action_set.create_action::<bool>(
            "left_primary_touch",
            "Left primary touch",
            &[],
        )?,
        right_primary_touch: action_set.create_action::<bool>(
            "right_primary_touch",
            "Right primary touch",
            &[],
        )?,
        left_secondary_touch: action_set.create_action::<bool>(
            "left_secondary_touch",
            "Left secondary touch",
            &[],
        )?,
        right_secondary_touch: action_set.create_action::<bool>(
            "right_secondary_touch",
            "Right secondary touch",
            &[],
        )?,
        left_menu: action_set.create_action::<bool>("left_menu", "Left menu", &[])?,
        right_menu: action_set.create_action::<bool>("right_menu", "Right menu", &[])?,
        left_thumbrest_touch: action_set.create_action::<bool>(
            "left_thumbrest_touch",
            "Left thumbrest touch",
            &[],
        )?,
        right_thumbrest_touch: action_set.create_action::<bool>(
            "right_thumbrest_touch",
            "Right thumbrest touch",
            &[],
        )?,
        left_select: action_set.create_action::<bool>("left_select", "Left select", &[])?,
        right_select: action_set.create_action::<bool>("right_select", "Right select", &[])?,
    })
}

/// Selects interaction profile [`xr::Path`] handles used when suggesting default bindings.
fn interaction_profiles_from_user_paths(paths: &UserAndProfilePaths) -> InteractionProfilePaths {
    InteractionProfilePaths {
        oculus_touch: paths.oculus_touch_profile,
        valve_index: paths.valve_index_profile,
        htc_vive: paths.htc_vive_profile,
        microsoft_motion: paths.microsoft_motion_profile,
        generic_controller: paths.generic_controller_profile,
        simple_controller: paths.simple_controller_profile,
        pico4_controller: paths.pico4_controller_profile,
        pico_neo3_controller: paths.pico_neo3_controller_profile,
        hp_reverb_g2: paths.hp_reverb_g2_profile,
        samsung_odyssey: paths.samsung_odyssey_profile,
        htc_vive_cosmos: paths.htc_vive_cosmos_profile,
        htc_vive_focus3: paths.htc_vive_focus3_profile,
        meta_touch_pro: paths.meta_touch_pro_profile,
        meta_touch_plus: paths.meta_touch_plus_profile,
    }
}

/// Borrow set for [`apply_suggested_interaction_bindings`].
fn action_refs_from_registered<'a>(
    grip_aim: &'a GripAimPoseActions,
    triggers: &'a TriggerActions,
    squeeze: &'a SqueezeActions,
    thumbstick: &'a ThumbstickActions,
    trackpad: &'a TrackpadActions,
    face: &'a FaceAndUtilityActions,
) -> ActionRefs<'a> {
    ActionRefs {
        left_grip_pose: &grip_aim.left_grip_pose,
        right_grip_pose: &grip_aim.right_grip_pose,
        left_aim_pose: &grip_aim.left_aim_pose,
        right_aim_pose: &grip_aim.right_aim_pose,
        left_trigger: &triggers.left_trigger,
        right_trigger: &triggers.right_trigger,
        left_trigger_touch: &triggers.left_trigger_touch,
        right_trigger_touch: &triggers.right_trigger_touch,
        left_trigger_click: &triggers.left_trigger_click,
        right_trigger_click: &triggers.right_trigger_click,
        left_squeeze: &squeeze.left_squeeze,
        right_squeeze: &squeeze.right_squeeze,
        left_squeeze_click: &squeeze.left_squeeze_click,
        right_squeeze_click: &squeeze.right_squeeze_click,
        left_thumbstick: &thumbstick.left_thumbstick,
        right_thumbstick: &thumbstick.right_thumbstick,
        left_thumbstick_touch: &thumbstick.left_thumbstick_touch,
        right_thumbstick_touch: &thumbstick.right_thumbstick_touch,
        left_thumbstick_click: &thumbstick.left_thumbstick_click,
        right_thumbstick_click: &thumbstick.right_thumbstick_click,
        left_trackpad: &trackpad.left_trackpad,
        right_trackpad: &trackpad.right_trackpad,
        left_trackpad_touch: &trackpad.left_trackpad_touch,
        right_trackpad_touch: &trackpad.right_trackpad_touch,
        left_trackpad_click: &trackpad.left_trackpad_click,
        right_trackpad_click: &trackpad.right_trackpad_click,
        left_trackpad_force: &trackpad.left_trackpad_force,
        right_trackpad_force: &trackpad.right_trackpad_force,
        left_primary: &face.left_primary,
        right_primary: &face.right_primary,
        left_secondary: &face.left_secondary,
        right_secondary: &face.right_secondary,
        left_primary_touch: &face.left_primary_touch,
        right_primary_touch: &face.right_primary_touch,
        left_secondary_touch: &face.left_secondary_touch,
        right_secondary_touch: &face.right_secondary_touch,
        left_menu: &face.left_menu,
        right_menu: &face.right_menu,
        left_thumbrest_touch: &face.left_thumbrest_touch,
        right_thumbrest_touch: &face.right_thumbrest_touch,
        left_select: &face.left_select,
        right_select: &face.right_select,
    }
}

fn create_grip_and_aim_spaces(
    session: &xr::Session<xr::Vulkan>,
    poses: &GripAimPoseActions,
) -> Result<(xr::Space, xr::Space, xr::Space, xr::Space), xr::sys::Result> {
    Ok((
        poses
            .left_grip_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        poses
            .right_grip_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        poses
            .left_aim_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
        poses
            .right_aim_pose
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?,
    ))
}

/// Creates the action set, suggests bindings for known interaction profiles, and builds grip/aim spaces.
///
/// `gates` describes which OpenXR extensions were enabled on the instance; profiles whose
/// extension is disabled are skipped to avoid suggesting bindings against paths the runtime does
/// not recognise.
pub(super) fn create_openxr_input_parts(
    instance: &xr::Instance,
    session: &xr::Session<xr::Vulkan>,
    gates: &ProfileExtensionGates,
) -> Result<OpenxrInputParts, xr::sys::Result> {
    let action_set = instance.create_action_set("renderide_input", "Renderide VR input", 0)?;
    let paths_bundle = resolve_user_and_profile_paths(instance)?;
    let binding_paths = resolve_binding_subpaths(instance)?;

    let grip_aim = register_grip_aim_pose_actions(&action_set)?;
    let triggers = register_trigger_actions(&action_set)?;
    let squeeze = register_squeeze_actions(&action_set)?;
    let thumbstick = register_thumbstick_actions(&action_set)?;
    let trackpad = register_trackpad_actions(&action_set)?;
    let face = register_face_and_utility_actions(&action_set)?;

    let interaction_profiles = interaction_profiles_from_user_paths(&paths_bundle);
    let action_refs = action_refs_from_registered(
        &grip_aim,
        &triggers,
        &squeeze,
        &thumbstick,
        &trackpad,
        &face,
    );

    apply_suggested_interaction_bindings(
        instance,
        &interaction_profiles,
        &binding_paths,
        &action_refs,
        gates,
    )?;

    session.attach_action_sets(&[&action_set])?;

    let (left_space, right_space, left_aim_space, right_aim_space) =
        create_grip_and_aim_spaces(session, &grip_aim)?;

    Ok(OpenxrInputParts {
        action_set,
        left_user_path: paths_bundle.left_user_path,
        right_user_path: paths_bundle.right_user_path,
        oculus_touch_profile: paths_bundle.oculus_touch_profile,
        valve_index_profile: paths_bundle.valve_index_profile,
        htc_vive_profile: paths_bundle.htc_vive_profile,
        microsoft_motion_profile: paths_bundle.microsoft_motion_profile,
        generic_controller_profile: paths_bundle.generic_controller_profile,
        simple_controller_profile: paths_bundle.simple_controller_profile,
        pico4_controller_profile: paths_bundle.pico4_controller_profile,
        pico_neo3_controller_profile: paths_bundle.pico_neo3_controller_profile,
        hp_reverb_g2_profile: paths_bundle.hp_reverb_g2_profile,
        samsung_odyssey_profile: paths_bundle.samsung_odyssey_profile,
        htc_vive_cosmos_profile: paths_bundle.htc_vive_cosmos_profile,
        htc_vive_focus3_profile: paths_bundle.htc_vive_focus3_profile,
        meta_touch_pro_profile: paths_bundle.meta_touch_pro_profile,
        meta_touch_plus_profile: paths_bundle.meta_touch_plus_profile,
        left_profile_cache: AtomicU8::new(0),
        right_profile_cache: AtomicU8::new(0),
        left_grip_pose: grip_aim.left_grip_pose,
        right_grip_pose: grip_aim.right_grip_pose,
        left_trigger: triggers.left_trigger,
        right_trigger: triggers.right_trigger,
        left_trigger_touch: triggers.left_trigger_touch,
        right_trigger_touch: triggers.right_trigger_touch,
        left_trigger_click: triggers.left_trigger_click,
        right_trigger_click: triggers.right_trigger_click,
        left_squeeze: squeeze.left_squeeze,
        right_squeeze: squeeze.right_squeeze,
        left_squeeze_click: squeeze.left_squeeze_click,
        right_squeeze_click: squeeze.right_squeeze_click,
        left_thumbstick: thumbstick.left_thumbstick,
        right_thumbstick: thumbstick.right_thumbstick,
        left_thumbstick_touch: thumbstick.left_thumbstick_touch,
        right_thumbstick_touch: thumbstick.right_thumbstick_touch,
        left_thumbstick_click: thumbstick.left_thumbstick_click,
        right_thumbstick_click: thumbstick.right_thumbstick_click,
        left_trackpad: trackpad.left_trackpad,
        right_trackpad: trackpad.right_trackpad,
        left_trackpad_touch: trackpad.left_trackpad_touch,
        right_trackpad_touch: trackpad.right_trackpad_touch,
        left_trackpad_click: trackpad.left_trackpad_click,
        right_trackpad_click: trackpad.right_trackpad_click,
        left_trackpad_force: trackpad.left_trackpad_force,
        right_trackpad_force: trackpad.right_trackpad_force,
        left_primary: face.left_primary,
        right_primary: face.right_primary,
        left_secondary: face.left_secondary,
        right_secondary: face.right_secondary,
        left_primary_touch: face.left_primary_touch,
        right_primary_touch: face.right_primary_touch,
        left_secondary_touch: face.left_secondary_touch,
        right_secondary_touch: face.right_secondary_touch,
        left_menu: face.left_menu,
        right_menu: face.right_menu,
        left_thumbrest_touch: face.left_thumbrest_touch,
        right_thumbrest_touch: face.right_thumbrest_touch,
        left_select: face.left_select,
        right_select: face.right_select,
        left_space,
        right_space,
        left_aim_space,
        right_aim_space,
    })
}
