//! OpenXR action creation, interaction profile paths, and grip/aim space setup.
//!
//! Extracted from [`super::OpenxrInput::new`] to keep the main input type focused on per-frame sampling.

use std::sync::atomic::AtomicU8;

use openxr as xr;

use super::bindings::{
    apply_suggested_interaction_bindings, ActionRefs, BindingPaths, InteractionProfilePaths,
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

/// Creates the action set, suggests bindings for known interaction profiles, and builds grip/aim spaces.
///
/// `runtime_supports_generic_controller` must match whether the OpenXR instance was created with
/// `XR_KHR_generic_controller` enabled; when `false`, generic controller binding suggestions are skipped.
///
/// `runtime_supports_bd_controller` must match whether `XR_BD_controller_interaction` was enabled
/// on the instance; when `false`, ByteDance Pico profile binding suggestions are skipped.
pub(super) fn create_openxr_input_parts(
    instance: &xr::Instance,
    session: &xr::Session<xr::Vulkan>,
    runtime_supports_generic_controller: bool,
    runtime_supports_bd_controller: bool,
) -> Result<OpenxrInputParts, xr::sys::Result> {
    let action_set = instance.create_action_set("renderide_input", "Renderide VR input", 0)?;
    let left_user_path = instance.string_to_path("/user/hand/left")?;
    let right_user_path = instance.string_to_path("/user/hand/right")?;
    let oculus_touch_profile =
        instance.string_to_path("/interaction_profiles/oculus/touch_controller")?;
    let valve_index_profile =
        instance.string_to_path("/interaction_profiles/valve/index_controller")?;
    let htc_vive_profile = instance.string_to_path("/interaction_profiles/htc/vive_controller")?;
    let microsoft_motion_profile =
        instance.string_to_path("/interaction_profiles/microsoft/motion_controller")?;
    let generic_controller_profile =
        instance.string_to_path("/interaction_profiles/khr/generic_controller")?;
    let simple_controller_profile =
        instance.string_to_path("/interaction_profiles/khr/simple_controller")?;
    let pico4_controller_profile =
        instance.string_to_path("/interaction_profiles/bytedance/pico4_controller")?;
    let pico_neo3_controller_profile =
        instance.string_to_path("/interaction_profiles/bytedance/pico_neo3_controller")?;
    let left_grip_pose =
        action_set.create_action::<xr::Posef>("left_grip_pose", "Left grip pose", &[])?;
    let right_grip_pose =
        action_set.create_action::<xr::Posef>("right_grip_pose", "Right grip pose", &[])?;
    let left_trigger = action_set.create_action::<f32>("left_trigger", "Left trigger", &[])?;
    let right_trigger = action_set.create_action::<f32>("right_trigger", "Right trigger", &[])?;
    let left_trigger_touch =
        action_set.create_action::<bool>("left_trigger_touch", "Left trigger touch", &[])?;
    let right_trigger_touch =
        action_set.create_action::<bool>("right_trigger_touch", "Right trigger touch", &[])?;
    let left_trigger_click =
        action_set.create_action::<bool>("left_trigger_click", "Left trigger click", &[])?;
    let right_trigger_click =
        action_set.create_action::<bool>("right_trigger_click", "Right trigger click", &[])?;
    let left_squeeze = action_set.create_action::<f32>("left_squeeze", "Left squeeze", &[])?;
    let right_squeeze = action_set.create_action::<f32>("right_squeeze", "Right squeeze", &[])?;
    let left_squeeze_click =
        action_set.create_action::<bool>("left_squeeze_click", "Left squeeze click", &[])?;
    let right_squeeze_click =
        action_set.create_action::<bool>("right_squeeze_click", "Right squeeze click", &[])?;
    let left_thumbstick =
        action_set.create_action::<xr::Vector2f>("left_thumbstick", "Left thumbstick", &[])?;
    let right_thumbstick =
        action_set.create_action::<xr::Vector2f>("right_thumbstick", "Right thumbstick", &[])?;
    let left_thumbstick_touch =
        action_set.create_action::<bool>("left_thumbstick_touch", "Left thumbstick touch", &[])?;
    let right_thumbstick_touch = action_set.create_action::<bool>(
        "right_thumbstick_touch",
        "Right thumbstick touch",
        &[],
    )?;
    let left_thumbstick_click =
        action_set.create_action::<bool>("left_thumbstick_click", "Left thumbstick click", &[])?;
    let right_thumbstick_click = action_set.create_action::<bool>(
        "right_thumbstick_click",
        "Right thumbstick click",
        &[],
    )?;
    let left_trackpad =
        action_set.create_action::<xr::Vector2f>("left_trackpad", "Left trackpad", &[])?;
    let right_trackpad =
        action_set.create_action::<xr::Vector2f>("right_trackpad", "Right trackpad", &[])?;
    let left_trackpad_touch =
        action_set.create_action::<bool>("left_trackpad_touch", "Left trackpad touch", &[])?;
    let right_trackpad_touch =
        action_set.create_action::<bool>("right_trackpad_touch", "Right trackpad touch", &[])?;
    let left_trackpad_click =
        action_set.create_action::<bool>("left_trackpad_click", "Left trackpad click", &[])?;
    let right_trackpad_click =
        action_set.create_action::<bool>("right_trackpad_click", "Right trackpad click", &[])?;
    let left_trackpad_force =
        action_set.create_action::<f32>("left_trackpad_force", "Left trackpad force", &[])?;
    let right_trackpad_force =
        action_set.create_action::<f32>("right_trackpad_force", "Right trackpad force", &[])?;
    let left_primary =
        action_set.create_action::<bool>("left_primary", "Left primary button", &[])?;
    let right_primary =
        action_set.create_action::<bool>("right_primary", "Right primary button", &[])?;
    let left_secondary =
        action_set.create_action::<bool>("left_secondary", "Left secondary button", &[])?;
    let right_secondary =
        action_set.create_action::<bool>("right_secondary", "Right secondary button", &[])?;
    let left_primary_touch =
        action_set.create_action::<bool>("left_primary_touch", "Left primary touch", &[])?;
    let right_primary_touch =
        action_set.create_action::<bool>("right_primary_touch", "Right primary touch", &[])?;
    let left_secondary_touch =
        action_set.create_action::<bool>("left_secondary_touch", "Left secondary touch", &[])?;
    let right_secondary_touch =
        action_set.create_action::<bool>("right_secondary_touch", "Right secondary touch", &[])?;
    let left_menu = action_set.create_action::<bool>("left_menu", "Left menu", &[])?;
    let right_menu = action_set.create_action::<bool>("right_menu", "Right menu", &[])?;
    let left_thumbrest_touch =
        action_set.create_action::<bool>("left_thumbrest_touch", "Left thumbrest touch", &[])?;
    let right_thumbrest_touch =
        action_set.create_action::<bool>("right_thumbrest_touch", "Right thumbrest touch", &[])?;
    let left_select = action_set.create_action::<bool>("left_select", "Left select", &[])?;
    let right_select = action_set.create_action::<bool>("right_select", "Right select", &[])?;
    let left_grip_pose_path = instance.string_to_path("/user/hand/left/input/grip/pose")?;
    let right_grip_pose_path = instance.string_to_path("/user/hand/right/input/grip/pose")?;
    let left_aim_pose =
        action_set.create_action::<xr::Posef>("left_aim_pose", "Left aim pose", &[])?;
    let right_aim_pose =
        action_set.create_action::<xr::Posef>("right_aim_pose", "Right aim pose", &[])?;
    let left_aim_pose_path = instance.string_to_path("/user/hand/left/input/aim/pose")?;
    let right_aim_pose_path = instance.string_to_path("/user/hand/right/input/aim/pose")?;
    let left_trigger_value_path = instance.string_to_path("/user/hand/left/input/trigger/value")?;
    let right_trigger_value_path =
        instance.string_to_path("/user/hand/right/input/trigger/value")?;
    let left_trigger_touch_path = instance.string_to_path("/user/hand/left/input/trigger/touch")?;
    let right_trigger_touch_path =
        instance.string_to_path("/user/hand/right/input/trigger/touch")?;
    let left_trigger_click_path = instance.string_to_path("/user/hand/left/input/trigger/click")?;
    let right_trigger_click_path =
        instance.string_to_path("/user/hand/right/input/trigger/click")?;
    let left_squeeze_value_path = instance.string_to_path("/user/hand/left/input/squeeze/value")?;
    let right_squeeze_value_path =
        instance.string_to_path("/user/hand/right/input/squeeze/value")?;
    let left_squeeze_click_path = instance.string_to_path("/user/hand/left/input/squeeze/click")?;
    let right_squeeze_click_path =
        instance.string_to_path("/user/hand/right/input/squeeze/click")?;
    let left_thumbstick_path = instance.string_to_path("/user/hand/left/input/thumbstick")?;
    let right_thumbstick_path = instance.string_to_path("/user/hand/right/input/thumbstick")?;
    let left_thumbstick_touch_path =
        instance.string_to_path("/user/hand/left/input/thumbstick/touch")?;
    let right_thumbstick_touch_path =
        instance.string_to_path("/user/hand/right/input/thumbstick/touch")?;
    let left_thumbstick_click_path =
        instance.string_to_path("/user/hand/left/input/thumbstick/click")?;
    let right_thumbstick_click_path =
        instance.string_to_path("/user/hand/right/input/thumbstick/click")?;
    let left_trackpad_path = instance.string_to_path("/user/hand/left/input/trackpad")?;
    let right_trackpad_path = instance.string_to_path("/user/hand/right/input/trackpad")?;
    let left_trackpad_touch_path =
        instance.string_to_path("/user/hand/left/input/trackpad/touch")?;
    let right_trackpad_touch_path =
        instance.string_to_path("/user/hand/right/input/trackpad/touch")?;
    let left_trackpad_click_path =
        instance.string_to_path("/user/hand/left/input/trackpad/click")?;
    let right_trackpad_click_path =
        instance.string_to_path("/user/hand/right/input/trackpad/click")?;
    let left_trackpad_force_path =
        instance.string_to_path("/user/hand/left/input/trackpad/force")?;
    let right_trackpad_force_path =
        instance.string_to_path("/user/hand/right/input/trackpad/force")?;
    let left_x_click_path = instance.string_to_path("/user/hand/left/input/x/click")?;
    let left_y_click_path = instance.string_to_path("/user/hand/left/input/y/click")?;
    let left_x_touch_path = instance.string_to_path("/user/hand/left/input/x/touch")?;
    let left_y_touch_path = instance.string_to_path("/user/hand/left/input/y/touch")?;
    let left_a_click_path = instance.string_to_path("/user/hand/left/input/a/click")?;
    let left_b_click_path = instance.string_to_path("/user/hand/left/input/b/click")?;
    let left_a_touch_path = instance.string_to_path("/user/hand/left/input/a/touch")?;
    let left_b_touch_path = instance.string_to_path("/user/hand/left/input/b/touch")?;
    let right_a_click_path = instance.string_to_path("/user/hand/right/input/a/click")?;
    let right_b_click_path = instance.string_to_path("/user/hand/right/input/b/click")?;
    let right_a_touch_path = instance.string_to_path("/user/hand/right/input/a/touch")?;
    let right_b_touch_path = instance.string_to_path("/user/hand/right/input/b/touch")?;
    let left_menu_click_path = instance.string_to_path("/user/hand/left/input/menu/click")?;
    let right_menu_click_path = instance.string_to_path("/user/hand/right/input/menu/click")?;
    let left_thumbrest_touch_path =
        instance.string_to_path("/user/hand/left/input/thumbrest/touch")?;
    let right_thumbrest_touch_path =
        instance.string_to_path("/user/hand/right/input/thumbrest/touch")?;
    let left_select_click_path = instance.string_to_path("/user/hand/left/input/select/click")?;
    let right_select_click_path = instance.string_to_path("/user/hand/right/input/select/click")?;

    let interaction_profiles = InteractionProfilePaths {
        oculus_touch: oculus_touch_profile,
        valve_index: valve_index_profile,
        htc_vive: htc_vive_profile,
        microsoft_motion: microsoft_motion_profile,
        generic_controller: generic_controller_profile,
        simple_controller: simple_controller_profile,
        pico4_controller: pico4_controller_profile,
    };

    let binding_paths = BindingPaths {
        left_grip_pose: left_grip_pose_path,
        right_grip_pose: right_grip_pose_path,
        left_aim_pose: left_aim_pose_path,
        right_aim_pose: right_aim_pose_path,
        left_trigger_value: left_trigger_value_path,
        right_trigger_value: right_trigger_value_path,
        left_trigger_touch: left_trigger_touch_path,
        right_trigger_touch: right_trigger_touch_path,
        left_trigger_click: left_trigger_click_path,
        right_trigger_click: right_trigger_click_path,
        left_squeeze_value: left_squeeze_value_path,
        right_squeeze_value: right_squeeze_value_path,
        left_squeeze_click: left_squeeze_click_path,
        right_squeeze_click: right_squeeze_click_path,
        left_thumbstick: left_thumbstick_path,
        right_thumbstick: right_thumbstick_path,
        left_thumbstick_touch: left_thumbstick_touch_path,
        right_thumbstick_touch: right_thumbstick_touch_path,
        left_thumbstick_click: left_thumbstick_click_path,
        right_thumbstick_click: right_thumbstick_click_path,
        left_trackpad: left_trackpad_path,
        right_trackpad: right_trackpad_path,
        left_trackpad_touch: left_trackpad_touch_path,
        right_trackpad_touch: right_trackpad_touch_path,
        left_trackpad_click: left_trackpad_click_path,
        right_trackpad_click: right_trackpad_click_path,
        left_trackpad_force: left_trackpad_force_path,
        right_trackpad_force: right_trackpad_force_path,
        left_x_click: left_x_click_path,
        left_y_click: left_y_click_path,
        left_x_touch: left_x_touch_path,
        left_y_touch: left_y_touch_path,
        left_a_click: left_a_click_path,
        left_b_click: left_b_click_path,
        left_a_touch: left_a_touch_path,
        left_b_touch: left_b_touch_path,
        right_a_click: right_a_click_path,
        right_b_click: right_b_click_path,
        right_a_touch: right_a_touch_path,
        right_b_touch: right_b_touch_path,
        left_menu_click: left_menu_click_path,
        right_menu_click: right_menu_click_path,
        left_thumbrest_touch: left_thumbrest_touch_path,
        right_thumbrest_touch: right_thumbrest_touch_path,
        left_select_click: left_select_click_path,
        right_select_click: right_select_click_path,
    };

    let action_refs = ActionRefs {
        left_grip_pose: &left_grip_pose,
        right_grip_pose: &right_grip_pose,
        left_aim_pose: &left_aim_pose,
        right_aim_pose: &right_aim_pose,
        left_trigger: &left_trigger,
        right_trigger: &right_trigger,
        left_trigger_touch: &left_trigger_touch,
        right_trigger_touch: &right_trigger_touch,
        left_trigger_click: &left_trigger_click,
        right_trigger_click: &right_trigger_click,
        left_squeeze: &left_squeeze,
        right_squeeze: &right_squeeze,
        left_squeeze_click: &left_squeeze_click,
        right_squeeze_click: &right_squeeze_click,
        left_thumbstick: &left_thumbstick,
        right_thumbstick: &right_thumbstick,
        left_thumbstick_touch: &left_thumbstick_touch,
        right_thumbstick_touch: &right_thumbstick_touch,
        left_thumbstick_click: &left_thumbstick_click,
        right_thumbstick_click: &right_thumbstick_click,
        left_trackpad: &left_trackpad,
        right_trackpad: &right_trackpad,
        left_trackpad_touch: &left_trackpad_touch,
        right_trackpad_touch: &right_trackpad_touch,
        left_trackpad_click: &left_trackpad_click,
        right_trackpad_click: &right_trackpad_click,
        left_trackpad_force: &left_trackpad_force,
        right_trackpad_force: &right_trackpad_force,
        left_primary: &left_primary,
        right_primary: &right_primary,
        left_secondary: &left_secondary,
        right_secondary: &right_secondary,
        left_primary_touch: &left_primary_touch,
        right_primary_touch: &right_primary_touch,
        left_secondary_touch: &left_secondary_touch,
        right_secondary_touch: &right_secondary_touch,
        left_menu: &left_menu,
        right_menu: &right_menu,
        left_thumbrest_touch: &left_thumbrest_touch,
        right_thumbrest_touch: &right_thumbrest_touch,
        left_select: &left_select,
        right_select: &right_select,
    };

    apply_suggested_interaction_bindings(
        instance,
        &interaction_profiles,
        &binding_paths,
        &action_refs,
        runtime_supports_generic_controller,
        runtime_supports_bd_controller,
    )?;

    session.attach_action_sets(&[&action_set])?;
    let left_space = left_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    let right_space = right_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    let left_aim_space =
        left_aim_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    let right_aim_space =
        right_aim_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
    Ok(OpenxrInputParts {
        action_set,
        left_user_path,
        right_user_path,
        oculus_touch_profile,
        valve_index_profile,
        htc_vive_profile,
        microsoft_motion_profile,
        generic_controller_profile,
        simple_controller_profile,
        pico4_controller_profile,
        pico_neo3_controller_profile,
        left_profile_cache: AtomicU8::new(0),
        right_profile_cache: AtomicU8::new(0),
        left_grip_pose,
        right_grip_pose,
        left_trigger,
        right_trigger,
        left_trigger_touch,
        right_trigger_touch,
        left_trigger_click,
        right_trigger_click,
        left_squeeze,
        right_squeeze,
        left_squeeze_click,
        right_squeeze_click,
        left_thumbstick,
        right_thumbstick,
        left_thumbstick_touch,
        right_thumbstick_touch,
        left_thumbstick_click,
        right_thumbstick_click,
        left_trackpad,
        right_trackpad,
        left_trackpad_touch,
        right_trackpad_touch,
        left_trackpad_click,
        right_trackpad_click,
        left_trackpad_force,
        right_trackpad_force,
        left_primary,
        right_primary,
        left_secondary,
        right_secondary,
        left_primary_touch,
        right_primary_touch,
        left_secondary_touch,
        right_secondary_touch,
        left_menu,
        right_menu,
        left_thumbrest_touch,
        right_thumbrest_touch,
        left_select,
        right_select,
        left_space,
        right_space,
        left_aim_space,
        right_aim_space,
    })
}
