//! OpenXR action set, interaction profile bindings, and per-frame VR controller sampling.

use std::sync::atomic::{AtomicU8, Ordering};

use glam::{Quat, Vec2, Vec3};
use openxr as xr;

use crate::shared::{Chirality, VRControllerState};

use super::bindings::{
    apply_suggested_interaction_bindings, ActionRefs, BindingPaths, InteractionProfilePaths,
};
use super::frame::{resolve_controller_frame, ControllerFrame};
use super::pose::pose_from_location;
use super::profile::{
    decode_profile_code, is_concrete_profile, log_profile_transition, profile_code,
    ActiveControllerProfile,
};
use super::state::build_controller_state;

/// OpenXR actions and derived spaces for headset/controller input used by the renderer IPC path.
pub struct OpenxrInput {
    action_set: xr::ActionSet,
    left_user_path: xr::Path,
    right_user_path: xr::Path,
    oculus_touch_profile: xr::Path,
    valve_index_profile: xr::Path,
    htc_vive_profile: xr::Path,
    microsoft_motion_profile: xr::Path,
    generic_controller_profile: xr::Path,
    simple_controller_profile: xr::Path,
    left_profile_cache: AtomicU8,
    right_profile_cache: AtomicU8,
    /// Kept alive for the OpenXR session; per-frame poses use the derived [`xr::Space`] handles.
    #[allow(dead_code)]
    left_grip_pose: xr::Action<xr::Posef>,
    /// Kept alive for the OpenXR session; per-frame poses use the derived [`xr::Space`] handles.
    #[allow(dead_code)]
    right_grip_pose: xr::Action<xr::Posef>,
    left_trigger: xr::Action<f32>,
    right_trigger: xr::Action<f32>,
    left_trigger_touch: xr::Action<bool>,
    right_trigger_touch: xr::Action<bool>,
    left_trigger_click: xr::Action<bool>,
    right_trigger_click: xr::Action<bool>,
    left_squeeze: xr::Action<f32>,
    right_squeeze: xr::Action<f32>,
    left_squeeze_click: xr::Action<bool>,
    right_squeeze_click: xr::Action<bool>,
    left_thumbstick: xr::Action<xr::Vector2f>,
    right_thumbstick: xr::Action<xr::Vector2f>,
    left_thumbstick_touch: xr::Action<bool>,
    right_thumbstick_touch: xr::Action<bool>,
    left_thumbstick_click: xr::Action<bool>,
    right_thumbstick_click: xr::Action<bool>,
    left_trackpad: xr::Action<xr::Vector2f>,
    right_trackpad: xr::Action<xr::Vector2f>,
    left_trackpad_touch: xr::Action<bool>,
    right_trackpad_touch: xr::Action<bool>,
    left_trackpad_click: xr::Action<bool>,
    right_trackpad_click: xr::Action<bool>,
    left_trackpad_force: xr::Action<f32>,
    right_trackpad_force: xr::Action<f32>,
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
    left_space: xr::Space,
    right_space: xr::Space,
    left_aim_space: xr::Space,
    right_aim_space: xr::Space,
}

impl OpenxrInput {
    /// Creates the action set, suggests bindings for known interaction profiles, and builds grip/aim spaces.
    ///
    /// `runtime_supports_generic_controller` must match whether the OpenXR instance was created with
    /// `XR_KHR_generic_controller` enabled; when `false`, generic controller binding suggestions are skipped.
    ///
    /// `runtime_supports_bd_controller` must match whether `XR_BD_controller_interaction` was enabled
    /// on the instance; when `false`, ByteDance Pico profile binding suggestions are skipped.
    pub fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
        runtime_supports_generic_controller: bool,
        runtime_supports_bd_controller: bool,
    ) -> Result<Self, xr::sys::Result> {
        let action_set = instance.create_action_set("renderide_input", "Renderide VR input", 0)?;
        let left_user_path = instance.string_to_path("/user/hand/left")?;
        let right_user_path = instance.string_to_path("/user/hand/right")?;
        let oculus_touch_profile =
            instance.string_to_path("/interaction_profiles/oculus/touch_controller")?;
        let valve_index_profile =
            instance.string_to_path("/interaction_profiles/valve/index_controller")?;
        let htc_vive_profile =
            instance.string_to_path("/interaction_profiles/htc/vive_controller")?;
        let microsoft_motion_profile =
            instance.string_to_path("/interaction_profiles/microsoft/motion_controller")?;
        let generic_controller_profile =
            instance.string_to_path("/interaction_profiles/khr/generic_controller")?;
        let simple_controller_profile =
            instance.string_to_path("/interaction_profiles/khr/simple_controller")?;
        let pico4_controller_profile =
            instance.string_to_path("/interaction_profiles/bytedance/pico4_controller")?;
        let left_grip_pose =
            action_set.create_action::<xr::Posef>("left_grip_pose", "Left grip pose", &[])?;
        let right_grip_pose =
            action_set.create_action::<xr::Posef>("right_grip_pose", "Right grip pose", &[])?;
        let left_trigger = action_set.create_action::<f32>("left_trigger", "Left trigger", &[])?;
        let right_trigger =
            action_set.create_action::<f32>("right_trigger", "Right trigger", &[])?;
        let left_trigger_touch =
            action_set.create_action::<bool>("left_trigger_touch", "Left trigger touch", &[])?;
        let right_trigger_touch =
            action_set.create_action::<bool>("right_trigger_touch", "Right trigger touch", &[])?;
        let left_trigger_click =
            action_set.create_action::<bool>("left_trigger_click", "Left trigger click", &[])?;
        let right_trigger_click =
            action_set.create_action::<bool>("right_trigger_click", "Right trigger click", &[])?;
        let left_squeeze = action_set.create_action::<f32>("left_squeeze", "Left squeeze", &[])?;
        let right_squeeze =
            action_set.create_action::<f32>("right_squeeze", "Right squeeze", &[])?;
        let left_squeeze_click =
            action_set.create_action::<bool>("left_squeeze_click", "Left squeeze click", &[])?;
        let right_squeeze_click =
            action_set.create_action::<bool>("right_squeeze_click", "Right squeeze click", &[])?;
        let left_thumbstick =
            action_set.create_action::<xr::Vector2f>("left_thumbstick", "Left thumbstick", &[])?;
        let right_thumbstick = action_set.create_action::<xr::Vector2f>(
            "right_thumbstick",
            "Right thumbstick",
            &[],
        )?;
        let left_thumbstick_touch = action_set.create_action::<bool>(
            "left_thumbstick_touch",
            "Left thumbstick touch",
            &[],
        )?;
        let right_thumbstick_touch = action_set.create_action::<bool>(
            "right_thumbstick_touch",
            "Right thumbstick touch",
            &[],
        )?;
        let left_thumbstick_click = action_set.create_action::<bool>(
            "left_thumbstick_click",
            "Left thumbstick click",
            &[],
        )?;
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
        let right_trackpad_touch = action_set.create_action::<bool>(
            "right_trackpad_touch",
            "Right trackpad touch",
            &[],
        )?;
        let left_trackpad_click =
            action_set.create_action::<bool>("left_trackpad_click", "Left trackpad click", &[])?;
        let right_trackpad_click = action_set.create_action::<bool>(
            "right_trackpad_click",
            "Right trackpad click",
            &[],
        )?;
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
        let left_secondary_touch = action_set.create_action::<bool>(
            "left_secondary_touch",
            "Left secondary touch",
            &[],
        )?;
        let right_secondary_touch = action_set.create_action::<bool>(
            "right_secondary_touch",
            "Right secondary touch",
            &[],
        )?;
        let left_menu = action_set.create_action::<bool>("left_menu", "Left menu", &[])?;
        let right_menu = action_set.create_action::<bool>("right_menu", "Right menu", &[])?;
        let left_thumbrest_touch = action_set.create_action::<bool>(
            "left_thumbrest_touch",
            "Left thumbrest touch",
            &[],
        )?;
        let right_thumbrest_touch = action_set.create_action::<bool>(
            "right_thumbrest_touch",
            "Right thumbrest touch",
            &[],
        )?;
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
        let left_trigger_value_path =
            instance.string_to_path("/user/hand/left/input/trigger/value")?;
        let right_trigger_value_path =
            instance.string_to_path("/user/hand/right/input/trigger/value")?;
        let left_trigger_touch_path =
            instance.string_to_path("/user/hand/left/input/trigger/touch")?;
        let right_trigger_touch_path =
            instance.string_to_path("/user/hand/right/input/trigger/touch")?;
        let left_trigger_click_path =
            instance.string_to_path("/user/hand/left/input/trigger/click")?;
        let right_trigger_click_path =
            instance.string_to_path("/user/hand/right/input/trigger/click")?;
        let left_squeeze_value_path =
            instance.string_to_path("/user/hand/left/input/squeeze/value")?;
        let right_squeeze_value_path =
            instance.string_to_path("/user/hand/right/input/squeeze/value")?;
        let left_squeeze_click_path =
            instance.string_to_path("/user/hand/left/input/squeeze/click")?;
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
        let left_select_click_path =
            instance.string_to_path("/user/hand/left/input/select/click")?;
        let right_select_click_path =
            instance.string_to_path("/user/hand/right/input/select/click")?;

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
        let left_space =
            left_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
        let right_space =
            right_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
        let left_aim_space =
            left_aim_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
        let right_aim_space =
            right_aim_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
        Ok(Self {
            action_set,
            left_user_path,
            right_user_path,
            oculus_touch_profile,
            valve_index_profile,
            htc_vive_profile,
            microsoft_motion_profile,
            generic_controller_profile,
            simple_controller_profile,
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

    fn detect_profile(
        &self,
        session: &xr::Session<xr::Vulkan>,
        hand_user_path: xr::Path,
    ) -> ActiveControllerProfile {
        let Ok(profile) = session.current_interaction_profile(hand_user_path) else {
            return ActiveControllerProfile::Generic;
        };
        if profile == self.oculus_touch_profile {
            ActiveControllerProfile::Touch
        } else if profile == self.valve_index_profile {
            ActiveControllerProfile::Index
        } else if profile == self.htc_vive_profile {
            ActiveControllerProfile::Vive
        } else if profile == self.microsoft_motion_profile {
            ActiveControllerProfile::WindowsMr
        } else if profile == self.generic_controller_profile {
            ActiveControllerProfile::Generic
        } else if profile == self.simple_controller_profile || profile == xr::Path::NULL {
            ActiveControllerProfile::Simple
        } else {
            ActiveControllerProfile::Generic
        }
    }

    fn active_profile(
        &self,
        session: &xr::Session<xr::Vulkan>,
        hand_user_path: xr::Path,
        side: Chirality,
    ) -> ActiveControllerProfile {
        let live = self.detect_profile(session, hand_user_path);
        let cache = match side {
            Chirality::left => &self.left_profile_cache,
            Chirality::right => &self.right_profile_cache,
        };
        if is_concrete_profile(live) {
            cache.store(profile_code(live), Ordering::Relaxed);
            return live;
        }
        decode_profile_code(cache.load(Ordering::Relaxed))
            .filter(|cached| is_concrete_profile(*cached))
            .unwrap_or(live)
    }

    /// Syncs actions, samples poses and digital/analog state, and returns left/right [`VRControllerState`] values.
    pub fn sync_and_sample(
        &self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        predicted_time: xr::Time,
    ) -> Result<Vec<VRControllerState>, xr::sys::Result> {
        session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)])?;
        let left_loc = self.left_space.locate(stage, predicted_time)?;
        let right_loc = self.right_space.locate(stage, predicted_time)?;
        let left_aim_loc = self.left_aim_space.locate(stage, predicted_time)?;
        let right_aim_loc = self.right_aim_space.locate(stage, predicted_time)?;
        let left_grip_pose = pose_from_location(&left_loc);
        let right_grip_pose = pose_from_location(&right_loc);
        let left_aim_pose = pose_from_location(&left_aim_loc);
        let right_aim_pose = pose_from_location(&right_aim_loc);
        let left_trigger = self.left_trigger.state(session, xr::Path::NULL)?;
        let right_trigger = self.right_trigger.state(session, xr::Path::NULL)?;
        let left_trigger_touch = self.left_trigger_touch.state(session, xr::Path::NULL)?;
        let right_trigger_touch = self.right_trigger_touch.state(session, xr::Path::NULL)?;
        let left_trigger_click = self.left_trigger_click.state(session, xr::Path::NULL)?;
        let right_trigger_click = self.right_trigger_click.state(session, xr::Path::NULL)?;
        let left_squeeze = self.left_squeeze.state(session, xr::Path::NULL)?;
        let right_squeeze = self.right_squeeze.state(session, xr::Path::NULL)?;
        let left_squeeze_click = self.left_squeeze_click.state(session, xr::Path::NULL)?;
        let right_squeeze_click = self.right_squeeze_click.state(session, xr::Path::NULL)?;
        let left_thumbstick = self.left_thumbstick.state(session, xr::Path::NULL)?;
        let right_thumbstick = self.right_thumbstick.state(session, xr::Path::NULL)?;
        let left_thumbstick_touch = self.left_thumbstick_touch.state(session, xr::Path::NULL)?;
        let right_thumbstick_touch = self.right_thumbstick_touch.state(session, xr::Path::NULL)?;
        let left_thumbstick_click = self.left_thumbstick_click.state(session, xr::Path::NULL)?;
        let right_thumbstick_click = self.right_thumbstick_click.state(session, xr::Path::NULL)?;
        let left_trackpad = self.left_trackpad.state(session, xr::Path::NULL)?;
        let right_trackpad = self.right_trackpad.state(session, xr::Path::NULL)?;
        let left_trackpad_touch = self.left_trackpad_touch.state(session, xr::Path::NULL)?;
        let right_trackpad_touch = self.right_trackpad_touch.state(session, xr::Path::NULL)?;
        let left_trackpad_click = self.left_trackpad_click.state(session, xr::Path::NULL)?;
        let right_trackpad_click = self.right_trackpad_click.state(session, xr::Path::NULL)?;
        let left_trackpad_force = self.left_trackpad_force.state(session, xr::Path::NULL)?;
        let right_trackpad_force = self.right_trackpad_force.state(session, xr::Path::NULL)?;
        let left_primary = self.left_primary.state(session, xr::Path::NULL)?;
        let right_primary = self.right_primary.state(session, xr::Path::NULL)?;
        let left_secondary = self.left_secondary.state(session, xr::Path::NULL)?;
        let right_secondary = self.right_secondary.state(session, xr::Path::NULL)?;
        let left_primary_touch = self.left_primary_touch.state(session, xr::Path::NULL)?;
        let right_primary_touch = self.right_primary_touch.state(session, xr::Path::NULL)?;
        let left_secondary_touch = self.left_secondary_touch.state(session, xr::Path::NULL)?;
        let right_secondary_touch = self.right_secondary_touch.state(session, xr::Path::NULL)?;
        let left_menu = self.left_menu.state(session, xr::Path::NULL)?;
        let right_menu = self.right_menu.state(session, xr::Path::NULL)?;
        let left_thumbrest_touch = self.left_thumbrest_touch.state(session, xr::Path::NULL)?;
        let right_thumbrest_touch = self.right_thumbrest_touch.state(session, xr::Path::NULL)?;
        let left_select = self.left_select.state(session, xr::Path::NULL)?;
        let right_select = self.right_select.state(session, xr::Path::NULL)?;
        let left_thumbstick_vec = Vec2::new(
            left_thumbstick.current_state.x,
            left_thumbstick.current_state.y,
        );
        let right_thumbstick_vec = Vec2::new(
            right_thumbstick.current_state.x,
            right_thumbstick.current_state.y,
        );
        let left_trackpad_vec =
            Vec2::new(left_trackpad.current_state.x, left_trackpad.current_state.y);
        let right_trackpad_vec = Vec2::new(
            right_trackpad.current_state.x,
            right_trackpad.current_state.y,
        );
        let left_profile = self.active_profile(session, self.left_user_path, Chirality::left);
        let right_profile = self.active_profile(session, self.right_user_path, Chirality::right);
        log_profile_transition(Chirality::left, left_profile);
        log_profile_transition(Chirality::right, right_profile);
        let left_frame =
            resolve_controller_frame(left_profile, Chirality::left, left_grip_pose, left_aim_pose);
        let right_frame = resolve_controller_frame(
            right_profile,
            Chirality::right,
            right_grip_pose,
            right_aim_pose,
        );
        let left = build_controller_state(
            left_profile,
            Chirality::left,
            left_frame.is_some(),
            left_frame.unwrap_or(ControllerFrame {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                has_bound_hand: false,
                hand_position: Vec3::ZERO,
                hand_rotation: Quat::IDENTITY,
            }),
            left_trigger.current_state,
            left_trigger_touch.current_state,
            left_trigger_click.current_state,
            left_squeeze.current_state,
            left_squeeze_click.current_state,
            left_thumbstick_vec,
            left_thumbstick_touch.current_state,
            left_thumbstick_click.current_state,
            left_trackpad_vec,
            left_trackpad_touch.current_state,
            left_trackpad_click.current_state,
            left_trackpad_force.current_state,
            left_primary.current_state,
            left_secondary.current_state,
            left_primary_touch.current_state,
            left_secondary_touch.current_state,
            left_menu.current_state,
            left_thumbrest_touch.current_state,
            left_select.current_state,
        );
        let right = build_controller_state(
            right_profile,
            Chirality::right,
            right_frame.is_some(),
            right_frame.unwrap_or(ControllerFrame {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                has_bound_hand: false,
                hand_position: Vec3::ZERO,
                hand_rotation: Quat::IDENTITY,
            }),
            right_trigger.current_state,
            right_trigger_touch.current_state,
            right_trigger_click.current_state,
            right_squeeze.current_state,
            right_squeeze_click.current_state,
            right_thumbstick_vec,
            right_thumbstick_touch.current_state,
            right_thumbstick_click.current_state,
            right_trackpad_vec,
            right_trackpad_touch.current_state,
            right_trackpad_click.current_state,
            right_trackpad_force.current_state,
            right_primary.current_state,
            right_secondary.current_state,
            right_primary_touch.current_state,
            right_secondary_touch.current_state,
            right_menu.current_state,
            right_thumbrest_touch.current_state,
            right_select.current_state,
        );
        Ok(vec![left, right])
    }

    /// Logs once (at trace level) if stereo view array order may not match left-then-right pose ordering.
    pub fn log_stereo_view_order_once(views: &[xr::View]) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static ONCE: AtomicBool = AtomicBool::new(false);
        if views.len() < 2 || ONCE.swap(true, Ordering::Relaxed) {
            return;
        }
        let x0 = views[0].pose.position.x;
        let x1 = views[1].pose.position.x;
        if x0 > x1 + 0.02 {
            logger::trace!(
                "OpenXR stereo: views[0].pose.x ({x0}) > views[1].pose.x ({x1}); runtime may use right-then-left ordering - verify eye mapping."
            );
        }
    }
}
