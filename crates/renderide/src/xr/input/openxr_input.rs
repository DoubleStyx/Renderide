//! OpenXR action set, interaction profile bindings, and per-frame VR controller sampling.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

use glam::{Quat, Vec2, Vec3};
use openxr as xr;

use crate::shared::{Chirality, VRControllerState};

use super::frame::{resolve_controller_frame, ControllerFrame};
use super::openxr_actions::{create_openxr_input_parts, OpenxrInputParts};
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
    pico4_controller_profile: xr::Path,
    pico_neo3_controller_profile: xr::Path,
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

impl From<OpenxrInputParts> for OpenxrInput {
    fn from(p: OpenxrInputParts) -> Self {
        Self {
            action_set: p.action_set,
            left_user_path: p.left_user_path,
            right_user_path: p.right_user_path,
            oculus_touch_profile: p.oculus_touch_profile,
            valve_index_profile: p.valve_index_profile,
            htc_vive_profile: p.htc_vive_profile,
            microsoft_motion_profile: p.microsoft_motion_profile,
            generic_controller_profile: p.generic_controller_profile,
            simple_controller_profile: p.simple_controller_profile,
            pico4_controller_profile: p.pico4_controller_profile,
            pico_neo3_controller_profile: p.pico_neo3_controller_profile,
            left_profile_cache: p.left_profile_cache,
            right_profile_cache: p.right_profile_cache,
            left_grip_pose: p.left_grip_pose,
            right_grip_pose: p.right_grip_pose,
            left_trigger: p.left_trigger,
            right_trigger: p.right_trigger,
            left_trigger_touch: p.left_trigger_touch,
            right_trigger_touch: p.right_trigger_touch,
            left_trigger_click: p.left_trigger_click,
            right_trigger_click: p.right_trigger_click,
            left_squeeze: p.left_squeeze,
            right_squeeze: p.right_squeeze,
            left_squeeze_click: p.left_squeeze_click,
            right_squeeze_click: p.right_squeeze_click,
            left_thumbstick: p.left_thumbstick,
            right_thumbstick: p.right_thumbstick,
            left_thumbstick_touch: p.left_thumbstick_touch,
            right_thumbstick_touch: p.right_thumbstick_touch,
            left_thumbstick_click: p.left_thumbstick_click,
            right_thumbstick_click: p.right_thumbstick_click,
            left_trackpad: p.left_trackpad,
            right_trackpad: p.right_trackpad,
            left_trackpad_touch: p.left_trackpad_touch,
            right_trackpad_touch: p.right_trackpad_touch,
            left_trackpad_click: p.left_trackpad_click,
            right_trackpad_click: p.right_trackpad_click,
            left_trackpad_force: p.left_trackpad_force,
            right_trackpad_force: p.right_trackpad_force,
            left_primary: p.left_primary,
            right_primary: p.right_primary,
            left_secondary: p.left_secondary,
            right_secondary: p.right_secondary,
            left_primary_touch: p.left_primary_touch,
            right_primary_touch: p.right_primary_touch,
            left_secondary_touch: p.left_secondary_touch,
            right_secondary_touch: p.right_secondary_touch,
            left_menu: p.left_menu,
            right_menu: p.right_menu,
            left_thumbrest_touch: p.left_thumbrest_touch,
            right_thumbrest_touch: p.right_thumbrest_touch,
            left_select: p.left_select,
            right_select: p.right_select,
            left_space: p.left_space,
            right_space: p.right_space,
            left_aim_space: p.left_aim_space,
            right_aim_space: p.right_aim_space,
        }
    }
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
        create_openxr_input_parts(
            instance,
            session,
            runtime_supports_generic_controller,
            runtime_supports_bd_controller,
        )
        .map(Into::into)
    }

    fn detect_profile(
        &self,
        session: &xr::Session<xr::Vulkan>,
        hand_user_path: xr::Path,
    ) -> ActiveControllerProfile {
        let Ok(profile) = session.current_interaction_profile(hand_user_path) else {
            return ActiveControllerProfile::Generic;
        };
        if profile == self.oculus_touch_profile
            || profile == self.pico4_controller_profile
            || profile == self.pico_neo3_controller_profile
        {
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
        Self::log_grip_missing_aim_valid_throttled(Chirality::left, left_grip_pose, left_aim_pose);
        Self::log_grip_missing_aim_valid_throttled(
            Chirality::right,
            right_grip_pose,
            right_aim_pose,
        );
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

    /// Logs at most once every 300 frames per hand when the grip space is untracked but the aim space is valid.
    ///
    /// Confirms on-device that [`super::frame::resolve_controller_frame`] can use the aim fallback path.
    fn log_grip_missing_aim_valid_throttled(
        side: Chirality,
        grip_pose: Option<(Vec3, Quat)>,
        aim_pose: Option<(Vec3, Quat)>,
    ) {
        if grip_pose.is_some() || aim_pose.is_none() {
            return;
        }
        static LEFT: AtomicU32 = AtomicU32::new(0);
        static RIGHT: AtomicU32 = AtomicU32::new(0);
        let slot = match side {
            Chirality::left => &LEFT,
            Chirality::right => &RIGHT,
        };
        let n = slot.fetch_add(1, Ordering::Relaxed);
        if n % 300 == 0 {
            logger::debug!(
                "OpenXR {side:?}: grip pose invalid or untracked but aim pose valid; resolving controller frame from aim for IPC"
            );
        }
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
