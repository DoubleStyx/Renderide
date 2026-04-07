//! OpenXR [`xr::ActionSet`] bindings for tracked controllers (Oculus / Meta Touch-style).
//!
//! [`OpenxrInput`] uses the `/interaction_profiles/oculus/touch_controller` profile and binds
//! grip poses, triggers, squeeze, thumbsticks, face buttons, and menu. If creation fails
//! (unsupported runtime), the renderer continues without controller input.

use glam::Vec2;
use openxr as xr;

use crate::shared::{
    BodyNode, Chirality, TouchControllerModel, TouchControllerState, VRControllerState,
};

use super::session::openxr_pose_to_host_tracking;

/// Action spaces and bindings for sampling controller poses and Touch-style controls.
///
/// Grip [`xr::Space`]s are created from pose actions; [`Self::sync_and_sample`] must run after
/// `wait_frame` and before rendering so the predicted display time matches the frame.
pub struct OpenxrInput {
    action_set: xr::ActionSet,
    #[allow(dead_code)]
    left_grip_pose: xr::Action<xr::Posef>,
    #[allow(dead_code)]
    right_grip_pose: xr::Action<xr::Posef>,
    left_trigger: xr::Action<f32>,
    right_trigger: xr::Action<f32>,
    left_squeeze: xr::Action<f32>,
    right_squeeze: xr::Action<f32>,
    left_thumbstick: xr::Action<xr::Vector2f>,
    right_thumbstick: xr::Action<xr::Vector2f>,
    left_trigger_click: xr::Action<bool>,
    right_trigger_click: xr::Action<bool>,
    left_thumbstick_click: xr::Action<bool>,
    right_thumbstick_click: xr::Action<bool>,
    left_x: xr::Action<bool>,
    left_y: xr::Action<bool>,
    right_a: xr::Action<bool>,
    right_b: xr::Action<bool>,
    left_menu: xr::Action<bool>,
    left_space: xr::Space,
    right_space: xr::Space,
}

impl OpenxrInput {
    /// Creates actions, suggests Touch bindings, attaches to `session`, and builds grip [`xr::Space`]s.
    pub fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
    ) -> Result<Self, xr::sys::Result> {
        let action_set = instance.create_action_set("renderide_input", "Renderide VR input", 0)?;

        let left_grip_pose =
            action_set.create_action::<xr::Posef>("left_grip_pose", "Left grip pose", &[])?;
        let right_grip_pose =
            action_set.create_action::<xr::Posef>("right_grip_pose", "Right grip pose", &[])?;
        let left_trigger = action_set.create_action::<f32>("left_trigger", "Left trigger", &[])?;
        let right_trigger =
            action_set.create_action::<f32>("right_trigger", "Right trigger", &[])?;
        let left_squeeze = action_set.create_action::<f32>("left_squeeze", "Left squeeze", &[])?;
        let right_squeeze =
            action_set.create_action::<f32>("right_squeeze", "Right squeeze", &[])?;
        let left_thumbstick =
            action_set.create_action::<xr::Vector2f>("left_thumbstick", "Left thumbstick", &[])?;
        let right_thumbstick = action_set.create_action::<xr::Vector2f>(
            "right_thumbstick",
            "Right thumbstick",
            &[],
        )?;
        let left_trigger_click =
            action_set.create_action::<bool>("left_trigger_click", "Left trigger click", &[])?;
        let right_trigger_click =
            action_set.create_action::<bool>("right_trigger_click", "Right trigger click", &[])?;
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
        let left_x = action_set.create_action::<bool>("left_x", "Left X", &[])?;
        let left_y = action_set.create_action::<bool>("left_y", "Left Y", &[])?;
        let right_a = action_set.create_action::<bool>("right_a", "Right A", &[])?;
        let right_b = action_set.create_action::<bool>("right_b", "Right B", &[])?;
        let left_menu = action_set.create_action::<bool>("left_menu", "Left menu", &[])?;

        let touch_profile =
            instance.string_to_path("/interaction_profiles/oculus/touch_controller")?;
        instance.suggest_interaction_profile_bindings(
            touch_profile,
            &[
                xr::Binding::new(
                    &left_grip_pose,
                    instance.string_to_path("/user/hand/left/input/grip/pose")?,
                ),
                xr::Binding::new(
                    &right_grip_pose,
                    instance.string_to_path("/user/hand/right/input/grip/pose")?,
                ),
                xr::Binding::new(
                    &left_trigger,
                    instance.string_to_path("/user/hand/left/input/trigger/value")?,
                ),
                xr::Binding::new(
                    &right_trigger,
                    instance.string_to_path("/user/hand/right/input/trigger/value")?,
                ),
                xr::Binding::new(
                    &left_squeeze,
                    instance.string_to_path("/user/hand/left/input/squeeze/value")?,
                ),
                xr::Binding::new(
                    &right_squeeze,
                    instance.string_to_path("/user/hand/right/input/squeeze/value")?,
                ),
                xr::Binding::new(
                    &left_thumbstick,
                    instance.string_to_path("/user/hand/left/input/thumbstick")?,
                ),
                xr::Binding::new(
                    &right_thumbstick,
                    instance.string_to_path("/user/hand/right/input/thumbstick")?,
                ),
                xr::Binding::new(
                    &left_trigger_click,
                    instance.string_to_path("/user/hand/left/input/trigger/click")?,
                ),
                xr::Binding::new(
                    &right_trigger_click,
                    instance.string_to_path("/user/hand/right/input/trigger/click")?,
                ),
                xr::Binding::new(
                    &left_thumbstick_click,
                    instance.string_to_path("/user/hand/left/input/thumbstick/click")?,
                ),
                xr::Binding::new(
                    &right_thumbstick_click,
                    instance.string_to_path("/user/hand/right/input/thumbstick/click")?,
                ),
                xr::Binding::new(
                    &left_x,
                    instance.string_to_path("/user/hand/left/input/x/click")?,
                ),
                xr::Binding::new(
                    &left_y,
                    instance.string_to_path("/user/hand/left/input/y/click")?,
                ),
                xr::Binding::new(
                    &right_a,
                    instance.string_to_path("/user/hand/right/input/a/click")?,
                ),
                xr::Binding::new(
                    &right_b,
                    instance.string_to_path("/user/hand/right/input/b/click")?,
                ),
                xr::Binding::new(
                    &left_menu,
                    instance.string_to_path("/user/hand/left/input/menu/click")?,
                ),
            ],
        )?;

        session.attach_action_sets(&[&action_set])?;

        let left_space =
            left_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;
        let right_space =
            right_grip_pose.create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)?;

        Ok(Self {
            action_set,
            left_grip_pose,
            right_grip_pose,
            left_trigger,
            right_trigger,
            left_squeeze,
            right_squeeze,
            left_thumbstick,
            right_thumbstick,
            left_trigger_click,
            right_trigger_click,
            left_thumbstick_click,
            right_thumbstick_click,
            left_x,
            left_y,
            right_a,
            right_b,
            left_menu,
            left_space,
            right_space,
        })
    }

    /// Syncs actions and returns two [`VRControllerState`] entries (Touch) for the predicted frame time.
    pub fn sync_and_sample(
        &self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        predicted_time: xr::Time,
    ) -> Result<Vec<VRControllerState>, xr::sys::Result> {
        session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)])?;

        let left_loc = self.left_space.locate(stage, predicted_time)?;
        let right_loc = self.right_space.locate(stage, predicted_time)?;

        let left_tracked = left_loc
            .location_flags
            .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
            && left_loc
                .location_flags
                .contains(xr::SpaceLocationFlags::POSITION_VALID);
        let right_tracked = right_loc
            .location_flags
            .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
            && right_loc
                .location_flags
                .contains(xr::SpaceLocationFlags::POSITION_VALID);

        let (lp, lr) = openxr_pose_to_host_tracking(&left_loc.pose);
        let (rp, rr) = openxr_pose_to_host_tracking(&right_loc.pose);

        let lt = self.left_trigger.state(session, xr::Path::NULL)?;
        let rt = self.right_trigger.state(session, xr::Path::NULL)?;
        let ls = self.left_squeeze.state(session, xr::Path::NULL)?;
        let rs = self.right_squeeze.state(session, xr::Path::NULL)?;
        let lj = self.left_thumbstick.state(session, xr::Path::NULL)?;
        let rj = self.right_thumbstick.state(session, xr::Path::NULL)?;
        let ltc = self.left_trigger_click.state(session, xr::Path::NULL)?;
        let rtc = self.right_trigger_click.state(session, xr::Path::NULL)?;
        let ljc = self.left_thumbstick_click.state(session, xr::Path::NULL)?;
        let rjc = self.right_thumbstick_click.state(session, xr::Path::NULL)?;
        let lx = self.left_x.state(session, xr::Path::NULL)?;
        let ly = self.left_y.state(session, xr::Path::NULL)?;
        let ra = self.right_a.state(session, xr::Path::NULL)?;
        let rb = self.right_b.state(session, xr::Path::NULL)?;
        let lm = self.left_menu.state(session, xr::Path::NULL)?;

        let left = TouchControllerState {
            model: TouchControllerModel::quest_and_rift_s,
            start: lm.current_state,
            button_yb: ly.current_state,
            button_xa: lx.current_state,
            button_yb_touch: false,
            button_xa_touch: false,
            thumbrest_touch: false,
            grip: ls.current_state,
            grip_click: ls.current_state > 0.85,
            joystick_raw: Vec2::new(lj.current_state.x, lj.current_state.y),
            joystick_touch: lj.current_state.x.abs() > 0.01 || lj.current_state.y.abs() > 0.01,
            joystick_click: ljc.current_state,
            trigger: lt.current_state,
            trigger_touch: lt.current_state > 0.01,
            trigger_click: ltc.current_state,
            device_id: None,
            device_model: Some("OpenXR Touch".to_string()),
            side: Chirality::left,
            body_node: BodyNode::left_controller,
            is_device_active: true,
            is_tracking: left_tracked,
            position: lp,
            rotation: lr,
            has_bound_hand: false,
            hand_position: lp,
            hand_rotation: lr,
            battery_level: 1.0,
            battery_charging: false,
        };

        let right = TouchControllerState {
            model: TouchControllerModel::quest_and_rift_s,
            start: false,
            button_yb: rb.current_state,
            button_xa: ra.current_state,
            button_yb_touch: false,
            button_xa_touch: false,
            thumbrest_touch: false,
            grip: rs.current_state,
            grip_click: rs.current_state > 0.85,
            joystick_raw: Vec2::new(rj.current_state.x, rj.current_state.y),
            joystick_touch: rj.current_state.x.abs() > 0.01 || rj.current_state.y.abs() > 0.01,
            joystick_click: rjc.current_state,
            trigger: rt.current_state,
            trigger_touch: rt.current_state > 0.01,
            trigger_click: rtc.current_state,
            device_id: None,
            device_model: Some("OpenXR Touch".to_string()),
            side: Chirality::right,
            body_node: BodyNode::right_controller,
            is_device_active: true,
            is_tracking: right_tracked,
            position: rp,
            rotation: rr,
            has_bound_hand: false,
            hand_position: rp,
            hand_rotation: rr,
            battery_level: 1.0,
            battery_charging: false,
        };

        Ok(vec![
            VRControllerState::touch_controller_state(left),
            VRControllerState::touch_controller_state(right),
        ])
    }

    /// Logs a one-time hint if stereo view X order does not match left-then-right (PRIMARY_STEREO convention).
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
                "OpenXR stereo: views[0].pose.x ({x0}) > views[1].pose.x ({x1}); runtime may use right-then-left ordering — verify eye mapping."
            );
        }
    }
}
