use glam::{Quat, Vec2, Vec3};
use openxr as xr;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::shared::{
    BodyNode, Chirality, GenericControllerState, IndexControllerState, TouchControllerModel,
    TouchControllerState, VRControllerState, ViveControllerState, WindowsMRControllerState,
};

use super::session::openxr_pose_to_host_tracking;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveControllerProfile {
    Touch,
    Index,
    Vive,
    WindowsMr,
    Generic,
    Simple,
}

fn profile_code(profile: ActiveControllerProfile) -> u8 {
    match profile {
        ActiveControllerProfile::Touch => 1,
        ActiveControllerProfile::Index => 2,
        ActiveControllerProfile::Vive => 3,
        ActiveControllerProfile::WindowsMr => 4,
        ActiveControllerProfile::Generic => 5,
        ActiveControllerProfile::Simple => 6,
    }
}

fn decode_profile_code(code: u8) -> Option<ActiveControllerProfile> {
    match code {
        1 => Some(ActiveControllerProfile::Touch),
        2 => Some(ActiveControllerProfile::Index),
        3 => Some(ActiveControllerProfile::Vive),
        4 => Some(ActiveControllerProfile::WindowsMr),
        5 => Some(ActiveControllerProfile::Generic),
        6 => Some(ActiveControllerProfile::Simple),
        _ => None,
    }
}

fn is_concrete_profile(profile: ActiveControllerProfile) -> bool {
    matches!(
        profile,
        ActiveControllerProfile::Touch
            | ActiveControllerProfile::Index
            | ActiveControllerProfile::Vive
            | ActiveControllerProfile::WindowsMr
    )
}

fn log_profile_transition(side: Chirality, profile: ActiveControllerProfile) {
    static LEFT: AtomicU8 = AtomicU8::new(0);
    static RIGHT: AtomicU8 = AtomicU8::new(0);
    let slot = match side {
        Chirality::left => &LEFT,
        Chirality::right => &RIGHT,
    };
    let code = profile_code(profile);
    let previous = slot.swap(code, Ordering::Relaxed);
    if previous != code {
        logger::info!("OpenXR {:?} controller profile: {:?}", side, profile);
    }
}

fn unity_euler_deg(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_rotation_y(y.to_radians())
        * Quat::from_rotation_x(x.to_radians())
        * Quat::from_rotation_z(z.to_radians())
}

fn touch_pose_correction(side: Chirality, position: Vec3, rotation: Quat) -> (Vec3, Quat) {
    let rotation = rotation * Quat::from_rotation_x(45.0_f32.to_radians());
    let offset = match side {
        Chirality::left => Vec3::new(-0.01, 0.04, 0.03),
        Chirality::right => Vec3::new(0.01, 0.04, 0.03),
    };
    (position - rotation * offset, rotation)
}

fn index_pose_correction(side: Chirality, position: Vec3, rotation: Quat) -> (Vec3, Quat) {
    let roll = match side {
        Chirality::left => 90.0_f32,
        Chirality::right => -90.0_f32,
    };
    (position, rotation * Quat::from_rotation_z(roll.to_radians()))
}

fn bound_hand_pose_defaults(profile: ActiveControllerProfile, side: Chirality) -> (bool, Vec3, Quat) {
    let generic_fix = unity_euler_deg(90.0, 90.0, 90.0).inverse();
    match (profile, side) {
        (ActiveControllerProfile::Touch, Chirality::left) => (
            true,
            Vec3::new(-0.04, -0.025, -0.1),
            unity_euler_deg(185.0, -95.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Touch, Chirality::right) => (
            true,
            Vec3::new(0.04, -0.025, -0.1),
            unity_euler_deg(5.0, -95.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Vive, Chirality::left)
        | (ActiveControllerProfile::Generic, Chirality::left)
        | (ActiveControllerProfile::Simple, Chirality::left) => (
            true,
            Vec3::new(-0.02, 0.0, -0.16),
            unity_euler_deg(140.0, -90.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Vive, Chirality::right)
        | (ActiveControllerProfile::Generic, Chirality::right)
        | (ActiveControllerProfile::Simple, Chirality::right) => (
            true,
            Vec3::new(0.02, 0.0, -0.16),
            unity_euler_deg(40.0, -90.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::WindowsMr, Chirality::left) => (
            true,
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (ActiveControllerProfile::WindowsMr, Chirality::right) => (
            true,
            Vec3::new(0.028, 0.0, -0.18),
            unity_euler_deg(30.0, -5.0, -100.0),
        ),
        (ActiveControllerProfile::Index, Chirality::left) => (
            true,
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (ActiveControllerProfile::Index, Chirality::right) => (
            true,
            Vec3::new(0.028, 0.0, -0.18),
            unity_euler_deg(30.0, -5.0, -100.0),
        ),
    }
}

#[cfg(test)]
fn transform_pose(
    base_position: Vec3,
    base_rotation: Quat,
    local_position: Vec3,
    local_rotation: Quat,
) -> (Vec3, Quat) {
    (
        base_position + base_rotation * local_position,
        (base_rotation * local_rotation).normalize(),
    )
}

fn inverse_transform_pose(
    base_position: Vec3,
    base_rotation: Quat,
    world_position: Vec3,
    world_rotation: Quat,
) -> (Vec3, Quat) {
    let inv = base_rotation.inverse();
    (
        inv * (world_position - base_position),
        (inv * world_rotation).normalize(),
    )
}

fn controller_pose_from_aim(position: Vec3, rotation: Quat) -> (Vec3, Quat) {
    let rotation = rotation.normalize();
    let tip_offset = Vec3::new(0.0, 0.0, 0.075);
    (position - rotation * tip_offset, rotation)
}

fn pose_from_location(location: &xr::SpaceLocation) -> Option<(Vec3, Quat)> {
    let tracked = location
        .location_flags
        .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID);
    tracked.then(|| openxr_pose_to_host_tracking(&location.pose))
}

#[derive(Clone, Copy)]
struct ControllerFrame {
    position: Vec3,
    rotation: Quat,
    has_bound_hand: bool,
    hand_position: Vec3,
    hand_rotation: Quat,
}

fn resolve_controller_frame(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    aim_pose: Option<(Vec3, Quat)>,
) -> Option<ControllerFrame> {
    let (has_bound_hand, hand_position_default, hand_rotation_default) =
        bound_hand_pose_defaults(profile, side);
    match profile {
        ActiveControllerProfile::Touch => {
            let (grip_position, grip_rotation) = grip_pose?;
            let (position, rotation) = touch_pose_correction(side, grip_position, grip_rotation);
            Some(ControllerFrame {
                position,
                rotation,
                has_bound_hand,
                hand_position: hand_position_default,
                hand_rotation: hand_rotation_default,
            })
        }
        ActiveControllerProfile::Index => {
            if let (Some((aim_position, aim_rotation)), Some((grip_position, grip_rotation))) =
                (aim_pose, grip_pose)
            {
                let (position, rotation) = controller_pose_from_aim(aim_position, aim_rotation);
                let (hand_world_position, hand_world_rotation) =
                    index_pose_correction(side, grip_position, grip_rotation);
                let (hand_position, hand_rotation) = inverse_transform_pose(
                    position,
                    rotation,
                    hand_world_position,
                    hand_world_rotation,
                );
                Some(ControllerFrame {
                    position,
                    rotation,
                    has_bound_hand,
                    hand_position,
                    hand_rotation,
                })
            } else if let Some((aim_position, aim_rotation)) = aim_pose {
                let (position, rotation) = controller_pose_from_aim(aim_position, aim_rotation);
                Some(ControllerFrame {
                    position,
                    rotation,
                    has_bound_hand,
                    hand_position: hand_position_default,
                    hand_rotation: hand_rotation_default,
                })
            } else if let Some((grip_position, grip_rotation)) = grip_pose {
                let (position, rotation) =
                    index_pose_correction(side, grip_position, grip_rotation);
                Some(ControllerFrame {
                    position,
                    rotation,
                    has_bound_hand,
                    hand_position: hand_position_default,
                    hand_rotation: hand_rotation_default,
                })
            } else {
                None
            }
        }
        _ => {
            let (position, rotation) = grip_pose?;
            Some(ControllerFrame {
                position,
                rotation,
                has_bound_hand,
                hand_position: hand_position_default,
                hand_rotation: hand_rotation_default,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let delta = (actual - expected).length();
        assert!(
            delta < 1e-4,
            "vec3 mismatch: actual={actual:?} expected={expected:?} delta={delta}"
        );
    }

    fn assert_quat_near(actual: Quat, expected: Quat) {
        let dot = actual.normalize().dot(expected.normalize()).abs();
        assert!(
            (1.0 - dot) < 1e-4,
            "quat mismatch: actual={actual:?} expected={expected:?} dot={dot}"
        );
    }

    #[test]
    fn index_frame_uses_aim_for_controller_and_grip_for_bound_hand() {
        let grip_position = Vec3::new(0.2, 1.3, -0.4);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let aim_position = Vec3::new(0.24, 1.34, -0.28);
        let aim_rotation = (Quat::from_rotation_y(0.75) * Quat::from_rotation_x(-0.1)).normalize();

        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");

        let (expected_controller_position, expected_controller_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        assert_vec3_near(frame.position, expected_controller_position);
        assert_quat_near(frame.rotation, expected_controller_rotation);

        let (hand_world_position, hand_world_rotation) = transform_pose(
            frame.position,
            frame.rotation,
            frame.hand_position,
            frame.hand_rotation,
        );
        let (expected_hand_position, expected_hand_rotation) =
            index_pose_correction(Chirality::left, grip_position, grip_rotation);
        assert_vec3_near(hand_world_position, expected_hand_position);
        assert_quat_near(hand_world_rotation, expected_hand_rotation);
    }
}

fn vec2_nonzero(v: Vec2) -> bool {
    v.length_squared() > 1e-6
}

fn choose_axis(thumbstick: Vec2, trackpad: Vec2) -> Vec2 {
    if vec2_nonzero(thumbstick) {
        thumbstick
    } else {
        trackpad
    }
}

fn body_node_for_side(side: Chirality) -> BodyNode {
    match side {
        Chirality::left => BodyNode::left_controller,
        Chirality::right => BodyNode::right_controller,
    }
}

fn device_label(profile: ActiveControllerProfile) -> &'static str {
    match profile {
        ActiveControllerProfile::Touch => "OpenXR Touch Controller",
        ActiveControllerProfile::Index => "OpenXR Index Controller",
        ActiveControllerProfile::Vive => "OpenXR Vive Controller",
        ActiveControllerProfile::WindowsMr => "OpenXR Windows MR Controller",
        ActiveControllerProfile::Generic => "OpenXR Generic Controller",
        ActiveControllerProfile::Simple => "OpenXR Simple Controller",
    }
}

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
    left_grip_pose: xr::Action<xr::Posef>,
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
    pub fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
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
        let mut any_bindings = false;
        let mut last_binding_err = None;
        let mut suggest = |profile: xr::Path, bindings: &[xr::Binding<'_>]| match instance
            .suggest_interaction_profile_bindings(profile, bindings)
        {
            Ok(()) => any_bindings = true,
            Err(e) => last_binding_err = Some(e),
        };
        suggest(
            oculus_touch_profile,
            &[
                xr::Binding::new(&left_grip_pose, left_grip_pose_path),
                xr::Binding::new(&right_grip_pose, right_grip_pose_path),
                xr::Binding::new(&left_aim_pose, left_aim_pose_path),
                xr::Binding::new(&right_aim_pose, right_aim_pose_path),
                xr::Binding::new(&left_trigger, left_trigger_value_path),
                xr::Binding::new(&right_trigger, right_trigger_value_path),
                xr::Binding::new(&left_trigger_touch, left_trigger_touch_path),
                xr::Binding::new(&right_trigger_touch, right_trigger_touch_path),
                xr::Binding::new(&left_squeeze, left_squeeze_value_path),
                xr::Binding::new(&right_squeeze, right_squeeze_value_path),
                xr::Binding::new(&left_thumbstick, left_thumbstick_path),
                xr::Binding::new(&right_thumbstick, right_thumbstick_path),
                xr::Binding::new(&left_thumbstick_touch, left_thumbstick_touch_path),
                xr::Binding::new(&right_thumbstick_touch, right_thumbstick_touch_path),
                xr::Binding::new(&left_thumbstick_click, left_thumbstick_click_path),
                xr::Binding::new(&right_thumbstick_click, right_thumbstick_click_path),
                xr::Binding::new(&left_primary, left_x_click_path),
                xr::Binding::new(&left_secondary, left_y_click_path),
                xr::Binding::new(&right_primary, right_a_click_path),
                xr::Binding::new(&right_secondary, right_b_click_path),
                xr::Binding::new(&left_primary_touch, left_x_touch_path),
                xr::Binding::new(&left_secondary_touch, left_y_touch_path),
                xr::Binding::new(&right_primary_touch, right_a_touch_path),
                xr::Binding::new(&right_secondary_touch, right_b_touch_path),
                xr::Binding::new(&left_menu, left_menu_click_path),
                xr::Binding::new(&left_thumbrest_touch, left_thumbrest_touch_path),
                xr::Binding::new(&right_thumbrest_touch, right_thumbrest_touch_path),
            ],
        );
        suggest(
            valve_index_profile,
            &[
                xr::Binding::new(&left_grip_pose, left_grip_pose_path),
                xr::Binding::new(&right_grip_pose, right_grip_pose_path),
                xr::Binding::new(&left_aim_pose, left_aim_pose_path),
                xr::Binding::new(&right_aim_pose, right_aim_pose_path),
                xr::Binding::new(&left_trigger, left_trigger_value_path),
                xr::Binding::new(&right_trigger, right_trigger_value_path),
                xr::Binding::new(&left_trigger_touch, left_trigger_touch_path),
                xr::Binding::new(&right_trigger_touch, right_trigger_touch_path),
                xr::Binding::new(&left_trigger_click, left_trigger_click_path),
                xr::Binding::new(&right_trigger_click, right_trigger_click_path),
                xr::Binding::new(&left_squeeze, left_squeeze_value_path),
                xr::Binding::new(&right_squeeze, right_squeeze_value_path),
                xr::Binding::new(&left_thumbstick, left_thumbstick_path),
                xr::Binding::new(&right_thumbstick, right_thumbstick_path),
                xr::Binding::new(&left_thumbstick_touch, left_thumbstick_touch_path),
                xr::Binding::new(&right_thumbstick_touch, right_thumbstick_touch_path),
                xr::Binding::new(&left_thumbstick_click, left_thumbstick_click_path),
                xr::Binding::new(&right_thumbstick_click, right_thumbstick_click_path),
                xr::Binding::new(&left_trackpad, left_trackpad_path),
                xr::Binding::new(&right_trackpad, right_trackpad_path),
                xr::Binding::new(&left_trackpad_touch, left_trackpad_touch_path),
                xr::Binding::new(&right_trackpad_touch, right_trackpad_touch_path),
                xr::Binding::new(&left_trackpad_force, left_trackpad_force_path),
                xr::Binding::new(&right_trackpad_force, right_trackpad_force_path),
                xr::Binding::new(&left_primary, left_a_click_path),
                xr::Binding::new(&left_secondary, left_b_click_path),
                xr::Binding::new(&right_primary, right_a_click_path),
                xr::Binding::new(&right_secondary, right_b_click_path),
                xr::Binding::new(&left_primary_touch, left_a_touch_path),
                xr::Binding::new(&left_secondary_touch, left_b_touch_path),
                xr::Binding::new(&right_primary_touch, right_a_touch_path),
                xr::Binding::new(&right_secondary_touch, right_b_touch_path),
            ],
        );
        suggest(
            htc_vive_profile,
            &[
                xr::Binding::new(&left_grip_pose, left_grip_pose_path),
                xr::Binding::new(&right_grip_pose, right_grip_pose_path),
                xr::Binding::new(&left_aim_pose, left_aim_pose_path),
                xr::Binding::new(&right_aim_pose, right_aim_pose_path),
                xr::Binding::new(&left_trigger, left_trigger_value_path),
                xr::Binding::new(&right_trigger, right_trigger_value_path),
                xr::Binding::new(&left_trigger_click, left_trigger_click_path),
                xr::Binding::new(&right_trigger_click, right_trigger_click_path),
                xr::Binding::new(&left_squeeze_click, left_squeeze_click_path),
                xr::Binding::new(&right_squeeze_click, right_squeeze_click_path),
                xr::Binding::new(&left_trackpad, left_trackpad_path),
                xr::Binding::new(&right_trackpad, right_trackpad_path),
                xr::Binding::new(&left_trackpad_touch, left_trackpad_touch_path),
                xr::Binding::new(&right_trackpad_touch, right_trackpad_touch_path),
                xr::Binding::new(&left_trackpad_click, left_trackpad_click_path),
                xr::Binding::new(&right_trackpad_click, right_trackpad_click_path),
                xr::Binding::new(&left_menu, left_menu_click_path),
                xr::Binding::new(&right_menu, right_menu_click_path),
            ],
        );
        suggest(
            microsoft_motion_profile,
            &[
                xr::Binding::new(&left_grip_pose, left_grip_pose_path),
                xr::Binding::new(&right_grip_pose, right_grip_pose_path),
                xr::Binding::new(&left_aim_pose, left_aim_pose_path),
                xr::Binding::new(&right_aim_pose, right_aim_pose_path),
                xr::Binding::new(&left_trigger, left_trigger_value_path),
                xr::Binding::new(&right_trigger, right_trigger_value_path),
                xr::Binding::new(&left_squeeze_click, left_squeeze_click_path),
                xr::Binding::new(&right_squeeze_click, right_squeeze_click_path),
                xr::Binding::new(&left_thumbstick, left_thumbstick_path),
                xr::Binding::new(&right_thumbstick, right_thumbstick_path),
                xr::Binding::new(&left_thumbstick_click, left_thumbstick_click_path),
                xr::Binding::new(&right_thumbstick_click, right_thumbstick_click_path),
                xr::Binding::new(&left_trackpad, left_trackpad_path),
                xr::Binding::new(&right_trackpad, right_trackpad_path),
                xr::Binding::new(&left_trackpad_touch, left_trackpad_touch_path),
                xr::Binding::new(&right_trackpad_touch, right_trackpad_touch_path),
                xr::Binding::new(&left_trackpad_click, left_trackpad_click_path),
                xr::Binding::new(&right_trackpad_click, right_trackpad_click_path),
                xr::Binding::new(&left_menu, left_menu_click_path),
                xr::Binding::new(&right_menu, right_menu_click_path),
            ],
        );
        suggest(
            generic_controller_profile,
            &[
                xr::Binding::new(&left_grip_pose, left_grip_pose_path),
                xr::Binding::new(&right_grip_pose, right_grip_pose_path),
                xr::Binding::new(&left_aim_pose, left_aim_pose_path),
                xr::Binding::new(&right_aim_pose, right_aim_pose_path),
                xr::Binding::new(&left_trigger, left_trigger_value_path),
                xr::Binding::new(&right_trigger, right_trigger_value_path),
                xr::Binding::new(&left_squeeze, left_squeeze_value_path),
                xr::Binding::new(&right_squeeze, right_squeeze_value_path),
                xr::Binding::new(&left_thumbstick, left_thumbstick_path),
                xr::Binding::new(&right_thumbstick, right_thumbstick_path),
                xr::Binding::new(&left_thumbstick_click, left_thumbstick_click_path),
                xr::Binding::new(&right_thumbstick_click, right_thumbstick_click_path),
                xr::Binding::new(&left_primary, left_select_click_path),
                xr::Binding::new(&right_primary, right_select_click_path),
                xr::Binding::new(&left_secondary, left_menu_click_path),
                xr::Binding::new(&right_secondary, right_menu_click_path),
            ],
        );
        suggest(
            simple_controller_profile,
            &[
                xr::Binding::new(&left_grip_pose, left_grip_pose_path),
                xr::Binding::new(&right_grip_pose, right_grip_pose_path),
                xr::Binding::new(&left_aim_pose, left_aim_pose_path),
                xr::Binding::new(&right_aim_pose, right_aim_pose_path),
                xr::Binding::new(&left_select, left_select_click_path),
                xr::Binding::new(&right_select, right_select_click_path),
                xr::Binding::new(&left_menu, left_menu_click_path),
                xr::Binding::new(&right_menu, right_menu_click_path),
            ],
        );
        if !any_bindings {
            return Err(last_binding_err.unwrap_or(xr::sys::Result::ERROR_PATH_UNSUPPORTED));
        }
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
        let left_frame = resolve_controller_frame(
            left_profile,
            Chirality::left,
            left_grip_pose,
            left_aim_pose,
        );
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

#[allow(clippy::too_many_arguments)]
fn build_controller_state(
    profile: ActiveControllerProfile,
    side: Chirality,
    is_tracking: bool,
    frame: ControllerFrame,
    trigger: f32,
    trigger_touch: bool,
    trigger_click: bool,
    squeeze: f32,
    squeeze_click: bool,
    thumbstick: Vec2,
    thumbstick_touch: bool,
    thumbstick_click: bool,
    trackpad: Vec2,
    trackpad_touch: bool,
    trackpad_click: bool,
    trackpad_force: f32,
    primary: bool,
    secondary: bool,
    primary_touch: bool,
    secondary_touch: bool,
    menu: bool,
    thumbrest_touch: bool,
    select: bool,
) -> VRControllerState {
    let device_id = Some(match side {
        Chirality::left => "OpenXR Left".to_string(),
        Chirality::right => "OpenXR Right".to_string(),
    });
    let device_model = Some(device_label(profile).to_string());
    let body_node = body_node_for_side(side);
    let trigger_touch = trigger_touch || trigger > 0.01;
    let trigger_click = trigger_click || trigger > 0.75;
    let grip_touch = squeeze_click || squeeze > 0.05;
    let grip_click = squeeze_click || squeeze > 0.85;
    let joystick_touch = thumbstick_touch || vec2_nonzero(thumbstick);
    let touchpad_touch = trackpad_touch || vec2_nonzero(trackpad) || trackpad_force > 0.01;
    let axis = choose_axis(thumbstick, trackpad);
    match profile {
        ActiveControllerProfile::Touch => {
            VRControllerState::touch_controller_state(TouchControllerState {
                model: TouchControllerModel::quest_and_rift_s,
                start: menu,
                button_yb: secondary,
                button_xa: primary,
                button_yb_touch: secondary_touch,
                button_xa_touch: primary_touch,
                thumbrest_touch,
                grip: squeeze,
                grip_click,
                joystick_raw: thumbstick,
                joystick_touch,
                joystick_click: thumbstick_click,
                trigger,
                trigger_touch,
                trigger_click,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Index => {
            VRControllerState::index_controller_state(IndexControllerState {
                grip: squeeze,
                grip_touch,
                grip_click,
                button_a: primary,
                button_b: secondary,
                button_atouch: primary_touch,
                button_btouch: secondary_touch,
                trigger,
                trigger_touch,
                trigger_click,
                joystick_raw: thumbstick,
                joystick_touch,
                joystick_click: thumbstick_click,
                touchpad: trackpad,
                touchpad_touch,
                touchpad_press: trackpad_click || trackpad_force > 0.3,
                touchpad_force: trackpad_force,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Vive => {
            VRControllerState::vive_controller_state(ViveControllerState {
                grip: squeeze_click || squeeze > 0.5,
                app: menu,
                trigger_hair: trigger_touch,
                trigger_click,
                trigger,
                touchpad_touch,
                touchpad_click: trackpad_click,
                touchpad: trackpad,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::WindowsMr => {
            VRControllerState::windows_mr_controller_state(WindowsMRControllerState {
                grip: squeeze_click || squeeze > 0.5,
                app: menu,
                trigger_hair: trigger_touch,
                trigger_click,
                trigger,
                touchpad_touch,
                touchpad_click: trackpad_click,
                touchpad: trackpad,
                joystick_click: thumbstick_click,
                joystick_raw: thumbstick,
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
        ActiveControllerProfile::Generic | ActiveControllerProfile::Simple => {
            VRControllerState::generic_controller_state(GenericControllerState {
                strength: if profile == ActiveControllerProfile::Simple && select {
                    1.0
                } else {
                    trigger
                },
                axis: if profile == ActiveControllerProfile::Simple {
                    Vec2::ZERO
                } else {
                    axis
                },
                touching_strength: trigger_touch
                    || (profile == ActiveControllerProfile::Simple && select),
                touching_axis: if profile == ActiveControllerProfile::Simple {
                    false
                } else {
                    joystick_touch || touchpad_touch
                },
                primary: if profile == ActiveControllerProfile::Simple {
                    select
                } else {
                    primary || trigger_click
                },
                menu,
                grab: grip_click,
                secondary: if profile == ActiveControllerProfile::Simple {
                    false
                } else {
                    secondary || trackpad_click
                },
                device_id,
                device_model,
                side,
                body_node,
                is_device_active: true,
                is_tracking,
                position: frame.position,
                rotation: frame.rotation,
                has_bound_hand: frame.has_bound_hand,
                hand_position: frame.hand_position,
                hand_rotation: frame.hand_rotation,
                battery_level: 1.0,
                battery_charging: false,
            })
        }
    }
}
