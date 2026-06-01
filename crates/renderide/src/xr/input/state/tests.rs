use glam::{Quat, Vec2, Vec3};

use crate::shared::{BodyNode, Chirality, VRControllerState};

use super::super::frame::ControllerFrame;
use super::super::profile::ActiveControllerProfile;
use super::axis::vec2_nonzero;
use super::{
    OpenxrControllerRawInputs, OpenxrControllerThresholdState, body_node_for_side,
    build_controller_state, build_controller_state_with_thresholds,
};

fn frame() -> ControllerFrame {
    ControllerFrame {
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_y(0.25),
        has_bound_hand: true,
        hand_position: Vec3::new(0.1, 0.2, 0.3),
        hand_rotation: Quat::from_rotation_x(0.5),
    }
}

fn raw(profile: ActiveControllerProfile, side: Chirality) -> OpenxrControllerRawInputs {
    OpenxrControllerRawInputs {
        profile,
        side,
        is_tracking: true,
        frame: frame(),
        trigger: 0.8,
        trigger_active: true,
        trigger_touch: false,
        trigger_click: false,
        squeeze: 0.9,
        squeeze_active: true,
        squeeze_click: false,
        thumbstick: Vec2::new(0.25, -0.5),
        thumbstick_touch: false,
        thumbstick_click: true,
        trackpad: Vec2::new(-0.2, 0.3),
        trackpad_touch: false,
        trackpad_click: true,
        trackpad_force: 0.35,
        primary: true,
        secondary: true,
        primary_touch: true,
        secondary_touch: false,
        menu: true,
        thumbrest_touch: true,
        select: false,
    }
}

#[test]
fn vec2_nonzero_uses_deadzone() {
    assert!(!vec2_nonzero(Vec2::ZERO));
    assert!(!vec2_nonzero(Vec2::splat(0.0001)));
    assert!(vec2_nonzero(Vec2::new(0.002, 0.0)));
    assert!(vec2_nonzero(Vec2::new(0.0, -0.002)));
}

#[test]
fn body_nodes_follow_chirality() {
    assert_eq!(
        body_node_for_side(Chirality::Left),
        BodyNode::LeftController
    );
    assert_eq!(
        body_node_for_side(Chirality::Right),
        BodyNode::RightController
    );
}

#[test]
fn touch_class_profiles_share_touch_payload_shape() {
    for profile in [
        ActiveControllerProfile::Touch,
        ActiveControllerProfile::Pico4,
        ActiveControllerProfile::PicoNeo3,
        ActiveControllerProfile::HpReverbG2,
        ActiveControllerProfile::ViveCosmos,
        ActiveControllerProfile::ViveFocus3,
        ActiveControllerProfile::Generic,
        ActiveControllerProfile::Simple,
    ] {
        let mut input = raw(profile, Chirality::Left);
        if profile == ActiveControllerProfile::HpReverbG2 {
            input.trigger = 0.95;
        }
        let state = build_controller_state(input);
        let VRControllerState::TouchControllerState(touch) = state else {
            panic!("profile {profile:?} should use touch payload");
        };
        assert_eq!(touch.side, Chirality::Left);
        assert_eq!(touch.body_node, BodyNode::LeftController);
        assert_eq!(touch.device_id.as_deref(), Some("OpenXR Left"));
        assert!(
            touch
                .device_model
                .unwrap_or_default()
                .starts_with("OpenXR ")
        );
        assert!(touch.trigger_touch);
        assert!(touch.trigger_click);
        assert!(touch.grip_click);
        assert!(touch.joystick_touch);
        assert!(touch.joystick_click);
        assert!(touch.button_xa);
        assert!(touch.button_yb);
        assert!(touch.thumbrest_touch);
        assert_eq!(touch.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(touch.hand_position, Vec3::new(0.1, 0.2, 0.3));
    }
}

#[test]
fn simple_select_folds_into_touch_trigger() {
    let mut input = raw(ActiveControllerProfile::Simple, Chirality::Right);
    input.trigger = 0.0;
    input.select = true;
    let VRControllerState::TouchControllerState(touch) = build_controller_state(input) else {
        panic!("simple profile should use touch payload");
    };
    assert_eq!(touch.side, Chirality::Right);
    assert_eq!(touch.body_node, BodyNode::RightController);
    assert_eq!(touch.trigger, 1.0);
    assert!(touch.trigger_touch);
    assert!(touch.trigger_click);
}

#[test]
fn index_profile_maps_trackpad_and_grip_axes() {
    let VRControllerState::IndexControllerState(index) =
        build_controller_state(raw(ActiveControllerProfile::Index, Chirality::Left))
    else {
        panic!("index profile should use index payload");
    };
    assert_eq!(index.grip, 0.9);
    assert!(index.grip_touch);
    assert!(index.grip_click);
    assert!(index.trigger_touch);
    assert!(index.trigger_click);
    assert_eq!(index.touchpad, Vec2::new(-0.2, 0.3));
    assert!(index.touchpad_touch);
    assert!(index.touchpad_press);
    assert_eq!(index.touchpad_force, 0.35);
    assert!(index.button_a);
    assert!(index.button_b);
}

#[test]
fn vive_profile_maps_menu_grip_trigger_and_trackpad() {
    let VRControllerState::ViveControllerState(vive) =
        build_controller_state(raw(ActiveControllerProfile::Vive, Chirality::Left))
    else {
        panic!("vive profile should use vive payload");
    };
    assert!(vive.grip);
    assert!(vive.app);
    assert!(vive.trigger_hair);
    assert!(vive.trigger_click);
    assert_eq!(vive.trigger, 0.8);
    assert!(vive.touchpad_touch);
    assert!(vive.touchpad_click);
    assert_eq!(vive.touchpad, Vec2::new(-0.2, 0.3));
}

#[test]
fn windows_mr_profile_maps_thumbstick_and_touchpad() {
    let mut input = raw(ActiveControllerProfile::WindowsMr, Chirality::Right);
    input.trigger = 0.95;
    let VRControllerState::WindowsMRControllerState(wmr) = build_controller_state(input) else {
        panic!("windows mr profile should use wmr payload");
    };
    assert_eq!(wmr.side, Chirality::Right);
    assert!(wmr.grip);
    assert!(wmr.app);
    assert!(wmr.trigger_hair);
    assert!(wmr.trigger_click);
    assert_eq!(wmr.joystick_raw, Vec2::new(0.25, -0.5));
    assert!(wmr.joystick_click);
    assert_eq!(wmr.touchpad, Vec2::new(-0.2, 0.3));
    assert!(wmr.touchpad_touch);
    assert!(wmr.touchpad_click);
}

#[test]
fn index_grip_click_uses_steamvr_threshold_hysteresis() {
    let mut thresholds = OpenxrControllerThresholdState::default();
    let mut input = raw(ActiveControllerProfile::Index, Chirality::Left);
    input.squeeze = 0.29;

    let VRControllerState::IndexControllerState(index) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("index profile should use index payload");
    };
    assert!(!index.grip_click);

    let mut input = raw(ActiveControllerProfile::Index, Chirality::Left);
    input.squeeze = 0.30;
    let VRControllerState::IndexControllerState(index) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("index profile should use index payload");
    };
    assert!(index.grip_click);

    let mut input = raw(ActiveControllerProfile::Index, Chirality::Left);
    input.squeeze = 0.26;
    let VRControllerState::IndexControllerState(index) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("index profile should use index payload");
    };
    assert!(index.grip_click);

    let mut input = raw(ActiveControllerProfile::Index, Chirality::Left);
    input.squeeze = 0.24;
    let VRControllerState::IndexControllerState(index) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("index profile should use index payload");
    };
    assert!(!index.grip_click);
}

#[test]
fn touch_class_trigger_and_grip_use_default_action_thresholds() {
    let mut thresholds = OpenxrControllerThresholdState::default();
    let mut input = raw(ActiveControllerProfile::Touch, Chirality::Left);
    input.trigger = 0.79;
    input.squeeze = 0.79;

    let VRControllerState::TouchControllerState(touch) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("touch profile should use touch payload");
    };
    assert!(!touch.trigger_click);
    assert!(!touch.grip_click);

    let mut input = raw(ActiveControllerProfile::Touch, Chirality::Left);
    input.trigger = 0.80;
    input.squeeze = 0.80;
    let VRControllerState::TouchControllerState(touch) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("touch profile should use touch payload");
    };
    assert!(touch.trigger_click);
    assert!(touch.grip_click);

    let mut input = raw(ActiveControllerProfile::Touch, Chirality::Left);
    input.trigger = 0.71;
    input.squeeze = 0.71;
    let VRControllerState::TouchControllerState(touch) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("touch profile should use touch payload");
    };
    assert!(touch.trigger_click);
    assert!(touch.grip_click);

    let mut input = raw(ActiveControllerProfile::Touch, Chirality::Left);
    input.trigger = 0.69;
    input.squeeze = 0.69;
    let VRControllerState::TouchControllerState(touch) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("touch profile should use touch payload");
    };
    assert!(!touch.trigger_click);
    assert!(!touch.grip_click);
}

#[test]
fn windows_mr_trigger_uses_high_action_threshold() {
    let mut thresholds = OpenxrControllerThresholdState::default();
    let mut input = raw(ActiveControllerProfile::WindowsMr, Chirality::Right);
    input.trigger = 0.94;

    let VRControllerState::WindowsMRControllerState(wmr) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("windows mr profile should use wmr payload");
    };
    assert!(!wmr.trigger_click);

    let mut input = raw(ActiveControllerProfile::WindowsMr, Chirality::Right);
    input.trigger = 0.95;
    let VRControllerState::WindowsMRControllerState(wmr) =
        build_controller_state_with_thresholds(input, &mut thresholds)
    else {
        panic!("windows mr profile should use wmr payload");
    };
    assert!(wmr.trigger_click);
}

#[test]
fn inactive_analog_actions_preserve_explicit_clicks() {
    let mut input = raw(ActiveControllerProfile::Touch, Chirality::Left);
    input.trigger = 0.0;
    input.trigger_active = false;
    input.trigger_click = true;
    input.squeeze = 0.0;
    input.squeeze_active = false;
    input.squeeze_click = true;

    let VRControllerState::TouchControllerState(touch) = build_controller_state(input) else {
        panic!("touch profile should use touch payload");
    };
    assert!(touch.trigger_click);
    assert!(touch.grip_click);
}
