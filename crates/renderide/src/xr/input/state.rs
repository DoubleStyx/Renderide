//! Maps OpenXR action state and resolved poses into host [`crate::shared::VRControllerState`].

use glam::Vec2;

use crate::shared::{
    BodyNode, Chirality, GenericControllerState, IndexControllerState, TouchControllerModel,
    TouchControllerState, VRControllerState, ViveControllerState, WindowsMRControllerState,
};

use super::frame::ControllerFrame;
use super::profile::{device_label, ActiveControllerProfile};

pub(super) fn vec2_nonzero(v: Vec2) -> bool {
    v.length_squared() > 1e-6
}

pub(super) fn choose_axis(thumbstick: Vec2, trackpad: Vec2) -> Vec2 {
    if vec2_nonzero(thumbstick) {
        thumbstick
    } else {
        trackpad
    }
}

pub(super) fn body_node_for_side(side: Chirality) -> BodyNode {
    match side {
        Chirality::left => BodyNode::left_controller,
        Chirality::right => BodyNode::right_controller,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_controller_state(
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
