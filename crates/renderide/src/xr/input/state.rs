//! Maps OpenXR action state and resolved poses into host [`crate::shared::VRControllerState`].

mod axis;
mod builders;

use glam::Vec2;

use crate::shared::{BodyNode, Chirality, VRControllerState};

use super::frame::ControllerFrame;
use super::profile::{ActiveControllerProfile, device_label};

use axis::{OpenxrAnalogAxes, derive_openxr_axis_button_flags};
use builders::{OpenxrHostControllerCtx, dispatch_openxr_profile_to_host_state};

pub(super) fn body_node_for_side(side: Chirality) -> BodyNode {
    match side {
        Chirality::Left => BodyNode::LeftController,
        Chirality::Right => BodyNode::RightController,
    }
}

/// Polled OpenXR actions and profile for [`build_controller_state`].
pub(super) struct OpenxrControllerRawInputs {
    pub profile: ActiveControllerProfile,
    pub side: Chirality,
    pub is_tracking: bool,
    pub frame: ControllerFrame,
    pub trigger: f32,
    pub trigger_touch: bool,
    pub trigger_click: bool,
    pub squeeze: f32,
    pub squeeze_click: bool,
    pub thumbstick: Vec2,
    pub thumbstick_touch: bool,
    pub thumbstick_click: bool,
    pub trackpad: Vec2,
    pub trackpad_touch: bool,
    pub trackpad_click: bool,
    pub trackpad_force: f32,
    pub primary: bool,
    pub secondary: bool,
    pub primary_touch: bool,
    pub secondary_touch: bool,
    pub menu: bool,
    pub thumbrest_touch: bool,
    pub select: bool,
}

/// Maps the active OpenXR profile to a host [`VRControllerState`] variant.
///
/// Every profile that lacks a dedicated host variant -- including
/// [`ActiveControllerProfile::Generic`] and [`ActiveControllerProfile::Simple`] -- is encoded as
/// [`VRControllerState::TouchControllerState`]. The host caches controllers by `device_id` and
/// casts the cached instance to the incoming variant's type; emitting the same polymorphic
/// shape across profile transitions is what keeps that cast valid when OpenXR transiently
/// reports an unbound profile (`xr::Path::NULL`) after a concrete profile was already observed.
pub(super) fn build_controller_state(inputs: OpenxrControllerRawInputs) -> VRControllerState {
    let device_id = Some(match inputs.side {
        Chirality::Left => "OpenXR Left".to_string(),
        Chirality::Right => "OpenXR Right".to_string(),
    });
    let device_model = Some(device_label(inputs.profile).to_string());
    let body_node = body_node_for_side(inputs.side);
    let derived = derive_openxr_axis_button_flags(&OpenxrAnalogAxes {
        trigger: inputs.trigger,
        trigger_touch: inputs.trigger_touch,
        trigger_click: inputs.trigger_click,
        squeeze: inputs.squeeze,
        squeeze_click: inputs.squeeze_click,
        thumbstick: inputs.thumbstick,
        thumbstick_touch: inputs.thumbstick_touch,
        trackpad: inputs.trackpad,
        trackpad_touch: inputs.trackpad_touch,
        trackpad_force: inputs.trackpad_force,
    });
    dispatch_openxr_profile_to_host_state(
        inputs.profile,
        OpenxrHostControllerCtx {
            frame: inputs.frame,
            is_tracking: inputs.is_tracking,
            device_id,
            device_model,
            side: inputs.side,
            body_node,
            trigger: inputs.trigger,
            trigger_touch: derived.trigger_touch,
            trigger_click: derived.trigger_click,
            squeeze: inputs.squeeze,
            squeeze_click: inputs.squeeze_click,
            grip_touch: derived.grip_touch,
            grip_click: derived.grip_click,
            joystick_touch: derived.joystick_touch,
            touchpad_touch: derived.touchpad_touch,
            thumbstick: inputs.thumbstick,
            thumbstick_click: inputs.thumbstick_click,
            trackpad: inputs.trackpad,
            trackpad_click: inputs.trackpad_click,
            trackpad_force: inputs.trackpad_force,
            primary: inputs.primary,
            secondary: inputs.secondary,
            primary_touch: inputs.primary_touch,
            secondary_touch: inputs.secondary_touch,
            menu: inputs.menu,
            thumbrest_touch: inputs.thumbrest_touch,
            select: inputs.select,
        },
    )
}

#[cfg(test)]
mod tests;
