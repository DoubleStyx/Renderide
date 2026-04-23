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

/// Raw analog axes and boolean touch hints before threshold expansion.
pub(super) struct OpenxrAnalogAxes {
    /// Trigger analog 0..1.
    pub trigger: f32,
    pub trigger_touch: bool,
    pub trigger_click: bool,
    /// Grip / squeeze analog.
    pub squeeze: f32,
    pub squeeze_click: bool,
    pub thumbstick: Vec2,
    pub thumbstick_touch: bool,
    pub trackpad: Vec2,
    pub trackpad_touch: bool,
    pub trackpad_force: f32,
}

/// Host-style booleans inferred from analog axes (Touch / OpenXR conventions).
struct OpenxrAxisDerivedButtons {
    trigger_touch: bool,
    trigger_click: bool,
    grip_touch: bool,
    grip_click: bool,
    /// Thumbstick deflection or explicit touch bit.
    joystick_touch: bool,
    /// Trackpad deflection, touch bit, or force.
    touchpad_touch: bool,
}

/// Expands analog thresholds into touch/click flags used across controller profiles.
fn derive_openxr_axis_button_flags(analog: &OpenxrAnalogAxes) -> OpenxrAxisDerivedButtons {
    OpenxrAxisDerivedButtons {
        trigger_touch: analog.trigger_touch || analog.trigger > 0.01,
        trigger_click: analog.trigger_click || analog.trigger > 0.75,
        grip_touch: analog.squeeze_click || analog.squeeze > 0.05,
        grip_click: analog.squeeze_click || analog.squeeze > 0.85,
        joystick_touch: analog.thumbstick_touch || vec2_nonzero(analog.thumbstick),
        touchpad_touch: analog.trackpad_touch
            || vec2_nonzero(analog.trackpad)
            || analog.trackpad_force > 0.01,
    }
}

pub(super) fn body_node_for_side(side: Chirality) -> BodyNode {
    match side {
        Chirality::Left => BodyNode::LeftController,
        Chirality::Right => BodyNode::RightController,
    }
}

/// Per-profile inputs after [`derive_openxr_axis_button_flags`] (grip/joystick/touchpad touch bits).
///
/// Bundles everything needed by the profile-specific `openxr_*_controller_state` builders.
struct OpenxrHostControllerCtx {
    frame: ControllerFrame,
    is_tracking: bool,
    device_id: Option<String>,
    device_model: Option<String>,
    side: Chirality,
    body_node: BodyNode,
    trigger: f32,
    trigger_touch: bool,
    trigger_click: bool,
    squeeze: f32,
    squeeze_click: bool,
    grip_touch: bool,
    grip_click: bool,
    joystick_touch: bool,
    touchpad_touch: bool,
    thumbstick: Vec2,
    thumbstick_click: bool,
    trackpad: Vec2,
    trackpad_click: bool,
    trackpad_force: f32,
    primary: bool,
    secondary: bool,
    primary_touch: bool,
    secondary_touch: bool,
    menu: bool,
    thumbrest_touch: bool,
    select: bool,
}

/// Oculus Touch–class mapping (and shared by [`ActiveControllerProfile::Generic`]).
fn host_state_touch_class_profile(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    openxr_touch_class_controller_state(ctx)
}

/// Valve Index mapping (thumbsticks + trackpads).
fn host_state_index_profile(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    openxr_index_controller_state(ctx)
}

/// HTC Vive wand mapping.
fn host_state_vive_profile(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    openxr_vive_controller_state(ctx)
}

/// Windows Mixed Reality mapping.
fn host_state_windows_mr_profile(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    openxr_windows_mr_controller_state(ctx)
}

/// Khronos simple controller profile.
fn host_state_simple_profile(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    openxr_simple_controller_state(ctx)
}

/// Dispatches to the concrete [`VRControllerState`] constructor for the active interaction profile.
///
/// Pico 4, Pico Neo3, HP Reverb G2, Vive Cosmos, and Vive Focus 3 all share the touch-class
/// payload because the host wire format has no dedicated variants for them. The `device_model`
/// string from [`super::profile::device_label`] is what tells the host which physical
/// controller the payload represents.
fn dispatch_openxr_profile_to_host_state(
    profile: ActiveControllerProfile,
    ctx: OpenxrHostControllerCtx,
) -> VRControllerState {
    match profile {
        ActiveControllerProfile::Touch
        | ActiveControllerProfile::Pico4
        | ActiveControllerProfile::PicoNeo3
        | ActiveControllerProfile::HpReverbG2
        | ActiveControllerProfile::ViveCosmos
        | ActiveControllerProfile::ViveFocus3
        | ActiveControllerProfile::Generic => host_state_touch_class_profile(ctx),
        ActiveControllerProfile::Index => host_state_index_profile(ctx),
        ActiveControllerProfile::Vive => host_state_vive_profile(ctx),
        ActiveControllerProfile::WindowsMr => host_state_windows_mr_profile(ctx),
        ActiveControllerProfile::Simple => host_state_simple_profile(ctx),
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
/// [`ActiveControllerProfile::Generic`] is encoded as [`VRControllerState::TouchControllerState`]
/// so the host input stack receives the same polymorphic shape as Touch controllers (Quest-class
/// paths) instead of [`VRControllerState::GenericControllerState`], which would deserialize to a
/// different controller type on the host.
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

/// Oculus Touch–class layout shared with [`ActiveControllerProfile::Generic`] (Quest-shaped host payload).
fn openxr_touch_class_controller_state(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    let OpenxrHostControllerCtx {
        frame,
        is_tracking,
        device_id,
        device_model,
        side,
        body_node,
        trigger,
        trigger_touch,
        trigger_click,
        squeeze,
        squeeze_click: _,
        grip_touch: _,
        grip_click,
        joystick_touch,
        touchpad_touch: _,
        thumbstick,
        thumbstick_click,
        trackpad: _,
        trackpad_click: _,
        trackpad_force: _,
        primary,
        secondary,
        primary_touch,
        secondary_touch,
        menu,
        thumbrest_touch,
        select: _,
    } = ctx;
    VRControllerState::TouchControllerState(TouchControllerState {
        model: TouchControllerModel::QuestAndRiftS,
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

fn openxr_index_controller_state(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    let OpenxrHostControllerCtx {
        frame,
        is_tracking,
        device_id,
        device_model,
        side,
        body_node,
        trigger,
        trigger_touch,
        trigger_click,
        squeeze,
        squeeze_click: _,
        grip_touch,
        grip_click,
        joystick_touch,
        touchpad_touch,
        thumbstick,
        thumbstick_click,
        trackpad,
        trackpad_click,
        trackpad_force,
        primary,
        secondary,
        primary_touch,
        secondary_touch,
        menu: _,
        thumbrest_touch: _,
        select: _,
    } = ctx;
    VRControllerState::IndexControllerState(IndexControllerState {
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

fn openxr_vive_controller_state(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    let OpenxrHostControllerCtx {
        frame,
        is_tracking,
        device_id,
        device_model,
        side,
        body_node,
        trigger,
        trigger_touch,
        trigger_click,
        squeeze,
        squeeze_click,
        grip_touch: _,
        grip_click: _,
        joystick_touch: _,
        touchpad_touch,
        thumbstick: _,
        thumbstick_click: _,
        trackpad,
        trackpad_click,
        trackpad_force: _,
        primary: _,
        secondary: _,
        primary_touch: _,
        secondary_touch: _,
        menu,
        thumbrest_touch: _,
        select: _,
    } = ctx;
    VRControllerState::ViveControllerState(ViveControllerState {
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

fn openxr_windows_mr_controller_state(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    let OpenxrHostControllerCtx {
        frame,
        is_tracking,
        device_id,
        device_model,
        side,
        body_node,
        trigger,
        trigger_touch,
        trigger_click,
        squeeze,
        squeeze_click,
        grip_touch: _,
        grip_click: _,
        joystick_touch: _,
        touchpad_touch,
        thumbstick,
        thumbstick_click,
        trackpad,
        trackpad_click,
        trackpad_force: _,
        primary: _,
        secondary: _,
        primary_touch: _,
        secondary_touch: _,
        menu,
        thumbrest_touch: _,
        select: _,
    } = ctx;
    VRControllerState::WindowsMRControllerState(WindowsMRControllerState {
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

fn openxr_simple_controller_state(ctx: OpenxrHostControllerCtx) -> VRControllerState {
    let OpenxrHostControllerCtx {
        frame,
        is_tracking,
        device_id,
        device_model,
        side,
        body_node,
        trigger,
        trigger_touch,
        trigger_click: _,
        squeeze: _,
        squeeze_click: _,
        grip_touch: _,
        grip_click,
        joystick_touch: _,
        touchpad_touch: _,
        thumbstick: _,
        thumbstick_click: _,
        trackpad: _,
        trackpad_click: _,
        trackpad_force: _,
        primary: _,
        secondary: _,
        primary_touch: _,
        secondary_touch: _,
        menu,
        thumbrest_touch: _,
        select,
    } = ctx;
    VRControllerState::GenericControllerState(GenericControllerState {
        strength: if select { 1.0 } else { trigger },
        axis: Vec2::ZERO,
        touching_strength: trigger_touch || select,
        touching_axis: false,
        primary: select,
        menu,
        grab: grip_click,
        secondary: false,
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
