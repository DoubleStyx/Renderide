//! Per-profile host `VRControllerState` payload builders.

use glam::Vec2;

use crate::shared::{
    BodyNode, Chirality, IndexControllerState, TouchControllerModel, TouchControllerState,
    VRControllerState, ViveControllerState, WindowsMRControllerState,
};

use super::super::frame::ControllerFrame;
use super::super::profile::ActiveControllerProfile;

/// Per-profile inputs after [`super::axis::derive_openxr_axis_button_flags`].
///
/// Bundles everything needed by the profile-specific `openxr_*_controller_state` builders.
pub(super) struct OpenxrHostControllerCtx {
    pub(super) frame: ControllerFrame,
    pub(super) is_tracking: bool,
    pub(super) device_id: Option<String>,
    pub(super) device_model: Option<String>,
    pub(super) side: Chirality,
    pub(super) body_node: BodyNode,
    pub(super) trigger: f32,
    pub(super) trigger_touch: bool,
    pub(super) trigger_click: bool,
    pub(super) squeeze: f32,
    pub(super) squeeze_click: bool,
    pub(super) grip_touch: bool,
    pub(super) grip_click: bool,
    pub(super) joystick_touch: bool,
    pub(super) touchpad_touch: bool,
    pub(super) thumbstick: Vec2,
    pub(super) thumbstick_click: bool,
    pub(super) trackpad: Vec2,
    pub(super) trackpad_click: bool,
    pub(super) trackpad_force: f32,
    pub(super) primary: bool,
    pub(super) secondary: bool,
    pub(super) primary_touch: bool,
    pub(super) secondary_touch: bool,
    pub(super) menu: bool,
    pub(super) thumbrest_touch: bool,
    pub(super) select: bool,
}

/// Oculus Touch-class mapping (also used by profiles without a dedicated host variant).
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

/// Dispatches to the concrete [`VRControllerState`] constructor for the active interaction profile.
///
/// Every profile without a dedicated host variant (Pico 4, Pico Neo3, HP Reverb G2, Vive Cosmos,
/// Vive Focus 3, Generic, Simple) routes through the touch-class payload. Holding the wire
/// variant constant across profile transitions is what prevents the host's per-`device_id`
/// controller cache from throwing `InvalidCastException` when OpenXR reports a transient
/// unbound profile after the user has already been assigned a concrete one. The
/// [`super::super::profile::device_label`] string is what tells the host which physical controller
/// the payload represents.
pub(super) fn dispatch_openxr_profile_to_host_state(
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
        | ActiveControllerProfile::Generic
        | ActiveControllerProfile::Simple => host_state_touch_class_profile(ctx),
        ActiveControllerProfile::Index => host_state_index_profile(ctx),
        ActiveControllerProfile::Vive => host_state_vive_profile(ctx),
        ActiveControllerProfile::WindowsMr => host_state_windows_mr_profile(ctx),
    }
}

/// Oculus Touch-class layout; the Quest-shaped host payload used by every OpenXR profile that
/// lacks a dedicated host [`VRControllerState`] variant.
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
        select,
    } = ctx;
    // The Khronos Simple profile only exposes `/input/select/click` and `/input/menu/click`, so
    // fold `select` into the Touch-class trigger/click channels. On profiles that bind trigger
    // directly this is a no-op (select is false).
    let trigger = trigger.max(if select { 1.0 } else { 0.0 });
    let trigger_touch = trigger_touch || select;
    let trigger_click = trigger_click || select;
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
