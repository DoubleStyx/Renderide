//! Analog-axis threshold derivation for OpenXR controller state.

use glam::Vec2;

use super::super::profile::ActiveControllerProfile;

const TOUCH_THRESHOLD: f32 = 0.01;
const TRACKPAD_FORCE_TOUCH_THRESHOLD: f32 = 0.01;

const INDEX_GRIP_CLICK: ActionThreshold = ActionThreshold::new(0.30, 0.25);
const DEFAULT_GRIP_CLICK: ActionThreshold = ActionThreshold::new(0.80, 0.70);
const DEFAULT_TRIGGER_CLICK: ActionThreshold = ActionThreshold::new(0.80, 0.70);
const VIVE_TRIGGER_CLICK: ActionThreshold = ActionThreshold::new(0.75, 0.70);
const WINDOWS_MR_TRIGGER_CLICK: ActionThreshold = ActionThreshold::new(0.95, 0.90);

pub(super) fn vec2_nonzero(v: Vec2) -> bool {
    v.length_squared() > 1e-6
}

#[derive(Clone, Copy)]
struct ActionThreshold {
    activate: f32,
    deactivate: f32,
}

impl ActionThreshold {
    const fn new(activate: f32, deactivate: f32) -> Self {
        Self {
            activate,
            deactivate,
        }
    }
}

#[derive(Default)]
struct ThresholdLatch {
    active: bool,
}

impl ThresholdLatch {
    fn sample(&mut self, raw_value: f32, threshold: ActionThreshold) -> bool {
        let value = normalized_action_value(raw_value);
        if self.active {
            self.active = value >= threshold.deactivate;
        } else {
            self.active = value >= threshold.activate;
        }
        self.active
    }

    fn reset(&mut self) {
        self.active = false;
    }
}

/// Stateful OpenXR analog action threshold latches for one controller.
#[derive(Default)]
pub(in crate::xr::input) struct OpenxrControllerThresholdState {
    trigger_click: ThresholdLatch,
    grip_click: ThresholdLatch,
}

/// Raw analog axes and boolean touch hints before threshold expansion.
pub(super) struct OpenxrAnalogAxes {
    /// Active OpenXR interaction profile for the sampled hand.
    pub profile: ActiveControllerProfile,
    /// Trigger analog 0..1.
    pub trigger: f32,
    /// Whether the trigger analog action is bound and active this frame.
    pub trigger_active: bool,
    pub trigger_touch: bool,
    pub trigger_click: bool,
    /// Grip / squeeze analog.
    pub squeeze: f32,
    /// Whether the grip / squeeze analog action is bound and active this frame.
    pub squeeze_active: bool,
    pub squeeze_click: bool,
    pub thumbstick: Vec2,
    pub thumbstick_touch: bool,
    pub trackpad: Vec2,
    pub trackpad_touch: bool,
    pub trackpad_force: f32,
}

/// Host-style booleans inferred from analog axes (Touch / OpenXR conventions).
pub(super) struct OpenxrAxisDerivedButtons {
    pub(super) trigger_touch: bool,
    pub(super) trigger_click: bool,
    pub(super) grip_touch: bool,
    pub(super) grip_click: bool,
    /// Thumbstick deflection or explicit touch bit.
    pub(super) joystick_touch: bool,
    /// Trackpad deflection, touch bit, or force.
    pub(super) touchpad_touch: bool,
}

/// Expands analog thresholds into touch/click flags used across controller profiles.
pub(super) fn derive_openxr_axis_button_flags(
    analog: &OpenxrAnalogAxes,
    thresholds: &mut OpenxrControllerThresholdState,
) -> OpenxrAxisDerivedButtons {
    OpenxrAxisDerivedButtons {
        trigger_touch: analog.trigger_touch || analog.trigger >= TOUCH_THRESHOLD,
        trigger_click: thresholded_trigger_click(analog, thresholds),
        grip_touch: analog.squeeze_click || analog.squeeze >= TOUCH_THRESHOLD,
        grip_click: thresholded_grip_click(analog, thresholds),
        joystick_touch: analog.thumbstick_touch || vec2_nonzero(analog.thumbstick),
        touchpad_touch: analog.trackpad_touch
            || vec2_nonzero(analog.trackpad)
            || analog.trackpad_force >= TRACKPAD_FORCE_TOUCH_THRESHOLD,
    }
}

fn thresholded_trigger_click(
    analog: &OpenxrAnalogAxes,
    thresholds: &mut OpenxrControllerThresholdState,
) -> bool {
    if !analog.trigger_active {
        thresholds.trigger_click.reset();
        return analog.trigger_click;
    }
    thresholds
        .trigger_click
        .sample(analog.trigger, trigger_click_threshold(analog.profile))
}

fn thresholded_grip_click(
    analog: &OpenxrAnalogAxes,
    thresholds: &mut OpenxrControllerThresholdState,
) -> bool {
    if !analog.squeeze_active {
        thresholds.grip_click.reset();
        return analog.squeeze_click;
    }
    thresholds
        .grip_click
        .sample(analog.squeeze, grip_click_threshold(analog.profile))
}

fn trigger_click_threshold(profile: ActiveControllerProfile) -> ActionThreshold {
    match profile {
        ActiveControllerProfile::Index | ActiveControllerProfile::Vive => VIVE_TRIGGER_CLICK,
        ActiveControllerProfile::HpReverbG2 | ActiveControllerProfile::WindowsMr => {
            WINDOWS_MR_TRIGGER_CLICK
        }
        ActiveControllerProfile::Touch
        | ActiveControllerProfile::Pico4
        | ActiveControllerProfile::PicoNeo3
        | ActiveControllerProfile::ViveCosmos
        | ActiveControllerProfile::ViveFocus3
        | ActiveControllerProfile::Generic
        | ActiveControllerProfile::Simple => DEFAULT_TRIGGER_CLICK,
    }
}

fn grip_click_threshold(profile: ActiveControllerProfile) -> ActionThreshold {
    match profile {
        ActiveControllerProfile::Index => INDEX_GRIP_CLICK,
        ActiveControllerProfile::Touch
        | ActiveControllerProfile::Pico4
        | ActiveControllerProfile::PicoNeo3
        | ActiveControllerProfile::HpReverbG2
        | ActiveControllerProfile::ViveCosmos
        | ActiveControllerProfile::ViveFocus3
        | ActiveControllerProfile::Generic
        | ActiveControllerProfile::Simple => DEFAULT_GRIP_CLICK,
        ActiveControllerProfile::Vive | ActiveControllerProfile::WindowsMr => DEFAULT_GRIP_CLICK,
    }
}

fn normalized_action_value(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
