//! Synthesises per-finger [`HandState`] data from controller input.
//!
//! Without this, the host would receive no hand-tracking data and its `HandPoser` would reset
//! every finger to its `OriginalRotation`, leaving the avatar playing the desktop idle pose while
//! the user is in VR.
//!
//! The presets below match the host's `Idle` and `Fist` finger-pose presets.
//! Segment layout matches the host's unpack loop: 24 entries indexed by
//! `BodyNode - LeftThumbMetacarpal`, ordered Thumb(Met,Prox,Dist,Tip), Index(Met,Prox,Inter,Dist,Tip),
//! Middle(..), Ring(..), Pinky(..). Right-hand [`HandState`] values reuse the same indexing but hold
//! right-hand data; the host mirrors via `bodyNode.GetSide(chirality)`.
//!
//! The curl blend is deliberately conservative: metacarpals are left at idle and we set
//! [`HandState::tracks_metacarpals`] to `false`, so the host overrides non-thumb metacarpals to the
//! avatar's own rest pose. Thumb is held at idle; index curl follows the trigger analog;
//! middle/ring/pinky follow the squeeze (grip) analog.

#[cfg(test)]
mod model_space_presets;
mod presets;

use glam::{Quat, Vec3};

use crate::shared::{Chirality, HandState, VRControllerState};

use presets::{
    FIST_POS_LEFT, FIST_POS_RIGHT, FIST_ROT_LEFT, FIST_ROT_RIGHT, IDLE_POS_LEFT, IDLE_POS_RIGHT,
    IDLE_ROT_LEFT, IDLE_ROT_RIGHT, LEFT_HAND_ID, RIGHT_HAND_ID, SEGMENT_COUNTS,
    TOTAL_SEGMENT_COUNT,
};

/// Which finger a [`HandState`] segment index (0..24) belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FingerKind {
    /// Thumb: segments 0..=3 (Metacarpal, Proximal, Distal, Tip -- no Intermediate).
    Thumb,
    /// Index finger: segments 4..=8.
    Index,
    /// Middle finger: segments 9..=13.
    Middle,
    /// Ring finger: segments 14..=18.
    Ring,
    /// Pinky: segments 19..=23.
    Pinky,
}

impl From<u8> for FingerKind {
    fn from(value: u8) -> Self {
        match value % 5 {
            0 => Self::Thumb,
            1 => Self::Index,
            2 => Self::Middle,
            3 => Self::Ring,
            4 => Self::Pinky,
            _ => Self::Pinky,
        }
    }
}

/// Controller-derived inputs used to drive the idle<->fist blend.
struct ControllerCurlInputs {
    /// Which hand this controller drives.
    side: Chirality,
    /// Tracking-space wrist position to report on [`HandState::wrist_position`]. When the runtime
    /// advertises a bound hand, this is the controller pose composed with the per-profile
    /// bound-hand offset (`controller.position + controller.rotation * controller.hand_position`),
    /// matching `TrackedDevicePositioner`'s own resolution of the
    /// `MappableTrackedObject.BodyNodePositionOffset` path in FrooxEngine. Otherwise it is the
    /// controller's tracking-space pose directly. `hand_position` / `hand_rotation` on
    /// [`VRControllerState`] are registration-time offsets (see
    /// [`crate::xr::input::pose::hand_pose_defaults`]), not tracking-space poses.
    wrist_position: Vec3,
    /// Tracking-space wrist rotation to report on [`HandState::wrist_rotation`]. Composed the same
    /// way as [`Self::wrist_position`] and normalised.
    wrist_rotation: Quat,
    /// Grip/squeeze analog in 0..=1. Drives middle, ring, and pinky curl.
    grip: f32,
    /// Trigger analog in 0..=1. Drives index finger curl.
    trigger: f32,
    /// Thumb boolean resting state. Drives thumb finger curl.
    thumb: bool,
}

/// Profile-agnostic pose + analog inputs feeding curl synthesis.
///
/// Each `VRControllerState` variant has the same set of pose fields and an analog trigger.
/// There are only a handful of profile-specific bits:
/// - `grip` is `f32` on Touch / Index but `bool` on Vive / WMR.
/// - `thumb` is conditioned on different sensors depending on controllers.
///
/// Match arms in [`extract_curl_inputs`] pre-coerce grip into `0.0` / `1.0` for the boolean
/// profiles, then call into [`curl_inputs_from_source`] uniformly.
struct ControllerCurlSource {
    is_tracking: bool,
    side: Chirality,
    position: Vec3,
    rotation: Quat,
    has_bound_hand: bool,
    hand_position: Vec3,
    hand_rotation: Quat,
    grip: f32,
    trigger: f32,
    thumb: bool,
}

/// Builds [`ControllerCurlSource`] from a profile-specific controller state with an explicit
/// grip expression. The other pose fields share the same names across every profile struct so
/// they forward 1:1.
macro_rules! curl_source {
    ($s:expr, grip = $grip:expr, thumb = $thumb:expr) => {
        ControllerCurlSource {
            is_tracking: $s.is_tracking,
            side: $s.side,
            position: $s.position,
            rotation: $s.rotation,
            has_bound_hand: $s.has_bound_hand,
            hand_position: $s.hand_position,
            hand_rotation: $s.hand_rotation,
            grip: $grip,
            trigger: $s.trigger,
            thumb: $thumb,
        }
    };
}

/// Builds [`ControllerCurlInputs`] from a profile-agnostic [`ControllerCurlSource`].
///
/// Returns `None` when the controller is not tracked (we do not want to feed the host random
/// hand poses). Grip and trigger are clamped to `0..=1` here so callers do not need to.
fn curl_inputs_from_source(src: ControllerCurlSource) -> Option<ControllerCurlInputs> {
    if !src.is_tracking {
        return None;
    }
    let (wrist_position, wrist_rotation) = if src.has_bound_hand {
        (
            src.position + src.rotation * src.hand_position,
            (src.rotation * src.hand_rotation).normalize(),
        )
    } else {
        (src.position, src.rotation)
    };
    Some(ControllerCurlInputs {
        side: src.side,
        wrist_position,
        wrist_rotation,
        grip: src.grip.clamp(0.0, 1.0),
        trigger: src.trigger.clamp(0.0, 1.0),
        thumb: src.thumb,
    })
}

/// Extracts the curl-driving inputs from a [`VRControllerState`] variant.
///
/// Returns `None` when the controller is not tracked or when the variant is not produced by
/// the current OpenXR dispatch. For controllers whose grip is a boolean (Vive wand, WMR), the
/// boolean is coerced to `0.0` / `1.0` before clamping.
fn extract_curl_inputs(controller: &VRControllerState) -> Option<ControllerCurlInputs> {
    match controller {
        VRControllerState::TouchControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = s.grip,
            thumb = s.thumbrest_touch || s.button_xa_touch || s.button_yb_touch || s.joystick_touch
        )),
        VRControllerState::IndexControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = s.grip,
            thumb = s.touchpad_touch || s.button_atouch || s.button_btouch || s.joystick_touch
        )),
        VRControllerState::ViveControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = if s.grip { 1.0 } else { 0.0 },
            thumb = s.touchpad_touch
        )),
        VRControllerState::WindowsMRControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = if s.grip { 1.0 } else { 0.0 },
            thumb = s.touchpad_touch
        )),
        VRControllerState::CosmosControllerState(_)
        | VRControllerState::GenericControllerState(_)
        | VRControllerState::HPReverbControllerState(_)
        | VRControllerState::PicoNeo2ControllerState(_) => {
            // These variants are not produced by the current OpenXR dispatch
            // (`crate::xr::input::state::dispatch_openxr_profile_to_host_state`). If they start
            // being emitted, add the analogous extractor here.
            None
        }
    }
}

/// Returns the idle<->fist blend factor for a given finger.
///
/// - Metacarpals are held at idle (`0.0`). Non-thumb metacarpals are overridden on the
///   host anyway when [`HandState::tracks_metacarpals`] is `false`, so their blend does not matter.
/// - Index curl follows `trigger`.
/// - Middle, ring, and pinky curl follow `grip`.
/// - Thumb curl follows `thumb`.
fn blend_factor_for_segment(finger: FingerKind, grip: f32, trigger: f32, thumb: f32) -> f32 {
    match finger {
        FingerKind::Thumb => thumb,
        FingerKind::Index => trigger,
        FingerKind::Middle | FingerKind::Ring | FingerKind::Pinky => grip,
    }
}

/// Builds a [`HandState`] for one controller by blending the idle and fist presets. Returns
/// `None` if the controller is untracked or not a variant we drive hands for.
fn synthesize_one_hand(controller: &VRControllerState) -> Option<HandState> {
    let inputs = extract_curl_inputs(controller)?;
    let thumb = if inputs.thumb { 1.0 } else { 0.0 };
    let (pos_idle, rot_idle, pos_fist, rot_fist, unique_id) = match inputs.side {
        Chirality::Left => (
            &IDLE_POS_LEFT,
            &IDLE_ROT_LEFT,
            &FIST_POS_LEFT,
            &FIST_ROT_LEFT,
            LEFT_HAND_ID,
        ),
        Chirality::Right => (
            &IDLE_POS_RIGHT,
            &IDLE_ROT_RIGHT,
            &FIST_POS_RIGHT,
            &FIST_ROT_RIGHT,
            RIGHT_HAND_ID,
        ),
    };
    let mut segment_positions = Vec::with_capacity(TOTAL_SEGMENT_COUNT);
    let mut segment_rotations = Vec::with_capacity(TOTAL_SEGMENT_COUNT);
    for (finger_index, &finger_segment_count) in SEGMENT_COUNTS.iter().enumerate() {
        let finger = FingerKind::from(finger_index as u8);
        let t = blend_factor_for_segment(finger, inputs.grip, inputs.trigger, thumb);
        let mut position = Vec3::ZERO;
        let mut rotation = Quat::IDENTITY;
        for i in 0..finger_segment_count {
            let pi = Vec3::from_array(pos_idle[finger_index][i]);
            let pf = Vec3::from_array(pos_fist[finger_index][i]);
            position += pi.lerp(pf, t);
            segment_positions.push(position);

            let ri = Quat::from_array(rot_idle[finger_index][i]);
            let rf = Quat::from_array(rot_fist[finger_index][i]);
            rotation = rotation.mul_quat(ri.slerp(rf, t));
            segment_rotations.push(rotation);
        }
    }
    Some(HandState {
        unique_id: Some(unique_id.to_string()),
        priority: 0,
        chirality: inputs.side,
        is_device_active: true,
        is_tracking: true,
        tracks_metacarpals: false,
        confidence: 1.0,
        wrist_position: inputs.wrist_position,
        wrist_rotation: inputs.wrist_rotation,
        segment_positions,
        segment_rotations,
    })
}

/// Produces one [`HandState`] per tracked VR controller in `controllers`, blending the idle and
/// fist presets using the controller's grip and trigger analogs.
///
/// Call this every XR frame after building the per-controller [`VRControllerState`] slice; the
/// returned vector belongs on [`crate::shared::VRInputsState::hands`].
pub fn synthesize_hand_states(controllers: &[VRControllerState]) -> Vec<HandState> {
    controllers.iter().filter_map(synthesize_one_hand).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{BodyNode, TouchControllerModel, TouchControllerState};

    fn touch_controller(
        side: Chirality,
        is_tracking: bool,
        grip: f32,
        trigger: f32,
        thumb: bool,
    ) -> VRControllerState {
        VRControllerState::TouchControllerState(touch_controller_state(
            side,
            is_tracking,
            grip,
            trigger,
            thumb,
        ))
    }

    fn touch_controller_state(
        side: Chirality,
        is_tracking: bool,
        grip: f32,
        trigger: f32,
        thumb: bool,
    ) -> TouchControllerState {
        TouchControllerState {
            model: TouchControllerModel::QuestAndRiftS,
            start: false,
            button_yb: false,
            button_xa: false,
            button_yb_touch: false,
            button_xa_touch: false,
            thumbrest_touch: thumb,
            grip,
            grip_click: false,
            joystick_raw: glam::Vec2::ZERO,
            joystick_touch: false,
            joystick_click: false,
            trigger,
            trigger_touch: false,
            trigger_click: false,
            device_id: None,
            device_model: None,
            side,
            body_node: match side {
                Chirality::Left => BodyNode::LeftController,
                Chirality::Right => BodyNode::RightController,
            },
            is_device_active: true,
            is_tracking,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            has_bound_hand: false,
            hand_position: Vec3::ZERO,
            hand_rotation: Quat::IDENTITY,
            battery_level: 1.0,
            battery_charging: false,
        }
    }

    fn idle_tracked_touch_controller_state() -> TouchControllerState {
        touch_controller_state(Chirality::Left, true, 0.0, 0.0, false)
    }

    #[test]
    fn produces_one_hand_per_tracked_controller() {
        let controllers = vec![
            touch_controller(Chirality::Left, true, 0.0, 0.0, false),
            touch_controller(Chirality::Right, true, 0.0, 0.0, false),
        ];
        let hands = synthesize_hand_states(&controllers);
        assert_eq!(hands.len(), 2);
        assert_eq!(hands[0].chirality, Chirality::Left);
        assert_eq!(hands[1].chirality, Chirality::Right);
    }

    #[test]
    fn skips_untracked_controllers() {
        let controllers = vec![
            touch_controller(Chirality::Left, false, 0.0, 0.0, false),
            touch_controller(Chirality::Right, true, 0.0, 0.0, false),
        ];
        let hands = synthesize_hand_states(&controllers);
        assert_eq!(hands.len(), 1);
        assert_eq!(hands[0].chirality, Chirality::Right);
    }

    #[test]
    fn segment_arrays_have_host_expected_length() {
        let hands =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.5, 0.5, false)]);
        let hand = &hands[0];
        assert_eq!(hand.segment_positions.len(), TOTAL_SEGMENT_COUNT);
        assert_eq!(hand.segment_rotations.len(), TOTAL_SEGMENT_COUNT);
        assert!(hand.is_tracking);
        assert!(!hand.tracks_metacarpals);
    }

    #[test]
    fn trigger_bends_index_but_not_other_fingers() {
        let idle =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
                .remove(0);
        let full_trigger =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 1.0, false)])
                .remove(0);
        let index_delta = (full_trigger.segment_rotations[6].to_array()[3]
            - idle.segment_rotations[6].to_array()[3])
            .abs();
        let middle_delta = (full_trigger.segment_rotations[11].to_array()[3]
            - idle.segment_rotations[11].to_array()[3])
            .abs();
        let thumb_delta = (full_trigger.segment_rotations[1].to_array()[3]
            - idle.segment_rotations[1].to_array()[3])
            .abs();
        assert!(
            index_delta > 0.05,
            "trigger should bend the index finger proximal joint (delta={index_delta})"
        );
        assert!(
            middle_delta < 1e-5,
            "trigger must not move the middle finger (delta={middle_delta})"
        );
        assert!(
            thumb_delta < 1e-5,
            "trigger must not move the thumb (delta={thumb_delta})"
        );
    }

    #[test]
    fn grip_bends_middle_ring_pinky_but_not_index_or_thumb() {
        let idle =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
                .remove(0);
        let full_grip =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 1.0, 0.0, false)])
                .remove(0);
        let middle_delta = (full_grip.segment_rotations[11].to_array()[3]
            - idle.segment_rotations[11].to_array()[3])
            .abs();
        let ring_delta = (full_grip.segment_rotations[16].to_array()[3]
            - idle.segment_rotations[16].to_array()[3])
            .abs();
        let pinky_delta = (full_grip.segment_rotations[21].to_array()[3]
            - idle.segment_rotations[21].to_array()[3])
            .abs();
        let index_delta = (full_grip.segment_rotations[6].to_array()[3]
            - idle.segment_rotations[6].to_array()[3])
            .abs();
        let thumb_delta = (full_grip.segment_rotations[1].to_array()[3]
            - idle.segment_rotations[1].to_array()[3])
            .abs();
        assert!(
            middle_delta > 0.05,
            "grip should bend middle (delta={middle_delta})"
        );
        assert!(
            ring_delta > 0.05,
            "grip should bend ring (delta={ring_delta})"
        );
        assert!(
            pinky_delta > 0.05,
            "grip should bend pinky (delta={pinky_delta})"
        );
        assert!(
            index_delta < 1e-5,
            "grip must not move the index finger (delta={index_delta})"
        );
        assert!(
            thumb_delta < 1e-5,
            "grip must not move the thumb (delta={thumb_delta})"
        );
    }

    #[test]
    fn left_and_right_hands_differ() {
        let hands = synthesize_hand_states(&[
            touch_controller(Chirality::Left, true, 0.5, 0.5, false),
            touch_controller(Chirality::Right, true, 0.5, 0.5, false),
        ]);
        let left_index_met_x = hands[0].segment_positions[4].x;
        let right_index_met_x = hands[1].segment_positions[4].x;
        assert!(
            (left_index_met_x - right_index_met_x).abs() > 1e-4,
            "left/right hand index metacarpals must use different preset data"
        );
        assert!(
            left_index_met_x.signum() != right_index_met_x.signum(),
            "left hand metacarpal x should be positive, right hand negative \
             (left={left_index_met_x}, right={right_index_met_x})"
        );
        assert_eq!(
            hands[0].unique_id.as_deref(),
            Some(LEFT_HAND_ID),
            "left hand should use stable LEFT_HAND_ID"
        );
        assert_eq!(
            hands[1].unique_id.as_deref(),
            Some(RIGHT_HAND_ID),
            "right hand should use stable RIGHT_HAND_ID"
        );
    }

    #[test]
    fn thumb_does_not_curl_other_fingers() {
        let idle =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
                .remove(0);
        let full_grip =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, true)])
                .remove(0);
        let middle_delta = (full_grip.segment_rotations[11].to_array()[3]
            - idle.segment_rotations[11].to_array()[3])
            .abs();
        let ring_delta = (full_grip.segment_rotations[16].to_array()[3]
            - idle.segment_rotations[16].to_array()[3])
            .abs();
        let pinky_delta = (full_grip.segment_rotations[21].to_array()[3]
            - idle.segment_rotations[21].to_array()[3])
            .abs();
        let index_delta = (full_grip.segment_rotations[6].to_array()[3]
            - idle.segment_rotations[6].to_array()[3])
            .abs();
        let thumb_delta = (full_grip.segment_rotations[1].to_array()[3]
            - idle.segment_rotations[1].to_array()[3])
            .abs();
        assert!(
            middle_delta < 1e-5,
            "thumb must not move middle (delta={middle_delta})"
        );
        assert!(
            ring_delta < 1e-5,
            "thumb must not move ring (delta={ring_delta})"
        );
        assert!(
            pinky_delta < 1e-5,
            "thumb must not move pinky (delta={pinky_delta})"
        );
        assert!(
            index_delta < 1e-5,
            "thumb must not move index (delta={index_delta})"
        );
        assert!(
            thumb_delta > 0.05,
            "thumb should bend (delta={thumb_delta})"
        );
    }

    fn touch_controller_with_pose(
        side: Chirality,
        position: Vec3,
        rotation: Quat,
        has_bound_hand: bool,
        hand_position: Vec3,
        hand_rotation: Quat,
    ) -> VRControllerState {
        VRControllerState::TouchControllerState(TouchControllerState {
            model: TouchControllerModel::QuestAndRiftS,
            start: false,
            button_yb: false,
            button_xa: false,
            button_yb_touch: false,
            button_xa_touch: false,
            thumbrest_touch: false,
            grip: 0.0,
            grip_click: false,
            joystick_raw: glam::Vec2::ZERO,
            joystick_touch: false,
            joystick_click: false,
            trigger: 0.0,
            trigger_touch: false,
            trigger_click: false,
            device_id: None,
            device_model: None,
            side,
            body_node: match side {
                Chirality::Left => BodyNode::LeftController,
                Chirality::Right => BodyNode::RightController,
            },
            is_device_active: true,
            is_tracking: true,
            position,
            rotation,
            has_bound_hand,
            hand_position,
            hand_rotation,
            battery_level: 1.0,
            battery_charging: false,
        })
    }

    #[test]
    fn bound_hand_wrist_is_controller_pose_composed_with_offset() {
        let position = Vec3::new(0.3, 1.4, -0.5);
        let rotation = Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2);
        let rotation = rotation.normalize();
        let hand_position = Vec3::new(-0.04, -0.025, -0.1);
        let hand_rotation = Quat::from_rotation_y(-1.57) * Quat::from_rotation_x(0.3);
        let hand_rotation = hand_rotation.normalize();

        let hands = synthesize_hand_states(&[touch_controller_with_pose(
            Chirality::Left,
            position,
            rotation,
            true,
            hand_position,
            hand_rotation,
        )]);
        let hand = &hands[0];

        let expected_pos = position + rotation * hand_position;
        let expected_rot = (rotation * hand_rotation).normalize();
        assert!(
            (hand.wrist_position - expected_pos).length() < 1e-5,
            "wrist_position should compose controller pose with bound-hand offset: \
             got {:?} expected {expected_pos:?}",
            hand.wrist_position,
        );
        assert!(
            hand.wrist_rotation.dot(expected_rot).abs() > 1.0 - 1e-5,
            "wrist_rotation should be (controller.rotation * hand_rotation).normalize(): \
             got {:?} expected {expected_rot:?}",
            hand.wrist_rotation,
        );
        assert!(
            hand.wrist_position.length() > 0.5,
            "wrist should be near the controller's tracking-space position, \
             not pinned near the origin (got {:?})",
            hand.wrist_position,
        );
    }

    #[test]
    fn unbound_hand_wrist_matches_controller_pose() {
        let position = Vec3::new(-0.2, 1.2, -0.3);
        let rotation = Quat::from_rotation_y(-0.4).normalize();
        let hands = synthesize_hand_states(&[touch_controller_with_pose(
            Chirality::Right,
            position,
            rotation,
            false,
            Vec3::ZERO,
            Quat::IDENTITY,
        )]);
        let hand = &hands[0];
        assert_eq!(hand.wrist_position, position);
        assert_eq!(hand.wrist_rotation, rotation);
    }

    #[test]
    fn touch_clamps_out_of_range_grip_and_trigger() {
        let VRControllerState::TouchControllerState(mut s) =
            touch_controller(Chirality::Left, true, 0.0, 0.0, false)
        else {
            panic!("expected touch controller state")
        };
        s.grip = 1.5;
        s.trigger = -0.5;
        let inputs = extract_curl_inputs(&VRControllerState::TouchControllerState(s))
            .expect("tracked controller should produce inputs");
        assert_eq!(inputs.grip, 1.0, "grip > 1 must clamp to 1");
        assert_eq!(inputs.trigger, 0.0, "trigger < 0 must clamp to 0");
    }

    #[test]
    fn touch_considers_all_face_buttons_and_sensors_for_thumb() {
        let thumbrest_touch_inputs = extract_curl_inputs(&VRControllerState::TouchControllerState(
            TouchControllerState {
                thumbrest_touch: true,
                ..idle_tracked_touch_controller_state()
            },
        ))
        .expect("tracked controller should produce inputs");
        assert!(
            thumbrest_touch_inputs.thumb,
            "thumbrest touch sensor should set thumb to true"
        );

        let button_xa_touch_inputs = extract_curl_inputs(&VRControllerState::TouchControllerState(
            TouchControllerState {
                button_xa_touch: true,
                ..idle_tracked_touch_controller_state()
            },
        ))
        .expect("tracked controller should produce inputs");
        assert!(
            button_xa_touch_inputs.thumb,
            "xa button touch sensor should set thumb to true"
        );

        let button_yb_touch_inputs = extract_curl_inputs(&VRControllerState::TouchControllerState(
            TouchControllerState {
                button_yb_touch: true,
                ..idle_tracked_touch_controller_state()
            },
        ))
        .expect("tracked controller should produce inputs");
        assert!(
            button_yb_touch_inputs.thumb,
            "yb button touch sensor should set thumb to true"
        );

        let joystick_touch_inputs = extract_curl_inputs(&VRControllerState::TouchControllerState(
            TouchControllerState {
                joystick_touch: true,
                ..idle_tracked_touch_controller_state()
            },
        ))
        .expect("tracked controller should produce inputs");
        assert!(
            joystick_touch_inputs.thumb,
            "joystick touch sensor should set thumb to true"
        );
    }

    #[test]
    fn index_clamps_out_of_range_grip_and_trigger() {
        use crate::shared::IndexControllerState;
        let s = IndexControllerState {
            side: Chirality::Right,
            body_node: BodyNode::RightController,
            is_device_active: true,
            is_tracking: true,
            grip: 2.0,
            trigger: -0.25,
            ..IndexControllerState::default()
        };
        let inputs = extract_curl_inputs(&VRControllerState::IndexControllerState(s))
            .expect("tracked controller should produce inputs");
        assert_eq!(inputs.grip, 1.0, "grip > 1 must clamp to 1");
        assert_eq!(inputs.trigger, 0.0, "trigger < 0 must clamp to 0");
    }

    #[test]
    fn vive_grip_bool_coerces_to_zero_or_one() {
        use crate::shared::ViveControllerState;
        let pressed = ViveControllerState {
            side: Chirality::Left,
            body_node: BodyNode::LeftController,
            is_device_active: true,
            is_tracking: true,
            grip: true,
            trigger: 0.4,
            ..ViveControllerState::default()
        };
        let released = ViveControllerState {
            grip: false,
            ..pressed.clone()
        };
        let pressed_inputs = extract_curl_inputs(&VRControllerState::ViveControllerState(pressed))
            .expect("tracked controller should produce inputs");
        let released_inputs =
            extract_curl_inputs(&VRControllerState::ViveControllerState(released))
                .expect("tracked controller should produce inputs");
        assert_eq!(pressed_inputs.grip, 1.0);
        assert_eq!(released_inputs.grip, 0.0);
        assert_eq!(pressed_inputs.trigger, 0.4);
    }

    #[test]
    fn windows_mr_grip_bool_coerces_to_zero_or_one() {
        use crate::shared::WindowsMRControllerState;
        let pressed = WindowsMRControllerState {
            side: Chirality::Right,
            body_node: BodyNode::RightController,
            is_device_active: true,
            is_tracking: true,
            grip: true,
            trigger: 0.7,
            ..WindowsMRControllerState::default()
        };
        let released = WindowsMRControllerState {
            grip: false,
            ..pressed.clone()
        };
        let pressed_inputs =
            extract_curl_inputs(&VRControllerState::WindowsMRControllerState(pressed))
                .expect("tracked controller should produce inputs");
        let released_inputs =
            extract_curl_inputs(&VRControllerState::WindowsMRControllerState(released))
                .expect("tracked controller should produce inputs");
        assert_eq!(pressed_inputs.grip, 1.0);
        assert_eq!(released_inputs.grip, 0.0);
        assert_eq!(pressed_inputs.trigger, 0.7);
    }

    #[test]
    fn untracked_vive_returns_none() {
        use crate::shared::ViveControllerState;
        let s = ViveControllerState {
            side: Chirality::Left,
            body_node: BodyNode::LeftController,
            is_device_active: true,
            is_tracking: false,
            grip: true,
            trigger: 1.0,
            ..ViveControllerState::default()
        };
        assert!(extract_curl_inputs(&VRControllerState::ViveControllerState(s)).is_none());
    }

    #[test]
    fn left_and_right_wrists_are_mirrored_under_mirrored_inputs() {
        // With identity controller rotations, mirrored controller positions plus mirrored
        // bound-hand offsets must produce X-mirrored wrists. This guards against one side's
        // composition getting sign-flipped in the future.
        let left_position = Vec3::new(-0.25, 1.4, -0.4);
        let right_position = Vec3::new(0.25, 1.4, -0.4);
        let left_offset = Vec3::new(-0.04, -0.025, -0.1);
        let right_offset = Vec3::new(0.04, -0.025, -0.1);

        let hands = synthesize_hand_states(&[
            touch_controller_with_pose(
                Chirality::Left,
                left_position,
                Quat::IDENTITY,
                true,
                left_offset,
                Quat::IDENTITY,
            ),
            touch_controller_with_pose(
                Chirality::Right,
                right_position,
                Quat::IDENTITY,
                true,
                right_offset,
                Quat::IDENTITY,
            ),
        ]);
        let left_wrist = hands[0].wrist_position;
        let right_wrist = hands[1].wrist_position;
        assert!(
            (left_wrist.x + right_wrist.x).abs() < 1e-4,
            "wrist X should be mirrored between hands under mirrored inputs: \
             left={left_wrist:?} right={right_wrist:?}",
        );
        assert!(
            (left_wrist.y - right_wrist.y).abs() < 1e-4,
            "wrist Y should match between hands when Y inputs match: \
             left={left_wrist:?} right={right_wrist:?}",
        );
        assert!(
            (left_wrist.z - right_wrist.z).abs() < 1e-4,
            "wrist Z should match between hands when Z inputs match: \
             left={left_wrist:?} right={right_wrist:?}",
        );
    }
}
