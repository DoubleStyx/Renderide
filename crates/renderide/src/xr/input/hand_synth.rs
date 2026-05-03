//! Synthesises per-finger [`HandState`] data from controller input.
//!
//! Without this, the host would receive no hand-tracking data and its `HandPoser` would reset
//! every finger to its `OriginalRotation`, leaving the avatar playing the desktop idle pose while
//! the user is in VR.
//!
//! The presets below are transcribed from the host's `Idle` and `Fist` finger-pose presets.
//! Segment layout matches the host's unpack loop: 24 entries indexed by
//! `BodyNode - LeftThumbMetacarpal`, ordered Thumb(Met,Prox,Dist,Tip), Index(Met,Prox,Inter,Dist,Tip),
//! Middle(..), Ring(..), Pinky(..). Right-hand [`HandState`] values reuse the same indexing but hold
//! right-hand data; the host mirrors via `bodyNode.GetSide(chirality)`.
//!
//! The curl blend is deliberately conservative: metacarpals are left at idle and we set
//! [`HandState::tracks_metacarpals`] to `false`, so the host overrides non-thumb metacarpals to the
//! avatar's own rest pose. Thumb is held at idle; index curl follows the trigger analog;
//! middle/ring/pinky follow the squeeze (grip) analog.

mod presets;

use glam::{Quat, Vec3};

use crate::shared::{Chirality, HandState, VRControllerState};

use presets::{
    FIST_POS_LEFT, FIST_POS_RIGHT, FIST_ROT_LEFT, FIST_ROT_RIGHT, IDLE_POS_LEFT, IDLE_POS_RIGHT,
    IDLE_ROT_LEFT, IDLE_ROT_RIGHT, LEFT_HAND_ID, RIGHT_HAND_ID, SEGMENT_COUNT,
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

/// Returns which finger the segment at `index` (0..24) belongs to, matching the
/// `BodyNode::LeftThumbMetacarpal..=LeftPinkyTip` layout.
fn finger_kind_for_segment(index: usize) -> FingerKind {
    match index {
        0..=3 => FingerKind::Thumb,
        4..=8 => FingerKind::Index,
        9..=13 => FingerKind::Middle,
        14..=18 => FingerKind::Ring,
        19..=23 => FingerKind::Pinky,
        _ => FingerKind::Pinky,
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
    /// [`crate::xr::input::pose::bound_hand_pose_defaults`]), not tracking-space poses.
    wrist_position: Vec3,
    /// Tracking-space wrist rotation to report on [`HandState::wrist_rotation`]. Composed the same
    /// way as [`Self::wrist_position`] and normalised.
    wrist_rotation: Quat,
    /// Grip/squeeze analog in 0..=1. Drives middle, ring, and pinky curl.
    grip: f32,
    /// Trigger analog in 0..=1. Drives index finger curl.
    trigger: f32,
}

/// Extracts the curl-driving inputs from a [`VRControllerState`] variant.
///
/// Returns `None` when the controller is not tracked (we do not want to feed the host random
/// hand poses). For controllers whose grip is a boolean (Vive wand, WMR), the boolean is
/// coerced to `0.0` / `1.0`.
fn extract_curl_inputs(controller: &VRControllerState) -> Option<ControllerCurlInputs> {
    match controller {
        VRControllerState::TouchControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (
                    s.position + s.rotation * s.hand_position,
                    (s.rotation * s.hand_rotation).normalize(),
                )
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: s.grip.clamp(0.0, 1.0),
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::IndexControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (
                    s.position + s.rotation * s.hand_position,
                    (s.rotation * s.hand_rotation).normalize(),
                )
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: s.grip.clamp(0.0, 1.0),
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::ViveControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (
                    s.position + s.rotation * s.hand_position,
                    (s.rotation * s.hand_rotation).normalize(),
                )
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: if s.grip { 1.0 } else { 0.0 },
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
        VRControllerState::WindowsMRControllerState(s) => {
            if !s.is_tracking {
                return None;
            }
            let (wrist_position, wrist_rotation) = if s.has_bound_hand {
                (
                    s.position + s.rotation * s.hand_position,
                    (s.rotation * s.hand_rotation).normalize(),
                )
            } else {
                (s.position, s.rotation)
            };
            Some(ControllerCurlInputs {
                side: s.side,
                wrist_position,
                wrist_rotation,
                grip: if s.grip { 1.0 } else { 0.0 },
                trigger: s.trigger.clamp(0.0, 1.0),
            })
        }
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

/// Returns the idle<->fist blend factor for a given segment index.
///
/// - Thumb and metacarpals are held at idle (`0.0`). Non-thumb metacarpals are overridden on the
///   host anyway when [`HandState::tracks_metacarpals`] is `false`, so their blend does not matter.
/// - Index curl follows `trigger`.
/// - Middle, ring, and pinky curl follow `grip`.
fn blend_factor_for_segment(index: usize, grip: f32, trigger: f32) -> f32 {
    match finger_kind_for_segment(index) {
        FingerKind::Thumb => 0.0,
        FingerKind::Index => trigger,
        FingerKind::Middle | FingerKind::Ring | FingerKind::Pinky => grip,
    }
}

/// Builds a [`HandState`] for one controller by blending the idle and fist presets. Returns
/// `None` if the controller is untracked or not a variant we drive hands for.
fn synthesize_one_hand(controller: &VRControllerState) -> Option<HandState> {
    let inputs = extract_curl_inputs(controller)?;
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
    let mut segment_positions = Vec::with_capacity(SEGMENT_COUNT);
    let mut segment_rotations = Vec::with_capacity(SEGMENT_COUNT);
    for i in 0..SEGMENT_COUNT {
        let t = blend_factor_for_segment(i, inputs.grip, inputs.trigger);
        let pi = Vec3::from_array(pos_idle[i]);
        let pf = Vec3::from_array(pos_fist[i]);
        let ri = Quat::from_array(rot_idle[i]);
        let rf = Quat::from_array(rot_fist[i]);
        segment_positions.push(pi.lerp(pf, t));
        segment_rotations.push(ri.slerp(rf, t));
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
    ) -> VRControllerState {
        VRControllerState::TouchControllerState(TouchControllerState {
            model: TouchControllerModel::QuestAndRiftS,
            start: false,
            button_yb: false,
            button_xa: false,
            button_yb_touch: false,
            button_xa_touch: false,
            thumbrest_touch: false,
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
        })
    }

    #[test]
    fn produces_one_hand_per_tracked_controller() {
        let controllers = vec![
            touch_controller(Chirality::Left, true, 0.0, 0.0),
            touch_controller(Chirality::Right, true, 0.0, 0.0),
        ];
        let hands = synthesize_hand_states(&controllers);
        assert_eq!(hands.len(), 2);
        assert_eq!(hands[0].chirality, Chirality::Left);
        assert_eq!(hands[1].chirality, Chirality::Right);
    }

    #[test]
    fn skips_untracked_controllers() {
        let controllers = vec![
            touch_controller(Chirality::Left, false, 0.0, 0.0),
            touch_controller(Chirality::Right, true, 0.0, 0.0),
        ];
        let hands = synthesize_hand_states(&controllers);
        assert_eq!(hands.len(), 1);
        assert_eq!(hands[0].chirality, Chirality::Right);
    }

    #[test]
    fn segment_arrays_have_host_expected_length() {
        let hands = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.5, 0.5)]);
        let hand = &hands[0];
        assert_eq!(hand.segment_positions.len(), SEGMENT_COUNT);
        assert_eq!(hand.segment_rotations.len(), SEGMENT_COUNT);
        assert!(hand.is_tracking);
        assert!(!hand.tracks_metacarpals);
    }

    #[test]
    fn trigger_bends_index_but_not_other_fingers() {
        let idle =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0)]).remove(0);
        let full_trigger =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 1.0)]).remove(0);
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
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0)]).remove(0);
        let full_grip =
            synthesize_hand_states(&[touch_controller(Chirality::Left, true, 1.0, 0.0)]).remove(0);
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
            touch_controller(Chirality::Left, true, 0.5, 0.5),
            touch_controller(Chirality::Right, true, 0.5, 0.5),
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
    fn thumb_metacarpal_always_at_idle_pose() {
        // Thumb is never blended, so thumb metacarpal position (segment 0) should always match the
        // idle preset regardless of grip/trigger.
        let hands = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 1.0, 1.0)]);
        let expected = Vec3::from_array(IDLE_POS_LEFT[0]);
        let actual = hands[0].segment_positions[0];
        assert!(
            (actual - expected).length() < 1e-6,
            "thumb metacarpal should stay at idle when grip=1, trigger=1"
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
