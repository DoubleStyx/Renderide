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

fn assert_vec3_close(actual: Vec3, expected: [f32; 3], label: &str) {
    let expected = Vec3::from_array(expected);
    let delta = (actual - expected).length();
    assert!(
        delta < 2e-5,
        "{label} position mismatch: got {actual:?}, expected {expected:?}, delta={delta}"
    );
}

fn assert_quat_close(actual: Quat, expected: [f32; 4], label: &str) {
    let expected = Quat::from_array(expected).normalize();
    let actual = actual.normalize();
    let dot = actual.dot(expected).abs();
    assert!(
        dot > 1.0 - 2e-5,
        "{label} rotation mismatch: got {actual:?}, expected {expected:?}, dot={dot}"
    );
}

fn assert_pose_matches_flat(
    positions: &[Vec3],
    rotations: &[Quat],
    expected_positions: &[[f32; 3]; flat_presets::SEGMENT_COUNT],
    expected_rotations: &[[f32; 4]; flat_presets::SEGMENT_COUNT],
    label: &str,
) {
    assert_eq!(positions.len(), TOTAL_SEGMENT_COUNT);
    assert_eq!(rotations.len(), TOTAL_SEGMENT_COUNT);
    for segment_index in 0..TOTAL_SEGMENT_COUNT {
        assert_vec3_close(
            positions[segment_index],
            expected_positions[segment_index],
            &format!("{label} segment {segment_index}"),
        );
        assert_quat_close(
            rotations[segment_index],
            expected_rotations[segment_index],
            &format!("{label} segment {segment_index}"),
        );
    }
}

fn assert_local_positions_match(
    actual: &presets::LocalPositionPresets,
    expected: &presets::LocalPositionPresets,
    label: &str,
) {
    for finger_index in 0..presets::FINGER_COUNT {
        for segment_index in 0..presets::MAX_SEGMENTS_PER_FINGER {
            assert_vec3_close(
                Vec3::from_array(actual[finger_index][segment_index]),
                expected[finger_index][segment_index],
                &format!("{label} finger {finger_index} segment {segment_index}"),
            );
        }
    }
}

fn assert_local_rotations_match(
    actual: &presets::LocalRotationPresets,
    expected: &presets::LocalRotationPresets,
    label: &str,
) {
    for finger_index in 0..presets::FINGER_COUNT {
        for segment_index in 0..presets::MAX_SEGMENTS_PER_FINGER {
            let expected_rotation = expected[finger_index][segment_index];
            if expected_rotation == [0.0; 4] {
                assert_eq!(
                    actual[finger_index][segment_index], expected_rotation,
                    "{label} finger {finger_index} segment {segment_index} padding mismatch"
                );
                continue;
            }
            assert_quat_close(
                Quat::from_array(actual[finger_index][segment_index]),
                expected_rotation,
                &format!("{label} finger {finger_index} segment {segment_index}"),
            );
        }
    }
}

fn flattened_local_pose(
    positions: &presets::LocalPositionPresets,
    rotations: &presets::LocalRotationPresets,
) -> (Vec<Vec3>, Vec<Quat>) {
    let mut flat_positions = Vec::with_capacity(TOTAL_SEGMENT_COUNT);
    let mut flat_rotations = Vec::with_capacity(TOTAL_SEGMENT_COUNT);
    for (finger_index, &segment_count) in SEGMENT_COUNTS.iter().enumerate() {
        append_blended_finger_pose(
            &mut flat_positions,
            &mut flat_rotations,
            LocalFingerBlend {
                idle_positions: &positions[finger_index],
                idle_rotations: &rotations[finger_index],
                fist_positions: &positions[finger_index],
                fist_rotations: &rotations[finger_index],
                segment_count,
                blend: 0.0,
            },
        );
    }
    (flat_positions, flat_rotations)
}

fn quat_angle_between(a: Quat, b: Quat) -> f32 {
    2.0 * a
        .normalize()
        .dot(b.normalize())
        .abs()
        .clamp(-1.0, 1.0)
        .acos()
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
    let hands = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.5, 0.5, false)]);
    let hand = &hands[0];
    assert_eq!(hand.segment_positions.len(), TOTAL_SEGMENT_COUNT);
    assert_eq!(hand.segment_rotations.len(), TOTAL_SEGMENT_COUNT);
    assert!(hand.is_tracking);
    assert!(!hand.tracks_metacarpals);
}

#[test]
fn local_presets_round_trip_to_flat_calibration_poses() {
    let (idle_left_positions, idle_left_rotations) =
        flattened_local_pose(&IDLE_POS_LEFT, &IDLE_ROT_LEFT);
    assert_pose_matches_flat(
        &idle_left_positions,
        &idle_left_rotations,
        &flat_presets::IDLE_POS_LEFT,
        &flat_presets::IDLE_ROT_LEFT,
        "idle left",
    );

    let (idle_right_positions, idle_right_rotations) =
        flattened_local_pose(&IDLE_POS_RIGHT, &IDLE_ROT_RIGHT);
    assert_pose_matches_flat(
        &idle_right_positions,
        &idle_right_rotations,
        &flat_presets::IDLE_POS_RIGHT,
        &flat_presets::IDLE_ROT_RIGHT,
        "idle right",
    );

    let (fist_left_positions, fist_left_rotations) =
        flattened_local_pose(&FIST_POS_LEFT, &FIST_ROT_LEFT);
    assert_pose_matches_flat(
        &fist_left_positions,
        &fist_left_rotations,
        &flat_presets::FIST_POS_LEFT,
        &flat_presets::FIST_ROT_LEFT,
        "fist left",
    );

    let (fist_right_positions, fist_right_rotations) =
        flattened_local_pose(&FIST_POS_RIGHT, &FIST_ROT_RIGHT);
    assert_pose_matches_flat(
        &fist_right_positions,
        &fist_right_rotations,
        &flat_presets::FIST_POS_RIGHT,
        &flat_presets::FIST_ROT_RIGHT,
        "fist right",
    );
}

#[test]
fn flat_source_presets_convert_to_checked_in_local_presets() {
    assert_local_positions_match(
        &flat_presets::flat_position_to_local_space(
            &flat_presets::IDLE_POS_LEFT,
            &flat_presets::IDLE_ROT_LEFT,
        ),
        &IDLE_POS_LEFT,
        "idle left",
    );
    assert_local_rotations_match(
        &flat_presets::flat_rotation_to_local_space(&flat_presets::IDLE_ROT_LEFT),
        &IDLE_ROT_LEFT,
        "idle left",
    );
    assert_local_positions_match(
        &flat_presets::flat_position_to_local_space(
            &flat_presets::IDLE_POS_RIGHT,
            &flat_presets::IDLE_ROT_RIGHT,
        ),
        &IDLE_POS_RIGHT,
        "idle right",
    );
    assert_local_rotations_match(
        &flat_presets::flat_rotation_to_local_space(&flat_presets::IDLE_ROT_RIGHT),
        &IDLE_ROT_RIGHT,
        "idle right",
    );
    assert_local_positions_match(
        &flat_presets::flat_position_to_local_space(
            &flat_presets::FIST_POS_LEFT,
            &flat_presets::FIST_ROT_LEFT,
        ),
        &FIST_POS_LEFT,
        "fist left",
    );
    assert_local_rotations_match(
        &flat_presets::flat_rotation_to_local_space(&flat_presets::FIST_ROT_LEFT),
        &FIST_ROT_LEFT,
        "fist left",
    );
    assert_local_positions_match(
        &flat_presets::flat_position_to_local_space(
            &flat_presets::FIST_POS_RIGHT,
            &flat_presets::FIST_ROT_RIGHT,
        ),
        &FIST_POS_RIGHT,
        "fist right",
    );
    assert_local_rotations_match(
        &flat_presets::flat_rotation_to_local_space(&flat_presets::FIST_ROT_RIGHT),
        &FIST_ROT_RIGHT,
        "fist right",
    );
}

#[test]
fn idle_input_emits_flat_idle_pose() {
    let left = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
        .remove(0);
    assert_pose_matches_flat(
        &left.segment_positions,
        &left.segment_rotations,
        &flat_presets::IDLE_POS_LEFT,
        &flat_presets::IDLE_ROT_LEFT,
        "left idle hand",
    );

    let right =
        synthesize_hand_states(&[touch_controller(Chirality::Right, true, 0.0, 0.0, false)])
            .remove(0);
    assert_pose_matches_flat(
        &right.segment_positions,
        &right.segment_rotations,
        &flat_presets::IDLE_POS_RIGHT,
        &flat_presets::IDLE_ROT_RIGHT,
        "right idle hand",
    );
}

#[test]
fn full_inputs_emit_flat_fist_pose_for_driven_fingers() {
    let full_trigger =
        synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 1.0, false)])
            .remove(0);
    for segment_index in 4..=8 {
        assert_vec3_close(
            full_trigger.segment_positions[segment_index],
            flat_presets::FIST_POS_LEFT[segment_index],
            "full trigger index",
        );
        assert_quat_close(
            full_trigger.segment_rotations[segment_index],
            flat_presets::FIST_ROT_LEFT[segment_index],
            "full trigger index",
        );
    }

    let full_grip =
        synthesize_hand_states(&[touch_controller(Chirality::Left, true, 1.0, 0.0, false)])
            .remove(0);
    for segment_index in 9..=23 {
        assert_vec3_close(
            full_grip.segment_positions[segment_index],
            flat_presets::FIST_POS_LEFT[segment_index],
            "full grip non-index finger",
        );
        assert_quat_close(
            full_grip.segment_rotations[segment_index],
            flat_presets::FIST_ROT_LEFT[segment_index],
            "full grip non-index finger",
        );
    }

    let full_thumb =
        synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, true)])
            .remove(0);
    for segment_index in 0..=3 {
        assert_vec3_close(
            full_thumb.segment_positions[segment_index],
            flat_presets::FIST_POS_LEFT[segment_index],
            "full thumb",
        );
        assert_quat_close(
            full_thumb.segment_rotations[segment_index],
            flat_presets::FIST_ROT_LEFT[segment_index],
            "full thumb",
        );
    }
}

#[test]
fn half_curl_uses_hierarchical_rotation_path() {
    let half_trigger =
        synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.5, false)])
            .remove(0);
    let segment_index = 7;
    let direct_flat_slerp = Quat::from_array(flat_presets::IDLE_ROT_LEFT[segment_index]).slerp(
        Quat::from_array(flat_presets::FIST_ROT_LEFT[segment_index]),
        0.5,
    );
    let angle_delta = quat_angle_between(
        half_trigger.segment_rotations[segment_index],
        direct_flat_slerp,
    );
    assert!(
        angle_delta > 0.05,
        "half curl should not direct-slerp flat distal rotations (angle_delta={angle_delta})"
    );
}

#[test]
fn trigger_bends_index_but_not_other_fingers() {
    let idle = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
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
    let idle = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
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
    let idle = synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, false)])
        .remove(0);
    let full_thumb =
        synthesize_hand_states(&[touch_controller(Chirality::Left, true, 0.0, 0.0, true)])
            .remove(0);
    let middle_delta = (full_thumb.segment_rotations[11].to_array()[3]
        - idle.segment_rotations[11].to_array()[3])
        .abs();
    let ring_delta = (full_thumb.segment_rotations[16].to_array()[3]
        - idle.segment_rotations[16].to_array()[3])
        .abs();
    let pinky_delta = (full_thumb.segment_rotations[21].to_array()[3]
        - idle.segment_rotations[21].to_array()[3])
        .abs();
    let index_delta = (full_thumb.segment_rotations[6].to_array()[3]
        - idle.segment_rotations[6].to_array()[3])
        .abs();
    let thumb_delta = (full_thumb.segment_rotations[1].to_array()[3]
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
    let idle_inputs = extract_curl_inputs(&VRControllerState::TouchControllerState(
        idle_tracked_touch_controller_state(),
    ))
    .expect("tracked controller should produce inputs");
    assert!(
        !idle_inputs.thumb,
        "idle touch controller should leave thumb false"
    );

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
fn index_considers_touchpad_face_buttons_and_joystick_for_thumb() {
    use crate::shared::IndexControllerState;

    let idle = IndexControllerState {
        side: Chirality::Left,
        body_node: BodyNode::LeftController,
        is_device_active: true,
        is_tracking: true,
        ..IndexControllerState::default()
    };
    let idle_inputs = extract_curl_inputs(&VRControllerState::IndexControllerState(idle.clone()))
        .expect("tracked controller should produce inputs");
    assert!(
        !idle_inputs.thumb,
        "idle index controller should leave thumb false"
    );

    for state in [
        IndexControllerState {
            touchpad_touch: true,
            ..idle.clone()
        },
        IndexControllerState {
            button_atouch: true,
            ..idle.clone()
        },
        IndexControllerState {
            button_btouch: true,
            ..idle.clone()
        },
        IndexControllerState {
            joystick_touch: true,
            ..idle
        },
    ] {
        let inputs = extract_curl_inputs(&VRControllerState::IndexControllerState(state))
            .expect("tracked controller should produce inputs");
        assert!(inputs.thumb, "index thumb sensor should set thumb true");
    }
}

#[test]
fn vive_and_windows_mr_use_touchpad_for_thumb() {
    use crate::shared::{ViveControllerState, WindowsMRControllerState};

    let vive_idle = ViveControllerState {
        side: Chirality::Left,
        body_node: BodyNode::LeftController,
        is_device_active: true,
        is_tracking: true,
        ..ViveControllerState::default()
    };
    let vive_idle_inputs =
        extract_curl_inputs(&VRControllerState::ViveControllerState(vive_idle.clone()))
            .expect("tracked controller should produce inputs");
    assert!(
        !vive_idle_inputs.thumb,
        "idle vive controller should leave thumb false"
    );
    let vive_touch_inputs = extract_curl_inputs(&VRControllerState::ViveControllerState(
        ViveControllerState {
            touchpad_touch: true,
            ..vive_idle
        },
    ))
    .expect("tracked controller should produce inputs");
    assert!(vive_touch_inputs.thumb);

    let wmr_idle = WindowsMRControllerState {
        side: Chirality::Right,
        body_node: BodyNode::RightController,
        is_device_active: true,
        is_tracking: true,
        ..WindowsMRControllerState::default()
    };
    let wmr_idle_inputs = extract_curl_inputs(&VRControllerState::WindowsMRControllerState(
        wmr_idle.clone(),
    ))
    .expect("tracked controller should produce inputs");
    assert!(
        !wmr_idle_inputs.thumb,
        "idle windows mr controller should leave thumb false"
    );
    let wmr_touch_inputs = extract_curl_inputs(&VRControllerState::WindowsMRControllerState(
        WindowsMRControllerState {
            touchpad_touch: true,
            ..wmr_idle
        },
    ))
    .expect("tracked controller should produce inputs");
    assert!(wmr_touch_inputs.thumb);
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
    let released_inputs = extract_curl_inputs(&VRControllerState::ViveControllerState(released))
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
    let pressed_inputs = extract_curl_inputs(&VRControllerState::WindowsMRControllerState(pressed))
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
