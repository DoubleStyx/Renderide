//! OpenXR hand-joint conversion for host-facing [`HandState`] input.

use glam::{Quat, Vec3};
use openxr as xr;

use crate::shared::{Chirality, HandState};
use crate::xr::input::pose::pose_from_joint;
use crate::xr::session::openxr_tracking_pose_to_host;

const LEFT_HAND_ID: &str = "openxr_left_hand";
const RIGHT_HAND_ID: &str = "openxr_right_hand";
const OPENXR_HAND_PRIORITY: i32 = 1;
const HAND_CONFIDENCE: f32 = 1.0;
const FIRST_SEGMENT_JOINT_INDEX: usize = 2;
const LAST_SEGMENT_JOINT_INDEX: usize = 25;
const HOST_FINGER_SEGMENT_COUNT: usize = 24;

/// Returns the stable host device identifier used for one OpenXR hand.
fn openxr_hand_id(chirality: Chirality) -> &'static str {
    match chirality {
        Chirality::Left => LEFT_HAND_ID,
        Chirality::Right => RIGHT_HAND_ID,
    }
}

/// Builds a host hand shell with the OpenXR device identity and tracking flags.
fn openxr_hand_base(chirality: Chirality, is_tracking: bool) -> HandState {
    HandState {
        unique_id: Some(openxr_hand_id(chirality).to_string()),
        priority: OPENXR_HAND_PRIORITY,
        chirality,
        is_device_active: is_tracking,
        is_tracking,
        tracks_metacarpals: true,
        confidence: HAND_CONFIDENCE,
        wrist_position: Vec3::ZERO,
        wrist_rotation: Quat::IDENTITY,
        segment_positions: Vec::new(),
        segment_rotations: Vec::new(),
    }
}

/// Builds an untracked state for a previously registered OpenXR hand.
pub(super) fn inactive_openxr_hand(chirality: Chirality) -> HandState {
    openxr_hand_base(chirality, false)
}

/// Converts an [`xr::HandJointLocation`] into a pose in host coordinate space.
fn joint_to_host(location: &xr::HandJointLocation) -> Option<(Vec3, Quat)> {
    let (pos, rot) = pose_from_joint(location)?;
    Some(openxr_tracking_pose_to_host(pos, rot))
}

/// Returns a normalized quaternion, falling back to identity for invalid values.
fn normalize_or_identity(rotation: Quat) -> Quat {
    let len_sq = rotation.length_squared();
    if len_sq.is_finite() && len_sq >= 1e-10 {
        rotation.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Builds a [`HandState`] from [`xr::HandJointLocations`].
///
/// Finger segments are reported wrist-local in the host's expected finger-node order.
pub(super) fn hand_from_openxr(
    joints: &xr::HandJointLocations,
    chirality: Chirality,
) -> Option<HandState> {
    let (wrist_position, wrist_rotation) = joint_to_host(&joints[xr::HandJoint::WRIST])?;
    let inv_wrist_rot = wrist_rotation.inverse();

    let (segment_positions, segment_rotations): (Vec<Vec3>, Vec<Quat>) = joints
        [FIRST_SEGMENT_JOINT_INDEX..=LAST_SEGMENT_JOINT_INDEX]
        .iter()
        .map(|loc| {
            let (pos, rot) = joint_to_host(loc)?;
            let local_pos = inv_wrist_rot * (pos - wrist_position);
            let local_rot = normalize_or_identity(inv_wrist_rot * rot);
            Some((local_pos, local_rot))
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .unzip();

    if segment_positions.len() != HOST_FINGER_SEGMENT_COUNT
        || segment_rotations.len() != HOST_FINGER_SEGMENT_COUNT
    {
        return None;
    }

    let mut hand = openxr_hand_base(chirality, true);
    hand.wrist_position = wrist_position;
    hand.wrist_rotation = wrist_rotation;
    hand.segment_positions = segment_positions;
    hand.segment_rotations = segment_rotations;
    Some(hand)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a valid OpenXR hand joint location for tests.
    fn valid_joint(position: Vec3, rotation: Quat) -> xr::HandJointLocation {
        xr::HandJointLocation {
            location_flags: xr::SpaceLocationFlags::ORIENTATION_VALID
                | xr::SpaceLocationFlags::POSITION_VALID,
            pose: xr::Posef {
                orientation: xr::Quaternionf {
                    x: rotation.x,
                    y: rotation.y,
                    z: rotation.z,
                    w: rotation.w,
                },
                position: xr::Vector3f {
                    x: position.x,
                    y: position.y,
                    z: position.z,
                },
            },
            radius: 0.01,
        }
    }

    /// Builds a complete hand-joint array with unique segment offsets.
    fn valid_joints() -> xr::HandJointLocations {
        let mut joints = [valid_joint(Vec3::ZERO, Quat::IDENTITY); xr::HAND_JOINT_COUNT];
        joints[xr::HandJoint::WRIST] = valid_joint(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        for (i, joint) in joints[FIRST_SEGMENT_JOINT_INDEX..=LAST_SEGMENT_JOINT_INDEX]
            .iter_mut()
            .enumerate()
        {
            *joint = valid_joint(Vec3::new(2.0 + i as f32, 2.0, 3.0), Quat::IDENTITY);
        }
        joints
    }

    /// Asserts two vectors are almost equal.
    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let delta = (actual - expected).length();
        assert!(
            delta < 1e-5,
            "vec3 mismatch: actual={actual:?} expected={expected:?} delta={delta}"
        );
    }

    #[test]
    fn active_hand_has_stable_host_identity() {
        let hand = hand_from_openxr(&valid_joints(), Chirality::Left).expect("tracked hand");

        assert_eq!(hand.unique_id.as_deref(), Some(LEFT_HAND_ID));
        assert_eq!(hand.priority, OPENXR_HAND_PRIORITY);
        assert_eq!(hand.chirality, Chirality::Left);
        assert!(hand.is_device_active);
        assert!(hand.is_tracking);
        assert!(hand.tracks_metacarpals);
        assert_eq!(hand.confidence, HAND_CONFIDENCE);
    }

    #[test]
    fn active_hand_uses_expected_segment_count() {
        let hand = hand_from_openxr(&valid_joints(), Chirality::Right).expect("tracked hand");

        assert_eq!(hand.segment_positions.len(), HOST_FINGER_SEGMENT_COUNT);
        assert_eq!(hand.segment_rotations.len(), HOST_FINGER_SEGMENT_COUNT);
        assert_eq!(hand.unique_id.as_deref(), Some(RIGHT_HAND_ID));
    }

    #[test]
    fn active_hand_converts_wrist_and_segments_to_host_space() {
        let hand = hand_from_openxr(&valid_joints(), Chirality::Left).expect("tracked hand");

        assert_vec3_near(hand.wrist_position, Vec3::new(1.0, 2.0, -3.0));
        assert_vec3_near(hand.segment_positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_vec3_near(hand.segment_positions[1], Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(hand.wrist_rotation, Quat::IDENTITY);
        assert_eq!(hand.segment_rotations[0], Quat::IDENTITY);
    }

    #[test]
    fn invalid_wrist_pose_rejects_hand() {
        let mut joints = valid_joints();
        joints[xr::HandJoint::WRIST].location_flags = xr::SpaceLocationFlags::from_raw(0);

        assert!(hand_from_openxr(&joints, Chirality::Left).is_none());
    }

    #[test]
    fn invalid_segment_pose_rejects_hand() {
        let mut joints = valid_joints();
        joints[FIRST_SEGMENT_JOINT_INDEX].location_flags = xr::SpaceLocationFlags::from_raw(0);

        assert!(hand_from_openxr(&joints, Chirality::Left).is_none());
    }

    #[test]
    fn inactive_hand_clears_registered_openxr_device() {
        let hand = inactive_openxr_hand(Chirality::Right);

        assert_eq!(hand.unique_id.as_deref(), Some(RIGHT_HAND_ID));
        assert_eq!(hand.priority, OPENXR_HAND_PRIORITY);
        assert_eq!(hand.chirality, Chirality::Right);
        assert!(!hand.is_device_active);
        assert!(!hand.is_tracking);
        assert!(hand.segment_positions.is_empty());
        assert!(hand.segment_rotations.is_empty());
    }
}
