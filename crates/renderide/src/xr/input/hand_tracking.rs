use crate::xr::input::pose::pose_from_joint;
use crate::xr::session::openxr_tracking_pose_to_host;
use glam::{Quat, Vec3};
use openxr as xr;
use renderide_shared::{Chirality, HandState};

const LEFT_HAND_ID: &str = "openxr_left_hand";
const RIGHT_HAND_ID: &str = "openxr_right_hand";

/// Converts an [`xr::HandJointLocation`] into a pose in host coordinate space.
fn joint_to_host(location: &xr::HandJointLocation) -> Option<(Vec3, Quat)> {
    let (pos, rot) = pose_from_joint(location)?;
    Some(openxr_tracking_pose_to_host(pos, rot))
}

/// Builds a [`HandState`] from [`xr::HandJointLocations`].
///
/// The priority field is always set to 1, that way it is preferred over the synthesized hand.
pub fn hand_from_openxr(joints: xr::HandJointLocations, chirality: Chirality) -> Option<HandState> {
    let (wrist_position, wrist_rotation) = joint_to_host(&joints[xr::HandJoint::WRIST])?;
    let inv_wrist_rot = wrist_rotation.inverse();

    // these OpenXR bones are in the same order as OpenVR and Resonite's segments, we can cheat :3
    let start = xr::HandJoint::THUMB_METACARPAL.into_raw() as usize;
    let end = xr::HandJoint::LITTLE_TIP.into_raw() as usize;

    let (segment_positions, segment_rotations) = joints[start..=end]
        .iter()
        .map(|loc| {
            let (pos, rot) = joint_to_host(loc)?;
            // make all segment joints relative to the wrist
            let local_pos = inv_wrist_rot * (pos - wrist_position);
            let local_rot = (inv_wrist_rot * rot).normalize();
            Some((local_pos, local_rot))
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .unzip();

    Some(HandState {
        unique_id: Some(match chirality {
            Chirality::Left => LEFT_HAND_ID.to_string(),
            Chirality::Right => RIGHT_HAND_ID.to_string(),
        }),
        priority: 1,
        chirality,
        is_device_active: true,
        is_tracking: true,
        tracks_metacarpals: true,
        confidence: 1.0,
        wrist_position,
        wrist_rotation,
        segment_positions,
        segment_rotations,
    })
}
