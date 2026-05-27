//! Controller palm pose math and OpenXR [`openxr::SpaceLocation`] conversion.

use glam::{Quat, Vec3};
use openxr as xr;

use crate::shared::Chirality;

use super::profile::ActiveControllerProfile;

/// Converts an [`xr::SpaceLocation`] into OpenXR tracking-space `(position, rotation)`.
///
/// Returns `None` when either position or orientation is invalid, so callers can fall back to
/// grip-derived poses or keep the previous frame's state.
pub(super) fn pose_from_location(location: &xr::SpaceLocation) -> Option<(Vec3, Quat)> {
    let tracked = location
        .location_flags
        .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID);
    tracked.then(|| {
        let pose = &location.pose;
        let position = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
        let orientation = pose.orientation;
        let rotation = Quat::from_xyzw(orientation.x, orientation.y, orientation.z, orientation.w);
        let len_sq = rotation.length_squared();
        let rotation = if len_sq.is_finite() && len_sq >= 1e-10 {
            rotation.normalize()
        } else {
            Quat::IDENTITY
        };
        (position, rotation)
    })
}

/// Converts an [`xr::HandJointLocation`] into OpenXR tracking-space `(position, rotation)`.
///
/// Returns `None` when either position or orientation is invalid.
pub(super) fn pose_from_joint(location: &xr::HandJointLocation) -> Option<(Vec3, Quat)> {
    let tracked = location
        .location_flags
        .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID);
    tracked.then(|| {
        let pose = &location.pose;
        let position = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
        let orientation = pose.orientation;
        let rotation = Quat::from_xyzw(orientation.x, orientation.y, orientation.z, orientation.w);
        let len_sq = rotation.length_squared();
        let rotation = if len_sq.is_finite() && len_sq >= 1e-10 {
            rotation.normalize()
        } else {
            Quat::IDENTITY
        };
        (position, rotation)
    })
}

/// Default `hand_position` / `hand_rotation` IPC offsets for bound-hand tracking.
///
/// These offsets are paired with the palm-pose controller frame emitted by
/// [`super::frame::resolve_controller_frame`].
pub(super) fn hand_pose_defaults(side: Chirality) -> (Vec3, Quat) {
    match side {
        Chirality::Left => (
            Vec3::new(-0.016_561_74, 0.017_465_11, -0.060_415_43),
            Quat::from_xyzw(0.034_154_01, -0.036_695_68, 0.659_639_54, 0.749_908_4),
        ),
        Chirality::Right => (
            Vec3::new(0.016_549_01, 0.017_472_06, -0.060_390_15),
            Quat::from_xyzw(-0.034_164_81, -0.036_687_03, 0.659_629_6, -0.749_917_1),
        ),
    }
}

/// Converts an OpenXR grip pose into a palm-ext pose using per-profile offsets.
///
/// Offsets are local to the OpenXR grip pose and are used only when `XR_EXT_palm_pose` is not
/// available or does not produce a valid tracked pose for the current frame.
pub(super) fn grip_to_palm_ext_pose(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: (Vec3, Quat),
) -> (Vec3, Quat) {
    let (palm_ext_rot_offset, palm_ext_pos_offset) = match (profile, side) {
        (ActiveControllerProfile::Index, Chirality::Left) => (
            Quat::from_xyzw(-0.46, -0.02, -0.01, 0.89).normalize(),
            Vec3::new(-0.015, 0.0, 0.001),
        ),
        (ActiveControllerProfile::Index, Chirality::Right) => (
            Quat::from_xyzw(-0.46, 0.02, 0.01, 0.89).normalize(),
            Vec3::new(0.015, 0.0, 0.001),
        ),
        (_, Chirality::Left) => (
            Quat::from_xyzw(-0.46, -0.02, -0.01, 0.89).normalize(),
            Vec3::new(-0.015, 0.0, 0.001),
        ),
        (_, Chirality::Right) => (
            Quat::from_xyzw(-0.46, 0.02, 0.01, 0.89).normalize(),
            Vec3::new(0.015, 0.0, 0.001),
        ),
    };

    let (position, rotation) = grip_pose;
    (
        position + rotation * palm_ext_pos_offset,
        (rotation * palm_ext_rot_offset).normalize(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let delta = (actual - expected).length();
        assert!(
            delta < 1e-4,
            "vec3 mismatch: actual={actual:?} expected={expected:?} delta={delta}"
        );
    }

    fn assert_quat_near(actual: Quat, expected: Quat) {
        let dot = actual.normalize().dot(expected.normalize()).abs();
        assert!(
            (1.0 - dot) < 1e-4,
            "quat mismatch: actual={actual:?} expected={expected:?} dot={dot}"
        );
    }

    #[test]
    fn pose_from_location_returns_openxr_tracking_pose() {
        let expected_rotation = Quat::from_rotation_y(0.35).normalize();
        let location = xr::SpaceLocation {
            location_flags: xr::SpaceLocationFlags::ORIENTATION_VALID
                | xr::SpaceLocationFlags::POSITION_VALID,
            pose: xr::Posef {
                orientation: xr::Quaternionf {
                    x: expected_rotation.x,
                    y: expected_rotation.y,
                    z: expected_rotation.z,
                    w: expected_rotation.w,
                },
                position: xr::Vector3f {
                    x: 1.0,
                    y: 2.0,
                    z: -3.0,
                },
            },
        };

        let (position, rotation) = pose_from_location(&location).expect("tracked pose");

        assert_vec3_near(position, Vec3::new(1.0, 2.0, -3.0));
        assert_quat_near(rotation, expected_rotation);
    }

    #[test]
    fn grip_to_palm_ext_pose_applies_profile_offset() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        let (position, rotation) = grip_to_palm_ext_pose(
            ActiveControllerProfile::Index,
            Chirality::Left,
            (grip_position, grip_rotation),
        );

        assert!((position - grip_position).length() > 0.001);
        assert!(rotation.normalize().dot(grip_rotation).abs() < 0.98);
    }
}
