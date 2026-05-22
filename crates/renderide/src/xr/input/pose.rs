//! Controller grip/aim pose math and OpenXR [`openxr::SpaceLocation`] conversion.
//!
//! Values follow the host-side pose convention expected by the `SteamVRDriver`-shaped
//! controller data path. The host (`VR_Manager`) writes the
//! received `position` / `rotation` straight into `RawPosition` / `RawRotation`, so the renderer
//! is responsible for delivering poses in the exact frame the host was authored against.

use glam::{EulerRot, Quat, Vec3};
use openxr as xr;

use crate::shared::Chirality;

use super::profile::ActiveControllerProfile;

/// Converts an [`xr::SpaceLocation`] into OpenXR tracking-space `(position, rotation)`.
///
/// Returns `None` when either position or orientation is invalid, so callers can fall back to
/// aim-derived poses or keep the previous frame's state.
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

/// Default `hand_position` / `hand_rotation` on the IPC controller state types in
/// [`crate::shared`] for bound-hand tracking (FrooxEngine `BodyNodePositionOffset` /
/// `BodyNodeRotationOffset` on the hand device).
///
/// These are currently hardcoded to match the OpenXR palm_ext pose.
pub(super) fn hand_pose_defaults(side: Chirality) -> (Vec3, Quat) {
    match side {
        Chirality::Left => (
            Vec3::new(0.0, 0.01, -0.08),
            Quat::from_euler(
                EulerRot::XYZ,
                11.5f32.to_radians(),
                0.5f32.to_radians(),
                93.7f32.to_radians(),
            ),
        ),
        Chirality::Right => (
            Vec3::new(0.0, 0.01, -0.08),
            Quat::from_euler(
                EulerRot::XYZ,
                11.5f32.to_radians(),
                0.5f32.to_radians(),
                -93.7f32.to_radians(),
            ),
        ),
    }
}

/// Converts an OpenXR grip pose into a palm_ext pose using per-profile offsets.
///
/// Offsets can be derived using https://github.com/ValveSoftware/OpenXR-Canonical-Pose-Tool
pub(super) fn grip_to_palm_ext_pose(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: (Vec3, Quat),
) -> (Vec3, Quat) {
    let (palm_ext_rot_offset, palm_ext_pos_offset) = match (profile, side) {
        (ActiveControllerProfile::Index, Chirality::Left) => (
            Quat::from_xyzw(-0.46, -0.02, -0.01, 0.89).normalize(),
            Vec3::new(-0.015, 0.000, 0.001),
        ),
        (ActiveControllerProfile::Index, Chirality::Right) => (
            Quat::from_xyzw(-0.46, 0.02, 0.01, 0.89).normalize(),
            Vec3::new(0.015, 0.000, 0.001),
        ),
        // generic fallback
        // for now these are based on index, but ideally they should be swapped out for touch controllers
        // later as most controllers out there are heavily inspired by them.
        (_, Chirality::Left) => (
            Quat::from_xyzw(-0.46, -0.02, -0.01, 0.89).normalize(),
            Vec3::new(-0.015, 0.000, 0.001),
        ),
        (_, Chirality::Right) => (
            Quat::from_xyzw(-0.46, 0.02, 0.01, 0.89).normalize(),
            Vec3::new(0.015, 0.000, 0.001),
        ),
    };

    let (pos, rot) = grip_pose;

    let surface_pos = pos + rot * palm_ext_pos_offset;
    let surface_rot = rot * palm_ext_rot_offset;

    (surface_pos, surface_rot)
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
}
