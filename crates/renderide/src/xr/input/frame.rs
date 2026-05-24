//! Per-frame resolved controller pose before IPC mapping.

use glam::{Quat, Vec3};

use crate::shared::Chirality;
use crate::xr::session::openxr_tracking_pose_to_host;

use super::pose::{grip_to_palm_ext_pose, hand_pose_defaults};
use super::profile::ActiveControllerProfile;

/// Resolved controller and optional bound-hand pose in tracking space.
#[derive(Clone, Copy)]
pub(super) struct ControllerFrame {
    /// Controller position in host-tracking space.
    pub(super) position: Vec3,
    /// Controller orientation in host-tracking space.
    pub(super) rotation: Quat,
    /// Whether the frame carries a calibrated bound-hand offset.
    pub(super) has_bound_hand: bool,
    /// Bound-hand position relative to the controller pose.
    pub(super) hand_position: Vec3,
    /// Bound-hand rotation relative to the controller pose.
    pub(super) hand_rotation: Quat,
}

/// Per-profile pose resolution for the palm-pose controller path.
///
/// When `XR_EXT_palm_pose` is available, `palm_ext` is reported directly as the controller pose.
/// Otherwise the OpenXR grip pose is converted into the same palm-oriented frame using fixed
/// profile offsets.
pub(super) fn resolve_controller_frame(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    palm_ext_pose: Option<(Vec3, Quat)>,
) -> Option<ControllerFrame> {
    let (position, rotation) = if let Some(palm_pose) = palm_ext_pose {
        openxr_tracking_pose_to_host(palm_pose.0, palm_pose.1)
    } else if let Some(grip_pose) = grip_pose {
        let palm_pose = grip_to_palm_ext_pose(profile, side, grip_pose);
        openxr_tracking_pose_to_host(palm_pose.0, palm_pose.1)
    } else {
        return None;
    };
    let (hand_position, hand_rotation) = hand_pose_defaults(side);

    Some(ControllerFrame {
        position,
        rotation,
        has_bound_hand: true,
        hand_position,
        hand_rotation,
    })
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::shared::Chirality;
    use crate::xr::session::openxr_tracking_pose_to_host;

    use super::super::pose::{grip_to_palm_ext_pose, hand_pose_defaults};
    use super::super::profile::ActiveControllerProfile;
    use super::resolve_controller_frame;

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

    fn rotation_delta_angle(a: Quat, b: Quat) -> f32 {
        2.0 * a
            .normalize()
            .dot(b.normalize())
            .abs()
            .clamp(-1.0, 1.0)
            .acos()
    }

    #[test]
    fn no_poses_available_returns_none() {
        assert!(
            resolve_controller_frame(
                ActiveControllerProfile::Generic,
                Chirality::Left,
                None,
                None
            )
            .is_none()
        );
    }

    #[test]
    fn grip_pose_falls_back_to_palm_ext_offset() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            None,
        )
        .expect("frame");
        let palm_pose = grip_to_palm_ext_pose(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            (grip_position, grip_rotation),
        );
        let (expected_position, expected_rotation) =
            openxr_tracking_pose_to_host(palm_pose.0, palm_pose.1);

        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
        assert!(frame.has_bound_hand);
    }

    #[test]
    fn palm_ext_pose_is_reported_directly_as_controller_pose() {
        let palm_position = Vec3::new(0.3, 1.2, -0.5);
        let palm_rotation = Quat::from_rotation_x(0.25).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            Some((Vec3::ZERO, Quat::IDENTITY)),
            Some((palm_position, palm_rotation)),
        )
        .expect("frame");
        let (expected_position, expected_rotation) =
            openxr_tracking_pose_to_host(palm_position, palm_rotation);

        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
        assert!(frame.hand_position.length() > 0.01);
        assert!(rotation_delta_angle(frame.hand_rotation, Quat::IDENTITY) > 0.2);
    }

    #[test]
    fn palm_ext_pose_takes_priority_over_grip() {
        let palm_position = Vec3::new(0.3, 1.2, -0.5);
        let palm_rotation = Quat::from_rotation_x(0.25).normalize();
        let grip_position = Vec3::new(0.9, 0.8, -0.1);
        let grip_rotation = Quat::from_rotation_y(0.8).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::Right,
            Some((grip_position, grip_rotation)),
            Some((palm_position, palm_rotation)),
        )
        .expect("frame");
        let (expected_position, expected_rotation) =
            openxr_tracking_pose_to_host(palm_position, palm_rotation);

        assert_vec3_near(frame.position, expected_position);
        assert_quat_near(frame.rotation, expected_rotation);
    }

    #[test]
    fn bound_hand_defaults_are_fixed_to_palm_pose_frame() {
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Touch,
            Chirality::Right,
            Some((Vec3::new(0.1, 1.0, -0.2), Quat::IDENTITY)),
            None,
        )
        .expect("frame");
        let (hand_position, hand_rotation) = hand_pose_defaults(Chirality::Right);

        assert_vec3_near(frame.hand_position, hand_position);
        assert_quat_near(frame.hand_rotation, hand_rotation);
    }
}
