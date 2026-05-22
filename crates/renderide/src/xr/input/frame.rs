//! Per-frame resolved controller pose (grip/aim) before IPC mapping.

use glam::{Quat, Vec3};

use super::profile::ActiveControllerProfile;
use crate::shared::Chirality;
use crate::xr::input::pose::{grip_to_palm_ext_pose, hand_pose_defaults};
use crate::xr::session::openxr_tracking_pose_to_host;

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

/// Per-profile pose resolution. If a palm_ext pose is available we use that, otherwise we convert
/// a grip pose into it using fixed offsets in [`grip_to_palm_ext_pose`].
///
/// The hand pose is currently hardcoded to match the palm_ext pose.
pub(super) fn resolve_controller_frame(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    palm_ext_pose: Option<(Vec3, Quat)>,
) -> Option<ControllerFrame> {
    let (pos, rot) = if let Some(p) = palm_ext_pose {
        p
    } else if let Some(p) = grip_pose {
        grip_to_palm_ext_pose(profile, side, p)
    } else {
        return None;
    };

    let (position, rotation) = openxr_tracking_pose_to_host(pos, rot);
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
    fn no_poses_available_return_none() {
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
    fn grip_pose_gets_offset() {
        let grip_position = Vec3::new(0.3, 1.2, -0.5);
        let grip_rotation = Quat::from_rotation_x(0.25).normalize();
        let frame = resolve_controller_frame(
            ActiveControllerProfile::Generic,
            Chirality::Left,
            Some((grip_position, grip_rotation)),
            None,
        )
        .expect("frame");
        let (original_position, original_rotation) =
            openxr_tracking_pose_to_host(grip_position, grip_rotation);

        assert_ne!(frame.rotation, original_rotation);
        assert_ne!(frame.position, original_position);
    }

    #[test]
    fn palm_ext_pose_gets_used_as_controller_pose_if_available() {
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
}
