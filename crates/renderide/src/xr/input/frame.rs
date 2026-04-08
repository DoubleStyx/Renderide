//! Per-frame resolved controller pose (grip/aim) before IPC mapping.

use glam::{Quat, Vec3};

use crate::shared::Chirality;

use super::pose::{
    bound_hand_pose_defaults, controller_pose_from_aim, index_pose_correction,
    inverse_transform_pose, touch_pose_correction,
};
use super::profile::ActiveControllerProfile;

/// Resolved controller and optional bound-hand pose in tracking space.
#[derive(Clone, Copy)]
pub(super) struct ControllerFrame {
    pub(super) position: Vec3,
    pub(super) rotation: Quat,
    pub(super) has_bound_hand: bool,
    pub(super) hand_position: Vec3,
    pub(super) hand_rotation: Quat,
}

pub(super) fn resolve_controller_frame(
    profile: ActiveControllerProfile,
    side: Chirality,
    grip_pose: Option<(Vec3, Quat)>,
    aim_pose: Option<(Vec3, Quat)>,
) -> Option<ControllerFrame> {
    let (has_bound_hand, hand_position_default, hand_rotation_default) =
        bound_hand_pose_defaults(profile, side);
    match profile {
        ActiveControllerProfile::Touch => {
            let (grip_position, grip_rotation) = grip_pose?;
            let (position, rotation) = touch_pose_correction(side, grip_position, grip_rotation);
            Some(ControllerFrame {
                position,
                rotation,
                has_bound_hand,
                hand_position: hand_position_default,
                hand_rotation: hand_rotation_default,
            })
        }
        ActiveControllerProfile::Index => {
            if let (Some((aim_position, aim_rotation)), Some((grip_position, grip_rotation))) =
                (aim_pose, grip_pose)
            {
                let (position, rotation) = controller_pose_from_aim(aim_position, aim_rotation);
                let (hand_world_position, hand_world_rotation) =
                    index_pose_correction(side, grip_position, grip_rotation);
                let (hand_position, hand_rotation) = inverse_transform_pose(
                    position,
                    rotation,
                    hand_world_position,
                    hand_world_rotation,
                );
                Some(ControllerFrame {
                    position,
                    rotation,
                    has_bound_hand,
                    hand_position,
                    hand_rotation,
                })
            } else if let Some((aim_position, aim_rotation)) = aim_pose {
                let (position, rotation) = controller_pose_from_aim(aim_position, aim_rotation);
                Some(ControllerFrame {
                    position,
                    rotation,
                    has_bound_hand,
                    hand_position: hand_position_default,
                    hand_rotation: hand_rotation_default,
                })
            } else if let Some((grip_position, grip_rotation)) = grip_pose {
                let (position, rotation) =
                    index_pose_correction(side, grip_position, grip_rotation);
                Some(ControllerFrame {
                    position,
                    rotation,
                    has_bound_hand,
                    hand_position: hand_position_default,
                    hand_rotation: hand_rotation_default,
                })
            } else {
                None
            }
        }
        _ => {
            let (position, rotation) = grip_pose?;
            Some(ControllerFrame {
                position,
                rotation,
                has_bound_hand,
                hand_position: hand_position_default,
                hand_rotation: hand_rotation_default,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::shared::Chirality;

    use super::super::pose::{controller_pose_from_aim, index_pose_correction, transform_pose};
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

    #[test]
    fn index_frame_uses_aim_for_controller_and_grip_for_bound_hand() {
        let grip_position = Vec3::new(0.2, 1.3, -0.4);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let aim_position = Vec3::new(0.24, 1.34, -0.28);
        let aim_rotation = (Quat::from_rotation_y(0.75) * Quat::from_rotation_x(-0.1)).normalize();

        let frame = resolve_controller_frame(
            ActiveControllerProfile::Index,
            Chirality::left,
            Some((grip_position, grip_rotation)),
            Some((aim_position, aim_rotation)),
        )
        .expect("frame");

        let (expected_controller_position, expected_controller_rotation) =
            controller_pose_from_aim(aim_position, aim_rotation);
        assert_vec3_near(frame.position, expected_controller_position);
        assert_quat_near(frame.rotation, expected_controller_rotation);

        let (hand_world_position, hand_world_rotation) = transform_pose(
            frame.position,
            frame.rotation,
            frame.hand_position,
            frame.hand_rotation,
        );
        let (expected_hand_position, expected_hand_rotation) =
            index_pose_correction(Chirality::left, grip_position, grip_rotation);
        assert_vec3_near(hand_world_position, expected_hand_position);
        assert_quat_near(hand_world_rotation, expected_hand_rotation);
    }
}
