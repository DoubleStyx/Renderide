//! Controller grip/aim pose math and OpenXR [`openxr::SpaceLocation`] conversion.

use glam::{Quat, Vec3};
use openxr as xr;

use crate::shared::Chirality;
use crate::xr::session::openxr_pose_to_host_tracking;

use super::profile::ActiveControllerProfile;

pub(super) fn unity_euler_deg(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_rotation_y(y.to_radians())
        * Quat::from_rotation_x(x.to_radians())
        * Quat::from_rotation_z(z.to_radians())
}

pub(super) fn touch_pose_correction(
    side: Chirality,
    position: Vec3,
    rotation: Quat,
) -> (Vec3, Quat) {
    let rotation = rotation * Quat::from_rotation_x(45.0_f32.to_radians());
    let offset = match side {
        Chirality::Left => Vec3::new(-0.01, 0.04, 0.03),
        Chirality::Right => Vec3::new(0.01, 0.04, 0.03),
    };
    (position - rotation * offset, rotation)
}

pub(super) fn index_pose_correction(
    side: Chirality,
    position: Vec3,
    rotation: Quat,
) -> (Vec3, Quat) {
    let roll = match side {
        Chirality::Left => 90.0_f32,
        Chirality::Right => -90.0_f32,
    };
    (
        position,
        rotation * Quat::from_rotation_z(roll.to_radians()),
    )
}

/// Default `hand_position` / `hand_rotation` on the IPC controller state types in
/// [`crate::shared`] for bound-hand tracking (FrooxEngine `BodyNodePositionOffset` /
/// `BodyNodeRotationOffset` on the hand device).
///
/// The host does not hardcode these: `VR_Manager` forwards IPC `handPosition` / `handRotation` into
/// `MappableTrackedObject.Initialize` at registration. Values here match the **SteamVR/OpenVR**
/// grip-frame convention: same Euler triples per controller class, plus
/// `handRotation *= Inverse(Euler(90°, 90°, 90°))` for Touch, Vive, and Generic (not for Index,
/// WMR, HP Reverb, Pico). The old Oculus runtime exposed its controllers via the OVR SDK's local
/// controller pose rather than a standard XR grip, so only the SteamVR/OpenVR pose is replicated.
///
/// `generic_fix` is `unity_euler_deg(90.0, 90.0, 90.0).inverse()` (equivalent to `Rx(-90°)`),
/// matching that SteamVR post-multiply.
pub(super) fn bound_hand_pose_defaults(
    profile: ActiveControllerProfile,
    side: Chirality,
) -> (bool, Vec3, Quat) {
    let generic_fix = unity_euler_deg(90.0, 90.0, 90.0).inverse();
    match (profile, side) {
        (ActiveControllerProfile::Touch, Chirality::Left) => (
            true,
            Vec3::new(-0.04, -0.025, -0.1),
            unity_euler_deg(185.0, -95.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Touch, Chirality::Right) => (
            true,
            Vec3::new(0.04, -0.025, -0.1),
            unity_euler_deg(5.0, -95.0, -90.0) * generic_fix,
        ),
        (
            ActiveControllerProfile::Vive
            | ActiveControllerProfile::Generic
            | ActiveControllerProfile::Simple,
            Chirality::Left,
        ) => (
            true,
            Vec3::new(-0.02, 0.0, -0.16),
            unity_euler_deg(140.0, -90.0, -90.0) * generic_fix,
        ),
        (
            ActiveControllerProfile::Vive
            | ActiveControllerProfile::Generic
            | ActiveControllerProfile::Simple,
            Chirality::Right,
        ) => (
            true,
            Vec3::new(0.02, 0.0, -0.16),
            unity_euler_deg(40.0, -90.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::WindowsMr, Chirality::Left) => (
            true,
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (ActiveControllerProfile::WindowsMr, Chirality::Right) => (
            true,
            Vec3::new(0.028, 0.0, -0.18),
            unity_euler_deg(30.0, -5.0, -100.0),
        ),
        (ActiveControllerProfile::Index, Chirality::Left) => (
            true,
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (ActiveControllerProfile::Index, Chirality::Right) => (
            true,
            Vec3::new(0.028, 0.0, -0.18),
            unity_euler_deg(30.0, -5.0, -100.0),
        ),
    }
}

/// Composes a parent pose with a child pose expressed in parent space (tests only).
#[cfg(test)]
pub(super) fn transform_pose(
    base_position: Vec3,
    base_rotation: Quat,
    local_position: Vec3,
    local_rotation: Quat,
) -> (Vec3, Quat) {
    (
        base_position + base_rotation * local_position,
        (base_rotation * local_rotation).normalize(),
    )
}

pub(super) fn inverse_transform_pose(
    base_position: Vec3,
    base_rotation: Quat,
    world_position: Vec3,
    world_rotation: Quat,
) -> (Vec3, Quat) {
    let inv = base_rotation.inverse();
    (
        inv * (world_position - base_position),
        (inv * world_rotation).normalize(),
    )
}

pub(super) fn controller_pose_from_aim(position: Vec3, rotation: Quat) -> (Vec3, Quat) {
    let rotation = rotation.normalize();
    let tip_offset = Vec3::new(0.0, 0.0, 0.075);
    (position - rotation * tip_offset, rotation)
}

/// Converts an [`xr::SpaceLocation`] into host-tracking-space `(position, rotation)` using only
/// [`openxr_pose_to_host_tracking`] (OpenXR RH → FrooxEngine/Unity LH). No extra grip-axis
/// correction is applied here; controller pose must match what [`bound_hand_pose_defaults`] was
/// authored against (SteamVR `Generic.Pose`–style tracking after the same conversion).
pub(super) fn pose_from_location(location: &xr::SpaceLocation) -> Option<(Vec3, Quat)> {
    let tracked = location
        .location_flags
        .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID);
    tracked.then(|| openxr_pose_to_host_tracking(&location.pose))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With `generic_fix` applied, the bound-hand rotation's local Y axis (palm normal) must
    /// have a dominant inward component (±X toward the other hand), not point forward (+Z) or
    /// strongly downward (-Y). This guards against regressions where `generic_fix` is
    /// accidentally removed.
    #[test]
    fn neutral_grip_palm_faces_inward_not_forward_generic() {
        for (side, expected_x_sign) in [(Chirality::Left, -1.0_f32), (Chirality::Right, 1.0_f32)] {
            let (_, _, hand_rot) = bound_hand_pose_defaults(ActiveControllerProfile::Generic, side);
            let palm_normal = hand_rot * Vec3::Y;
            assert!(
                palm_normal.x * expected_x_sign > 0.5,
                "{side:?}: palm normal {palm_normal:?} should have significant inward X component \
                 (expected sign {expected_x_sign}), got X={}",
                palm_normal.x,
            );
            assert!(
                palm_normal.z.abs() < 0.5,
                "{side:?}: palm normal {palm_normal:?} should not point strongly forward/back",
            );
        }
    }

    #[test]
    fn neutral_grip_palm_faces_inward_not_forward_touch() {
        for (side, expected_x_sign) in [(Chirality::Left, -1.0_f32), (Chirality::Right, 1.0_f32)] {
            let (_, _, hand_rot) = bound_hand_pose_defaults(ActiveControllerProfile::Touch, side);
            let palm_normal = hand_rot * Vec3::Y;
            assert!(
                palm_normal.x * expected_x_sign > 0.3,
                "{side:?}: palm normal {palm_normal:?} should have inward X component \
                 (expected sign {expected_x_sign}), got X={}",
                palm_normal.x,
            );
        }
    }

    #[test]
    fn bound_hand_chirality_mirrors_x_component() {
        let (_, pos_l, rot_l) =
            bound_hand_pose_defaults(ActiveControllerProfile::Generic, Chirality::Left);
        let (_, pos_r, rot_r) =
            bound_hand_pose_defaults(ActiveControllerProfile::Generic, Chirality::Right);
        assert!(
            (pos_l.x + pos_r.x).abs() < 1e-4,
            "position X should be mirrored: left={}, right={}",
            pos_l.x,
            pos_r.x,
        );
        let palm_l = rot_l * Vec3::Y;
        let palm_r = rot_r * Vec3::Y;
        assert!(
            (palm_l.x + palm_r.x).abs() < 0.15,
            "palm normal X should be approximately mirrored: left={palm_l:?}, right={palm_r:?}",
        );
    }
}
