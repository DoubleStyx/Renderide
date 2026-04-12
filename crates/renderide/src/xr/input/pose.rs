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
        Chirality::left => Vec3::new(-0.01, 0.04, 0.03),
        Chirality::right => Vec3::new(0.01, 0.04, 0.03),
    };
    (position - rotation * offset, rotation)
}

pub(super) fn index_pose_correction(
    side: Chirality,
    position: Vec3,
    rotation: Quat,
) -> (Vec3, Quat) {
    let roll = match side {
        Chirality::left => 90.0_f32,
        Chirality::right => -90.0_f32,
    };
    (
        position,
        rotation * Quat::from_rotation_z(roll.to_radians()),
    )
}

/// Default `BodyNodeRotationOffset` and position for the bound-hand body node relative to the
/// controller's tracked pose. The rotation orients the avatar hand bone in the controller's local
/// frame; `generic_fix` (`Rx(-90°)`) rotates the palm normal from the controller's +Z (forward)
/// axis to ±X (inward toward the other hand).
pub(super) fn bound_hand_pose_defaults(
    profile: ActiveControllerProfile,
    side: Chirality,
) -> (bool, Vec3, Quat) {
    let generic_fix = unity_euler_deg(90.0, 90.0, 90.0).inverse();
    match (profile, side) {
        (ActiveControllerProfile::Touch, Chirality::left) => (
            true,
            Vec3::new(-0.04, -0.025, -0.1),
            unity_euler_deg(185.0, -95.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Touch, Chirality::right) => (
            true,
            Vec3::new(0.04, -0.025, -0.1),
            unity_euler_deg(5.0, -95.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Vive, Chirality::left)
        | (ActiveControllerProfile::Generic, Chirality::left)
        | (ActiveControllerProfile::Simple, Chirality::left) => (
            true,
            Vec3::new(-0.02, 0.0, -0.16),
            unity_euler_deg(140.0, -90.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Vive, Chirality::right)
        | (ActiveControllerProfile::Generic, Chirality::right)
        | (ActiveControllerProfile::Simple, Chirality::right) => (
            true,
            Vec3::new(0.02, 0.0, -0.16),
            unity_euler_deg(40.0, -90.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::WindowsMr, Chirality::left) => (
            true,
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (ActiveControllerProfile::WindowsMr, Chirality::right) => (
            true,
            Vec3::new(0.028, 0.0, -0.18),
            unity_euler_deg(30.0, -5.0, -100.0),
        ),
        (ActiveControllerProfile::Index, Chirality::left) => (
            true,
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (ActiveControllerProfile::Index, Chirality::right) => (
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

/// Universal correction applied to every OpenXR grip/aim pose after the RH-to-LH conversion
/// (`openxr_pose_to_host_tracking`) to align with the host convention. Set to [`Quat::IDENTITY`]
/// when no systematic offset exists between OpenXR grip orientation and the old driver convention.
const GRIP_TO_HOST_CORRECTION: Quat = Quat::IDENTITY;

pub(super) fn pose_from_location(location: &xr::SpaceLocation) -> Option<(Vec3, Quat)> {
    let tracked = location
        .location_flags
        .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID);
    tracked.then(|| {
        let (pos, rot) = openxr_pose_to_host_tracking(&location.pose);
        (pos, (rot * GRIP_TO_HOST_CORRECTION).normalize())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default bound-hand rotations must not pitch the palm strongly downward (-Y). The exact
    /// palm direction (X vs Z emphasis) follows [`unity_euler_deg`] and profile-specific defaults;
    /// this guards against regressions like the old `generic_fix` that pitched the palm down.
    #[test]
    fn neutral_grip_palm_faces_inward_not_down_generic() {
        for side in [Chirality::left, Chirality::right] {
            let (_, _, hand_rot) = bound_hand_pose_defaults(ActiveControllerProfile::Generic, side);
            let palm_normal = hand_rot * Vec3::Y;
            assert!(
                palm_normal.y > -0.4,
                "{side:?}: palm normal {palm_normal:?} should not point strongly downward",
            );
            let horizontal = (palm_normal.x * palm_normal.x + palm_normal.z * palm_normal.z).sqrt();
            assert!(
                horizontal > palm_normal.y.abs(),
                "{side:?}: palm horizontal span should exceed |Y| (not tipped to floor): {palm_normal:?}",
            );
        }
    }

    #[test]
    fn neutral_grip_palm_faces_inward_not_down_touch() {
        for side in [Chirality::left, Chirality::right] {
            let (_, _, hand_rot) = bound_hand_pose_defaults(ActiveControllerProfile::Touch, side);
            let palm_normal = hand_rot * Vec3::Y;
            assert!(
                palm_normal.y > -0.4,
                "{side:?}: palm normal {palm_normal:?} should not point strongly downward",
            );
            let horizontal = (palm_normal.x * palm_normal.x + palm_normal.z * palm_normal.z).sqrt();
            assert!(
                horizontal > palm_normal.y.abs(),
                "{side:?}: palm horizontal span should exceed |Y| (not tipped to floor): {palm_normal:?}",
            );
        }
    }

    #[test]
    fn bound_hand_chirality_mirrors_x_component() {
        let (_, pos_l, rot_l) =
            bound_hand_pose_defaults(ActiveControllerProfile::Generic, Chirality::left);
        let (_, pos_r, rot_r) =
            bound_hand_pose_defaults(ActiveControllerProfile::Generic, Chirality::right);
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

    #[test]
    fn grip_to_host_correction_is_identity() {
        assert_eq!(
            GRIP_TO_HOST_CORRECTION,
            Quat::IDENTITY,
            "GRIP_TO_HOST_CORRECTION should be identity unless empirical testing shows otherwise",
        );
    }
}
