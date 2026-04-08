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

pub(super) fn pose_from_location(location: &xr::SpaceLocation) -> Option<(Vec3, Quat)> {
    let tracked = location
        .location_flags
        .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        && location
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID);
    tracked.then(|| openxr_pose_to_host_tracking(&location.pose))
}
