//! Controller grip/aim pose math and OpenXR [`openxr::SpaceLocation`] conversion.
//!
//! Values are a port of the host-side pose convention from the original Unity renderer's
//! `SteamVRDriver` (`UpdateController` and `InitHandData`). The host (`VR_Manager`) writes the
//! received `position` / `rotation` straight into `RawPosition` / `RawRotation`, so the renderer
//! is responsible for delivering poses in the exact frame the host was authored against.

use glam::{Quat, Vec3};
use openxr as xr;

use crate::shared::Chirality;
use crate::xr::session::openxr_pose_to_host_tracking;

use super::profile::ActiveControllerProfile;

/// Builds a quaternion with the same Y-X-Z composition order Unity's `Quaternion.Euler` uses,
/// so per-profile rotation constants taken from `SteamVRDriver.InitHandData` can be transcribed
/// verbatim.
pub(super) fn unity_euler_deg(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_rotation_y(y.to_radians())
        * Quat::from_rotation_x(x.to_radians())
        * Quat::from_rotation_z(z.to_radians())
}

/// Touch-only grip correction from `SteamVRDriver.UpdateController`: the real Oculus Touch is the
/// one device SteamVR's raw pose was noticeably tilted and offset relative to where the host's
/// hand anchor should sit.
///
/// OpenXR grip pose is not identical to SteamVR's raw pose, so a residual per-runtime offset is
/// expected after this correction. Other controllers apply no grip correction in Unity and none
/// here either.
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

/// Default `hand_position` / `hand_rotation` on the IPC controller state types in
/// [`crate::shared`] for bound-hand tracking (FrooxEngine `BodyNodePositionOffset` /
/// `BodyNodeRotationOffset` on the hand device).
///
/// Ported from the original Unity renderer's `SteamVRDriver.InitHandData` (decompiled
/// `Assembly-CSharp` lines 259-414; source project lines 309-476). The host does not hardcode
/// these: `VR_Manager` forwards IPC `handPosition` / `handRotation` into
/// `MappableTrackedObject.Initialize` at registration.
///
/// `generic_fix = unity_euler_deg(90, 90, 90).inverse()` reproduces Unity's post-multiply
/// `Quaternion.Inverse(Quaternion.Euler(90,90,90))`, applied only to Vive / Touch / Generic in
/// that driver.
///
/// Divergence from Unity: `hasBoundHand` is returned `true` for every profile. Unity ties the
/// flag to a runtime `DisableSkeletalModel` config (default `false`) for Index / Vive / Touch /
/// Cosmos, falling back to skeletal hand tracking otherwise. The renderer has no skeletal hand
/// tracking, so the bound-hand defaults are always surfaced; Index / Cosmos / ViveFocus3 return
/// identity so the hand visual sits at the grip rather than at a wrist offset that was never
/// calibrated for them.
pub(super) fn bound_hand_pose_defaults(
    profile: ActiveControllerProfile,
    side: Chirality,
) -> (bool, Vec3, Quat) {
    let generic_fix = unity_euler_deg(90.0, 90.0, 90.0).inverse();
    let (position, rotation) = match (profile, side) {
        (ActiveControllerProfile::Touch, Chirality::Left) => (
            Vec3::new(-0.04, -0.025, -0.1),
            unity_euler_deg(185.0, -95.0, -90.0) * generic_fix,
        ),
        (ActiveControllerProfile::Touch, Chirality::Right) => (
            Vec3::new(0.04, -0.025, -0.1),
            unity_euler_deg(5.0, -95.0, -90.0) * generic_fix,
        ),
        (
            ActiveControllerProfile::Vive
            | ActiveControllerProfile::Generic
            | ActiveControllerProfile::Simple,
            Chirality::Left,
        ) => (
            Vec3::new(-0.02, 0.0, -0.16),
            unity_euler_deg(140.0, -90.0, -90.0) * generic_fix,
        ),
        (
            ActiveControllerProfile::Vive
            | ActiveControllerProfile::Generic
            | ActiveControllerProfile::Simple,
            Chirality::Right,
        ) => (
            Vec3::new(0.02, 0.0, -0.16),
            unity_euler_deg(40.0, -90.0, -90.0) * generic_fix,
        ),
        (
            ActiveControllerProfile::WindowsMr
            | ActiveControllerProfile::HpReverbG2
            | ActiveControllerProfile::Pico4
            | ActiveControllerProfile::PicoNeo3,
            Chirality::Left,
        ) => (
            Vec3::new(-0.028, 0.0, -0.18),
            unity_euler_deg(30.0, 5.0, 100.0),
        ),
        (
            ActiveControllerProfile::WindowsMr
            | ActiveControllerProfile::HpReverbG2
            | ActiveControllerProfile::Pico4
            | ActiveControllerProfile::PicoNeo3,
            Chirality::Right,
        ) => (
            Vec3::new(0.028, 0.0, -0.18),
            unity_euler_deg(30.0, -5.0, -100.0),
        ),
        (
            ActiveControllerProfile::Index
            | ActiveControllerProfile::ViveCosmos
            | ActiveControllerProfile::ViveFocus3,
            _,
        ) => (Vec3::ZERO, Quat::IDENTITY),
    };
    (true, position, rotation)
}

/// Derives a controller grip-style pose from an aim pose by stepping back along the forward axis
/// to the approximate grip origin. Used only when the OpenXR grip pose is invalid but aim is
/// still tracked.
pub(super) fn controller_pose_from_aim(position: Vec3, rotation: Quat) -> (Vec3, Quat) {
    let rotation = rotation.normalize();
    let tip_offset = Vec3::new(0.0, 0.0, 0.075);
    (position - rotation * tip_offset, rotation)
}

/// Converts an [`xr::SpaceLocation`] into host-tracking-space `(position, rotation)` using only
/// [`openxr_pose_to_host_tracking`] (OpenXR RH -> FrooxEngine/Unity LH).
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
    tracked.then(|| openxr_pose_to_host_tracking(&location.pose))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bound-hand rotations for profiles Unity post-multiplies with `generic_fix` must have a
    /// palm normal pointing inward (+/-X toward the other hand), not forward or down. Guards
    /// against accidentally dropping `generic_fix`.
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
        for profile in [
            ActiveControllerProfile::Generic,
            ActiveControllerProfile::Touch,
            ActiveControllerProfile::Vive,
            ActiveControllerProfile::WindowsMr,
            ActiveControllerProfile::HpReverbG2,
            ActiveControllerProfile::Pico4,
            ActiveControllerProfile::PicoNeo3,
        ] {
            let (_, pos_l, rot_l) = bound_hand_pose_defaults(profile, Chirality::Left);
            let (_, pos_r, rot_r) = bound_hand_pose_defaults(profile, Chirality::Right);
            assert!(
                (pos_l.x + pos_r.x).abs() < 1e-4,
                "{profile:?}: position X should be mirrored: left={}, right={}",
                pos_l.x,
                pos_r.x,
            );
            let palm_l = rot_l * Vec3::Y;
            let palm_r = rot_r * Vec3::Y;
            assert!(
                (palm_l.x + palm_r.x).abs() < 0.15,
                "{profile:?}: palm normal X should be approximately mirrored: left={palm_l:?}, right={palm_r:?}",
            );
        }
    }

    /// Profiles without calibrated hand offsets (Index, ViveCosmos, ViveFocus3) return identity
    /// pose so the hand visual sits at the grip origin instead of at a wrong wrist offset.
    #[test]
    fn identity_profiles_have_zero_offset_and_identity_rotation() {
        for profile in [
            ActiveControllerProfile::Index,
            ActiveControllerProfile::ViveCosmos,
            ActiveControllerProfile::ViveFocus3,
        ] {
            for side in [Chirality::Left, Chirality::Right] {
                let (has, pos, rot) = bound_hand_pose_defaults(profile, side);
                assert!(has, "{profile:?} {side:?}: has_bound_hand must be true");
                assert!(
                    pos.length() < 1e-6,
                    "{profile:?} {side:?}: expected zero position, got {pos:?}",
                );
                let dot = rot.normalize().dot(Quat::IDENTITY).abs();
                assert!(
                    (1.0 - dot) < 1e-6,
                    "{profile:?} {side:?}: expected identity rotation, got {rot:?}",
                );
            }
        }
    }

    /// Pico4 / PicoNeo3 / HpReverbG2 share the Windows MR bound-hand values (matching Unity's
    /// PicoNeo2 / HPReverb / WindowsMR entries).
    #[test]
    fn pico_and_reverb_share_windowsmr_defaults() {
        let reference = [
            bound_hand_pose_defaults(ActiveControllerProfile::WindowsMr, Chirality::Left),
            bound_hand_pose_defaults(ActiveControllerProfile::WindowsMr, Chirality::Right),
        ];
        for profile in [
            ActiveControllerProfile::HpReverbG2,
            ActiveControllerProfile::Pico4,
            ActiveControllerProfile::PicoNeo3,
        ] {
            for (side, expected) in [
                (Chirality::Left, reference[0]),
                (Chirality::Right, reference[1]),
            ] {
                let got = bound_hand_pose_defaults(profile, side);
                assert_eq!(got.0, expected.0, "{profile:?} {side:?}");
                assert!(
                    (got.1 - expected.1).length() < 1e-6,
                    "{profile:?} {side:?}: position {:?} vs {:?}",
                    got.1,
                    expected.1,
                );
                let dot = got.2.normalize().dot(expected.2.normalize()).abs();
                assert!(
                    (1.0 - dot) < 1e-6,
                    "{profile:?} {side:?}: rotation {:?} vs {:?}",
                    got.2,
                    expected.2,
                );
            }
        }
    }
}
