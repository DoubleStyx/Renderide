//! Controller grip/aim pose math and OpenXR [`openxr::SpaceLocation`] conversion.
//!
//! Values follow the host-side pose convention expected by the `SteamVRDriver`-shaped
//! controller data path. The host (`VR_Manager`) writes the
//! received `position` / `rotation` straight into `RawPosition` / `RawRotation`, so the renderer
//! is responsible for delivering poses in the exact frame the host was authored against.

use glam::{EulerRot, Quat, Vec3};
use openxr as xr;

use crate::shared::Chirality;
use crate::xr::session::openxr_tracking_pose_to_host;

use super::profile::ActiveControllerProfile;

/// Builds a quaternion with the same Y-X-Z composition order Unity's `Quaternion.Euler` uses,
/// so per-profile rotation constants align with the host controller calibration data.
pub(super) fn unity_euler_deg(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_rotation_y(y.to_radians())
        * Quat::from_rotation_x(x.to_radians())
        * Quat::from_rotation_z(z.to_radians())
}

/// Local OpenXR-space transform from standardized grip pose to SteamVR-style raw controller pose.
#[derive(Clone, Copy)]
pub(super) struct RawFromGripTransform {
    rotation: Quat,
    translation: Vec3,
}

impl RawFromGripTransform {
    fn identity() -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        }
    }

    fn from_grip_from_raw(rotation: Quat, translation: Vec3) -> Self {
        let rotation = rotation.normalize();
        let raw_from_grip_rotation = rotation.inverse();
        Self {
            rotation: raw_from_grip_rotation,
            translation: raw_from_grip_rotation * -translation,
        }
    }

    fn apply(self, position: Vec3, rotation: Quat) -> (Vec3, Quat) {
        (
            position + rotation * self.translation,
            (rotation * self.rotation).normalize(),
        )
    }
}

/// Returns the local OpenXR-space transform from grip pose to SteamVR raw controller pose.
pub(super) fn steamvr_raw_from_openxr_grip(
    profile: ActiveControllerProfile,
    side: Chirality,
) -> RawFromGripTransform {
    let (grip_from_raw_rotation, grip_from_raw_translation) = match (profile, side) {
        (ActiveControllerProfile::Index, Chirality::Left) => (
            Quat::from_euler(
                EulerRot::XYZ,
                15.392_f32.to_radians(),
                -2.071_f32.to_radians(),
                0.303_f32.to_radians(),
            ),
            Vec3::new(0.0, -0.015, 0.13),
        ),
        (ActiveControllerProfile::Index, Chirality::Right) => (
            Quat::from_euler(
                EulerRot::XYZ,
                15.392_f32.to_radians(),
                2.071_f32.to_radians(),
                -0.303_f32.to_radians(),
            ),
            Vec3::new(0.0, -0.015, 0.13),
        ),
        (ActiveControllerProfile::Touch | ActiveControllerProfile::ViveFocus3, Chirality::Left) => {
            (
                Quat::from_euler(EulerRot::XYZ, 20.6_f32.to_radians(), 0.0, 0.0),
                Vec3::new(0.007, -0.001_829_41, 0.101_948_2),
            )
        }
        (
            ActiveControllerProfile::Touch | ActiveControllerProfile::ViveFocus3,
            Chirality::Right,
        ) => (
            Quat::from_euler(EulerRot::XYZ, 20.6_f32.to_radians(), 0.0, 0.0),
            Vec3::new(-0.007, -0.001_829_41, 0.101_948_2),
        ),
        _ => return RawFromGripTransform::identity(),
    };
    RawFromGripTransform::from_grip_from_raw(grip_from_raw_rotation, grip_from_raw_translation)
}

/// Converts an OpenXR grip pose into the SteamVR raw controller pose expected by host IPC.
///
/// OpenXR `/input/grip/pose` is a standardized hand grip frame. SteamVR raw poses are device
/// model frames, so controller models where those frames differ need a fixed local transform
/// before host-specific controller corrections are applied.
///
/// Calibration is composed in OpenXR right-handed tracking space first. The calibrated raw pose is
/// then converted into the host left-handed basis so controller frames and HMD frames enter
/// FrooxEngine in the same tracking space.
pub(super) fn openxr_grip_to_steamvr_raw(
    profile: ActiveControllerProfile,
    side: Chirality,
    position: Vec3,
    rotation: Quat,
) -> (Vec3, Quat) {
    let (raw_position, raw_rotation) =
        steamvr_raw_from_openxr_grip(profile, side).apply(position, rotation);
    openxr_tracking_pose_to_host(raw_position, raw_rotation)
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
/// The host does not hardcode these: `VR_Manager` forwards IPC `handPosition` / `handRotation` into
/// `MappableTrackedObject.Initialize` at registration.
///
/// `generic_fix = unity_euler_deg(90, 90, 90).inverse()` reproduces Unity's post-multiply
/// `Quaternion.Inverse(Quaternion.Euler(90,90,90))`, applied only to Vive / Touch / Generic in
/// that driver.
///
/// `hasBoundHand` is returned `true` for every profile. The host can otherwise prefer skeletal
/// hand tracking for some profiles, but the renderer has no skeletal hand tracking, so the
/// bound-hand defaults are always surfaced; Index / Cosmos / ViveFocus3 return identity so the
/// hand visual sits at the grip rather than at a wrist offset that was never calibrated for them.
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

/// Derives a controller grip-style pose from an OpenXR aim pose by stepping back along the
/// OpenXR forward axis to the approximate grip origin.
pub(super) fn controller_pose_from_aim(position: Vec3, rotation: Quat) -> (Vec3, Quat) {
    let rotation = rotation.normalize();
    let tip_offset = Vec3::new(0.0, 0.0, -0.075);
    (position - rotation * tip_offset, rotation)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::xr::session::openxr_tracking_pose_to_host;

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

    fn rotation_angle_from_identity(rotation: Quat) -> f32 {
        2.0 * rotation.normalize().w.abs().clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn calibrated_raw_from_grip_transforms_match_profile_table() {
        for (profile, side, grip_from_raw_rotation, grip_from_raw_translation) in [
            (
                ActiveControllerProfile::Index,
                Chirality::Left,
                Quat::from_euler(
                    EulerRot::XYZ,
                    15.392_f32.to_radians(),
                    -2.071_f32.to_radians(),
                    0.303_f32.to_radians(),
                ),
                Vec3::new(0.0, -0.015, 0.13),
            ),
            (
                ActiveControllerProfile::Index,
                Chirality::Right,
                Quat::from_euler(
                    EulerRot::XYZ,
                    15.392_f32.to_radians(),
                    2.071_f32.to_radians(),
                    -0.303_f32.to_radians(),
                ),
                Vec3::new(0.0, -0.015, 0.13),
            ),
            (
                ActiveControllerProfile::Touch,
                Chirality::Left,
                Quat::from_euler(EulerRot::XYZ, 20.6_f32.to_radians(), 0.0, 0.0),
                Vec3::new(0.007, -0.001_829_41, 0.101_948_2),
            ),
            (
                ActiveControllerProfile::Touch,
                Chirality::Right,
                Quat::from_euler(EulerRot::XYZ, 20.6_f32.to_radians(), 0.0, 0.0),
                Vec3::new(-0.007, -0.001_829_41, 0.101_948_2),
            ),
            (
                ActiveControllerProfile::ViveFocus3,
                Chirality::Left,
                Quat::from_euler(EulerRot::XYZ, 20.6_f32.to_radians(), 0.0, 0.0),
                Vec3::new(0.007, -0.001_829_41, 0.101_948_2),
            ),
            (
                ActiveControllerProfile::ViveFocus3,
                Chirality::Right,
                Quat::from_euler(EulerRot::XYZ, 20.6_f32.to_radians(), 0.0, 0.0),
                Vec3::new(-0.007, -0.001_829_41, 0.101_948_2),
            ),
        ] {
            let actual = steamvr_raw_from_openxr_grip(profile, side);
            let expected = RawFromGripTransform::from_grip_from_raw(
                grip_from_raw_rotation,
                grip_from_raw_translation,
            );
            assert_vec3_near(actual.translation, expected.translation);
            assert_quat_near(actual.rotation, expected.rotation);
        }
    }

    #[test]
    fn identity_calibration_profiles_emit_openxr_grip_unchanged_before_host_conversion() {
        let grip_position = Vec3::new(0.25, 1.4, -0.35);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();

        for profile in [
            ActiveControllerProfile::Vive,
            ActiveControllerProfile::WindowsMr,
            ActiveControllerProfile::HpReverbG2,
            ActiveControllerProfile::Pico4,
            ActiveControllerProfile::PicoNeo3,
            ActiveControllerProfile::ViveCosmos,
            ActiveControllerProfile::Generic,
            ActiveControllerProfile::Simple,
        ] {
            for side in [Chirality::Left, Chirality::Right] {
                let (position, rotation) =
                    steamvr_raw_from_openxr_grip(profile, side).apply(grip_position, grip_rotation);
                assert_vec3_near(position, grip_position);
                assert_quat_near(rotation, grip_rotation);
            }
        }
    }

    #[test]
    fn identity_calibration_profiles_convert_result_to_host_tracking_space() {
        let grip_position = Vec3::new(0.25, 1.4, -0.35);
        let grip_rotation = (Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.2)).normalize();
        let (expected_position, expected_rotation) =
            openxr_tracking_pose_to_host(grip_position, grip_rotation);

        let (position, rotation) = openxr_grip_to_steamvr_raw(
            ActiveControllerProfile::Vive,
            Chirality::Left,
            grip_position,
            grip_rotation,
        );

        assert_vec3_near(position, expected_position);
        assert_quat_near(rotation, expected_rotation);
    }

    #[test]
    fn raw_pose_offset_magnitudes_match_supported_profiles() {
        for (profile, expected_translation, angle_min, angle_max) in [
            (ActiveControllerProfile::Index, 0.130_86, 0.26, 0.28),
            (ActiveControllerProfile::Touch, 0.102_2, 0.35, 0.37),
            (ActiveControllerProfile::ViveFocus3, 0.102_2, 0.35, 0.37),
        ] {
            for side in [Chirality::Left, Chirality::Right] {
                let (position, rotation) =
                    steamvr_raw_from_openxr_grip(profile, side).apply(Vec3::ZERO, Quat::IDENTITY);
                assert!(
                    (position.length() - expected_translation).abs() < 1e-4,
                    "{profile:?} {side:?}: position offset {position:?}",
                );
                let angle = rotation_angle_from_identity(rotation);
                assert!(
                    (angle_min..=angle_max).contains(&angle),
                    "{profile:?} {side:?}: rotation angle {angle}",
                );
            }
        }
    }

    #[test]
    fn raw_pose_offsets_mirror_across_hands() {
        for profile in [
            ActiveControllerProfile::Index,
            ActiveControllerProfile::Touch,
            ActiveControllerProfile::ViveFocus3,
        ] {
            let (left_position, left_rotation) =
                steamvr_raw_from_openxr_grip(profile, Chirality::Left)
                    .apply(Vec3::ZERO, Quat::IDENTITY);
            let (right_position, right_rotation) =
                steamvr_raw_from_openxr_grip(profile, Chirality::Right)
                    .apply(Vec3::ZERO, Quat::IDENTITY);

            assert!(
                (left_position.x + right_position.x).abs() < 1e-4,
                "{profile:?}: position X should mirror: left={left_position:?}, right={right_position:?}",
            );
            assert!(
                (left_position.y - right_position.y).abs() < 1e-4,
                "{profile:?}: position Y should match: left={left_position:?}, right={right_position:?}",
            );
            assert!(
                (left_position.z - right_position.z).abs() < 1e-4,
                "{profile:?}: position Z should match: left={left_position:?}, right={right_position:?}",
            );
            let left_angle = rotation_angle_from_identity(left_rotation);
            let right_angle = rotation_angle_from_identity(right_rotation);
            assert!(
                (left_angle - right_angle).abs() < 1e-4,
                "{profile:?}: rotation angle should match: left={left_angle}, right={right_angle}",
            );
        }
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

    /// Pico4 / PicoNeo3 / HpReverbG2 share the Windows MR bound-hand values.
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
