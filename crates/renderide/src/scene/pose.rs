//! Pose validation and identity [`RenderTransform`](crate::shared::RenderTransform).

use glam::{Quat, Vec3};

use crate::shared::RenderTransform;

/// Maximum absolute value for position or scale before treating the pose as corrupt.
pub(super) const POSE_VALIDATION_THRESHOLD: f32 = 1e6;

/// Validates a host pose (NaN / inf / huge components).
pub struct PoseValidation<'a> {
    /// Pose under test.
    pub pose: &'a RenderTransform,
}

impl PoseValidation<'_> {
    /// Returns `true` if position, scale, and rotation quaternion are finite and within threshold.
    pub fn is_valid(&self) -> bool {
        let pos_ok = self.pose.position.x.is_finite()
            && self.pose.position.y.is_finite()
            && self.pose.position.z.is_finite()
            && self.pose.position.x.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.position.y.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.position.z.abs() < POSE_VALIDATION_THRESHOLD;
        if !pos_ok {
            return false;
        }
        let scale_ok = self.pose.scale.x.is_finite()
            && self.pose.scale.y.is_finite()
            && self.pose.scale.z.is_finite()
            && self.pose.scale.x.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.scale.y.abs() < POSE_VALIDATION_THRESHOLD
            && self.pose.scale.z.abs() < POSE_VALIDATION_THRESHOLD;
        if !scale_ok {
            return false;
        }

        self.pose.rotation.x.is_finite()
            && self.pose.rotation.y.is_finite()
            && self.pose.rotation.z.is_finite()
            && self.pose.rotation.w.is_finite()
    }
}

/// Identity local pose: origin, unit scale, identity rotation (`RenderTransform` / Unity TRS).
pub(super) fn render_transform_identity() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

#[cfg(test)]
mod tests {
    use super::{POSE_VALIDATION_THRESHOLD, PoseValidation, render_transform_identity};
    use glam::{Quat, Vec3};

    use crate::shared::RenderTransform;

    /// A sensible baseline pose that passes [`PoseValidation::is_valid`] so individual tests can
    /// mutate exactly one axis under test.
    fn baseline() -> RenderTransform {
        render_transform_identity()
    }

    /// The identity pose is always valid.
    #[test]
    fn identity_pose_is_valid() {
        let pose = baseline();
        assert!(PoseValidation { pose: &pose }.is_valid());
    }

    /// Any NaN on any component rejects the pose.
    #[test]
    fn nan_components_are_rejected() {
        let mutators: [fn(&mut RenderTransform); 10] = [
            |p: &mut RenderTransform| p.position.x = f32::NAN,
            |p: &mut RenderTransform| p.position.y = f32::NAN,
            |p: &mut RenderTransform| p.position.z = f32::NAN,
            |p: &mut RenderTransform| p.scale.x = f32::NAN,
            |p: &mut RenderTransform| p.scale.y = f32::NAN,
            |p: &mut RenderTransform| p.scale.z = f32::NAN,
            |p: &mut RenderTransform| p.rotation.x = f32::NAN,
            |p: &mut RenderTransform| p.rotation.y = f32::NAN,
            |p: &mut RenderTransform| p.rotation.z = f32::NAN,
            |p: &mut RenderTransform| p.rotation.w = f32::NAN,
        ];
        for mutate in mutators {
            let mut pose = baseline();
            mutate(&mut pose);
            assert!(
                !PoseValidation { pose: &pose }.is_valid(),
                "NaN mutation should invalidate pose: {pose:?}"
            );
        }
    }

    /// Infinite components are also rejected (position + scale + rotation all via `is_finite`).
    #[test]
    fn infinite_components_are_rejected() {
        let mut pose = baseline();
        pose.position = Vec3::new(f32::INFINITY, 0.0, 0.0);
        assert!(!PoseValidation { pose: &pose }.is_valid());

        pose = baseline();
        pose.scale = Vec3::new(1.0, f32::NEG_INFINITY, 1.0);
        assert!(!PoseValidation { pose: &pose }.is_valid());

        pose = baseline();
        pose.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, f32::INFINITY);
        assert!(!PoseValidation { pose: &pose }.is_valid());
    }

    /// The threshold check is strict (`<`), so values at exactly the threshold are rejected but
    /// values just below it are accepted.
    #[test]
    fn position_and_scale_threshold_is_strict() {
        let mut pose = baseline();
        pose.position.x = POSE_VALIDATION_THRESHOLD;
        assert!(!PoseValidation { pose: &pose }.is_valid());

        pose = baseline();
        pose.position.x = POSE_VALIDATION_THRESHOLD * 0.5;
        assert!(PoseValidation { pose: &pose }.is_valid());

        pose = baseline();
        pose.scale.y = -POSE_VALIDATION_THRESHOLD;
        assert!(!PoseValidation { pose: &pose }.is_valid());
    }

    /// The rotation quaternion is only required to be finite -- no normalization or threshold check
    /// applies. Rotations with large or non-unit components are still accepted.
    #[test]
    fn rotation_threshold_is_not_applied() {
        let mut pose = baseline();
        pose.rotation = Quat::from_xyzw(1e9, 1e9, 1e9, 1e9);
        assert!(PoseValidation { pose: &pose }.is_valid());
    }
}
