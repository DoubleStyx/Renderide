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
