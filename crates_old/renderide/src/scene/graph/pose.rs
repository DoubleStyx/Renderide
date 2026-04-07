//! Pose validation and identity transform helpers.

use nalgebra::{UnitQuaternion, Vector3};

use crate::shared::RenderTransform;

/// Maximum allowed absolute value for position or scale components before rejecting as corrupt.
pub(super) const POSE_VALIDATION_THRESHOLD: f32 = 1e6;

/// Validates pose data; rejects NaN, inf, or values with abs > 1e6.
pub struct PoseValidation<'a> {
    /// Pose to validate.
    pub pose: &'a RenderTransform,
    /// Frame index for error logging context.
    pub frame_index: i32,
    /// Scene ID for error logging context.
    pub scene_id: i32,
    /// Transform ID for error logging context.
    pub transform_id: i32,
}

impl PoseValidation<'_> {
    /// Returns true if the pose has no inf, NaN, or absurdly large values.
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

        self.pose.rotation.i.is_finite()
            && self.pose.rotation.j.is_finite()
            && self.pose.rotation.k.is_finite()
            && self.pose.rotation.w.is_finite()
    }
}

/// Returns a RenderTransform with identity rotation, zero position, and unit scale.
pub(super) fn render_transform_identity() -> RenderTransform {
    RenderTransform {
        position: Vector3::zeros(),
        scale: Vector3::new(1.0, 1.0, 1.0),
        rotation: UnitQuaternion::identity().into_inner(),
    }
}
