//! Pose repair and identity [`RenderTransform`](crate::shared::RenderTransform).

use glam::{Quat, Vec3};

use crate::shared::RenderTransform;

/// Maximum absolute value for position or scale before repairing the pose component.
pub(super) const POSE_VALIDATION_THRESHOLD: f32 = 1e6;

/// Largest absolute position or scale value written after pose repair.
pub(super) const POSE_REPAIR_CLAMP_LIMIT: f32 = POSE_VALIDATION_THRESHOLD - 1.0;

/// Repairs a host pose so scene state stores finite, bounded position / scale and finite rotation.
pub(super) fn repair_render_transform(
    pose: &RenderTransform,
    fallback: &RenderTransform,
) -> RenderTransform {
    RenderTransform {
        position: Vec3::new(
            repair_bounded_component(pose.position.x, fallback.position.x, 0.0),
            repair_bounded_component(pose.position.y, fallback.position.y, 0.0),
            repair_bounded_component(pose.position.z, fallback.position.z, 0.0),
        ),
        scale: Vec3::new(
            repair_bounded_component(pose.scale.x, fallback.scale.x, 1.0),
            repair_bounded_component(pose.scale.y, fallback.scale.y, 1.0),
            repair_bounded_component(pose.scale.z, fallback.scale.z, 1.0),
        ),
        rotation: Quat::from_xyzw(
            repair_rotation_component(pose.rotation.x, fallback.rotation.x, 0.0),
            repair_rotation_component(pose.rotation.y, fallback.rotation.y, 0.0),
            repair_rotation_component(pose.rotation.z, fallback.rotation.z, 0.0),
            repair_rotation_component(pose.rotation.w, fallback.rotation.w, 1.0),
        ),
    }
}

fn repair_bounded_component(value: f32, fallback: f32, default: f32) -> f32 {
    if value.is_nan() {
        return repair_bounded_fallback(fallback, default);
    }
    value.clamp(-POSE_REPAIR_CLAMP_LIMIT, POSE_REPAIR_CLAMP_LIMIT)
}

fn repair_bounded_fallback(fallback: f32, default: f32) -> f32 {
    if fallback.is_nan() {
        return default;
    }
    fallback.clamp(-POSE_REPAIR_CLAMP_LIMIT, POSE_REPAIR_CLAMP_LIMIT)
}

fn repair_rotation_component(value: f32, fallback: f32, default: f32) -> f32 {
    if value.is_finite() {
        return value;
    }
    if fallback.is_finite() {
        return fallback;
    }
    default
}

/// Identity local pose: origin, unit scale, identity rotation (`RenderTransform` / Unity TRS).
pub(in crate::scene) fn render_transform_identity() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

#[cfg(test)]
mod tests {
    use super::{POSE_REPAIR_CLAMP_LIMIT, render_transform_identity, repair_render_transform};
    use glam::{Quat, Vec3};

    use crate::shared::RenderTransform;

    /// A sensible baseline pose so individual tests can mutate exactly one axis under test.
    fn baseline() -> RenderTransform {
        render_transform_identity()
    }

    fn repaired_components_are_valid(pose: &RenderTransform) {
        assert!(pose.position.x.is_finite());
        assert!(pose.position.y.is_finite());
        assert!(pose.position.z.is_finite());
        assert!(pose.position.x.abs() <= POSE_REPAIR_CLAMP_LIMIT);
        assert!(pose.position.y.abs() <= POSE_REPAIR_CLAMP_LIMIT);
        assert!(pose.position.z.abs() <= POSE_REPAIR_CLAMP_LIMIT);
        assert!(pose.scale.x.is_finite());
        assert!(pose.scale.y.is_finite());
        assert!(pose.scale.z.is_finite());
        assert!(pose.scale.x.abs() <= POSE_REPAIR_CLAMP_LIMIT);
        assert!(pose.scale.y.abs() <= POSE_REPAIR_CLAMP_LIMIT);
        assert!(pose.scale.z.abs() <= POSE_REPAIR_CLAMP_LIMIT);
        assert!(pose.rotation.x.is_finite());
        assert!(pose.rotation.y.is_finite());
        assert!(pose.rotation.z.is_finite());
        assert!(pose.rotation.w.is_finite());
    }

    /// Identity input repairs to itself.
    #[test]
    fn identity_pose_repairs_to_identity() {
        let pose = baseline();
        assert_eq!(
            repair_render_transform(&pose, &baseline()).position,
            pose.position
        );
        assert_eq!(
            repair_render_transform(&pose, &baseline()).scale,
            pose.scale
        );
        assert_eq!(
            repair_render_transform(&pose, &baseline()).rotation,
            pose.rotation
        );
    }

    /// Huge finite and infinite bounded components clamp to the valid range.
    #[test]
    fn bounded_components_are_clamped() {
        let mut pose = baseline();
        pose.position = Vec3::new(f32::INFINITY, -2.0e7, 3.0);
        pose.scale = Vec3::new(4.0e8, f32::NEG_INFINITY, -5.0);

        let repaired = repair_render_transform(&pose, &baseline());

        assert_eq!(repaired.position.x, POSE_REPAIR_CLAMP_LIMIT);
        assert_eq!(repaired.position.y, -POSE_REPAIR_CLAMP_LIMIT);
        assert_eq!(repaired.position.z, 3.0);
        assert_eq!(repaired.scale.x, POSE_REPAIR_CLAMP_LIMIT);
        assert_eq!(repaired.scale.y, -POSE_REPAIR_CLAMP_LIMIT);
        assert_eq!(repaired.scale.z, -5.0);
        repaired_components_are_valid(&repaired);
    }

    /// NaN bounded components fall back component-wise to the prior scene pose.
    #[test]
    fn nan_bounded_components_use_valid_fallback_components() {
        let mut pose = baseline();
        pose.position.x = f32::NAN;
        pose.scale.y = f32::NAN;
        let fallback = RenderTransform {
            position: Vec3::new(9.0, 8.0, 7.0),
            scale: Vec3::new(6.0, 5.0, 4.0),
            rotation: Quat::from_xyzw(0.1, 0.2, 0.3, 0.9),
        };

        let repaired = repair_render_transform(&pose, &fallback);

        assert_eq!(repaired.position.x, 9.0);
        assert_eq!(repaired.position.y, 0.0);
        assert_eq!(repaired.position.z, 0.0);
        assert_eq!(repaired.scale.x, 1.0);
        assert_eq!(repaired.scale.y, 5.0);
        assert_eq!(repaired.scale.z, 1.0);
        repaired_components_are_valid(&repaired);
    }

    /// Non-finite rotation components fall back component-wise because rotation has no clamp range.
    #[test]
    fn non_finite_rotation_components_use_fallback_or_identity() {
        let mut pose = baseline();
        pose.rotation = Quat::from_xyzw(f32::NAN, f32::INFINITY, 0.25, f32::NEG_INFINITY);
        let fallback = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::from_xyzw(0.4, f32::NAN, 0.6, 0.8),
        };

        let repaired = repair_render_transform(&pose, &fallback);

        assert_eq!(repaired.rotation.x, 0.4);
        assert_eq!(repaired.rotation.y, 0.0);
        assert_eq!(repaired.rotation.z, 0.25);
        assert_eq!(repaired.rotation.w, 0.8);
        repaired_components_are_valid(&repaired);
    }
}
