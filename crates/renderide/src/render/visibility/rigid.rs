//! Frustum culling for rigid (non-skinned) meshes.
//!
//! Transforms the draw's local [`RenderBoundingBox`] through its model matrix to produce a
//! world-space AABB, then tests every corner against the view-projection clip volume.
//!
//! # Conservative behaviour
//! When bounds are degenerate (near-zero or non-finite extents) or the world AABB corners cannot
//! be computed (non-finite model matrix output), the draw is **kept** (returns `true`).
//! This matches the policy in the collect loop: prefer false-positives over false-negatives.

use glam::{Mat4, Vec3};

use crate::shared::RenderBoundingBox;

use super::{mesh_bounds_degenerate_for_cull, world_aabb_visible_in_homogeneous_clip};

/// Transforms a local center/extents AABB through `model_matrix` into a world-space AABB.
///
/// Returns `None` when:
/// - Any of `center` or `extents` is non-finite (corrupt mesh metadata), or
/// - Any transformed corner is non-finite (degenerate or huge model matrix).
fn world_aabb_from_local_bounds(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
) -> Option<(Vec3, Vec3)> {
    let c = bounds.center;
    let e = bounds.extents;
    if !(c.x.is_finite()
        && c.y.is_finite()
        && c.z.is_finite()
        && e.x.is_finite()
        && e.y.is_finite()
        && e.z.is_finite())
    {
        return None;
    }
    let ex = e.x.abs();
    let ey = e.y.abs();
    let ez = e.z.abs();
    let min_l = Vec3::new(c.x - ex, c.y - ey, c.z - ez);
    let max_l = Vec3::new(c.x + ex, c.y + ey, c.z + ez);

    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for x in [min_l.x, max_l.x] {
        for y in [min_l.y, max_l.y] {
            for z in [min_l.z, max_l.z] {
                let p = model_matrix.transform_point3(Vec3::new(x, y, z));
                if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
                    return None;
                }
                wmin = wmin.min(p);
                wmax = wmax.max(p);
            }
        }
    }
    Some((wmin, wmax))
}

/// Returns `true` if the rigid mesh draw should be submitted (visible or indeterminate).
///
/// Tests the local [`RenderBoundingBox`] transformed by `model_matrix` against the
/// view-projection frustum. Conservatively returns `true` when:
/// - Bounds are degenerate (see [`mesh_bounds_degenerate_for_cull`]).
/// - Any world AABB corner is non-finite.
pub fn rigid_mesh_potentially_visible(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
    view_proj: Mat4,
) -> bool {
    if mesh_bounds_degenerate_for_cull(bounds) {
        return true;
    }
    let Some((wmin, wmax)) = world_aabb_from_local_bounds(bounds, model_matrix) else {
        return true;
    };
    world_aabb_visible_in_homogeneous_clip(view_proj, wmin, wmax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::pass::reverse_z_projection;
    use crate::scene::math::matrix_na_to_glam;
    use glam::Mat4;
    use nalgebra::{Matrix4 as NaMat4, Point3, Vector3 as NaVec3};

    fn look_vp_naive() -> Mat4 {
        let view = NaMat4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 0.0),
            &NaVec3::new(0.0, 1.0, 0.0),
        );
        let proj = reverse_z_projection(1.0, 60f32.to_radians(), 0.1, 100.0);
        matrix_na_to_glam(&(proj * view))
    }

    fn make_bounds(
        cx: f32,
        cy: f32,
        cz: f32,
        ex: f32,
        ey: f32,
        ez: f32,
    ) -> crate::shared::RenderBoundingBox {
        crate::shared::RenderBoundingBox {
            center: NaVec3::new(cx, cy, cz),
            extents: NaVec3::new(ex, ey, ez),
        }
    }

    #[test]
    fn box_in_front_of_camera_visible() {
        let vp = look_vp_naive();
        let b = make_bounds(0.0, 0.0, 0.0, 0.5, 0.5, 0.5);
        assert!(rigid_mesh_potentially_visible(&b, Mat4::IDENTITY, vp));
    }

    #[test]
    fn box_behind_camera_culled() {
        let vp = look_vp_naive();
        let b = make_bounds(0.0, 0.0, 20.0, 0.5, 0.5, 0.5);
        assert!(!rigid_mesh_potentially_visible(&b, Mat4::IDENTITY, vp));
    }

    #[test]
    fn box_far_left_culled() {
        let vp = look_vp_naive();
        let b = make_bounds(50.0, 0.0, 0.0, 0.5, 0.5, 0.5);
        assert!(!rigid_mesh_potentially_visible(&b, Mat4::IDENTITY, vp));
    }

    #[test]
    fn degenerate_zero_extents_conservative_not_culled() {
        let vp = look_vp_naive();
        // Tight box far left is culled.
        let tight = make_bounds(50.0, 0.0, 0.0, 0.5, 0.5, 0.5);
        assert!(!rigid_mesh_potentially_visible(&tight, Mat4::IDENTITY, vp));
        // Same position but zero extents is NOT culled (degenerate = conservative).
        let degenerate = make_bounds(50.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(rigid_mesh_potentially_visible(
            &degenerate,
            Mat4::IDENTITY,
            vp
        ));
    }

    #[test]
    fn bounds_degenerate_for_cull_detects_zero_extents() {
        use crate::render::visibility::mesh_bounds_degenerate_for_cull;
        let b_zero = make_bounds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(mesh_bounds_degenerate_for_cull(&b_zero));
        let b_good = make_bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert!(!mesh_bounds_degenerate_for_cull(&b_good));
    }
}
