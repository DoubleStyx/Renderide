//! Shared CPU bounding-volume math.

use glam::{Mat4, Vec3, Vec4Swizzles};

use crate::shared::RenderBoundingBox;

/// Epsilon for treating a model matrix bottom row as affine `[0, 0, 0, 1]`.
const MODEL_MATRIX_AFFINE_BOTTOM_EPS: f32 = 1e-4;

fn model_matrix_is_affine_bottom_row(m: Mat4) -> bool {
    let r = m.row(3);
    r.x.abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
        && r.y.abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
        && r.z.abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
        && (r.w - 1.0).abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
}

fn world_aabb_from_local_bounds_affine(
    bounds: &RenderBoundingBox,
    m: Mat4,
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

    let center_w = m.transform_point3(Vec3::new(c.x, c.y, c.z));
    if !(center_w.x.is_finite() && center_w.y.is_finite() && center_w.z.is_finite()) {
        return None;
    }

    let c0 = m.x_axis.xyz();
    let c1 = m.y_axis.xyz();
    let c2 = m.z_axis.xyz();

    let hx =
        c2.x.abs()
            .mul_add(ez, c0.x.abs().mul_add(ex, c1.x.abs() * ey));
    let hy =
        c2.y.abs()
            .mul_add(ez, c0.y.abs().mul_add(ex, c1.y.abs() * ey));
    let hz =
        c2.z.abs()
            .mul_add(ez, c0.z.abs().mul_add(ex, c1.z.abs() * ey));

    if !(hx.is_finite() && hy.is_finite() && hz.is_finite()) {
        return None;
    }

    let half = Vec3::new(hx, hy, hz);
    let wmin = center_w - half;
    let wmax = center_w + half;
    if !(wmin.x.is_finite()
        && wmin.y.is_finite()
        && wmin.z.is_finite()
        && wmax.x.is_finite()
        && wmax.y.is_finite()
        && wmax.z.is_finite())
    {
        return None;
    }
    Some((wmin, wmax))
}

fn world_aabb_from_local_bounds_bruteforce(
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

/// Transforms a local center/extents AABB through `model_matrix` into a world-space AABB.
pub fn world_aabb_from_local_bounds(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
) -> Option<(Vec3, Vec3)> {
    if model_matrix_is_affine_bottom_row(model_matrix) {
        world_aabb_from_local_bounds_affine(bounds, model_matrix)
    } else {
        world_aabb_from_local_bounds_bruteforce(bounds, model_matrix)
    }
}
