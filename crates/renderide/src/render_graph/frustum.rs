//! Six-plane view frustum and world AABB tests for CPU mesh culling.
//!
//! **Production culling** uses [`world_aabb_visible_in_homogeneous_clip`], which matches
//! `clip = view_proj * vec4(world, 1)` exactly (same as WGSL `mat4x4` × `vec4`).
//!
//! [`Frustum`] / six-plane tests remain for debugging and optional comparisons.

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

use crate::shared::RenderBoundingBox;

/// Epsilon for homogeneous clip half-space tests (aligned with legacy visibility checks).
pub const HOMOGENEOUS_CLIP_EPS: f32 = 1e-5;

/// Maximum absolute half-extent below which uploaded mesh bounds are treated as **untrusted** for culling.
pub(crate) const DEGENERATE_MESH_BOUNDS_EXTENT_EPS: f32 = 1e-8;

/// Epsilon for treating a model matrix bottom row as affine `[0, 0, 0, 1]`.
const MODEL_MATRIX_AFFINE_BOTTOM_EPS: f32 = 1e-4;

/// Returns `true` when bounds should not be used for culling (keep the draw).
pub fn mesh_bounds_degenerate_for_cull(bounds: &RenderBoundingBox) -> bool {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return true;
    }
    let m = e.x.abs().max(e.y.abs()).max(e.z.abs());
    m < DEGENERATE_MESH_BOUNDS_EXTENT_EPS
}

/// Largest absolute half-extent along any axis; `0` if extents are non-finite.
pub fn mesh_bounds_max_half_extent(bounds: &RenderBoundingBox) -> f32 {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return 0.0;
    }
    e.x.abs().max(e.y.abs()).max(e.z.abs())
}

/// A plane `n · x + d = 0` with unit `n`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    /// Outward-facing unit normal of the clip half-space.
    pub normal: Vec3,
    /// Signed distance term in `n · x + d = 0`.
    pub distance: f32,
}

impl Plane {
    /// Builds a plane from a row of the transposed clip matrix `(a, b, c, w)` and normalizes.
    pub fn from_clip_row(v: Vec4) -> Self {
        let n = v.truncate();
        let len = n.length();
        if len < 1e-20 || !len.is_finite() {
            return Self {
                normal: Vec3::Y,
                distance: 0.0,
            };
        }
        Self {
            normal: n / len,
            distance: v.w / len,
        }
    }

    /// Signed distance from `p` to this plane; negative inside the frustum half-space for clip planes.
    #[inline]
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.distance
    }
}

/// Six clip planes extracted from a column-major `view_proj` matching `clip = view_proj * vec4(world, 1)`.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// Left, right, bottom, top, near, far clip planes in world space.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extracts frustum planes from `view_proj` using the transpose + row combination method
    /// (Gribb–Hartmann style), matching common HLSL references for column-major matrices.
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let m = view_proj.transpose();
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);
        Self {
            planes: [
                Plane::from_clip_row(r3 + r0),
                Plane::from_clip_row(r3 - r0),
                Plane::from_clip_row(r3 + r1),
                Plane::from_clip_row(r3 - r1),
                Plane::from_clip_row(r3 + r2),
                Plane::from_clip_row(r3 - r2),
            ],
        }
    }

    /// Returns `true` if the axis-aligned box may intersect the frustum (conservative).
    #[inline]
    pub fn intersects_aabb(&self, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        for plane in &self.planes {
            let p = Vec3::new(
                if plane.normal.x >= 0.0 {
                    aabb_max.x
                } else {
                    aabb_min.x
                },
                if plane.normal.y >= 0.0 {
                    aabb_max.y
                } else {
                    aabb_min.y
                },
                if plane.normal.z >= 0.0 {
                    aabb_max.z
                } else {
                    aabb_min.z
                },
            );
            if plane.signed_distance(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

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

    let hx = c0.x.abs() * ex + c1.x.abs() * ey + c2.x.abs() * ez;
    let hy = c0.y.abs() * ex + c1.y.abs() * ey + c2.y.abs() * ez;
    let hz = c0.z.abs() * ex + c1.z.abs() * ey + c2.z.abs() * ez;

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

/// Returns `true` if the axis-aligned world box may intersect the clip volume.
///
/// Transforms all eight corners with the same **`view_proj`** used for [`crate::gpu::PaddedPerDrawUniforms`]
/// (`projection * view`, no model matrix). For each clip-space half-space, if **all** corners lie
/// outside, the box is culled. Matches reverse-Z clip (`z` vs `w`) used by the renderer.
pub fn world_aabb_visible_in_homogeneous_clip(
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> bool {
    let xs = [world_min.x, world_max.x];
    let ys = [world_min.y, world_max.y];
    let zs = [world_min.z, world_max.z];

    let mut clip_corners = [Vec4::ZERO; 8];
    let mut i = 0usize;
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                clip_corners[i] = view_proj * Vec4::new(x, y, z, 1.0);
                i += 1;
            }
        }
    }

    if clip_corners.iter().all(|p| p.w <= HOMOGENEOUS_CLIP_EPS) {
        return false;
    }

    if clip_corners
        .iter()
        .all(|p| p.x + p.w < -HOMOGENEOUS_CLIP_EPS)
    {
        return false;
    }
    if clip_corners
        .iter()
        .all(|p| p.w - p.x < -HOMOGENEOUS_CLIP_EPS)
    {
        return false;
    }
    if clip_corners
        .iter()
        .all(|p| p.y + p.w < -HOMOGENEOUS_CLIP_EPS)
    {
        return false;
    }
    if clip_corners
        .iter()
        .all(|p| p.w - p.y < -HOMOGENEOUS_CLIP_EPS)
    {
        return false;
    }
    if clip_corners.iter().all(|p| p.z < -HOMOGENEOUS_CLIP_EPS) {
        return false;
    }
    if clip_corners
        .iter()
        .all(|p| p.z - p.w > HOMOGENEOUS_CLIP_EPS)
    {
        return false;
    }

    true
}

/// Conservative world AABB for skinning: union of bone palette origins expanded by max half-extent.
pub fn world_aabb_from_skinned_bone_origins(
    bounds: &RenderBoundingBox,
    bone_palette: &[Mat4],
) -> Option<(Vec3, Vec3)> {
    if bone_palette.is_empty() {
        return None;
    }
    let pad = mesh_bounds_max_half_extent(bounds);
    if !pad.is_finite() || pad < 0.0 {
        return None;
    }

    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for m in bone_palette {
        let t = m.col(3);
        let p = Vec3::new(t.x, t.y, t.z);
        if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
            wmin = wmin.min(p);
            wmax = wmax.max(p);
        }
    }
    if !(wmin.x.is_finite() && wmax.x.is_finite()) {
        return None;
    }
    let pad_v = Vec3::splat(pad);
    Some((wmin - pad_v, wmax + pad_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_plane_cross_check_matches_homogeneous_clip_random_boxes() {
        let proj = crate::render_graph::camera::reverse_z_perspective(
            16.0 / 9.0,
            60f32.to_radians(),
            0.1,
            100.0,
        );
        let view = Mat4::look_at_rh(Vec3::new(0.0, 1.5, 4.0), Vec3::ZERO, Vec3::Y);
        let view_proj = proj * view;

        let frustum = Frustum::from_view_proj(view_proj);

        let boxes = [
            (Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5)),
            (Vec3::new(50.0, 50.0, 50.0), Vec3::new(51.0, 51.0, 51.0)),
            (
                Vec3::new(-100.0, -100.0, -100.0),
                Vec3::new(-99.0, -99.0, -99.0),
            ),
        ];

        for (mn, mx) in boxes {
            let clip = world_aabb_visible_in_homogeneous_clip(view_proj, mn, mx);
            let planes = frustum.intersects_aabb(mn, mx);
            assert_eq!(clip, planes, "mismatch for aabb {:?} {:?}", mn, mx);
        }
    }

    #[test]
    fn frustum_rejects_box_fully_outside_left() {
        let proj =
            crate::render_graph::camera::reverse_z_perspective(1.0, 60f32.to_radians(), 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let view_proj = proj * view;

        let frustum = Frustum::from_view_proj(view_proj);
        // Far to the right of the frustum in world space (rough heuristic; box should be outside)
        let mn = Vec3::new(50.0, 0.0, -5.0);
        let mx = Vec3::new(55.0, 1.0, 5.0);
        assert!(!frustum.intersects_aabb(mn, mx));
        assert!(!world_aabb_visible_in_homogeneous_clip(view_proj, mn, mx));
    }

    #[test]
    fn degenerate_bounds_detected() {
        let mut b = RenderBoundingBox::default();
        assert!(mesh_bounds_degenerate_for_cull(&b));
        b.extents = glam::Vec3::new(1.0, 0.0, 0.0);
        assert!(!mesh_bounds_degenerate_for_cull(&b));
    }
}
