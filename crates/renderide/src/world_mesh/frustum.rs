//! Six-plane view frustum and world AABB tests for CPU mesh culling.
//!
//! **Production culling** uses [`world_aabb_visible_in_homogeneous_clip`], which matches
//! `clip = view_proj * vec4(world, 1)` exactly (same as WGSL `mat4x4` × `vec4`).
//!
//! [`Frustum`] / six-plane tests remain for debugging and optional comparisons.

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

use crate::shared::RenderBoundingBox;

/// Epsilon for homogeneous clip half-space tests used by frustum culling.
pub const HOMOGENEOUS_CLIP_EPS: f32 = 1e-5;

/// Maximum absolute half-extent below which uploaded mesh bounds are treated as **untrusted** for culling.
pub(crate) const DEGENERATE_MESH_BOUNDS_EXTENT_EPS: f32 = 1e-8;

/// Epsilon for treating a model matrix bottom row as affine `[0, 0, 0, 1]`.
const MODEL_MATRIX_AFFINE_BOTTOM_EPS: f32 = 1e-4;

/// Returns `true` when bounds should not be used for culling (keep the draw).
#[inline]
pub fn mesh_bounds_degenerate_for_cull(bounds: &RenderBoundingBox) -> bool {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return true;
    }
    let m = e.x.abs().max(e.y.abs()).max(e.z.abs());
    m < DEGENERATE_MESH_BOUNDS_EXTENT_EPS
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

/// Returns `true` if the axis-aligned world box may intersect the clip volume.
///
/// Transforms all eight corners with the same **`view_proj`** used for [`crate::gpu::PaddedPerDrawUniforms`]
/// (`projection * view`, no model matrix). For each clip-space half-space, if **all** corners lie
/// outside, the box is culled. Matches reverse-Z clip (`z` vs `w`) used by the renderer.
///
/// Implementation packs the eight corners as two SoA `Vec4`-of-4 (low / high half), evaluating each
/// `view_proj` row once per half so all four lanes of the same component (`clip.x`, `.y`, `.z`,
/// `.w`) sit in registers ready for the half-space reductions. Each predicate then collapses to a
/// single `Vec4::max`/`min` followed by `max_element`/`min_element`, replacing the prior 8
/// sequential scalar compares per predicate. Glam's `Vec4` is `__m128`-backed on x86_64 / `float32x4_t`
/// on aarch64 / scalar elsewhere, so this path is auto-vectorised on the supported targets without a
/// new dependency.
///
/// `world_aabb_visible_in_homogeneous_clip_scalar_for_tests` (private) preserves the original
/// corner-by-corner scalar implementation as a parity reference for the property test below.
pub fn world_aabb_visible_in_homogeneous_clip(
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> bool {
    // Eight corners are the Cartesian product of (min/max).{x,y,z}. Lay them out as two halves of
    // four corners each, with the x coordinate constant inside each half (`min.x` for `lo`,
    // `max.x` for `hi`) and y/z varying as a 2×2 sub-product. The two halves share the y/z
    // pattern, so we only need one (ys, zs) Vec4 pair.
    let xs_lo = Vec4::splat(world_min.x);
    let xs_hi = Vec4::splat(world_max.x);
    let ys = Vec4::new(world_min.y, world_min.y, world_max.y, world_max.y);
    let zs = Vec4::new(world_min.z, world_max.z, world_min.z, world_max.z);

    // `view_proj` is column-major; `view_proj.row(i).dot(corner)` yields the i-th clip component.
    // Row 0 contributes to clip.x, row 1 to clip.y, row 2 to clip.z, row 3 to clip.w.
    let r0 = view_proj.row(0);
    let r1 = view_proj.row(1);
    let r2 = view_proj.row(2);
    let r3 = view_proj.row(3);

    // For a row `r` and the four corners in a half (xs, ys, zs), the per-corner dot products are
    //     r.x * xs + r.y * ys + r.z * zs + r.w * 1.0
    // Each lane of the resulting Vec4 holds one corner's component, so the four lanes are SoA
    // ready for the half-space tests below. Inlining the multiplies keeps everything in vector
    // registers — glam Vec4 ops compile to SIMD on x86_64 / aarch64.
    let dot4 = |r: Vec4, xs: Vec4| -> Vec4 {
        Vec4::splat(r.x) * xs + Vec4::splat(r.y) * ys + Vec4::splat(r.z) * zs + Vec4::splat(r.w)
    };

    let cx_lo = dot4(r0, xs_lo);
    let cy_lo = dot4(r1, xs_lo);
    let cz_lo = dot4(r2, xs_lo);
    let cw_lo = dot4(r3, xs_lo);
    let cx_hi = dot4(r0, xs_hi);
    let cy_hi = dot4(r1, xs_hi);
    let cz_hi = dot4(r2, xs_hi);
    let cw_hi = dot4(r3, xs_hi);

    // For each half-space test "all corners satisfy expr ?op? threshold", reduce both halves with
    // `Vec4::max` / `Vec4::min`, then collapse to a scalar via `max_element` / `min_element`.
    //   "all w <= EPS"     ⇔ max(w over all corners) <= EPS
    //   "all x + w < -EPS" ⇔ max(x + w over all corners) < -EPS
    //   "all w - x < -EPS" ⇔ max(w - x over all corners) < -EPS  (equivalently min(x - w) > EPS)
    //   …and similar for y, z half-spaces and the reverse-Z near/far tests.
    let max_w = cw_lo.max(cw_hi).max_element();
    if max_w <= HOMOGENEOUS_CLIP_EPS {
        return false;
    }

    let max_x_plus_w = (cx_lo + cw_lo).max(cx_hi + cw_hi).max_element();
    if max_x_plus_w < -HOMOGENEOUS_CLIP_EPS {
        return false;
    }
    let max_w_minus_x = (cw_lo - cx_lo).max(cw_hi - cx_hi).max_element();
    if max_w_minus_x < -HOMOGENEOUS_CLIP_EPS {
        return false;
    }
    let max_y_plus_w = (cy_lo + cw_lo).max(cy_hi + cw_hi).max_element();
    if max_y_plus_w < -HOMOGENEOUS_CLIP_EPS {
        return false;
    }
    let max_w_minus_y = (cw_lo - cy_lo).max(cw_hi - cy_hi).max_element();
    if max_w_minus_y < -HOMOGENEOUS_CLIP_EPS {
        return false;
    }
    // Reverse-Z near plane: scalar reference rejected when `all p.z < -EPS`, so we max over `cz`.
    let max_z = cz_lo.max(cz_hi).max_element();
    if max_z < -HOMOGENEOUS_CLIP_EPS {
        return false;
    }
    // Reverse-Z far plane: scalar reference rejected when `all p.z - p.w > +EPS`, so we min and
    // compare against +EPS.
    let min_z_minus_w = (cz_lo - cw_lo).min(cz_hi - cw_hi).min_element();
    if min_z_minus_w > HOMOGENEOUS_CLIP_EPS {
        return false;
    }

    true
}

/// Scalar reference implementation of [`world_aabb_visible_in_homogeneous_clip`] — corner-by-corner
/// `view_proj` multiply followed by seven sequential `iter().all(...)` predicates. Retained for the
/// SIMD-vs-scalar parity property test below.
#[cfg(test)]
fn world_aabb_visible_in_homogeneous_clip_scalar_for_tests(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_plane_cross_check_matches_homogeneous_clip_random_boxes() {
        let proj = crate::camera::reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 100.0);
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
            assert_eq!(clip, planes, "mismatch for aabb {mn:?} {mx:?}");
        }
    }

    #[test]
    fn frustum_rejects_box_fully_outside_left() {
        let proj = crate::camera::reverse_z_perspective(1.0, 60f32.to_radians(), 0.1, 100.0);
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
        b.extents = Vec3::new(1.0, 0.0, 0.0);
        assert!(!mesh_bounds_degenerate_for_cull(&b));
    }

    /// Deterministic LCG for the property test; avoids pulling `rand` into the dep tree just for
    /// this one test and keeps failures bit-reproducible across machines.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn lcg_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let bits = (lcg_next(state) >> 32) as u32;
        let unit = (bits as f32) / (u32::MAX as f32);
        lo + unit * (hi - lo)
    }

    /// SIMD-vs-scalar parity: the new SoA-Vec4 path must agree with the corner-by-corner scalar
    /// reference for every input, since the cull stat counters and per-view filter masks downstream
    /// depend on bit-for-bit identical accept/reject decisions. Sweeps a wide mix of camera
    /// positions, AABB sizes, and AABB centers (including boxes intentionally near the clip planes
    /// where edge cases live).
    #[test]
    fn simd_aabb_clip_test_matches_scalar_reference_across_random_inputs() {
        let mut state: u64 = 0xA1B2_C3D4_E5F6_0789;

        let projections = [
            crate::camera::reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 100.0),
            crate::camera::reverse_z_perspective(1.0, 90f32.to_radians(), 0.05, 500.0),
            crate::camera::reverse_z_perspective(4.0 / 3.0, 45f32.to_radians(), 0.5, 50.0),
        ];

        let mut mismatches = 0usize;
        let mut samples = 0usize;
        for proj in projections {
            for _ in 0..3_400 {
                let cam = Vec3::new(
                    lcg_f32(&mut state, -10.0, 10.0),
                    lcg_f32(&mut state, -5.0, 5.0),
                    lcg_f32(&mut state, -10.0, 10.0),
                );
                let target = Vec3::new(
                    lcg_f32(&mut state, -5.0, 5.0),
                    lcg_f32(&mut state, -2.0, 2.0),
                    lcg_f32(&mut state, -5.0, 5.0),
                );
                if (cam - target).length_squared() < 0.01 {
                    continue;
                }
                let view = Mat4::look_at_rh(cam, target, Vec3::Y);
                let view_proj = proj * view;

                let center = Vec3::new(
                    lcg_f32(&mut state, -20.0, 20.0),
                    lcg_f32(&mut state, -20.0, 20.0),
                    lcg_f32(&mut state, -20.0, 20.0),
                );
                let half = Vec3::new(
                    lcg_f32(&mut state, 0.05, 5.0),
                    lcg_f32(&mut state, 0.05, 5.0),
                    lcg_f32(&mut state, 0.05, 5.0),
                );
                let mn = center - half;
                let mx = center + half;

                let simd = world_aabb_visible_in_homogeneous_clip(view_proj, mn, mx);
                let scalar =
                    world_aabb_visible_in_homogeneous_clip_scalar_for_tests(view_proj, mn, mx);
                if simd != scalar {
                    mismatches += 1;
                }
                samples += 1;
            }
        }
        assert!(samples >= 10_000, "expected >= 10k samples, got {samples}");
        assert_eq!(
            mismatches, 0,
            "SIMD vs scalar mismatch on {mismatches}/{samples} random inputs"
        );
    }
}
