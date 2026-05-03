//! Projection of a world-space AABB into a Hi-Z screen-space footprint (UV rect + reverse-Z bound).

use glam::{Mat4, Vec3, Vec4};

/// Screen-space rectangle (UV in `[0, 1]`) plus the closest reverse-Z depth a world AABB can reach.
///
/// `uv_min` / `uv_max` are clamped to the unit square by the caller as needed; this struct stores
/// the raw extents so footprint sizing math sees the true clip-space coverage. Y has already been
/// flipped to image-space (top-down).
#[derive(Clone, Copy, Debug)]
pub(super) struct AabbScreenFootprint {
    /// Top-left UV corner of the projected AABB.
    pub uv_min: (f32, f32),
    /// Bottom-right UV corner of the projected AABB.
    pub uv_max: (f32, f32),
    /// Maximum (closest, reverse-Z) NDC z across the eight projected corners.
    pub max_ndc_z: f32,
}

/// Projects the eight AABB corners through `view_proj` and gathers the screen-space footprint.
///
/// Returns `None` (caller should keep the draw) when any corner has `clip.w <= 0` (straddles the
/// near plane / behind the camera) or the projection produces non-finite NDC values -- these are
/// the same conservative early-outs the original corner-by-corner implementation used.
///
/// Implementation packs the eight corners as two SoA `Vec4`-of-4 (low / high half), evaluating
/// each `view_proj` row once per half so all four lanes of the same component (`clip.x`, `.y`,
/// `.z`, `.w`) sit in registers ready for the reductions. Mirrors the SIMD layout used by
/// `world_aabb_visible_in_homogeneous_clip` in `world_mesh/culling/frustum.rs`. Glam's `Vec4` is
/// `__m128`-backed on x86_64 / `float32x4_t` on aarch64 / scalar elsewhere, so this path is
/// auto-vectorised on the supported targets without a new dependency.
///
/// `project_aabb_to_screen_scalar_for_tests` (private, `#[cfg(test)]`) preserves the original
/// corner-by-corner scalar implementation as a parity reference for the property test below.
#[inline]
pub(super) fn project_aabb_to_screen(
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> Option<AabbScreenFootprint> {
    // Pack the eight AABB corners as the Cartesian product of (min/max).{x,y,z}, split into two
    // halves of four corners each. The lo half holds corners with x = world_min.x; the hi half
    // holds those with x = world_max.x. The y/z lane pattern is shared between halves so we only
    // build (ys, zs) once.
    let xs_lo = Vec4::splat(world_min.x);
    let xs_hi = Vec4::splat(world_max.x);
    let ys = Vec4::new(world_min.y, world_min.y, world_max.y, world_max.y);
    let zs = Vec4::new(world_min.z, world_max.z, world_min.z, world_max.z);

    let r0 = view_proj.row(0);
    let r1 = view_proj.row(1);
    let r2 = view_proj.row(2);
    let r3 = view_proj.row(3);

    // For a row `r` and the four corners in a half (xs, ys, zs), the per-corner dot products are
    //     r.x * xs + r.y * ys + r.z * zs + r.w * 1.0
    // Sum left-to-right (`((r.x*xs + r.y*ys) + r.z*zs) + r.w`) so the per-lane summation order
    // matches glam's `Mat4::mul_vec4` reduction (`((m.x*v.x + m.y*v.y) + m.z*v.z) + m.w*v.w`).
    // That keeps every clip component bit-identical to the scalar reference path used by the
    // parity test below, so cull stat counters and per-view filter masks downstream don't drift.
    let dot4 = |r: Vec4, xs: Vec4| -> Vec4 {
        Vec4::splat(r.x) * xs + Vec4::splat(r.y) * ys + Vec4::splat(r.z) * zs + Vec4::splat(r.w)
    };

    let cw_lo = dot4(r3, xs_lo);
    let cw_hi = dot4(r3, xs_hi);
    if !cw_lo.is_finite() || !cw_hi.is_finite() {
        return None;
    }
    if cw_lo.min_element() <= 0.0 || cw_hi.min_element() <= 0.0 {
        return None;
    }

    let cx_lo = dot4(r0, xs_lo);
    let cy_lo = dot4(r1, xs_lo);
    let cz_lo = dot4(r2, xs_lo);
    let cx_hi = dot4(r0, xs_hi);
    let cy_hi = dot4(r1, xs_hi);
    let cz_hi = dot4(r2, xs_hi);

    let inv_w_lo = Vec4::ONE / cw_lo;
    let inv_w_hi = Vec4::ONE / cw_hi;
    let ndc_x_lo = cx_lo * inv_w_lo;
    let ndc_x_hi = cx_hi * inv_w_hi;
    let ndc_y_lo = cy_lo * inv_w_lo;
    let ndc_y_hi = cy_hi * inv_w_hi;
    let ndc_z_lo = cz_lo * inv_w_lo;
    let ndc_z_hi = cz_hi * inv_w_hi;
    if !ndc_x_lo.is_finite()
        || !ndc_x_hi.is_finite()
        || !ndc_y_lo.is_finite()
        || !ndc_y_hi.is_finite()
        || !ndc_z_lo.is_finite()
        || !ndc_z_hi.is_finite()
    {
        return None;
    }

    let min_ndc_x = ndc_x_lo.min(ndc_x_hi).min_element();
    let max_ndc_x = ndc_x_lo.max(ndc_x_hi).max_element();
    let min_ndc_y = ndc_y_lo.min(ndc_y_hi).min_element();
    let max_ndc_y = ndc_y_lo.max(ndc_y_hi).max_element();
    let max_ndc_z = ndc_z_lo.max(ndc_z_hi).max_element();

    let u0 = min_ndc_x.mul_add(0.5, 0.5);
    let u1 = max_ndc_x.mul_add(0.5, 0.5);
    let v0 = 1.0 - max_ndc_y.mul_add(0.5, 0.5);
    let v1 = 1.0 - min_ndc_y.mul_add(0.5, 0.5);

    Some(AabbScreenFootprint {
        uv_min: (u0.min(u1), v0.min(v1)),
        uv_max: (u0.max(u1), v0.max(v1)),
        max_ndc_z,
    })
}

#[cfg(test)]
fn aabb_corners(min: Vec3, max: Vec3) -> [Vec4; 8] {
    [
        Vec4::new(min.x, min.y, min.z, 1.0),
        Vec4::new(max.x, min.y, min.z, 1.0),
        Vec4::new(min.x, max.y, min.z, 1.0),
        Vec4::new(max.x, max.y, min.z, 1.0),
        Vec4::new(min.x, min.y, max.z, 1.0),
        Vec4::new(max.x, min.y, max.z, 1.0),
        Vec4::new(min.x, max.y, max.z, 1.0),
        Vec4::new(max.x, max.y, max.z, 1.0),
    ]
}

/// Scalar reference implementation of [`project_aabb_to_screen`] -- corner-by-corner `view_proj`
/// multiply followed by sequential min/max reductions. Retained for the SIMD-vs-scalar parity
/// property test below.
#[cfg(test)]
fn project_aabb_to_screen_scalar_for_tests(
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> Option<AabbScreenFootprint> {
    let corners = aabb_corners(world_min, world_max);
    let mut max_ndc_z = f32::MIN;
    let mut min_ndc_x = f32::MAX;
    let mut max_ndc_x = f32::MIN;
    let mut min_ndc_y = f32::MAX;
    let mut max_ndc_y = f32::MIN;

    for c in &corners {
        let clip = view_proj * *c;
        if !clip.w.is_finite() || clip.w <= 0.0 {
            return None;
        }
        let inv_w = 1.0 / clip.w;
        let ndc_x = clip.x * inv_w;
        let ndc_y = clip.y * inv_w;
        let ndc_z = clip.z * inv_w;
        if !ndc_x.is_finite() || !ndc_y.is_finite() || !ndc_z.is_finite() {
            return None;
        }
        max_ndc_z = max_ndc_z.max(ndc_z);
        min_ndc_x = min_ndc_x.min(ndc_x);
        max_ndc_x = max_ndc_x.max(ndc_x);
        min_ndc_y = min_ndc_y.min(ndc_y);
        max_ndc_y = max_ndc_y.max(ndc_y);
    }

    let u0 = min_ndc_x.mul_add(0.5, 0.5);
    let u1 = max_ndc_x.mul_add(0.5, 0.5);
    let v0 = 1.0 - max_ndc_y.mul_add(0.5, 0.5);
    let v1 = 1.0 - min_ndc_y.mul_add(0.5, 0.5);

    Some(AabbScreenFootprint {
        uv_min: (u0.min(u1), v0.min(v1)),
        uv_max: (u0.max(u1), v0.max(v1)),
        max_ndc_z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footprint_eq(a: AabbScreenFootprint, b: AabbScreenFootprint) -> bool {
        a.uv_min == b.uv_min && a.uv_max == b.uv_max && a.max_ndc_z == b.max_ndc_z
    }

    /// Deterministic LCG for the property test; mirrors the helper in
    /// `world_mesh/culling/frustum.rs` so failures are bit-reproducible across machines and
    /// don't pull `rand` into the dep tree just for this test.
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

    #[test]
    fn footprint_rejects_box_straddling_near_plane() {
        let proj = crate::camera::reverse_z_perspective(1.0, 60f32.to_radians(), 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
        let view_proj = proj * view;
        // Box envelopes the camera position (z=1 in world), so several corners have clip.w <= 0.
        let mn = Vec3::new(-1.0, -1.0, -1.0);
        let mx = Vec3::new(1.0, 1.0, 2.0);
        let simd = project_aabb_to_screen(view_proj, mn, mx);
        let scalar = project_aabb_to_screen_scalar_for_tests(view_proj, mn, mx);
        assert!(simd.is_none(), "expected None for near-plane straddle");
        assert!(scalar.is_none(), "scalar reference should also reject");
    }

    #[test]
    fn footprint_rejects_box_fully_behind_camera() {
        let proj = crate::camera::reverse_z_perspective(1.0, 60f32.to_radians(), 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        let view_proj = proj * view;
        // Look down -Z; this box sits at +Z so it's behind the camera.
        let mn = Vec3::new(-1.0, -1.0, 5.0);
        let mx = Vec3::new(1.0, 1.0, 6.0);
        let simd = project_aabb_to_screen(view_proj, mn, mx);
        let scalar = project_aabb_to_screen_scalar_for_tests(view_proj, mn, mx);
        assert!(simd.is_none());
        assert!(scalar.is_none());
    }

    #[test]
    fn footprint_in_frustum_matches_scalar_exactly() {
        let proj = crate::camera::reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 1.5, 4.0), Vec3::ZERO, Vec3::Y);
        let view_proj = proj * view;
        let mn = Vec3::new(-0.5, 0.0, -0.5);
        let mx = Vec3::new(0.5, 1.0, 0.5);

        let simd =
            project_aabb_to_screen(view_proj, mn, mx).expect("box is in front of the camera");
        let scalar = project_aabb_to_screen_scalar_for_tests(view_proj, mn, mx)
            .expect("scalar reference should agree");
        assert!(
            footprint_eq(simd, scalar),
            "simd={simd:?} scalar={scalar:?}"
        );
    }

    /// SIMD-vs-scalar parity across a wide sweep of projections, camera positions, and AABB
    /// shapes. Both paths execute the same arithmetic in the same order on the same hardware so
    /// every sample must match bit-for-bit; if a future glam release breaks that, this surfaces
    /// the regression immediately rather than letting cull stats silently drift.
    #[test]
    fn simd_aabb_screen_footprint_matches_scalar_reference_across_random_inputs() {
        let mut state: u64 = 0xF1E2_D3C4_B5A6_9788;

        let projections = [
            crate::camera::reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), 0.1, 100.0),
            crate::camera::reverse_z_perspective(1.0, 90f32.to_radians(), 0.05, 500.0),
            crate::camera::reverse_z_perspective(4.0 / 3.0, 45f32.to_radians(), 0.5, 50.0),
        ];

        let mut samples = 0usize;
        let mut mismatches = 0usize;
        let mut some_count = 0usize;
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

                let simd = project_aabb_to_screen(view_proj, mn, mx);
                let scalar = project_aabb_to_screen_scalar_for_tests(view_proj, mn, mx);
                match (simd, scalar) {
                    (None, None) => {}
                    (Some(s), Some(r)) => {
                        if !footprint_eq(s, r) {
                            mismatches += 1;
                        }
                        some_count += 1;
                    }
                    _ => {
                        mismatches += 1;
                    }
                }
                samples += 1;
            }
        }
        assert!(samples >= 10_000, "expected >= 10k samples, got {samples}");
        assert!(
            some_count >= 1_000,
            "sweep should land at least 1k boxes in front of the camera so the reduction path is \
             actually exercised; got {some_count}/{samples}"
        );
        assert_eq!(
            mismatches, 0,
            "SIMD vs scalar mismatch on {mismatches}/{samples} random inputs"
        );
    }
}
