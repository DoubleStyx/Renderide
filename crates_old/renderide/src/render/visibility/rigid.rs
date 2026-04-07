//! Frustum culling for rigid (non-skinned) meshes.
//!
//! Transforms the draw's local [`RenderBoundingBox`] through its model matrix to produce a
//! world-space AABB, then tests every corner against the view-projection clip volume.
//!
//! Affine model matrices use a closed-form world AABB (one point transform plus absolute 3×3
//! column combination). Non-affine bottom rows fall back to transforming all eight corners.
//!
//! # Conservative behaviour
//! When bounds are degenerate (near-zero or non-finite extents) or the world AABB corners cannot
//! be computed (non-finite model matrix output), the draw is **kept** (returns `true`).
//! This matches the policy in the collect loop: prefer false-positives over false-negatives.

use std::collections::HashMap;

use glam::{Mat4, Vec3, Vec4Swizzles};

use crate::shared::RenderBoundingBox;

use super::{mesh_bounds_degenerate_for_cull, world_aabb_visible_in_homogeneous_clip};

/// Epsilon for treating a model matrix bottom row as affine \([0,0,0,1]\).
const MODEL_MATRIX_AFFINE_BOTTOM_EPS: f32 = 1e-4;

/// Bit pattern tag of [`RenderBoundingBox`] center and extents for cache invalidation when upload bounds change.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RigidFrustumCullBoundsTag(pub [u32; 6]);

impl RigidFrustumCullBoundsTag {
    /// Builds a tag from mesh upload bounds so cache entries invalidate if bounds change for the same asset id.
    pub fn from_bounds(b: &RenderBoundingBox) -> Self {
        Self([
            b.center.x.to_bits(),
            b.center.y.to_bits(),
            b.center.z.to_bits(),
            b.extents.x.to_bits(),
            b.extents.y.to_bits(),
            b.extents.z.to_bits(),
        ])
    }
}

/// Key for [`RigidFrustumCullCache`]: space, drawable transform, mesh asset, and bounds fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RigidFrustumCullCacheKey {
    /// Render space id ([`SpaceDrawBatch::space_id`](crate::render::batch::SpaceDrawBatch::space_id)).
    pub space_id: i32,
    /// Drawable's scene transform id.
    pub node_id: i32,
    pub mesh_asset_id: i32,
    pub bounds_tag: RigidFrustumCullBoundsTag,
}

impl RigidFrustumCullCacheKey {
    /// Creates a cache key for a rigid draw in a batch.
    pub fn new(
        space_id: i32,
        node_id: i32,
        mesh_asset_id: i32,
        bounds: &RenderBoundingBox,
    ) -> Self {
        Self {
            space_id,
            node_id,
            mesh_asset_id,
            bounds_tag: RigidFrustumCullBoundsTag::from_bounds(bounds),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RigidFrustumCullCacheEntry {
    model_matrix: Mat4,
    bounds_tag: RigidFrustumCullBoundsTag,
    wmin: Vec3,
    wmax: Vec3,
}

/// Caches world-space AABBs for rigid frustum tests when the model matrix (and bounds tag) match the previous frame.
#[derive(Debug, Default)]
pub struct RigidFrustumCullCache {
    entries: HashMap<RigidFrustumCullCacheKey, RigidFrustumCullCacheEntry>,
}

impl RigidFrustumCullCache {
    /// Removes all entries (e.g. after large scene teardown if memory is a concern).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Returns `true` when the fourth row of `m` is approximately `[0, 0, 0, 1]` (column-vector convention).
fn model_matrix_is_affine_bottom_row(m: Mat4) -> bool {
    let r = m.row(3);
    r.x.abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
        && r.y.abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
        && r.z.abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
        && (r.w - 1.0).abs() <= MODEL_MATRIX_AFFINE_BOTTOM_EPS
}

/// World-space AABB for a local axis-aligned box under an affine transform (no perspective in `m`).
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

/// Transforms eight local AABB corners; used when the model matrix may be projective.
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
///
/// Returns `None` when:
/// - Any of `center` or `extents` is non-finite (corrupt mesh metadata), or
/// - Any transformed corner is non-finite (degenerate or huge model matrix).
pub(crate) fn world_aabb_from_local_bounds(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
) -> Option<(Vec3, Vec3)> {
    if model_matrix_is_affine_bottom_row(model_matrix) {
        world_aabb_from_local_bounds_affine(bounds, model_matrix)
    } else {
        world_aabb_from_local_bounds_bruteforce(bounds, model_matrix)
    }
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

/// Like [`rigid_mesh_potentially_visible`], but reuses the world-space AABB in `cache` when the model matrix and bounds tag match.
pub fn rigid_mesh_potentially_visible_cached(
    bounds: &RenderBoundingBox,
    model_matrix: Mat4,
    view_proj: Mat4,
    cache_key: RigidFrustumCullCacheKey,
    cache: &mut RigidFrustumCullCache,
) -> bool {
    if mesh_bounds_degenerate_for_cull(bounds) {
        return true;
    }
    let tag = cache_key.bounds_tag;
    if let Some(e) = cache.entries.get(&cache_key)
        && e.model_matrix == model_matrix
        && e.bounds_tag == tag
    {
        return world_aabb_visible_in_homogeneous_clip(view_proj, e.wmin, e.wmax);
    }
    let Some((wmin, wmax)) = world_aabb_from_local_bounds(bounds, model_matrix) else {
        return true;
    };
    cache.entries.insert(
        cache_key,
        RigidFrustumCullCacheEntry {
            model_matrix,
            bounds_tag: tag,
            wmin,
            wmax,
        },
    );
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

    fn assert_aabb_close(a: Option<(Vec3, Vec3)>, b: Option<(Vec3, Vec3)>, eps: f32) {
        match (a, b) {
            (Some((amin, amax)), Some((bmin, bmax))) => {
                assert!(
                    (amin - bmin).length() < eps && (amax - bmax).length() < eps,
                    "a=({:?},{:?}) b=({:?},{:?})",
                    amin,
                    amax,
                    bmin,
                    bmax
                );
            }
            (None, None) => {}
            (x, y) => panic!("mismatch: {:?} vs {:?}", x, y),
        }
    }

    #[test]
    fn affine_aabb_matches_bruteforce_random_affine_matrices() {
        let bounds = make_bounds(0.25, -0.5, 0.75, 1.0, 2.0, 0.5);
        let mut t = 0u32;
        let mut next_f32 = || {
            t = t.wrapping_mul(1103515245).wrapping_add(12345);
            let u = (t >> 8) & 0xffff;
            (u as f32 / 32768.0) - 1.0
        };
        for _ in 0..200 {
            let mut cols = [0f32; 16];
            for c in &mut cols {
                *c = next_f32() * 2.0;
            }
            cols[12] = next_f32() * 10.0;
            cols[13] = next_f32() * 10.0;
            cols[14] = next_f32() * 10.0;
            cols[15] = 1.0;
            let m = Mat4::from_cols_array(&cols);
            let affine = model_matrix_is_affine_bottom_row(m);
            let fast = world_aabb_from_local_bounds_affine(&bounds, m);
            let brute = world_aabb_from_local_bounds_bruteforce(&bounds, m);
            if affine {
                assert_aabb_close(fast, brute, 1e-3);
            }
            let combined = world_aabb_from_local_bounds(&bounds, m);
            assert_eq!(combined, brute);
        }
    }

    #[test]
    fn non_affine_bottom_row_uses_bruteforce() {
        let bounds = make_bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        let m = Mat4::from_cols(
            glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
            glam::Vec4::new(0.0, 0.0, 0.0, 0.5),
        );
        assert!(!model_matrix_is_affine_bottom_row(m));
        let brute = world_aabb_from_local_bounds_bruteforce(&bounds, m);
        assert_eq!(world_aabb_from_local_bounds(&bounds, m), brute);
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
        let tight = make_bounds(50.0, 0.0, 0.0, 0.5, 0.5, 0.5);
        assert!(!rigid_mesh_potentially_visible(&tight, Mat4::IDENTITY, vp));
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

    #[test]
    fn frustum_cull_cache_reuses_aabb_when_model_unchanged() {
        let vp1 = look_vp_naive();
        let view_away = NaMat4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 20.0),
            &NaVec3::new(0.0, 1.0, 0.0),
        );
        let proj = reverse_z_projection(1.0, 60f32.to_radians(), 0.1, 100.0);
        let vp2 = matrix_na_to_glam(&(proj * view_away));
        let bounds = make_bounds(0.0, 0.0, 0.0, 0.5, 0.5, 0.5);
        let key = RigidFrustumCullCacheKey::new(1, 2, 3, &bounds);
        let mut cache = RigidFrustumCullCache::default();
        let m = Mat4::IDENTITY;
        let v1 = rigid_mesh_potentially_visible_cached(&bounds, m, vp1, key, &mut cache);
        assert!(v1);
        assert_eq!(cache.entries.len(), 1);
        let v2 = rigid_mesh_potentially_visible_cached(&bounds, m, vp2, key, &mut cache);
        assert!(
            !v2,
            "camera looking +Z should not see box near origin; cache should reuse AABB"
        );
        assert_eq!(cache.entries.len(), 1);
    }
}
