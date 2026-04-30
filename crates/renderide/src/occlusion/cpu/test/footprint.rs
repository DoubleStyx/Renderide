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
/// near plane / behind the camera) or the projection produces non-finite NDC values — these are
/// the same conservative early-outs the previous inline implementation used.
#[inline]
pub(super) fn project_aabb_to_screen(
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
