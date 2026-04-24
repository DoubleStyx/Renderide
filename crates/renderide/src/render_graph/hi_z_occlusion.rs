//! CPU hierarchical-Z occlusion test (reverse-Z depth buffer).
//!
//! Hi-Z pyramids come from the **previous** frame; tests must use [`HiZTemporalState`] view–projection
//! from the frame that produced that depth, not the current frame.
//!
//! Set `RENDERIDE_HIZ_TRACE=1` to emit [`logger::trace`] lines when a draw is classified as fully
//! occluded (can be verbose).

use std::env;
use std::sync::OnceLock;

use glam::{Mat4, Vec3, Vec4};

use super::hi_z_cpu::{mip_byte_offset_floats, mip_dimensions, HiZCpuSnapshot};
use super::world_mesh_cull::WorldMeshCullProjParams;

/// Small bias to reduce mip / quantization flicker at occlusion boundaries (reverse-Z).
const HI_Z_BIAS: f32 = 5e-5;

/// Extra reverse-Z slack before declaring full occlusion (reduces view-dependent popping at depth edges).
const HI_Z_OCCLUSION_MARGIN: f32 = 5e-4;

/// Caps Hi-Z mip used for the occlusion test so coarse mips do not swing the decision when the
/// footprint crosses discrete mip boundaries.
const HI_Z_OCCLUSION_MAX_MIP: u32 = 2;

fn hiz_trace_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        env::var_os("RENDERIDE_HIZ_TRACE").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

/// Picks a mip level from an approximate footprint (in **base** Hi-Z texels, not full viewport pixels).
///
/// Matches the previous single-center-sample Hi-Z path: coarser mips for larger screen extents.
#[inline]
fn hi_z_mip_for_pixel_extent(extent_base_px: f32) -> u32 {
    if !extent_base_px.is_finite() || extent_base_px <= 1.0 {
        return 0;
    }
    let mut e = extent_base_px.max(1.0);
    let mut level = 0u32;
    while e > 2.0 && level < 15 {
        e *= 0.5;
        level += 1;
    }
    level
}

/// Builds view–projection matrices for Hi-Z tests (same rules as frustum culling, using **previous**
/// frame data from [`super::HiZTemporalState`]).
pub fn hi_z_view_proj_matrices(
    prev: &WorldMeshCullProjParams,
    prev_view: Mat4,
    is_overlay: bool,
) -> Vec<Mat4> {
    if let Some((sl, sr)) = prev.vr_stereo {
        if is_overlay {
            return vec![prev.overlay_proj * prev_view];
        }
        return vec![sl, sr];
    }
    let base = if is_overlay {
        prev.overlay_proj
    } else {
        prev.world_proj
    };
    vec![base * prev_view]
}

/// Returns `true` when the axis-aligned world bounds are **fully occluded** by `snapshot` for `view_proj`.
///
/// Conservative: if **any** corner has `clip.w <= 0` (straddles the near plane / behind the camera),
/// returns `false` (keep the draw). Compares the AABB **closest** depth (maximum NDC Z in reverse-Z)
/// to the **minimum** depth in a 2×2 texel neighborhood at the footprint center (weakest occluder in
/// that block in reverse-Z, reducing single-texel and mip-boundary popping). Mip level is capped;
/// an extra margin is required before culling.
#[inline]
pub fn mesh_fully_occluded_in_hiz(
    snapshot: &HiZCpuSnapshot,
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> bool {
    let corners = aabb_corners(world_min, world_max);
    let mut max_ndc_z = f32::MIN;
    let mut min_ndc_x = f32::MAX;
    let mut max_ndc_x = f32::MIN;
    let mut min_ndc_y = f32::MAX;
    let mut max_ndc_y = f32::MIN;

    for c in &corners {
        let clip = view_proj * *c;
        if !clip.w.is_finite() || clip.w <= 0.0 {
            return false;
        }
        let inv_w = 1.0 / clip.w;
        let ndc_x = clip.x * inv_w;
        let ndc_y = clip.y * inv_w;
        let ndc_z = clip.z * inv_w;
        if !ndc_x.is_finite() || !ndc_y.is_finite() || !ndc_z.is_finite() {
            return false;
        }
        max_ndc_z = max_ndc_z.max(ndc_z);
        min_ndc_x = min_ndc_x.min(ndc_x);
        max_ndc_x = max_ndc_x.max(ndc_x);
        min_ndc_y = min_ndc_y.min(ndc_y);
        max_ndc_y = max_ndc_y.max(ndc_y);
    }

    let base_w = snapshot.base_width.max(1) as f32;
    let base_h = snapshot.base_height.max(1) as f32;

    let u0 = min_ndc_x.mul_add(0.5, 0.5);
    let u1 = max_ndc_x.mul_add(0.5, 0.5);
    let v0 = 1.0 - max_ndc_y.mul_add(0.5, 0.5);
    let v1 = 1.0 - min_ndc_y.mul_add(0.5, 0.5);

    let px_min = u0.min(u1);
    let px_max = u0.max(u1);
    let py_min = v0.min(v1);
    let py_max = v0.max(v1);

    // px_max >= px_min and py_max >= py_min by construction (min/max of two values).
    let du_base = (px_max - px_min) * base_w;
    let dv_base = (py_max - py_min) * base_h;
    let extent_base = du_base.max(dv_base).max(1.0);
    let mip = hi_z_mip_for_pixel_extent(extent_base)
        .min(HI_Z_OCCLUSION_MAX_MIP)
        .min(snapshot.mip_levels.saturating_sub(1));

    let Some((mw, mh)) = mip_dimensions(snapshot.base_width, snapshot.base_height, mip) else {
        return false;
    };
    if mw == 0 || mh == 0 {
        return false;
    }

    let uc = ((px_min + px_max) * 0.5).clamp(0.0, 1.0);
    let vc = ((py_min + py_max) * 0.5).clamp(0.0, 1.0);
    let sx = ((uc * mw as f32).floor() as u32).min(mw.saturating_sub(1));
    let sy = ((vc * mh as f32).floor() as u32).min(mh.saturating_sub(1));

    let Some(hiz_min) = hiz_min_in_2x2(snapshot, mip, sx, sy, mw, mh) else {
        return false;
    };

    // Reverse-Z: farther = smaller NDC Z. Fully occluded if the closest AABB point is still farther than the occluder.
    let occluded = max_ndc_z + HI_Z_BIAS + HI_Z_OCCLUSION_MARGIN < hiz_min;
    if occluded && hiz_trace_enabled() {
        logger::trace!(
            "Hi-Z full occluder: mip={} extent_base={} max_ndc_z={} hiz_min_2x2={}",
            mip,
            extent_base,
            max_ndc_z,
            hiz_min
        );
    }
    occluded
}

/// Minimum depth in a 2×2 block anchored at `(sx, sy)` (clamped), reverse-Z.
///
/// Using the **minimum** (farthest surface in the neighborhood) is visibility-conservative: fewer
/// false-positive culls when a single texel spikes closer than its neighbors.
#[inline]
fn hiz_min_in_2x2(
    snapshot: &HiZCpuSnapshot,
    mip: u32,
    sx: u32,
    sy: u32,
    mw: u32,
    mh: u32,
) -> Option<f32> {
    let mip_base = mip_byte_offset_floats(snapshot.base_width, snapshot.base_height, mip);
    let mut vmin = f32::MAX;
    let mut any = false;
    for dy in 0..=1u32 {
        for dx in 0..=1u32 {
            let x = (sx + dx).min(mw.saturating_sub(1));
            let y = (sy + dy).min(mh.saturating_sub(1));
            let idx = mip_base + (y * mw + x) as usize;
            if let Some(&v) = snapshot.mips.get(idx) {
                if v.is_finite() {
                    vmin = vmin.min(v);
                    any = true;
                }
            }
        }
    }
    if any {
        Some(vmin)
    } else {
        None
    }
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

/// Stereo Hi-Z policy: keep the draw unless **both** eyes report full occlusion (matches frustum OR across eyes).
#[inline]
pub fn stereo_hiz_keeps_draw(occluded_left: bool, occluded_right: bool) -> bool {
    !(occluded_left && occluded_right)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn hi_z_mip_for_pixel_extent_levels() {
        assert_eq!(hi_z_mip_for_pixel_extent(1.0), 0);
        assert_eq!(hi_z_mip_for_pixel_extent(2.0), 0);
        assert_eq!(hi_z_mip_for_pixel_extent(2.1), 1);
        assert_eq!(hi_z_mip_for_pixel_extent(4.0), 1);
        assert_eq!(hi_z_mip_for_pixel_extent(8.0), 2);
    }

    #[test]
    fn raw_mip_extent_can_exceed_occlusion_cap() {
        assert!(
            hi_z_mip_for_pixel_extent(1024.0) > HI_Z_OCCLUSION_MAX_MIP,
            "large footprints choose deep mips; mesh_fully_occluded_in_hiz caps with HI_Z_OCCLUSION_MAX_MIP"
        );
    }

    /// Borderline depth: without [`HI_Z_OCCLUSION_MARGIN`] the object could be classified occluded;
    /// margin requires a clearer gap (reduces popping).
    #[test]
    fn occlusion_margin_blocks_borderline_cull() {
        let vp = Mat4::IDENTITY;
        let mips = vec![0.92f32; 21];
        let snap = HiZCpuSnapshot {
            base_width: 4,
            base_height: 4,
            mip_levels: 3,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        // Closest point ~0.9195; uniform Hi-Z 0.92 — gap smaller than HI_Z_OCCLUSION_MARGIN + bias.
        let wmin = Vec3::new(-0.01, -0.01, 0.9195);
        let wmax = Vec3::new(0.01, 0.01, 0.91);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax),
            "margin should avoid cull when barely behind the Hi-Z plane"
        );
    }

    #[test]
    fn clearly_behind_uniform_hiz_is_fully_occluded() {
        let vp = Mat4::IDENTITY;
        let mips = vec![0.92f32; 21];
        let snap = HiZCpuSnapshot {
            base_width: 4,
            base_height: 4,
            mip_levels: 3,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        let wmin = Vec3::new(-0.01, -0.01, 0.85);
        let wmax = Vec3::new(0.01, 0.01, 0.80);
        assert!(mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax));
    }

    /// A hole (far / low reverse-Z) in the 2×2 block lowers `hiz_min`, so we do not cull.
    #[test]
    fn hiz_min_2x2_sees_farther_occluder_in_block() {
        let vp = Mat4::IDENTITY;
        // mip0 4×4 row-major: center anchor (sx,sy)=(2,2) uses indices 10..=15; put a hole at 10.
        let mut mips = vec![0.95f32; 21];
        mips[10] = 0.35;
        let snap = HiZCpuSnapshot {
            base_width: 4,
            base_height: 4,
            mip_levels: 3,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        let wmin = Vec3::new(-0.01, -0.01, 0.90);
        let wmax = Vec3::new(0.01, 0.01, 0.88);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax),
            "2×2 min must include the farther sample so we keep the draw"
        );
    }

    #[test]
    fn straddling_near_plane_not_fully_occluded() {
        // Last row [0,0,0,-1] makes clip.w = -w; corners with w>0 and w<=0 in the same AABB → keep draw.
        let vp = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, -1.0,
        ]);
        let mips = vec![0.5f32; 21];
        let snap = HiZCpuSnapshot {
            base_width: 4,
            base_height: 4,
            mip_levels: 3,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        let wmin = Vec3::new(0.0, 0.0, 0.0);
        let wmax = Vec3::new(1.0, 1.0, 1.0);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax),
            "must not cull when any corner has clip.w <= 0"
        );
    }

    #[test]
    fn stereo_hiz_keeps_if_either_eye_not_fully_occluded() {
        assert!(stereo_hiz_keeps_draw(false, false));
        assert!(stereo_hiz_keeps_draw(true, false));
        assert!(stereo_hiz_keeps_draw(false, true));
        assert!(!stereo_hiz_keeps_draw(true, true));
    }

    /// Regression: a single center texel at the chosen mip avoids pulling unrelated **near** depth
    /// from a wide footprint (the old rect `max` path caused false-positive culls).
    #[test]
    fn fully_occluded_uses_closest_corner_not_farthest() {
        let vp = Mat4::IDENTITY;
        // Uniform Hi-Z plane slightly farther than the front of the box (reverse-Z: smaller = farther).
        // 4×4 + 2×2 + 1×1 = 21 floats for three mips.
        let mips = vec![0.92f32; 21];
        let snap = HiZCpuSnapshot {
            base_width: 4,
            base_height: 4,
            mip_levels: 3,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        // Front of AABB at z=0.99 (closer than Hi-Z 0.92), back at z=0.05. Must not cull on back alone.
        let near = Vec3::new(-0.01, -0.01, 0.99);
        let far = Vec3::new(0.01, 0.01, 0.05);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, near, far),
            "closest point still in front of occluder"
        );
    }
}
