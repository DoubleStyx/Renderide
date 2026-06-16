//! CPU hierarchical-Z occlusion test (reverse-Z depth buffer).
//!
//! Hi-Z pyramids come from the **previous** frame; tests must use [`crate::cull_contract::HiZTemporalState`]
//! view-projection from the frame that produced that depth, not the current frame.
//!
//! Set `RENDERIDE_HIZ_TRACE=1` to emit [`logger::trace`] lines when a draw is classified as fully
//! occluded (can be verbose).

mod footprint;
mod sampling;

use std::env;
use std::sync::LazyLock;

use glam::{Mat4, Vec3};

use super::snapshot::HiZCpuSnapshot;
use crate::camera::overlay_camera_view_matrix;
use crate::cull_contract::WorldMeshCullProjParams;
use footprint::project_aabb_to_screen;
use sampling::{HiZUvRect, sample_hiz_rect};

/// Small bias to reduce mip / quantization flicker at occlusion boundaries (reverse-Z).
const HI_Z_BIAS: f32 = 5e-5;

/// Extra reverse-Z slack before declaring full occlusion (reduces view-dependent popping at depth edges).
const HI_Z_OCCLUSION_MARGIN: f32 = 5e-4;

fn hiz_trace_enabled() -> bool {
    static FLAG: LazyLock<bool> = LazyLock::new(|| {
        env::var_os("RENDERIDE_HIZ_TRACE").is_some_and(|v| !v.is_empty() && v != "0")
    });
    *FLAG
}

/// Builds view-projection matrices for Hi-Z tests (same rules as frustum culling, using **previous**
/// frame data from [`crate::world_mesh::HiZTemporalState`]).
pub fn hi_z_view_proj_matrices(
    prev: &WorldMeshCullProjParams,
    prev_view: Mat4,
    is_overlay: bool,
) -> Vec<Mat4> {
    if is_overlay {
        return vec![prev.overlay_proj * overlay_camera_view_matrix()];
    }
    if let Some((sl, sr)) = prev.vr_stereo {
        return vec![sl, sr];
    }
    vec![prev.world_proj * prev_view]
}

/// Returns `true` when the axis-aligned world bounds are **fully occluded** by `snapshot` for `view_proj`.
///
/// Conservative: if **any** corner has `clip.w <= 0` (straddles the near plane / behind the camera),
/// returns `false` (keep the draw). Compares the AABB **closest** depth (maximum NDC Z in reverse-Z)
/// to the farthest depth stored in a padded projected-rectangle query. The query starts at the mip
/// matching the projected footprint size, walks back to mip0, and fails open when the rectangle
/// would require too many samples.
pub fn mesh_fully_occluded_in_hiz(
    snapshot: &HiZCpuSnapshot,
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> bool {
    let Some(footprint) = project_aabb_to_screen(view_proj, world_min, world_max) else {
        return false;
    };
    let Some(rect) = HiZUvRect::from_raw(footprint.uv_min, footprint.uv_max) else {
        return false;
    };

    let threshold = footprint.max_ndc_z + HI_Z_BIAS + HI_Z_OCCLUSION_MARGIN;
    if !threshold.is_finite() {
        return false;
    }

    // Reverse-Z: farther = smaller NDC Z. Fully occluded if the closest AABB point is still farther than the occluder.
    let query = sample_hiz_rect(snapshot, rect, threshold);
    if query.occluded && hiz_trace_enabled() {
        logger::trace!(
            "Hi-Z full occluder: mip={} samples={} rect={:?} max_ndc_z={} threshold={} farthest_depth={:?} reason={:?}",
            query.mip,
            query.total_samples,
            query.rect,
            footprint.max_ndc_z,
            threshold,
            query.farthest_depth,
            query.reason,
        );
    }
    query.occluded
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
        // Closest point ~0.9195; uniform Hi-Z 0.92 -- gap smaller than HI_Z_OCCLUSION_MARGIN + bias.
        let wmin = Vec3::new(-0.01, -0.01, 0.91);
        let wmax = Vec3::new(0.01, 0.01, 0.9195);
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
        let wmin = Vec3::new(-0.01, -0.01, 0.80);
        let wmax = Vec3::new(0.01, 0.01, 0.85);
        assert!(mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax));
    }

    fn snapshot_from_base(
        base_width: u32,
        base_height: u32,
        levels: u32,
        base: Vec<f32>,
    ) -> HiZCpuSnapshot {
        assert_eq!(base.len(), (base_width * base_height) as usize);
        let mut all = base.clone();
        let mut prev = base;
        let mut prev_w = base_width;
        let mut prev_h = base_height;
        for _mip in 1..levels {
            let next_w = (prev_w >> 1).max(1);
            let next_h = (prev_h >> 1).max(1);
            let mut next = vec![1.0f32; (next_w * next_h) as usize];
            for y in 0..next_h {
                let y0 = y * prev_h / next_h;
                let y1 = ((y + 1) * prev_h).div_ceil(next_h).min(prev_h);
                for x in 0..next_w {
                    let x0 = x * prev_w / next_w;
                    let x1 = ((x + 1) * prev_w).div_ceil(next_w).min(prev_w);
                    let mut farthest = f32::MAX;
                    for sy in y0..y1.max(y0 + 1) {
                        for sx in x0..x1.max(x0 + 1) {
                            let idx = (sy.min(prev_h - 1) * prev_w + sx.min(prev_w - 1)) as usize;
                            farthest = farthest.min(prev[idx]);
                        }
                    }
                    next[(y * next_w + x) as usize] = farthest;
                }
            }
            all.extend_from_slice(&next);
            prev = next;
            prev_w = next_w;
            prev_h = next_h;
        }
        HiZCpuSnapshot {
            base_width,
            base_height,
            mip_levels: levels,
            mips: Arc::from(all),
        }
    }

    /// A far reverse-Z hole anywhere in the projected rectangle keeps the draw visible.
    #[test]
    fn hiz_projected_rect_sees_farther_hole_at_edge() {
        let vp = Mat4::IDENTITY;
        let mut base = vec![0.95f32; 16 * 16];
        base[2 * 16 + 2] = 0.35;
        let snap = snapshot_from_base(16, 16, 5, base);
        assert!(snap.validate().is_some());
        let wmin = Vec3::new(-0.6, -0.6, 0.88);
        let wmax = Vec3::new(0.6, 0.6, 0.90);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax),
            "the projected rectangle query must include the far edge sample so we keep the draw"
        );
    }

    #[test]
    fn offscreen_hiz_projection_samples_flipped_depth_region() {
        let mut base = vec![0.95f32; 16 * 16];
        for y in 8..16 {
            for x in 0..16 {
                base[y * 16 + x] = 0.0;
            }
        }
        let snap = snapshot_from_base(16, 16, 5, base);
        assert!(snap.validate().is_some());

        let wmin = Vec3::new(-0.05, 0.45, 0.82);
        let wmax = Vec3::new(0.05, 0.55, 0.85);
        assert!(
            mesh_fully_occluded_in_hiz(&snap, Mat4::IDENTITY, wmin, wmax),
            "raw projection samples the top occluder and demonstrates the offscreen mismatch"
        );

        let offscreen_view_proj = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, offscreen_view_proj, wmin, wmax),
            "offscreen projection samples the vertically flipped visible depth region"
        );
    }

    #[test]
    fn broad_visible_rect_exceeding_sample_budget_is_kept() {
        let vp = Mat4::IDENTITY;
        let snap = snapshot_from_base(128, 128, 8, vec![0.0f32; 128 * 128]);
        assert!(snap.validate().is_some());
        let wmin = Vec3::new(-1.0, -1.0, 0.48);
        let wmax = Vec3::new(1.0, 1.0, 0.50);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, wmin, wmax),
            "sample budget overflow must fail open"
        );
    }

    #[test]
    fn straddling_near_plane_not_fully_occluded() {
        // Last row [0,0,0,-1] makes clip.w = -w; corners with w>0 and w<=0 in the same AABB -> keep draw.
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
        // 4x4 + 2x2 + 1x1 = 21 floats for three mips.
        let mips = vec![0.92f32; 21];
        let snap = HiZCpuSnapshot {
            base_width: 4,
            base_height: 4,
            mip_levels: 3,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        // Front of AABB at z=0.99 (closer than Hi-Z 0.92), back at z=0.05. Must not cull on back alone.
        let far = Vec3::new(-0.01, -0.01, 0.05);
        let near = Vec3::new(0.01, 0.01, 0.99);
        assert!(
            !mesh_fully_occluded_in_hiz(&snap, vp, far, near),
            "closest point still in front of occluder"
        );
    }
}
