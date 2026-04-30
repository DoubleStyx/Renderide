//! CPU hierarchical-Z occlusion test (reverse-Z depth buffer).
//!
//! Hi-Z pyramids come from the **previous** frame; tests must use [`crate::world_mesh::HiZTemporalState`]
//! view–projection from the frame that produced that depth, not the current frame.
//!
//! Set `RENDERIDE_HIZ_TRACE=1` to emit [`logger::trace`] lines when a draw is classified as fully
//! occluded (can be verbose).

mod footprint;
mod sampling;

use std::env;
use std::sync::LazyLock;

use glam::{Mat4, Vec3};

use super::snapshot::HiZCpuSnapshot;
use crate::world_mesh::cull::WorldMeshCullProjParams;
use footprint::project_aabb_to_screen;
use sampling::{hiz_min_in_2x2, mip_extent, select_hi_z_mip};

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

/// Builds view–projection matrices for Hi-Z tests (same rules as frustum culling, using **previous**
/// frame data from [`crate::world_mesh::HiZTemporalState`]).
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
    let Some(footprint) = project_aabb_to_screen(view_proj, world_min, world_max) else {
        return false;
    };

    let base_w = snapshot.base_width.max(1) as f32;
    let base_h = snapshot.base_height.max(1) as f32;
    let du_base = (footprint.uv_max.0 - footprint.uv_min.0) * base_w;
    let dv_base = (footprint.uv_max.1 - footprint.uv_min.1) * base_h;
    let extent_base = du_base.max(dv_base).max(1.0);
    let mip = select_hi_z_mip(extent_base, snapshot.mip_levels);

    let Some((mw, mh)) = mip_extent(snapshot, mip) else {
        return false;
    };

    let uc = ((footprint.uv_min.0 + footprint.uv_max.0) * 0.5).clamp(0.0, 1.0);
    let vc = ((footprint.uv_min.1 + footprint.uv_max.1) * 0.5).clamp(0.0, 1.0);
    let sx = ((uc * mw as f32).floor() as u32).min(mw.saturating_sub(1));
    let sy = ((vc * mh as f32).floor() as u32).min(mh.saturating_sub(1));

    let Some(hiz_min) = hiz_min_in_2x2(snapshot, mip, sx, sy, mw, mh) else {
        return false;
    };

    // Reverse-Z: farther = smaller NDC Z. Fully occluded if the closest AABB point is still farther than the occluder.
    let occluded = footprint.max_ndc_z + HI_Z_BIAS + HI_Z_OCCLUSION_MARGIN < hiz_min;
    if occluded && hiz_trace_enabled() {
        logger::trace!(
            "Hi-Z full occluder: mip={} extent_base={} max_ndc_z={} hiz_min_2x2={}",
            mip,
            extent_base,
            footprint.max_ndc_z,
            hiz_min
        );
    }
    occluded
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
