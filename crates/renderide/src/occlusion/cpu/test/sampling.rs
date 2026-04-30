//! Hi-Z mip selection and 2×2-block min sampling for AABB occlusion tests.

use crate::occlusion::cpu::pyramid::{mip_byte_offset_floats, mip_dimensions};
use crate::occlusion::cpu::snapshot::HiZCpuSnapshot;

/// Caps the Hi-Z mip used for the occlusion test so coarse mips do not swing the decision when
/// the footprint crosses discrete mip boundaries.
const HI_Z_OCCLUSION_MAX_MIP: u32 = 2;

/// Picks a mip from an approximate footprint extent expressed in **base** Hi-Z texels.
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

/// Returns the Hi-Z mip level that should be sampled for a footprint of `extent_base_px` base
/// texels, capped both by [`HI_Z_OCCLUSION_MAX_MIP`] and by the pyramid's actual mip count.
#[inline]
pub(super) fn select_hi_z_mip(extent_base_px: f32, snapshot_mip_levels: u32) -> u32 {
    hi_z_mip_for_pixel_extent(extent_base_px)
        .min(HI_Z_OCCLUSION_MAX_MIP)
        .min(snapshot_mip_levels.saturating_sub(1))
}

/// Minimum depth in a 2×2 block anchored at `(sx, sy)` (clamped), reverse-Z.
///
/// Using the **minimum** (farthest surface in the neighborhood) is visibility-conservative: fewer
/// false-positive culls when a single texel spikes closer than its neighbors.
#[inline]
pub(super) fn hiz_min_in_2x2(
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
            if let Some(&v) = snapshot.mips.get(idx)
                && v.is_finite()
            {
                vmin = vmin.min(v);
                any = true;
            }
        }
    }
    any.then_some(vmin)
}

/// Looks up `(width, height)` for `mip`, returning `None` (skip the test) for degenerate snapshots.
#[inline]
pub(super) fn mip_extent(snapshot: &HiZCpuSnapshot, mip: u32) -> Option<(u32, u32)> {
    let (mw, mh) = mip_dimensions(snapshot.base_width, snapshot.base_height, mip)?;
    if mw == 0 || mh == 0 {
        None
    } else {
        Some((mw, mh))
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn select_hi_z_mip_clamps_to_snapshot_mips() {
        assert_eq!(select_hi_z_mip(1024.0, 2), 1);
        assert_eq!(select_hi_z_mip(1024.0, 1), 0);
    }
}
