//! Hi-Z mip selection and padded-rectangle sampling for AABB occlusion tests.

use crate::hi_z_cpu::pyramid::{mip_byte_offset_floats, mip_dimensions};
use crate::hi_z_cpu::snapshot::HiZCpuSnapshot;

const HI_Z_RECT_SAMPLE_LIMIT: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HiZRectReason {
    Occluded,
    VisibleDepth,
    SampleBudgetExceeded,
    Offscreen,
    EmptySnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HiZSampleRect {
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}

impl HiZSampleRect {
    #[inline]
    fn sample_count(self) -> u32 {
        (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HiZRectQuery {
    pub occluded: bool,
    pub reason: HiZRectReason,
    pub mip: u32,
    pub total_samples: u32,
    pub rect: Option<HiZSampleRect>,
    pub farthest_depth: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HiZUvRect {
    pub min_u: f32,
    pub min_v: f32,
    pub max_u: f32,
    pub max_v: f32,
}

impl HiZUvRect {
    #[inline]
    pub(super) fn from_raw(uv_min: (f32, f32), uv_max: (f32, f32)) -> Option<Self> {
        let min_u = uv_min.0.min(uv_max.0);
        let max_u = uv_min.0.max(uv_max.0);
        let min_v = uv_min.1.min(uv_max.1);
        let max_v = uv_min.1.max(uv_max.1);
        if !min_u.is_finite() || !max_u.is_finite() || !min_v.is_finite() || !max_v.is_finite() {
            return None;
        }
        if max_u < 0.0 || min_u > 1.0 || max_v < 0.0 || min_v > 1.0 {
            return None;
        }
        Some(Self {
            min_u: min_u.clamp(0.0, 1.0),
            min_v: min_v.clamp(0.0, 1.0),
            max_u: max_u.clamp(0.0, 1.0),
            max_v: max_v.clamp(0.0, 1.0),
        })
    }

    #[inline]
    fn extent_base_px(self, base_width: u32, base_height: u32) -> f32 {
        let du = (self.max_u - self.min_u) * base_width.max(1) as f32;
        let dv = (self.max_v - self.min_v) * base_height.max(1) as f32;
        du.max(dv).max(1.0)
    }
}

/// Picks the first mip from an approximate footprint extent expressed in base Hi-Z texels.
///
/// Starts coarse enough for the projected rectangle, then walks toward mip0 until a level proves
/// full occlusion or the query fails open.
#[inline]
fn hi_z_mip_for_pixel_extent(extent_base_px: f32) -> u32 {
    if !extent_base_px.is_finite() || extent_base_px <= 1.0 {
        return 0;
    }
    extent_base_px.log2().ceil().max(0.0) as u32
}

/// Returns the coarsest Hi-Z mip level sampled first for a projected footprint.
#[inline]
pub(super) fn select_hi_z_start_mip(extent_base_px: f32, snapshot_mip_levels: u32) -> u32 {
    hi_z_mip_for_pixel_extent(extent_base_px).min(snapshot_mip_levels.saturating_sub(1))
}

#[inline]
fn padded_rect_for_mip(rect: HiZUvRect, width: u32, height: u32) -> Option<HiZSampleRect> {
    if width == 0 || height == 0 {
        return None;
    }
    let max_x = width - 1;
    let max_y = height - 1;
    let min_x = ((rect.min_u * width as f32) - 1.0)
        .floor()
        .clamp(0.0, max_x as f32) as u32;
    let max_x = ((rect.max_u * width as f32) + 1.0)
        .floor()
        .clamp(0.0, max_x as f32) as u32;
    let min_y = ((rect.min_v * height as f32) - 1.0)
        .floor()
        .clamp(0.0, max_y as f32) as u32;
    let max_y = ((rect.max_v * height as f32) + 1.0)
        .floor()
        .clamp(0.0, max_y as f32) as u32;
    Some(HiZSampleRect {
        min_x,
        max_x,
        min_y,
        max_y,
    })
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

/// Tests a projected rectangle against a farthest-depth reverse-Z pyramid.
pub(super) fn sample_hiz_rect(
    snapshot: &HiZCpuSnapshot,
    rect: HiZUvRect,
    closest_depth_threshold: f32,
) -> HiZRectQuery {
    let Some((base_w, base_h)) = mip_extent(snapshot, 0) else {
        return visible_result(HiZRectReason::EmptySnapshot, 0, 0, None, None);
    };
    let start_mip = select_hi_z_start_mip(rect.extent_base_px(base_w, base_h), snapshot.mip_levels);
    let mut total_samples = 0u32;
    let mut last_rect = None;
    let mut last_farthest_depth = None;

    for mip in (0..=start_mip).rev() {
        let Some((mw, mh)) = mip_extent(snapshot, mip) else {
            return visible_result(
                HiZRectReason::EmptySnapshot,
                mip,
                total_samples,
                last_rect,
                last_farthest_depth,
            );
        };
        let Some(sample_rect) = padded_rect_for_mip(rect, mw, mh) else {
            return visible_result(
                HiZRectReason::Offscreen,
                mip,
                total_samples,
                last_rect,
                last_farthest_depth,
            );
        };
        total_samples = total_samples.saturating_add(sample_rect.sample_count());
        if total_samples > HI_Z_RECT_SAMPLE_LIMIT {
            return visible_result(
                HiZRectReason::SampleBudgetExceeded,
                mip,
                total_samples,
                Some(sample_rect),
                last_farthest_depth,
            );
        }

        let mip_base = mip_byte_offset_floats(snapshot.base_width, snapshot.base_height, mip);
        let mut visible_at_this_mip = false;
        let mut farthest_depth = f32::MAX;
        'samples: for y in sample_rect.min_y..=sample_rect.max_y {
            for x in sample_rect.min_x..=sample_rect.max_x {
                let idx = mip_base + (y * mw + x) as usize;
                let Some(&depth) = snapshot.mips.get(idx) else {
                    return visible_result(
                        HiZRectReason::EmptySnapshot,
                        mip,
                        total_samples,
                        Some(sample_rect),
                        last_farthest_depth,
                    );
                };
                if !depth.is_finite() {
                    return visible_result(
                        HiZRectReason::VisibleDepth,
                        mip,
                        total_samples,
                        Some(sample_rect),
                        Some(depth),
                    );
                }
                farthest_depth = farthest_depth.min(depth);
                if depth <= closest_depth_threshold {
                    visible_at_this_mip = true;
                    break 'samples;
                }
            }
        }

        last_rect = Some(sample_rect);
        last_farthest_depth = Some(farthest_depth);
        if !visible_at_this_mip {
            return HiZRectQuery {
                occluded: true,
                reason: HiZRectReason::Occluded,
                mip,
                total_samples,
                rect: Some(sample_rect),
                farthest_depth: Some(farthest_depth),
            };
        }
    }

    visible_result(
        HiZRectReason::VisibleDepth,
        0,
        total_samples,
        last_rect,
        last_farthest_depth,
    )
}

#[inline]
fn visible_result(
    reason: HiZRectReason,
    mip: u32,
    total_samples: u32,
    rect: Option<HiZSampleRect>,
    farthest_depth: Option<f32>,
) -> HiZRectQuery {
    HiZRectQuery {
        occluded: false,
        reason,
        mip,
        total_samples,
        rect,
        farthest_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hi_z_mip_for_pixel_extent_levels() {
        assert_eq!(hi_z_mip_for_pixel_extent(1.0), 0);
        assert_eq!(hi_z_mip_for_pixel_extent(2.0), 1);
        assert_eq!(hi_z_mip_for_pixel_extent(2.1), 2);
        assert_eq!(hi_z_mip_for_pixel_extent(4.0), 2);
        assert_eq!(hi_z_mip_for_pixel_extent(8.0), 3);
    }

    #[test]
    fn select_hi_z_mip_clamps_to_snapshot_mips() {
        assert_eq!(select_hi_z_start_mip(1024.0, 2), 1);
        assert_eq!(select_hi_z_start_mip(1024.0, 1), 0);
    }

    #[test]
    fn padded_rect_expands_by_one_texel() {
        let rect = HiZUvRect::from_raw((0.25, 0.25), (0.5, 0.5)).unwrap();
        assert_eq!(
            padded_rect_for_mip(rect, 8, 8),
            Some(HiZSampleRect {
                min_x: 1,
                max_x: 5,
                min_y: 1,
                max_y: 5
            })
        );
    }
}
