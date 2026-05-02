//! Hi-Z mip-pyramid layout math (pure: no `wgpu` types, no allocation in hot helpers).

/// Maximum length of the **longer** side of Hi-Z mip0 (downscaled from the depth attachment).
///
/// Previously 256; halved to **128** to cut pyramid area (~4x fewer mip0 texels), reducing GPU
/// compute, readback size, and CPU unpacking at the cost of coarser occlusion tests.
pub const HI_Z_PYRAMID_MAX_LONG_EDGE: u32 = 128;

/// `(width, height)` for `mip` given mip0 size.
#[inline]
pub fn mip_dimensions(base_width: u32, base_height: u32, mip: u32) -> Option<(u32, u32)> {
    if base_width == 0 || base_height == 0 {
        return None;
    }
    let w = (base_width >> mip).max(1);
    let h = (base_height >> mip).max(1);
    Some((w, h))
}

/// Total `f32` count for a full mip chain down to 1x1 or `mip_levels` slices.
pub fn total_float_count(base_width: u32, base_height: u32, mip_levels: u32) -> usize {
    let mut n = 0usize;
    for m in 0..mip_levels {
        let (w, h) = mip_dimensions(base_width, base_height, m).unwrap_or((0, 0));
        n += (w * h) as usize;
    }
    n
}

/// Offset in **float elements** from the start of `mips` to the first texel of `mip`.
#[inline]
pub fn mip_byte_offset_floats(base_width: u32, base_height: u32, mip: u32) -> usize {
    let mut off = 0usize;
    for k in 0..mip {
        let (w, h) = mip_dimensions(base_width, base_height, k).unwrap_or((0, 0));
        off += (w * h) as usize;
    }
    off
}

/// Hi-Z mip0 dimensions derived from full depth attachment size (long edge capped for cost).
///
/// Matches the GPU pyramid base used for occlusion readback: scales down so the longest side is at
/// most [`HI_Z_PYRAMID_MAX_LONG_EDGE`] texels (same factor on both axes).
pub fn hi_z_pyramid_dimensions(depth_w: u32, depth_h: u32) -> (u32, u32) {
    let max_dim = depth_w.max(depth_h).max(1);
    let scale = max_dim.div_ceil(HI_Z_PYRAMID_MAX_LONG_EDGE).max(1);
    let bw = depth_w.div_ceil(scale).max(1);
    let bh = depth_h.div_ceil(scale).max(1);
    (bw, bh)
}

/// Number of mips for a full chain until both dimensions reach 1, capped.
pub fn mip_levels_for_extent(base_width: u32, base_height: u32, max_mips: u32) -> u32 {
    if base_width == 0 || base_height == 0 {
        return 0;
    }
    let mut w = base_width;
    let mut h = base_height;
    let mut levels = 1u32;
    while levels < max_mips && (w > 1 || h > 1) {
        w = (w >> 1).max(1);
        h = (h >> 1).max(1);
        levels += 1;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_dimensions_halves_until_one() {
        assert_eq!(mip_dimensions(8, 8, 0), Some((8, 8)));
        assert_eq!(mip_dimensions(8, 8, 3), Some((1, 1)));
    }

    #[test]
    fn total_float_count_matches_manual() {
        // 4x4 + 2x2 + 1x1 = 16+4+1 = 21
        assert_eq!(total_float_count(4, 4, 3), 21);
    }

    #[test]
    fn hi_z_pyramid_dimensions_caps_long_edge() {
        let (w, h) = hi_z_pyramid_dimensions(1920, 1080);
        assert!(w <= HI_Z_PYRAMID_MAX_LONG_EDGE && h <= HI_Z_PYRAMID_MAX_LONG_EDGE);
        assert!(w >= 1 && h >= 1);
    }
}
