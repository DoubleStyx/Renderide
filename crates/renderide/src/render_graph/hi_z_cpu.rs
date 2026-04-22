//! CPU-side hierarchical depth pyramid snapshot after GPU readback.

/// Packed reverse-Z depth values (greater = closer) for one eye / one desktop pyramid.
///
/// `mips` stores mip0 row-major, then mip1, … each mip is `max(1, base_width >> k) × max(1, base_height >> k)`.
#[derive(Clone, Debug)]
pub struct HiZCpuSnapshot {
    /// Width of mip0 (matches main depth attachment width).
    pub base_width: u32,
    /// Height of mip0 (matches main depth attachment height).
    pub base_height: u32,
    /// Number of mips present in `mips` (including mip0).
    pub mip_levels: u32,
    /// Row-major `f32` samples for all mips concatenated.
    pub mips: Vec<f32>,
}

impl HiZCpuSnapshot {
    /// Returns `None` when dimensions or mip count are inconsistent with `mips` length.
    pub fn validate(&self) -> Option<()> {
        let expected = total_float_count(self.base_width, self.base_height, self.mip_levels);
        if expected != self.mips.len() {
            return None;
        }
        Some(())
    }

    /// Linear index of texel `(x, y)` at `mip` (clamped dimensions).
    pub fn texel_index(&self, mip: u32, x: u32, y: u32) -> Option<usize> {
        let (w, h) = mip_dimensions(self.base_width, self.base_height, mip)?;
        if x >= w || y >= h {
            return None;
        }
        let base = mip_byte_offset_floats(self.base_width, self.base_height, mip);
        Some(base + (y * w + x) as usize)
    }

    /// Samples a depth value at integer texel coordinates for `mip`, or `None` if out of range.
    pub fn sample_texel(&self, mip: u32, x: u32, y: u32) -> Option<f32> {
        let i = self.texel_index(mip, x, y)?;
        self.mips.get(i).copied()
    }
}

/// Owned Hi-Z pyramids for [`super::world_mesh_cull::WorldMeshCullInput`] (cloned once per frame from [`crate::backend::RenderBackend`]).
#[derive(Clone, Debug)]
pub enum HiZCullData {
    /// Single pyramid from desktop / mirror depth.
    Desktop(HiZCpuSnapshot),
    /// Left / right pyramids aligned with [`super::world_mesh_cull::WorldMeshCullProjParams::vr_stereo`] order.
    Stereo {
        /// Hi-Z pyramid for the left eye.
        left: HiZCpuSnapshot,
        /// Hi-Z pyramid for the right eye.
        right: HiZCpuSnapshot,
    },
}

/// Per-eye CPU pyramids for stereo Hi-Z (layer order matches [`crate::xr::swapchain::XR_VIEW_COUNT`]).
#[derive(Clone, Debug)]
pub struct HiZStereoCpuSnapshot {
    /// Layer 0 (left eye).
    pub left: HiZCpuSnapshot,
    /// Layer 1 (right eye).
    pub right: HiZCpuSnapshot,
}

/// Total `f32` count for a full mip chain down to 1×1 or `mip_levels` slices.
pub fn total_float_count(base_width: u32, base_height: u32, mip_levels: u32) -> usize {
    let mut n = 0usize;
    for m in 0..mip_levels {
        let (w, h) = mip_dimensions(base_width, base_height, m).unwrap_or((0, 0));
        n += (w * h) as usize;
    }
    n
}

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

/// Maximum length of the **longer** side of Hi-Z mip0 (downscaled from the depth attachment).
///
/// Previously 256; halved to **128** to cut pyramid area (~4× fewer mip0 texels), reducing GPU
/// compute, readback size, and CPU unpacking at the cost of coarser occlusion tests.
pub const HI_Z_PYRAMID_MAX_LONG_EDGE: u32 = 128;

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

/// Unpacks a **linear** row-major buffer (no row padding) into [`HiZCpuSnapshot`].
pub fn hi_z_snapshot_from_linear_linear(
    base_width: u32,
    base_height: u32,
    mip_levels: u32,
    mips: Vec<f32>,
) -> Option<HiZCpuSnapshot> {
    let snap = HiZCpuSnapshot {
        base_width,
        base_height,
        mip_levels,
        mips,
    };
    snap.validate()?;
    Some(snap)
}

/// Unpacks GPU readback with `bytes_per_row` alignment (256-byte aligned rows) into dense `mips`.
///
/// The output is pre-allocated to [`total_float_count`] and row reads use `chunks_exact(4)` so the
/// inner loop avoids per-byte bounds checks. Bytes are interpreted via [`f32::from_le_bytes`] to
/// stay correct on misaligned [`Vec<u8>`] buffers from `wgpu::BufferSlice::get_mapped_range`.
pub fn unpack_linear_rows_to_mips(
    base_width: u32,
    base_height: u32,
    mip_levels: u32,
    staging: &[u8],
) -> Option<Vec<f32>> {
    let expected = total_float_count(base_width, base_height, mip_levels);
    let mut out: Vec<f32> = Vec::with_capacity(expected);
    let mut staging_off = 0usize;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_width, base_height, mip)?;
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
        let mip_bytes = row_pitch * h as usize;
        if staging_off + mip_bytes > staging.len() {
            return None;
        }
        let dense_row_bytes = (w as usize) * 4;
        for row in 0..h {
            let row_start = staging_off + row as usize * row_pitch;
            let row_bytes = staging.get(row_start..row_start + dense_row_bytes)?;
            out.extend(
                row_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        staging_off += mip_bytes;
    }
    if out.len() != expected {
        return None;
    }
    Some(out)
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

    #[test]
    fn mip_offset_roundtrip() {
        let base_w = 4u32;
        let base_h = 4u32;
        let levels = 3u32;
        let n = total_float_count(base_w, base_h, levels);
        let mut mips = vec![0.0f32; n];
        let mut k = 0.0f32;
        for mip in 0..levels {
            let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let idx = mip_byte_offset_floats(base_w, base_h, mip) + (y * w + x) as usize;
                    mips[idx] = k;
                    k += 1.0;
                }
            }
        }
        let snap = HiZCpuSnapshot {
            base_width: base_w,
            base_height: base_h,
            mip_levels: levels,
            mips,
        };
        assert!(snap.validate().is_some());
        assert_eq!(snap.sample_texel(0, 0, 0), Some(0.0));
        assert_eq!(snap.sample_texel(2, 0, 0), Some(20.0));
    }
}
