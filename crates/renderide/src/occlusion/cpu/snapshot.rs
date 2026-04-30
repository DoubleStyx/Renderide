//! CPU-side hierarchical depth pyramid data types produced from GPU readback.

use std::sync::Arc;

use super::pyramid::{mip_dimensions, total_float_count};

/// Packed reverse-Z depth values (greater = closer) for one eye / one desktop pyramid.
///
/// `mips` stores mip0 row-major, then mip1, … each mip is `max(1, base_width >> k) × max(1, base_height >> k)`.
///
/// The pyramid buffer is shared via `Arc<[f32]>` so per-view and per-secondary-camera `Clone`s are
/// refcount bumps rather than full `Vec<f32>` copies. Producers construct the data once in a
/// `Vec<f32>` (see [`super::readback::unpack_linear_rows_to_mips`]) and hand it to
/// [`super::readback::hi_z_snapshot_from_linear_linear`], which converts into the shared representation.
#[derive(Clone, Debug)]
pub struct HiZCpuSnapshot {
    /// Width of mip0 (matches main depth attachment width).
    pub base_width: u32,
    /// Height of mip0 (matches main depth attachment height).
    pub base_height: u32,
    /// Number of mips present in `mips` (including mip0).
    pub mip_levels: u32,
    /// Row-major `f32` samples for all mips concatenated (shared; cloning is cheap).
    pub mips: Arc<[f32]>,
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
        let base = super::pyramid::mip_byte_offset_floats(self.base_width, self.base_height, mip);
        Some(base + (y * w + x) as usize)
    }

    /// Samples a depth value at integer texel coordinates for `mip`, or `None` if out of range.
    pub fn sample_texel(&self, mip: u32, x: u32, y: u32) -> Option<f32> {
        let i = self.texel_index(mip, x, y)?;
        self.mips.get(i).copied()
    }
}

/// Owned Hi-Z pyramids for [`crate::world_mesh::cull::WorldMeshCullInput`] (cloned once per frame from [`crate::backend::RenderBackend`]).
#[derive(Clone, Debug)]
pub enum HiZCullData {
    /// Single pyramid from desktop / mirror depth.
    Desktop(HiZCpuSnapshot),
    /// Left / right pyramids aligned with [`crate::world_mesh::cull::WorldMeshCullProjParams::vr_stereo`] order.
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

#[cfg(test)]
mod tests {
    use super::*;

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
                    let idx = super::super::pyramid::mip_byte_offset_floats(base_w, base_h, mip)
                        + (y * w + x) as usize;
                    mips[idx] = k;
                    k += 1.0;
                }
            }
        }
        let snap = HiZCpuSnapshot {
            base_width: base_w,
            base_height: base_h,
            mip_levels: levels,
            mips: Arc::from(mips),
        };
        assert!(snap.validate().is_some());
        assert_eq!(snap.sample_texel(0, 0, 0), Some(0.0));
        assert_eq!(snap.sample_texel(2, 0, 0), Some(20.0));
    }
}
