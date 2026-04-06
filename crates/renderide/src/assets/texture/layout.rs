//! Mip sizing, block dimensions, and validation for host [`SetTexture2DData`](crate::shared::SetTexture2DData).
//!
//! Layout rules mirror block-compressed strides used by Unity/Renderite texture uploads.

use crate::shared::{SetTexture2DData, TextureFormat};

/// Whether the host enum uses block-compressed packing for a mip chain.
pub fn host_format_is_compressed(format: TextureFormat) -> bool {
    bytes_per_compressed_block(format).is_some()
}

/// Texel block dimensions for a host [`TextureFormat`] (width, height in texels per block).
pub fn block_extent(format: TextureFormat) -> (u32, u32) {
    match format {
        TextureFormat::bc1
        | TextureFormat::bc2
        | TextureFormat::bc3
        | TextureFormat::bc4
        | TextureFormat::bc5
        | TextureFormat::bc6_h
        | TextureFormat::bc7
        | TextureFormat::etc2_rgb
        | TextureFormat::etc2_rgba1
        | TextureFormat::etc2_rgba8 => (4, 4),
        TextureFormat::astc_4x4 => (4, 4),
        TextureFormat::astc_5x5 => (5, 5),
        TextureFormat::astc_6x6 => (6, 6),
        TextureFormat::astc_8x8 => (8, 8),
        TextureFormat::astc_10x10 => (10, 10),
        TextureFormat::astc_12x12 => (12, 12),
        _ => (1, 1),
    }
}

/// Bytes occupied by one compressed block for `format`, or `None` for uncompressed (use [`mip_uncompressed_byte_len`]).
pub fn bytes_per_compressed_block(format: TextureFormat) -> Option<u32> {
    match format {
        TextureFormat::bc1 => Some(8),
        TextureFormat::bc2 | TextureFormat::bc3 => Some(16),
        TextureFormat::bc4 | TextureFormat::bc5 => Some(8),
        TextureFormat::bc6_h | TextureFormat::bc7 => Some(16),
        TextureFormat::etc2_rgb | TextureFormat::etc2_rgba1 => Some(8),
        TextureFormat::etc2_rgba8 => Some(16),
        TextureFormat::astc_4x4
        | TextureFormat::astc_5x5
        | TextureFormat::astc_6x6
        | TextureFormat::astc_8x8
        | TextureFormat::astc_10x10
        | TextureFormat::astc_12x12 => Some(16),
        _ => None,
    }
}

/// Bytes required to store one mip level of an uncompressed format (`width` × `height` texels).
pub fn mip_uncompressed_byte_len(format: TextureFormat, width: u32, height: u32) -> Option<u64> {
    let w = u64::from(width);
    let h = u64::from(height);
    let px = w.checked_mul(h)?;
    let bpp = match format {
        TextureFormat::alpha8 | TextureFormat::r8 => 1,
        TextureFormat::rgb565 | TextureFormat::bgr565 => 2,
        TextureFormat::rgb24 => 3,
        TextureFormat::rgba32
        | TextureFormat::argb32
        | TextureFormat::bgra32
        | TextureFormat::rgba_float
        | TextureFormat::argb_float => 4,
        TextureFormat::rgba_half | TextureFormat::argb_half => 8,
        TextureFormat::r_half => 2,
        TextureFormat::rg_half => 4,
        TextureFormat::r_float => 4,
        TextureFormat::rg_float => 8,
        TextureFormat::unknown => return None,
        // Compressed formats should use [`mip_compressed_byte_len`].
        _ => return None,
    };
    px.checked_mul(bpp)
}

/// Bytes for one mip of a block-compressed format.
pub fn mip_compressed_byte_len(format: TextureFormat, width: u32, height: u32) -> Option<u64> {
    let (bw, bh) = block_extent(format);
    let bpb = bytes_per_compressed_block(format)?;
    let blocks_x = u64::from(width.div_ceil(bw));
    let blocks_y = u64::from(height.div_ceil(bh));
    let blocks = blocks_x.checked_mul(blocks_y)?;
    blocks.checked_mul(u64::from(bpb))
}

/// Storage size for one mip level in the host packing (compressed vs uncompressed).
pub fn mip_byte_len(format: TextureFormat, width: u32, height: u32) -> Option<u64> {
    if bytes_per_compressed_block(format).is_some() {
        mip_compressed_byte_len(format, width, height)
    } else {
        mip_uncompressed_byte_len(format, width, height)
    }
}

/// Returns the width and height of mip `level` in a standard mip chain (matches wgpu/WebGPU mip sizing).
///
/// `level` 0 is the base size; each subsequent level halves each dimension, clamped to at least 1.
pub fn mip_dimensions_at_level(base_w: u32, base_h: u32, level: u32) -> (u32, u32) {
    let mut w = base_w;
    let mut h = base_h;
    for _ in 0..level {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    (w, h)
}

/// Sum of byte lengths for mips 0..`mipmap_count` for `base_w`×`base_h`.
pub fn total_mip_chain_byte_len(
    format: TextureFormat,
    base_w: u32,
    base_h: u32,
    mipmap_count: u32,
) -> Option<u64> {
    let mut total = 0u64;
    let mut w = base_w;
    let mut h = base_h;
    for _ in 0..mipmap_count {
        let mip = mip_byte_len(format, w, h)?;
        total = total.checked_add(mip)?;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    Some(total)
}

/// Approximate GPU bytes for a 2D texture (full mip chain) in `wgpu_format` (used for VRAM accounting).
pub fn estimate_gpu_texture_bytes(
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> u64 {
    let (block_w, block_h) = wgpu_format.block_dimensions();
    let block_w = block_w.max(1);
    let block_h = block_h.max(1);
    let bpp = u64::from(wgpu_format.block_copy_size(None).unwrap_or(0));
    let mut sum = 0u64;
    let mut w = width;
    let mut h = height;
    for _ in 0..mip_levels {
        let bx = u64::from(w.div_ceil(block_w));
        let by = u64::from(h.div_ceil(block_h));
        sum = sum.saturating_add(bx.saturating_mul(by).saturating_mul(bpp));
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    sum
}

/// Validates `mip_map_sizes` / `mip_starts` against `data.data.length` (payload window).
pub fn validate_mip_upload_layout(
    format: TextureFormat,
    data: &SetTexture2DData,
) -> Result<(), &'static str> {
    if data.mip_map_sizes.len() != data.mip_starts.len() {
        return Err("mip_map_sizes and mip_starts length mismatch");
    }
    if data.mip_map_sizes.is_empty() {
        return Err("no mips in upload");
    }
    let buf_len = u64::try_from(data.data.length.max(0)).map_err(|_| "buffer length overflow")?;

    for (i, sz) in data.mip_map_sizes.iter().enumerate() {
        if sz.x <= 0 || sz.y <= 0 {
            return Err("non-positive mip dimensions");
        }
        let w = u32::try_from(sz.x).map_err(|_| "mip width overflow")?;
        let h = u32::try_from(sz.y).map_err(|_| "mip height overflow")?;
        let mip_len =
            mip_byte_len(format, w, h).ok_or("unknown or unsupported format for mip size")?;
        let start_rel = data.mip_starts[i];
        if start_rel < 0 {
            return Err("negative mip_starts");
        }
        let start = u64::try_from(start_rel).map_err(|_| "mip start overflow")?;
        let end = start.checked_add(mip_len).ok_or("mip end overflow")?;
        if end > buf_len {
            return Err("mip region exceeds shared memory descriptor");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::SetTexture2DData;
    use glam::IVec2;

    #[test]
    fn validate_mip_layout_accepts_contiguous_payload() {
        let mut d = SetTexture2DData::default();
        d.data.length = 4 * 4 * 4 + 2 * 2 * 4; // rgba32 mip0 + mip1
        d.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        d.mip_starts = vec![0, 64];
        assert!(validate_mip_upload_layout(TextureFormat::rgba32, &d).is_ok());
    }

    #[test]
    fn validate_mip_layout_rejects_overflow() {
        let mut d = SetTexture2DData::default();
        d.data.length = 10;
        d.mip_map_sizes = vec![IVec2::new(4, 4)];
        d.mip_starts = vec![0];
        assert!(validate_mip_upload_layout(TextureFormat::rgba32, &d).is_err());
    }

    #[test]
    fn bc1_mip0_128_bytes_for_32x32() {
        let b = mip_byte_len(TextureFormat::bc1, 32, 32).expect("bc1");
        assert_eq!(b, (32 / 4) * (32 / 4) * 8);
    }

    #[test]
    fn rgba32_mip0_byte_len() {
        assert_eq!(
            mip_byte_len(TextureFormat::rgba32, 16, 16).unwrap(),
            16 * 16 * 4
        );
    }

    #[test]
    fn mip_dimensions_at_level_halves_each_step() {
        assert_eq!(mip_dimensions_at_level(114, 200, 0), (114, 200));
        assert_eq!(mip_dimensions_at_level(114, 200, 1), (57, 100));
        assert_eq!(mip_dimensions_at_level(114, 200, 2), (28, 50));
        assert_eq!(mip_dimensions_at_level(1, 1, 5), (1, 1));
    }
}
