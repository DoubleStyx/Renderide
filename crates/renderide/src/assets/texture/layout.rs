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
        TextureFormat::BC1
        | TextureFormat::BC2
        | TextureFormat::BC3
        | TextureFormat::BC4
        | TextureFormat::BC5
        | TextureFormat::BC6H
        | TextureFormat::BC7
        | TextureFormat::ETC2RGB
        | TextureFormat::ETC2RGBA1
        | TextureFormat::ETC2RGBA8 => (4, 4),
        TextureFormat::ASTC4x4 => (4, 4),
        TextureFormat::ASTC5x5 => (5, 5),
        TextureFormat::ASTC6x6 => (6, 6),
        TextureFormat::ASTC8x8 => (8, 8),
        TextureFormat::ASTC10x10 => (10, 10),
        TextureFormat::ASTC12x12 => (12, 12),
        _ => (1, 1),
    }
}

/// Bytes occupied by one compressed block for `format`, or `None` for uncompressed (use [`mip_uncompressed_byte_len`]).
pub fn bytes_per_compressed_block(format: TextureFormat) -> Option<u32> {
    match format {
        TextureFormat::BC1 => Some(8),
        TextureFormat::BC2 | TextureFormat::BC3 => Some(16),
        TextureFormat::BC4 => Some(8),
        // Two BC4-style halves (R then G), 16 bytes total per 4×4 block (matches `wgpu::TextureFormat::Bc5RgUnorm`).
        TextureFormat::BC5 => Some(16),
        TextureFormat::BC6H | TextureFormat::BC7 => Some(16),
        TextureFormat::ETC2RGB | TextureFormat::ETC2RGBA1 => Some(8),
        TextureFormat::ETC2RGBA8 => Some(16),
        TextureFormat::ASTC4x4
        | TextureFormat::ASTC5x5
        | TextureFormat::ASTC6x6
        | TextureFormat::ASTC8x8
        | TextureFormat::ASTC10x10
        | TextureFormat::ASTC12x12 => Some(16),
        _ => None,
    }
}

/// Bytes required to store one mip level of an uncompressed format (`width` × `height` texels).
pub fn mip_uncompressed_byte_len(format: TextureFormat, width: u32, height: u32) -> Option<u64> {
    let w = u64::from(width);
    let h = u64::from(height);
    let px = w.checked_mul(h)?;
    let bpp = match format {
        TextureFormat::Alpha8 | TextureFormat::R8 => 1,
        TextureFormat::RGB565 | TextureFormat::BGR565 => 2,
        TextureFormat::RGB24 => 3,
        TextureFormat::RGBA32 | TextureFormat::ARGB32 | TextureFormat::BGRA32 => 4,
        // Matches `wgpu::TextureFormat::Rgba32Float` (four f32 per texel).
        TextureFormat::RGBAFloat | TextureFormat::ARGBFloat => 16,
        TextureFormat::RGBAHalf | TextureFormat::ARGBHalf => 8,
        TextureFormat::RHalf => 2,
        TextureFormat::RGHalf => 4,
        TextureFormat::RFloat => 4,
        TextureFormat::RGFloat => 8,
        TextureFormat::Unknown => return None,
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

/// Bytes per texel for a tightly packed mip slice (`mip_bytes == width × height × result`).
pub fn mip_tight_bytes_per_texel(mip_bytes: usize, width: u32, height: u32) -> Option<usize> {
    let px = (width as usize).checked_mul(height as usize)?;
    if px == 0 {
        return None;
    }
    if !mip_bytes.is_multiple_of(px) {
        return None;
    }
    Some(mip_bytes / px)
}

/// Storage size for one mip level in the host packing (compressed vs uncompressed).
pub fn mip_byte_len(format: TextureFormat, width: u32, height: u32) -> Option<u64> {
    if bytes_per_compressed_block(format).is_some() {
        mip_compressed_byte_len(format, width, height)
    } else {
        mip_uncompressed_byte_len(format, width, height)
    }
}

/// Swaps the four 12-bit alpha index rows in a BC3/BC4 48-bit index block (bytes `[2..8]` of an
/// 8-byte BC4 block, or the alpha half of BC3).
fn flip_bc3_bc4_alpha_index_rows(block: &mut [u8]) {
    debug_assert!(block.len() >= 8);
    let mut bits = 0u64;
    for i in 0..6 {
        bits |= u64::from(block[2 + i]) << (8 * i);
    }
    bits &= (1u64 << 48) - 1;
    let r0 = bits & 0xFFF;
    let r1 = (bits >> 12) & 0xFFF;
    let r2 = (bits >> 24) & 0xFFF;
    let r3 = (bits >> 36) & 0xFFF;
    // Vertical flip: top row ↔ bottom, middle rows ↔ each other.
    let new_bits = r3 | (r2 << 12) | (r1 << 24) | (r0 << 36);
    for i in 0..6 {
        block[2 + i] = (new_bits >> (8 * i)) as u8;
    }
}

/// Reverses BC1 color selector rows (bytes `[4..8]`): one byte per texel row, 2 bits per texel.
fn flip_bc1_color_selector_rows(block: &mut [u8]) {
    debug_assert!(block.len() >= 8);
    block.swap(4, 7);
    block.swap(5, 6);
}

/// Reverses vertical texel rows inside one BC1 block (`8` bytes).
fn flip_bc1_block_in_place(block: &mut [u8]) {
    debug_assert_eq!(block.len(), 8);
    flip_bc1_color_selector_rows(block);
}

/// Reverses vertical texel rows inside one BC2 block (`16` bytes): explicit alpha then BC1 color.
fn flip_bc2_block_in_place(block: &mut [u8]) {
    debug_assert_eq!(block.len(), 16);
    block.swap(0, 6);
    block.swap(1, 7);
    block.swap(2, 4);
    block.swap(3, 5);
    flip_bc1_color_selector_rows(&mut block[8..16]);
}

/// Reverses vertical texel rows inside one BC3 block (`16` bytes): BC3 alpha + BC1 color.
fn flip_bc3_block_in_place(block: &mut [u8]) {
    debug_assert_eq!(block.len(), 16);
    flip_bc3_bc4_alpha_index_rows(&mut block[0..8]);
    flip_bc1_color_selector_rows(&mut block[8..16]);
}

/// Reverses vertical texel rows inside one BC4 block (`8` bytes).
fn flip_bc4_block_in_place(block: &mut [u8]) {
    debug_assert_eq!(block.len(), 8);
    flip_bc3_bc4_alpha_index_rows(block);
}

/// Reverses vertical texel rows inside one BC5 block (`16` bytes): two BC4 halves (RG).
fn flip_bc5_block_in_place(block: &mut [u8]) {
    debug_assert_eq!(block.len(), 16);
    flip_bc4_block_in_place(&mut block[0..8]);
    flip_bc4_block_in_place(&mut block[8..16]);
}

/// Reverses vertical texel rows inside one compressed block when the format has a known in-place
/// flip ([`TextureFormat`] **BC1–BC5**). For **BC6H**, **BC7**, **ETC2** the in-block flip is mode
/// dependent and is intentionally skipped — the caller still reorders block rows so the macro
/// orientation is correct, accepting a residual 4-pixel zigzag inside each block row.
fn flip_compressed_block_in_place(block: &mut [u8], format: TextureFormat) {
    match format {
        TextureFormat::BC1 => flip_bc1_block_in_place(block),
        TextureFormat::BC2 => flip_bc2_block_in_place(block),
        TextureFormat::BC3 => flip_bc3_block_in_place(block),
        TextureFormat::BC4 => flip_bc4_block_in_place(block),
        TextureFormat::BC5 => flip_bc5_block_in_place(block),
        _ => {}
    }
}

/// Whether [`flip_compressed_mip_block_rows_y`] can produce a fully flipped mip for `format`.
///
/// **BC1–BC5** support both block-row reordering and the in-block texel flip. **BC6H**, **BC7**, and
/// **ETC2** use mode-dependent block layouts, so the function reorders block rows only and returns
/// the result with a documented residual zigzag — those still resolve to `true` here because the
/// upload path *is* able to make the texture macroscopically upright.
///
/// **ASTC** has block sizes from 4×4 up to 12×12 texels; a block-row-only reorder would expose
/// up-to-12-pixel zigzag artifacts which are too visible to be considered "flipped". ASTC stays
/// unsupported and [`flip_compressed_mip_block_rows_y`] returns [`None`] for it so callers surface
/// a hard upload error.
pub fn flip_compressed_mip_block_rows_y_supported(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::BC1
            | TextureFormat::BC2
            | TextureFormat::BC3
            | TextureFormat::BC4
            | TextureFormat::BC5
            | TextureFormat::BC6H
            | TextureFormat::BC7
            | TextureFormat::ETC2RGB
            | TextureFormat::ETC2RGBA1
            | TextureFormat::ETC2RGBA8
    )
}

/// Reverses the vertical order of **block rows** in a tight-packed compressed mip (D3D-style layout:
/// row-major blocks; see [`crate::assets::texture::decode::decode_bc1_to_rgba8`] outer `byi` loop).
///
/// For **BC1–BC5** also reverses **texel rows inside each block** so the result is pixel-exact.
///
/// For **BC6H**, **BC7**, and **ETC2** the per-block bit layout depends on a per-block mode field
/// and a generic in-place flip is not possible without a full decoder. Those formats are flipped
/// at block granularity only — macroscopically upright but with a residual ≤4-pixel zigzag along
/// horizontal block edges. This is preferred to the prior silent unflipped fallback (which left
/// the entire texture inverted under the engine's V-flip shaders) and to a hard upload error
/// (which would make those textures disappear entirely). All 4×4-block formats are handled this
/// way; **ASTC** (block size up to 12×12) is not handled and returns [`None`].
///
/// Returns [`None`] for **ASTC** and any other unsupported format, or when the mip length does not
/// match [`mip_byte_len`].
///
/// Used when host data is top-down and [`crate::shared::SetTexture2DData::flip_y`] requests
/// conversion to GPU bottom-up storage while the [`wgpu::TextureFormat`] is native block-compressed
/// (per-texel row flips do not apply).
pub fn flip_compressed_mip_block_rows_y(
    format: TextureFormat,
    width: u32,
    height: u32,
    mip_src: &[u8],
) -> Option<Vec<u8>> {
    if !flip_compressed_mip_block_rows_y_supported(format) {
        return None;
    }
    let bpb = bytes_per_compressed_block(format)? as usize;
    let (bw, bh) = block_extent(format);
    let expected = mip_byte_len(format, width, height)? as usize;
    if mip_src.len() != expected {
        return None;
    }
    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let row_stride = (blocks_x as usize).checked_mul(bpb)?;
    let mut out = vec![0u8; expected];
    if blocks_y >= 2 {
        for byi in 0..blocks_y {
            let src_off = (byi as usize).checked_mul(row_stride)?;
            let dst_off = ((blocks_y - 1 - byi) as usize).checked_mul(row_stride)?;
            let row = mip_src.get(src_off..src_off + row_stride)?;
            out.get_mut(dst_off..dst_off + row_stride)?
                .copy_from_slice(row);
        }
    } else {
        out.copy_from_slice(mip_src);
    }
    for byi in 0..blocks_y {
        for bxi in 0..blocks_x {
            let idx = (byi as usize)
                .checked_mul(blocks_x as usize)?
                .checked_add(bxi as usize)?;
            let off = idx.checked_mul(bpb)?;
            let block = out.get_mut(off..off.checked_add(bpb)?)?;
            flip_compressed_block_in_place(block, format);
        }
    }
    Some(out)
}

/// Converts `mip_starts[i]` (after subtracting any descriptor rebasing bias) from host **linear texel**
/// addressing into a **byte offset** into the tight-packed mip payload.
///
/// Shared-memory uploads may still report offsets in texel units; this maps them using the size of a
/// minimal tile (`1×1` texels in host packing, or one compressed block) relative to [`block_extent`].
pub fn host_mip_payload_byte_offset(
    format: TextureFormat,
    start_texel_linear: usize,
) -> Option<usize> {
    let (bw, bh) = block_extent(format);
    let texels_per_tile = (bw as usize).checked_mul(bh as usize)?;
    if texels_per_tile == 0 {
        return None;
    }
    let tile_bytes = mip_byte_len(format, 1, 1)? as usize;
    let numer = start_texel_linear.checked_mul(tile_bytes)?;
    Some(numer.div_ceil(texels_per_tile))
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

/// Width, height, and depth at `level` in a standard 3D mip chain.
pub fn mip_dimensions_at_level_3d(
    base_w: u32,
    base_h: u32,
    base_d: u32,
    level: u32,
) -> (u32, u32, u32) {
    let mut w = base_w;
    let mut h = base_h;
    let mut d = base_d;
    for _ in 0..level {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        d = (d / 2).max(1);
    }
    (w, h, d)
}

/// Sum of byte lengths for mips 0..`mipmap_count` for a 3D volume (`base_w`×`base_h`×`base_d`).
pub fn total_mip_chain_volume_byte_len(
    format: TextureFormat,
    base_w: u32,
    base_h: u32,
    base_d: u32,
    mipmap_count: u32,
) -> Option<u64> {
    let mut total = 0u64;
    for level in 0..mipmap_count {
        let (w, h, d) = mip_dimensions_at_level_3d(base_w, base_h, base_d, level);
        let slice = mip_byte_len(format, w, h)?;
        total = total.checked_add(slice.checked_mul(u64::from(d))?)?;
    }
    Some(total)
}

/// Approximate GPU bytes for a 3D texture (full mip chain) in `wgpu_format`.
pub fn estimate_gpu_texture3d_bytes(
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32,
) -> u64 {
    let mut sum = 0u64;
    let mut w = width;
    let mut h = height;
    let mut d = depth;
    for _ in 0..mip_levels {
        sum = sum.saturating_add(
            estimate_gpu_texture_bytes(wgpu_format, w, h, 1).saturating_mul(u64::from(d)),
        );
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        d = (d / 2).max(1);
    }
    sum
}

/// Approximate GPU bytes for a cubemap (six faces, full mip chain per face).
pub fn estimate_gpu_cubemap_bytes(
    wgpu_format: wgpu::TextureFormat,
    face_size: u32,
    mip_levels: u32,
) -> u64 {
    estimate_gpu_texture_bytes(wgpu_format, face_size, face_size, mip_levels).saturating_mul(6)
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
#[expect(
    clippy::map_err_ignore,
    reason = "TryFromIntError payload adds no detail beyond the overflow label already emitted"
)]
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
