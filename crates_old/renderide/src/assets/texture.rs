//! Texture asset: CPU-side mip0 after host `SetTexture2D*` commands.
//!
//! GPU [`wgpu::TextureView`] creation lives in [`crate::gpu::GpuState::ensure_texture2d_gpu`].

use crate::shared::TextureFormat;

use super::Asset;
use super::AssetId;

/// Stored 2D texture data for GPU upload (mip level 0, RGBA8).
pub struct TextureAsset {
    /// Unique identifier for this texture.
    pub id: AssetId,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// Decoded mip0: `width * height * 4` bytes, sRGB-ready RGBA8.
    pub rgba8_mip0: Vec<u8>,
    /// Monotonic content revision for this row; bumps on `SetTexture2DFormat` and successful `SetTexture2DData`.
    ///
    /// [`crate::gpu::state::ensure_texture2d_gpu_view`] compares this to the last GPU upload so
    /// unchanged mip0 is not re-copied every frame.
    pub data_version: u64,
}

impl TextureAsset {
    /// Returns true when [`crate::gpu::GpuState::ensure_texture2d_gpu`] can build a GPU texture.
    pub fn ready_for_gpu(&self) -> bool {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        self.width > 0 && self.height > 0 && self.rgba8_mip0.len() >= expected
    }
}

impl Asset for TextureAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}

/// Converts raw mip0 bytes from the host into tight RGBA8 (row-major, top row first after optional flip).
///
/// `rgb565` uses `RRRRR GGGGGG BBBBB` in the 16-bit word; `bgr565` uses `BBBBB GGGGGG RRRRR`.
///
/// `bc1` (DXT1) and `bc3` (DXT5) decode 4×4 blocks to RGBA8 on the CPU (same layout as D3D11 BC formats).
pub fn decode_texture_mip0_to_rgba8(
    format: TextureFormat,
    width: u32,
    height: u32,
    flip_y: bool,
    raw: &[u8],
) -> Option<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let count = w.checked_mul(h)?;
    match format {
        TextureFormat::rgb24 => {
            let need = count.checked_mul(3)?;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            for p in raw[..need].chunks_exact(3) {
                out.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::rgba32 => {
            let need = count.checked_mul(4)?;
            if raw.len() < need {
                return None;
            }
            let mut out = raw[..need].to_vec();
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::argb32 => {
            let need = count.checked_mul(4)?;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(need);
            for p in raw[..need].chunks_exact(4) {
                out.extend_from_slice(&[p[1], p[2], p[3], p[0]]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::bgra32 => {
            let need = count.checked_mul(4)?;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(need);
            for p in raw[..need].chunks_exact(4) {
                out.push(p[2]);
                out.push(p[1]);
                out.push(p[0]);
                out.push(p[3]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::r8 => {
            let need = count;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            for &g in &raw[..need] {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::alpha8 => {
            let need = count;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            for &a in &raw[..need] {
                out.extend_from_slice(&[255, 255, 255, a]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::rgb565 | TextureFormat::bgr565 => {
            let need = count.checked_mul(2)?;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            for chunk in raw[..need].chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let (r5, g6, b5) = if format == TextureFormat::bgr565 {
                    let b5 = (v >> 11) & 0x1f;
                    let g6 = (v >> 5) & 0x3f;
                    let r5 = v & 0x1f;
                    (r5, g6, b5)
                } else {
                    let r5 = (v >> 11) & 0x1f;
                    let g6 = (v >> 5) & 0x3f;
                    let b5 = v & 0x1f;
                    (r5, g6, b5)
                };
                let r = ((u32::from(r5) * 255 + 15) / 31) as u8;
                let g = ((u32::from(g6) * 255 + 31) / 63) as u8;
                let b = ((u32::from(b5) * 255 + 15) / 31) as u8;
                out.extend_from_slice(&[r, g, b, 255]);
            }
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            Some(out)
        }
        TextureFormat::bc1 => decode_bc1_to_rgba8(w, h, raw).map(|mut out| {
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            out
        }),
        TextureFormat::bc3 => decode_bc3_to_rgba8(w, h, raw).map(|mut out| {
            if flip_y {
                flip_rgba_image_rows(&mut out, w, h);
            }
            out
        }),
        _ => None,
    }
}

/// Expands a 16-bit RGB565 value to 8-bit sRGB-ish components (matches other formats in this module).
fn rgb565_to_rgb8(c: u16) -> (u8, u8, u8) {
    let r5 = (c >> 11) & 0x1f;
    let g6 = (c >> 5) & 0x3f;
    let b5 = c & 0x1f;
    let r = ((u32::from(r5) * 255 + 15) / 31) as u8;
    let g = ((u32::from(g6) * 255 + 31) / 63) as u8;
    let b = ((u32::from(b5) * 255 + 15) / 31) as u8;
    (r, g, b)
}

/// Decodes one BC1 (DXT1) 8-byte block into 16 RGBA8 pixels (row-major within the 4×4 tile).
fn decode_bc1_block(block: &[u8; 8], tile_rgba: &mut [u8; 64]) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let (r0, g0, b0) = rgb565_to_rgb8(c0);
    let (r1, g1, b1) = rgb565_to_rgb8(c1);
    let colors: [[u8; 4]; 4] = if c0 > c1 {
        [
            [r0, g0, b0, 255],
            [r1, g1, b1, 255],
            [
                ((2 * u32::from(r0) + u32::from(r1)) / 3) as u8,
                ((2 * u32::from(g0) + u32::from(g1)) / 3) as u8,
                ((2 * u32::from(b0) + u32::from(b1)) / 3) as u8,
                255,
            ],
            [
                ((u32::from(r0) + 2 * u32::from(r1)) / 3) as u8,
                ((u32::from(g0) + 2 * u32::from(g1)) / 3) as u8,
                ((u32::from(b0) + 2 * u32::from(b1)) / 3) as u8,
                255,
            ],
        ]
    } else {
        [
            [r0, g0, b0, 255],
            [r1, g1, b1, 255],
            [
                ((u32::from(r0) + u32::from(r1)) / 2) as u8,
                ((u32::from(g0) + u32::from(g1)) / 2) as u8,
                ((u32::from(b0) + u32::from(b1)) / 2) as u8,
                255,
            ],
            [0, 0, 0, 0],
        ]
    };
    for i in 0..16 {
        let code = ((bits >> (i * 2)) & 3) as usize;
        let px = colors[code];
        tile_rgba[i * 4..(i + 1) * 4].copy_from_slice(&px);
    }
}

/// Decodes the first 8 bytes of a BC3 block as DXT5-style alpha (16× 8-bit alpha values).
fn decode_bc3_alpha_block(block_alpha: &[u8; 8], out_alpha: &mut [u8; 16]) {
    let a0 = u32::from(block_alpha[0]);
    let a1 = u32::from(block_alpha[1]);
    let mut bits = 0u64;
    for i in 0..6 {
        bits |= u64::from(block_alpha[2 + i]) << (8 * i);
    }
    let lut: [u8; 8] = if a0 > a1 {
        [
            a0 as u8,
            a1 as u8,
            ((6 * a0 + a1) / 7) as u8,
            ((5 * a0 + 2 * a1) / 7) as u8,
            ((4 * a0 + 3 * a1) / 7) as u8,
            ((3 * a0 + 4 * a1) / 7) as u8,
            ((2 * a0 + 5 * a1) / 7) as u8,
            ((a0 + 6 * a1) / 7) as u8,
        ]
    } else {
        [
            a0 as u8,
            a1 as u8,
            ((4 * a0 + a1) / 5) as u8,
            ((3 * a0 + 2 * a1) / 5) as u8,
            ((2 * a0 + 3 * a1) / 5) as u8,
            ((a0 + 4 * a1) / 5) as u8,
            0,
            255,
        ]
    };
    for (i, slot) in out_alpha.iter_mut().enumerate().take(16) {
        let code = ((bits >> (i * 3)) & 7) as usize;
        *slot = lut[code];
    }
}

/// Decodes mip0 BC1 (DXT1) to tight RGBA8. Expects `ceil(w/4)*ceil(h/4)*8` bytes.
fn decode_bc1_to_rgba8(width: usize, height: usize, raw: &[u8]) -> Option<Vec<u8>> {
    if width == 0 || height == 0 {
        return None;
    }
    let bx = width.div_ceil(4);
    let by = height.div_ceil(4);
    let need = bx.checked_mul(by)?.checked_mul(8)?;
    if raw.len() < need {
        return None;
    }
    let mut out = vec![0u8; width.checked_mul(height)?.checked_mul(4)?];
    for byi in 0..by {
        for bxi in 0..bx {
            let off = (byi * bx + bxi) * 8;
            let block: &[u8; 8] = raw.get(off..off + 8)?.try_into().ok()?;
            let mut tile = [0u8; 64];
            decode_bc1_block(block, &mut tile);
            for y in 0..4 {
                for x in 0..4 {
                    let gx = bxi * 4 + x;
                    let gy = byi * 4 + y;
                    if gx < width && gy < height {
                        let ti = (y * 4 + x) * 4;
                        let dst = (gy * width + gx) * 4;
                        out[dst..dst + 4].copy_from_slice(&tile[ti..ti + 4]);
                    }
                }
            }
        }
    }
    Some(out)
}

/// Decodes mip0 BC3 (DXT5) to tight RGBA8. Expects `ceil(w/4)*ceil(h/4)*16` bytes.
fn decode_bc3_to_rgba8(width: usize, height: usize, raw: &[u8]) -> Option<Vec<u8>> {
    if width == 0 || height == 0 {
        return None;
    }
    let bx = width.div_ceil(4);
    let by = height.div_ceil(4);
    let need = bx.checked_mul(by)?.checked_mul(16)?;
    if raw.len() < need {
        return None;
    }
    let mut out = vec![0u8; width.checked_mul(height)?.checked_mul(4)?];
    for byi in 0..by {
        for bxi in 0..bx {
            let off = (byi * bx + bxi) * 16;
            let chunk = raw.get(off..off + 16)?;
            let alpha: &[u8; 8] = chunk.get(0..8)?.try_into().ok()?;
            let color: &[u8; 8] = chunk.get(8..16)?.try_into().ok()?;
            let mut tile = [0u8; 64];
            decode_bc1_block(color, &mut tile);
            let mut alphas = [0u8; 16];
            decode_bc3_alpha_block(alpha, &mut alphas);
            for i in 0..16 {
                tile[i * 4 + 3] = alphas[i];
            }
            for y in 0..4 {
                for x in 0..4 {
                    let gx = bxi * 4 + x;
                    let gy = byi * 4 + y;
                    if gx < width && gy < height {
                        let ti = (y * 4 + x) * 4;
                        let dst = (gy * width + gx) * 4;
                        out[dst..dst + 4].copy_from_slice(&tile[ti..ti + 4]);
                    }
                }
            }
        }
    }
    Some(out)
}

fn flip_rgba_image_rows(buf: &mut [u8], width: usize, height: usize) {
    let row = width.saturating_mul(4);
    if row == 0 || height < 2 {
        return;
    }
    let mut tmp = vec![0u8; row];
    for y in 0..height / 2 {
        let a = y * row;
        let b = (height - 1 - y) * row;
        let (before, after) = buf.split_at_mut(b);
        let row_a = &mut before[a..a + row];
        let row_b = &mut after[..row];
        tmp.copy_from_slice(row_a);
        row_a.copy_from_slice(row_b);
        row_b.copy_from_slice(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_texture_mip0_to_rgba8, flip_rgba_image_rows};
    use crate::shared::TextureFormat;

    #[test]
    fn rgba32_roundtrip_size() {
        let raw: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 255];
        let out =
            decode_texture_mip0_to_rgba8(TextureFormat::rgba32, 2, 1, false, &raw).expect("ok");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn rgb24_expands_to_rgba() {
        let raw = vec![10u8, 20, 30, 40, 50, 60];
        let out =
            decode_texture_mip0_to_rgba8(TextureFormat::rgb24, 2, 1, false, &raw).expect("ok");
        assert_eq!(out, vec![10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn argb32_swizzles_to_rgba() {
        let raw = vec![255u8, 1, 2, 3];
        let out =
            decode_texture_mip0_to_rgba8(TextureFormat::argb32, 1, 1, false, &raw).expect("ok");
        assert_eq!(out, vec![1, 2, 3, 255]);
    }

    #[test]
    fn flip_y_swaps_rows() {
        let mut v = vec![
            1, 0, 0, 0, 2, 0, 0, 0, //
            3, 0, 0, 0, 4, 0, 0, 0,
        ];
        flip_rgba_image_rows(&mut v, 2, 2);
        assert_eq!(v[0..4], [3, 0, 0, 0]);
        assert_eq!(v[4..8], [4, 0, 0, 0]);
    }

    /// Full red in RGB565: `R=31`, `G=0`, `B=0` → `0xF800` LE `[0x00, 0xF8]`.
    #[test]
    fn rgb565_decodes_to_rgba() {
        let raw = vec![0x00u8, 0xF8];
        let out =
            decode_texture_mip0_to_rgba8(TextureFormat::rgb565, 1, 1, false, &raw).expect("ok");
        assert_eq!(out.len(), 4);
        assert!(out[0] >= 250 && out[1] < 5 && out[2] < 5 && out[3] == 255);
    }

    /// Full blue in BGR565: `B=31`, `G=0`, `R=0` → high 5 bits blue → `0xF800` LE `[0x00, 0xF8]`.
    #[test]
    fn bgr565_decodes_to_rgba() {
        let raw = vec![0x00u8, 0xF8];
        let out =
            decode_texture_mip0_to_rgba8(TextureFormat::bgr565, 1, 1, false, &raw).expect("ok");
        assert_eq!(out.len(), 4);
        assert!(out[2] >= 250 && out[1] < 5 && out[0] < 5 && out[3] == 255);
    }

    /// Solid red DXT1 block: color0 red (`0xF800`), color1 black, all indices 0.
    #[test]
    fn bc1_decodes_red_1x1() {
        let raw = vec![0x00u8, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let out = decode_texture_mip0_to_rgba8(TextureFormat::bc1, 1, 1, false, &raw).expect("ok");
        assert_eq!(out.len(), 4);
        assert!(out[0] >= 250 && out[1] < 5 && out[2] < 5 && out[3] == 255);
    }

    /// DXT5: alpha `a0=255`, `a1=254` (so `a0 > a1`), indices 0 → opaque; color block same as [`bc1_decodes_red_1x1`].
    #[test]
    fn bc3_decodes_red_opaque_1x1() {
        let raw = vec![
            255, 254, 0, 0, 0, 0, 0, 0, //
            0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let out = decode_texture_mip0_to_rgba8(TextureFormat::bc3, 1, 1, false, &raw).expect("ok");
        assert_eq!(out.len(), 4);
        assert!(out[0] >= 250 && out[1] < 5 && out[2] < 5 && out[3] >= 250);
    }
}
