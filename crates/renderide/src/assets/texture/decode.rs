//! Transient CPU decode paths for host [`TextureFormat`] when GPU-native storage is unavailable or swizzle is required.

use crate::shared::TextureFormat;

/// Decodes one mip level from `raw` to tightly packed RGBA8 (row-major, top-first after optional flip).
///
/// Used as a fallback for missing compression features or packed formats without a direct `wgpu` layout match.
pub fn decode_mip_to_rgba8(
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
        TextureFormat::r8 | TextureFormat::alpha8 => {
            let need = count;
            if raw.len() < need {
                return None;
            }
            let mut out = Vec::with_capacity(count * 4);
            if format == TextureFormat::r8 {
                for &g in &raw[..need] {
                    out.extend_from_slice(&[g, g, g, 255]);
                }
            } else {
                for &a in &raw[..need] {
                    out.extend_from_slice(&[255, 255, 255, a]);
                }
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

/// Returns true if host packing differs from tight RGBA8 used by `wgpu::TextureFormat::Rgba8Unorm`.
pub fn needs_rgba8_decode_before_upload(host: TextureFormat) -> bool {
    matches!(
        host,
        TextureFormat::rgb24
            | TextureFormat::argb32
            | TextureFormat::bgra32
            | TextureFormat::r8
            | TextureFormat::alpha8
            | TextureFormat::rgb565
            | TextureFormat::bgr565
            | TextureFormat::bc1
            | TextureFormat::bc3
    )
}

fn rgb565_to_rgb8(c: u16) -> (u8, u8, u8) {
    let r5 = (c >> 11) & 0x1f;
    let g6 = (c >> 5) & 0x3f;
    let b5 = c & 0x1f;
    let r = ((u32::from(r5) * 255 + 15) / 31) as u8;
    let g = ((u32::from(g6) * 255 + 31) / 63) as u8;
    let b = ((u32::from(b5) * 255 + 15) / 31) as u8;
    (r, g, b)
}

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

/// Flips rows for tightly packed mip with `bytes_per_pixel` stride (e.g. 16 for RGBA16F).
pub fn flip_mip_rows(buf: &mut [u8], width: u32, height: u32, bytes_per_pixel: usize) {
    let w = width as usize;
    let h = height as usize;
    let row = w.saturating_mul(bytes_per_pixel);
    if row == 0 || h < 2 {
        return;
    }
    let mut tmp = vec![0u8; row];
    for y in 0..h / 2 {
        let a = y * row;
        let b = (h - 1 - y) * row;
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
    use super::*;

    #[test]
    fn argb32_swizzles_to_rgba() {
        let raw = vec![255u8, 1, 2, 3];
        let out = decode_mip_to_rgba8(TextureFormat::argb32, 1, 1, false, &raw).expect("ok");
        assert_eq!(out, vec![1, 2, 3, 255]);
    }

    #[test]
    fn bc1_decodes_red_1x1() {
        let raw = vec![0x00u8, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let out = decode_mip_to_rgba8(TextureFormat::bc1, 1, 1, false, &raw).expect("ok");
        assert!(out[0] >= 250 && out[1] < 5 && out[2] < 5 && out[3] == 255);
    }
}
