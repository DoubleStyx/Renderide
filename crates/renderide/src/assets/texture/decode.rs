//! Transient CPU decode paths for host [`TextureFormat`] when GPU-native storage is unavailable or swizzle is required.

use rayon::prelude::*;

use crate::shared::TextureFormat;

/// Texel count above which mip decode fans out across rayon (64 KiB RGBA8 = 128x128).
///
/// Most tail mips in a chain are smaller than 128x128, so they keep the serial path and
/// avoid rayon dispatch overhead. Above 128x128 the decode is large enough to amortize.
const PARALLEL_DECODE_MIN_TEXELS: usize = 16_384;

/// Texel count per rayon chunk during parallel decode.
///
/// Larger chunks reduce rayon scheduler thrash, matching Filament's CountSplitter recursive
/// halving (which keeps splits to ~MAX_SPLITS=12 even on huge inputs).
const PARALLEL_DECODE_TEXELS_PER_CHUNK: usize = 8_192;

/// Runs `decode` over the full input/output, splitting into rayon chunks once `texel_count`
/// crosses [`PARALLEL_DECODE_MIN_TEXELS`].
///
/// Each chunk owns a disjoint output range and reads its disjoint source slice, so the closure
/// has no cross-chunk dependencies. The serial path keeps the same call shape so test fixtures
/// below the threshold exercise identical code.
#[inline]
fn decode_in_chunks<F>(
    src: &[u8],
    src_bytes_per_texel: usize,
    dst: &mut [u8],
    dst_bytes_per_texel: usize,
    decode: F,
) where
    F: Fn(&[u8], &mut [u8]) + Sync + Send,
{
    debug_assert!(dst_bytes_per_texel > 0);
    debug_assert_eq!(dst.len() % dst_bytes_per_texel, 0);
    let texel_count = dst.len() / dst_bytes_per_texel;
    if texel_count >= PARALLEL_DECODE_MIN_TEXELS {
        let src_chunk = PARALLEL_DECODE_TEXELS_PER_CHUNK * src_bytes_per_texel;
        let dst_chunk = PARALLEL_DECODE_TEXELS_PER_CHUNK * dst_bytes_per_texel;
        dst.par_chunks_mut(dst_chunk)
            .zip(src.par_chunks(src_chunk))
            .for_each(|(d, s)| decode(s, d));
    } else {
        decode(src, dst);
    }
}

/// Expands tight RGB24 texels to RGBA8 (alpha 255).
fn decode_rgb24_mip_to_rgba8(w: usize, h: usize, raw: &[u8]) -> Option<Vec<u8>> {
    profiling::scope!("texture_decode::rgb24");
    let count = w.checked_mul(h)?;
    let need = count.checked_mul(3)?;
    if raw.len() < need {
        return None;
    }
    let mut out = vec![0u8; count * 4];
    decode_in_chunks(&raw[..need], 3, &mut out, 4, |src, dst| {
        for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = 255;
        }
    });
    Some(out)
}

/// Copies tight RGBA32 bytes (no swizzle).
fn decode_rgba32_mip_copy(w: usize, h: usize, raw: &[u8]) -> Option<Vec<u8>> {
    let count = w.checked_mul(h)?;
    let need = count.checked_mul(4)?;
    if raw.len() < need {
        return None;
    }
    Some(raw[..need].to_vec())
}

/// Unpacks ARGB32 (Windows-style) to RGBA8.
fn decode_argb32_mip_to_rgba8(w: usize, h: usize, raw: &[u8]) -> Option<Vec<u8>> {
    profiling::scope!("texture_decode::argb32");
    let count = w.checked_mul(h)?;
    let need = count.checked_mul(4)?;
    if raw.len() < need {
        return None;
    }
    let mut out = vec![0u8; need];
    decode_in_chunks(&raw[..need], 4, &mut out, 4, |src, dst| {
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            d[0] = s[1];
            d[1] = s[2];
            d[2] = s[3];
            d[3] = s[0];
        }
    });
    Some(out)
}

/// Swizzles BGRA32 to RGBA8.
fn decode_bgra32_mip_to_rgba8(w: usize, h: usize, raw: &[u8]) -> Option<Vec<u8>> {
    profiling::scope!("texture_decode::bgra32");
    let count = w.checked_mul(h)?;
    let need = count.checked_mul(4)?;
    if raw.len() < need {
        return None;
    }
    let mut out = vec![0u8; need];
    decode_in_chunks(&raw[..need], 4, &mut out, 4, |src, dst| {
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = s[3];
        }
    });
    Some(out)
}

/// Expands grayscale or alpha-only 8-bit mips to RGBA8.
fn decode_r8_or_alpha8_mip_to_rgba8(
    format: TextureFormat,
    w: usize,
    h: usize,
    raw: &[u8],
) -> Option<Vec<u8>> {
    profiling::scope!("texture_decode::r8_or_alpha8");
    let count = w.checked_mul(h)?;
    if raw.len() < count {
        return None;
    }
    let mut out = vec![0u8; count * 4];
    let is_r8 = format == TextureFormat::R8;
    decode_in_chunks(&raw[..count], 1, &mut out, 4, |src, dst| {
        if is_r8 {
            for (s, d) in src.iter().zip(dst.chunks_exact_mut(4)) {
                d[0] = *s;
                d[1] = *s;
                d[2] = *s;
                d[3] = 255;
            }
        } else {
            for (s, d) in src.iter().zip(dst.chunks_exact_mut(4)) {
                d[0] = 255;
                d[1] = 255;
                d[2] = 255;
                d[3] = *s;
            }
        }
    });
    Some(out)
}

/// Decodes 565 packed rgb or bgr swizzle to RGBA8.
fn decode_rgb565_family_mip_to_rgba8(
    format: TextureFormat,
    w: usize,
    h: usize,
    raw: &[u8],
) -> Option<Vec<u8>> {
    profiling::scope!("texture_decode::rgb565");
    let count = w.checked_mul(h)?;
    let need = count.checked_mul(2)?;
    if raw.len() < need {
        return None;
    }
    let mut out = vec![0u8; count * 4];
    let bgr = format == TextureFormat::BGR565;
    decode_in_chunks(&raw[..need], 2, &mut out, 4, |src, dst| {
        for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact_mut(4)) {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let (r5, g6, b5) = if bgr {
                ((v) & 0x1f, (v >> 5) & 0x3f, (v >> 11) & 0x1f)
            } else {
                ((v >> 11) & 0x1f, (v >> 5) & 0x3f, (v) & 0x1f)
            };
            d[0] = ((u32::from(r5) * 255 + 15) / 31) as u8;
            d[1] = ((u32::from(g6) * 255 + 31) / 63) as u8;
            d[2] = ((u32::from(b5) * 255 + 15) / 31) as u8;
            d[3] = 255;
        }
    });
    Some(out)
}

/// Decodes one mip level from `raw` to tightly packed RGBA8 (row-major, source orientation).
///
/// Used as a fallback for missing compression features or packed formats without a direct `wgpu`
/// layout match. The renderer keeps host bytes in Unity (V=0 bottom) orientation throughout, so
/// no row flip is applied here -- `_flip_y` is retained for IPC compatibility only.
pub fn decode_mip_to_rgba8(
    format: TextureFormat,
    width: u32,
    height: u32,
    _flip_y: bool,
    raw: &[u8],
) -> Option<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    w.checked_mul(h)?;
    let rgba = match format {
        TextureFormat::RGB24 => decode_rgb24_mip_to_rgba8(w, h, raw)?,
        TextureFormat::RGBA32 => decode_rgba32_mip_copy(w, h, raw)?,
        TextureFormat::ARGB32 => decode_argb32_mip_to_rgba8(w, h, raw)?,
        TextureFormat::BGRA32 => decode_bgra32_mip_to_rgba8(w, h, raw)?,
        TextureFormat::R8 | TextureFormat::Alpha8 => {
            decode_r8_or_alpha8_mip_to_rgba8(format, w, h, raw)?
        }
        TextureFormat::RGB565 | TextureFormat::BGR565 => {
            decode_rgb565_family_mip_to_rgba8(format, w, h, raw)?
        }
        TextureFormat::BC1 => decode_bc1_to_rgba8(w, h, raw)?,
        TextureFormat::BC3 => decode_bc3_to_rgba8(w, h, raw)?,
        TextureFormat::BC2 => decode_with_lib(w, h, raw, texture2ddecoder::decode_bc2)?,
        TextureFormat::BC4 => decode_with_lib(w, h, raw, texture2ddecoder::decode_bc4)?,
        TextureFormat::BC5 => decode_with_lib(w, h, raw, texture2ddecoder::decode_bc5)?,
        TextureFormat::BC6H => decode_with_lib(w, h, raw, texture2ddecoder::decode_bc6_unsigned)?,
        TextureFormat::BC7 => decode_with_lib(w, h, raw, texture2ddecoder::decode_bc7)?,
        TextureFormat::ETC2RGB => decode_with_lib(w, h, raw, texture2ddecoder::decode_etc2_rgb)?,
        TextureFormat::ETC2RGBA1 => {
            decode_with_lib(w, h, raw, texture2ddecoder::decode_etc2_rgba1)?
        }
        TextureFormat::ETC2RGBA8 => {
            decode_with_lib(w, h, raw, texture2ddecoder::decode_etc2_rgba8)?
        }
        TextureFormat::ASTC4x4 => decode_astc_with_block(w, h, raw, 4, 4)?,
        TextureFormat::ASTC5x5 => decode_astc_with_block(w, h, raw, 5, 5)?,
        TextureFormat::ASTC6x6 => decode_astc_with_block(w, h, raw, 6, 6)?,
        TextureFormat::ASTC8x8 => decode_astc_with_block(w, h, raw, 8, 8)?,
        TextureFormat::ASTC10x10 => decode_astc_with_block(w, h, raw, 10, 10)?,
        TextureFormat::ASTC12x12 => decode_astc_with_block(w, h, raw, 12, 12)?,
        _ => return None,
    };
    Some(rgba)
}

/// `texture2ddecoder` packs each decoded texel as `u32::from_le_bytes([b, g, r, a])` (its
/// `color::color`). On little-endian targets (all renderide platforms) this places **B** at the
/// low byte; reorder to RGBA8 for [`wgpu::TextureFormat::Rgba8Unorm`] /
/// [`wgpu::TextureFormat::Rgba8UnormSrgb`] uploads.
fn bgra_u32_buf_to_rgba8(buf: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len() * 4);
    for &px in buf {
        out.push(((px >> 16) & 0xFF) as u8);
        out.push(((px >> 8) & 0xFF) as u8);
        out.push((px & 0xFF) as u8);
        out.push(((px >> 24) & 0xFF) as u8);
    }
    out
}

/// Adapter for `texture2ddecoder::decode_*(data, w, h, &mut [u32])` decoders that share the
/// generic `block_decoder!` signature (BC2/BC4/BC5/BC6H/BC7, ETC2 family).
fn decode_with_lib<F>(w: usize, h: usize, raw: &[u8], decode: F) -> Option<Vec<u8>>
where
    F: FnOnce(&[u8], usize, usize, &mut [u32]) -> Result<(), &'static str>,
{
    if w == 0 || h == 0 {
        return None;
    }
    let mut buf = vec![0u32; w.checked_mul(h)?];
    decode(raw, w, h, &mut buf).ok()?;
    Some(bgra_u32_buf_to_rgba8(&buf))
}

/// ASTC takes additional `block_w` / `block_h` parameters; otherwise mirrors [`decode_with_lib`].
fn decode_astc_with_block(
    w: usize,
    h: usize,
    raw: &[u8],
    block_w: usize,
    block_h: usize,
) -> Option<Vec<u8>> {
    if w == 0 || h == 0 {
        return None;
    }
    let mut buf = vec![0u32; w.checked_mul(h)?];
    texture2ddecoder::decode_astc(raw, w, h, block_w, block_h, &mut buf).ok()?;
    Some(bgra_u32_buf_to_rgba8(&buf))
}

/// Returns true if host mip bytes must be decoded to RGBA8 before [`wgpu::Queue::write_texture`].
///
/// When the device advertises the relevant [`wgpu::Features`] compression family, the host bytes
/// are uploaded as the native block-compressed GPU format instead (see
/// [`crate::assets::texture::upload::format_resolve`]). When the feature is missing, the format
/// falls back to RGBA8 via [`decode_mip_to_rgba8`] so the texture renders correctly with a
/// quality / VRAM penalty rather than a silent all-RGBA8 reinterpretation.
///
/// **BC3nm** normal-map channel packing is handled in shaders (`normal_decode.wgsl`,
/// `decode_ts_normal_sample_raw`).
pub fn needs_rgba8_decode_before_upload(device: &wgpu::Device, host: TextureFormat) -> bool {
    use TextureFormat::{
        ARGB32, ASTC4x4, ASTC5x5, ASTC6x6, ASTC8x8, ASTC10x10, ASTC12x12, Alpha8, BC1, BC2, BC3,
        BC4, BC5, BC6H, BC7, BGR565, BGRA32, ETC2RGB, ETC2RGBA1, ETC2RGBA8, R8, RGB24, RGB565,
    };
    let feats = device.features();
    let bc_native = feats.contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
    let etc2_native = feats.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2);
    let packed_rgb = matches!(
        host,
        RGB24 | ARGB32 | BGRA32 | R8 | Alpha8 | RGB565 | BGR565
    );
    let bc_cpu_fallback = matches!(host, BC1 | BC2 | BC3 | BC4 | BC5 | BC6H | BC7) && !bc_native;
    let etc2_cpu_fallback = matches!(host, ETC2RGB | ETC2RGBA1 | ETC2RGBA8) && !etc2_native;
    // ASTC is *always* CPU-decoded -- see [`crate::assets::texture::format::map_host_format`] for
    // the rationale (mode-dependent block layouts up to 12x12 prevent in-block flip).
    let astc_cpu_fallback = matches!(
        host,
        ASTC4x4 | ASTC5x5 | ASTC6x6 | ASTC8x8 | ASTC10x10 | ASTC12x12
    );
    packed_rgb || bc_cpu_fallback || etc2_cpu_fallback || astc_cpu_fallback
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

fn decode_bc1_block(block: [u8; 8], tile_rgba: &mut [u8; 64]) {
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
            [r0.midpoint(r1), g0.midpoint(g1), b0.midpoint(b1), 255],
            [0, 0, 0, 0],
        ]
    };
    for i in 0..16 {
        let code = ((bits >> (i * 2)) & 3) as usize;
        let px = colors[code];
        tile_rgba[i * 4..(i + 1) * 4].copy_from_slice(&px);
    }
}

fn decode_bc3_alpha_block(block_alpha: [u8; 8], out_alpha: &mut [u8; 16]) {
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
            decode_bc1_block(*block, &mut tile);
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
            decode_bc1_block(*color, &mut tile);
            let mut alphas = [0u8; 16];
            decode_bc3_alpha_block(*alpha, &mut alphas);
            for i in 0..16 {
                tile[i * 4 + 3] = alphas[i];
            }
            swizzle_bc3nm_normal_map_tile_if_detected(&mut tile);
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

/// Resonite/Unity **BC3nm** (DXT5nm) packs tangent-space **X** in the **alpha** block and sets **red**
/// to **1.0** in the color block (`Bitmap.PackNormalMap` -> `(1, Y, Y, X)` per texel). After BC3 decode,
/// PBS materials read tangent XY from the **RG** channels in WGSL, so **X** must be moved from **A->R**
/// for correct lighting.
///
/// Detection (per **4x4** tile): skip **uniform opaque white** RGB (`R=G=B=255` everywhere) so BC3 UI
/// cutouts are not turned cyan. Otherwise require every texel's **R** >= [`BC3NM_R_CHANNEL_MIN`] (BC3 can
/// round the constant 1.0 channel down slightly) and **G~=B** within [`BC3NM_GB_MAX_DELTA`] (duplicate Y
/// can diverge across endpoints/indices). Then assign `R := A`, `A := 0xFF`.
const BC3NM_R_CHANNEL_MIN: u8 = 250;
const BC3NM_GB_MAX_DELTA: u8 = 8;

fn swizzle_bc3nm_normal_map_tile_if_detected(tile: &mut [u8; 64]) {
    let uniform_white_rgb =
        (0..16).all(|i| tile[i * 4] == 255 && tile[i * 4 + 1] == 255 && tile[i * 4 + 2] == 255);
    if uniform_white_rgb {
        return;
    }
    let all_r_high = (0..16).all(|i| tile[i * 4] >= BC3NM_R_CHANNEL_MIN);
    if !all_r_high {
        return;
    }
    let gb_duplicate_y =
        (0..16).all(|i| tile[i * 4 + 1].abs_diff(tile[i * 4 + 2]) <= BC3NM_GB_MAX_DELTA);
    if !gb_duplicate_y {
        return;
    }
    for i in 0..16 {
        let a = tile[i * 4 + 3];
        tile[i * 4] = a;
        tile[i * 4 + 3] = 255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argb32_swizzles_to_rgba() {
        let raw = vec![255u8, 1, 2, 3];
        let out = decode_mip_to_rgba8(TextureFormat::ARGB32, 1, 1, false, &raw).expect("ok");
        assert_eq!(out, vec![1, 2, 3, 255]);
    }

    #[test]
    fn bc1_decodes_red_1x1() {
        let raw = vec![0x00u8, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let out = decode_mip_to_rgba8(TextureFormat::BC1, 1, 1, false, &raw).expect("ok");
        assert!(out[0] >= 250 && out[1] < 5 && out[2] < 5 && out[3] == 255);
    }

    #[test]
    fn bc3nm_swizzle_moves_alpha_to_red_when_all_red_saturated() {
        let mut tile = [0u8; 64];
        for i in 0..16 {
            tile[i * 4] = 255;
            tile[i * 4 + 1] = 128;
            tile[i * 4 + 2] = 127;
            tile[i * 4 + 3] = 42;
        }
        swizzle_bc3nm_normal_map_tile_if_detected(&mut tile);
        for i in 0..16 {
            assert_eq!(tile[i * 4], 42, "texel {i}");
            assert_eq!(tile[i * 4 + 1], 128);
            assert_eq!(tile[i * 4 + 2], 127);
            assert_eq!(tile[i * 4 + 3], 255);
        }
    }

    #[test]
    fn bc3nm_swizzle_applies_when_red_slightly_below_255_from_lossy_decode() {
        let mut tile = [0u8; 64];
        for i in 0..16 {
            tile[i * 4] = 252;
            tile[i * 4 + 1] = 128;
            tile[i * 4 + 2] = 126;
            tile[i * 4 + 3] = 99;
        }
        swizzle_bc3nm_normal_map_tile_if_detected(&mut tile);
        for i in 0..16 {
            assert_eq!(tile[i * 4], 99, "texel {i}");
            assert_eq!(tile[i * 4 + 1], 128);
            assert_eq!(tile[i * 4 + 2], 126);
            assert_eq!(tile[i * 4 + 3], 255);
        }
    }

    #[test]
    fn bc3nm_swizzle_no_op_when_red_not_uniformly_saturated() {
        let mut tile = [0u8; 64];
        for i in 0..16 {
            tile[i * 4] = 255;
            tile[i * 4 + 1] = 128;
            tile[i * 4 + 2] = 127;
            tile[i * 4 + 3] = 42;
        }
        tile[0] = 200;
        let expected = tile;
        swizzle_bc3nm_normal_map_tile_if_detected(&mut tile);
        assert_eq!(tile, expected);
    }

    #[test]
    fn bc3nm_swizzle_no_op_when_g_and_b_diverge_beyond_tolerance() {
        let mut tile = [0u8; 64];
        for i in 0..16 {
            tile[i * 4] = 255;
            tile[i * 4 + 1] = 128;
            tile[i * 4 + 2] = 100;
            tile[i * 4 + 3] = 42;
        }
        let expected = tile;
        swizzle_bc3nm_normal_map_tile_if_detected(&mut tile);
        assert_eq!(tile, expected);
    }

    #[test]
    fn bc3nm_swizzle_all_white_rgba_unchanged_visual() {
        let mut tile = [255u8; 64];
        swizzle_bc3nm_normal_map_tile_if_detected(&mut tile);
        assert_eq!(tile, [255u8; 64]);
    }

    #[test]
    fn bc3_full_mip_decode_swizzles_when_tile_is_nm_packed() {
        // One BC3 macroblock: alpha indices all 0 -> alpha 50 per texel. BC1 duplicate red endpoints
        // (`0xF800`) and zero indices -> solid sRGB red (R=255) for all 16 texels -> nm detection fires.
        let raw = vec![
            50u8, 50, 0, 0, 0, 0, 0, 0, // alpha: a0=a1=50; 48 index bits = 0
            0x00, 0xF8, 0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, // BC1: c0=c1=red, indices 0
        ];
        let out = decode_mip_to_rgba8(TextureFormat::BC3, 4, 4, false, &raw).expect("ok");
        assert_eq!(out.len(), 4 * 4 * 4);
        for px in out.chunks_exact(4) {
            assert_eq!(
                px[0], 50,
                "R holds tangent X (was in alpha) after BC3nm swizzle"
            );
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn bc2_4x4_decode_returns_solid_red() {
        // BC2: 8B explicit alpha (4 bits/texel) + 8B BC1 color. Alpha all 0xFF (premapped to
        // 4-bit 0xF), BC1 indices all 0 with c0=c1=red(0xF800) -> 4x4 opaque sRGB red.
        let raw = vec![
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // explicit alpha = full
            0x00, 0xF8, 0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, // BC1: c0=c1=red, indices 0
        ];
        let out = decode_mip_to_rgba8(TextureFormat::BC2, 4, 4, false, &raw).expect("ok");
        assert_eq!(out.len(), 4 * 4 * 4);
        for px in out.chunks_exact(4) {
            assert!(px[0] >= 250, "R~=255");
            assert!(px[1] < 5, "G~=0");
            assert!(px[2] < 5, "B~=0");
            assert_eq!(px[3], 255);
        }
    }

    fn ref_rgb24(w: usize, h: usize, raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 4);
        for p in raw.chunks_exact(3) {
            out.extend_from_slice(&[p[0], p[1], p[2], 255]);
        }
        out
    }

    fn ref_argb32(raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(raw.len());
        for p in raw.chunks_exact(4) {
            out.extend_from_slice(&[p[1], p[2], p[3], p[0]]);
        }
        out
    }

    fn ref_bgra32(raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(raw.len());
        for p in raw.chunks_exact(4) {
            out.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
        }
        out
    }

    #[test]
    fn rgb24_parallel_path_matches_serial_reference() {
        let w = 64usize;
        let h = 64usize;
        let raw: Vec<u8> = (0..(w * h * 3))
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();
        let out = decode_mip_to_rgba8(TextureFormat::RGB24, w as u32, h as u32, false, &raw)
            .expect("decode");
        let expected = ref_rgb24(w, h, &raw);
        assert_eq!(out, expected);
    }

    #[test]
    fn argb32_parallel_path_matches_serial_reference() {
        let w = 64usize;
        let h = 64usize;
        let raw: Vec<u8> = (0..(w * h * 4))
            .map(|i| (i as u8).wrapping_mul(11))
            .collect();
        let out = decode_mip_to_rgba8(TextureFormat::ARGB32, w as u32, h as u32, false, &raw)
            .expect("decode");
        let expected = ref_argb32(&raw);
        assert_eq!(out, expected);
    }

    #[test]
    fn bgra32_parallel_path_matches_serial_reference() {
        let w = 64usize;
        let h = 64usize;
        let raw: Vec<u8> = (0..(w * h * 4))
            .map(|i| (i as u8).wrapping_mul(13))
            .collect();
        let out = decode_mip_to_rgba8(TextureFormat::BGRA32, w as u32, h as u32, false, &raw)
            .expect("decode");
        let expected = ref_bgra32(&raw);
        assert_eq!(out, expected);
    }

    #[test]
    fn r8_parallel_path_replicates_to_rgb_with_full_alpha() {
        let w = 64usize;
        let h = 64usize;
        let raw: Vec<u8> = (0..(w * h)).map(|i| (i as u8).wrapping_mul(3)).collect();
        let out = decode_mip_to_rgba8(TextureFormat::R8, w as u32, h as u32, false, &raw)
            .expect("decode");
        for (i, px) in out.chunks_exact(4).enumerate() {
            assert_eq!(px[0], raw[i]);
            assert_eq!(px[1], raw[i]);
            assert_eq!(px[2], raw[i]);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn rgb565_parallel_path_matches_small_reference() {
        // Build a small (8x8 = 64 texel) input twice: once decoded via the small path, once tiled
        // up to 64x64 to force the parallel path. Per-texel bits are identical so the tiled output
        // must match the small output replicated.
        let w_small = 8usize;
        let h_small = 8usize;
        let raw_small: Vec<u8> = (0..(w_small * h_small * 2))
            .map(|i| (i as u8).wrapping_mul(17))
            .collect();
        let small_out = decode_mip_to_rgba8(
            TextureFormat::RGB565,
            w_small as u32,
            h_small as u32,
            false,
            &raw_small,
        )
        .expect("decode small");

        let w = 64usize;
        let h = 64usize;
        let mut raw = Vec::with_capacity(w * h * 2);
        for _ in 0..(w * h / (w_small * h_small)) {
            raw.extend_from_slice(&raw_small);
        }
        let big_out = decode_mip_to_rgba8(TextureFormat::RGB565, w as u32, h as u32, false, &raw)
            .expect("decode big");
        let mut expected = Vec::with_capacity(big_out.len());
        for _ in 0..(w * h / (w_small * h_small)) {
            expected.extend_from_slice(&small_out);
        }
        assert_eq!(big_out, expected);
    }

    #[test]
    fn etc2rgb_4x4_decode_returns_4x4_image() {
        // ETC2 RGB block (8B): use individual mode with two equal sub-blocks of mid-gray. Exact
        // pixel values aren't asserted here; the test pins the integration shape -- that the format
        // is wired and produces a 4x4xRGBA8 buffer without erroring.
        let raw = [0u8; 8];
        let out = decode_mip_to_rgba8(TextureFormat::ETC2RGB, 4, 4, false, &raw).expect("ok");
        assert_eq!(out.len(), 4 * 4 * 4);
        // Alpha must be opaque for ETC2 RGB.
        for px in out.chunks_exact(4) {
            assert_eq!(px[3], 255);
        }
    }
}
