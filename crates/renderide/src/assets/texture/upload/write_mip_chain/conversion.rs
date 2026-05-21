//! CPU-side conversion helpers for 2D mip-chain uploads.

use std::ops::Shr;

use rayon::prelude::*;
use wide::u32x4;

use super::super::TextureUploadError;
use super::super::mip_write_common::{
    MipUploadFormatCtx, MipUploadLabel, MipUploadPixels,
    mip_src_to_upload_pixels as shared_mip_src_to_upload_pixels,
};

/// Destination texels assigned to one downsample worker chunk.
const DOWNSAMPLE_PARALLEL_CHUNK_TEXELS: usize = 4_096;
/// Destination texel count above which downsampling may use Rayon.
const DOWNSAMPLE_PARALLEL_MIN_TEXELS: usize = DOWNSAMPLE_PARALLEL_CHUNK_TEXELS * 2;
/// Destination rows required before row-parallel downsampling can produce multiple chunks.
const DOWNSAMPLE_PARALLEL_MIN_ROWS: usize = 2;

#[inline]
fn downsample_parallel_rows_per_chunk(dst_w: usize) -> usize {
    DOWNSAMPLE_PARALLEL_CHUNK_TEXELS.div_ceil(dst_w.max(1))
}

#[inline]
fn should_parallelize_downsample(dst_w: usize, dst_h: usize) -> bool {
    dst_h >= DOWNSAMPLE_PARALLEL_MIN_ROWS
        && dst_w.saturating_mul(dst_h) >= DOWNSAMPLE_PARALLEL_MIN_TEXELS
}

/// Converts host mip bytes into a buffer suitable for [`write_one_mip`] (decode, optional row flip).
pub(super) fn mip_src_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    gw: u32,
    gh: u32,
    flip: bool,
    mip_src: &[u8],
    mip_index: usize,
) -> Result<MipUploadPixels, TextureUploadError> {
    shared_mip_src_to_upload_pixels(
        ctx,
        gw,
        gh,
        flip,
        mip_src,
        MipUploadLabel::texture2d(mip_index),
    )
}

/// Downsamples one RGBA8 mip into the next level using a simple box average.
pub(super) fn downsample_rgba8_box(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>, TextureUploadError> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err("zero-sized RGBA8 mip".into());
    }
    let expected = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| TextureUploadError::from("RGBA8 mip byte size overflow"))?;
    if src.len() != expected {
        return Err(TextureUploadError::from(format!(
            "RGBA8 mip len {} != expected {} ({}x{})",
            src.len(),
            expected,
            src_w,
            src_h
        )));
    }

    let dst_len = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| TextureUploadError::from("RGBA8 target mip byte size overflow"))?;
    let mut out = vec![0u8; dst_len];
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    if sw == dw * 2 && sh == dh * 2 {
        downsample_rgba8_box_2x2(src, &mut out, sw, dw);
    } else {
        downsample_rgba8_box_general(src, &mut out, sw, sh, dw, dh);
    }

    Ok(out)
}

/// Loads four contiguous RGBA8 bytes at `i` into a [`u32x4`] (one channel per lane, widened).
#[expect(
    clippy::inline_always,
    reason = "inner SIMD helper on the per-texel downsample hot path"
)]
#[inline(always)]
fn load_rgba8_lanes(src: &[u8], i: usize) -> u32x4 {
    u32x4::new([
        u32::from(src[i]),
        u32::from(src[i + 1]),
        u32::from(src[i + 2]),
        u32::from(src[i + 3]),
    ])
}

/// Stores the low byte of each [`u32x4`] lane into the four `dst` slots.
#[expect(
    clippy::inline_always,
    reason = "inner SIMD helper on the per-texel downsample hot path"
)]
#[inline(always)]
fn store_rgba8_lanes(dst: &mut [u8], i: usize, lanes: u32x4) {
    let arr = lanes.to_array();
    dst[i] = arr[0] as u8;
    dst[i + 1] = arr[1] as u8;
    dst[i + 2] = arr[2] as u8;
    dst[i + 3] = arr[3] as u8;
}

/// Common case: each output texel averages a 2x2 source neighborhood.
///
/// Medium and larger outputs fan rows out across rayon. The inner per-texel accumulator uses [`wide::u32x4`] so the
/// four channels widen and add through one SIMD register instead of four scalar operations.
fn downsample_rgba8_box_2x2(src: &[u8], dst: &mut [u8], sw: usize, dw: usize) {
    let two = u32x4::splat(2);
    let row_bytes = dw * 4;
    let dst_h = dst.len() / row_bytes;
    let process_row = |(dy, row): (usize, &mut [u8])| {
        let y0 = dy * 2;
        let row_y0 = y0 * sw * 4;
        let row_y1 = (y0 + 1) * sw * 4;
        for dx in 0..dw {
            let x0 = dx * 2;
            let i00 = row_y0 + x0 * 4;
            let i01 = i00 + 4;
            let i10 = row_y1 + x0 * 4;
            let i11 = i10 + 4;
            let s00 = load_rgba8_lanes(src, i00);
            let s01 = load_rgba8_lanes(src, i01);
            let s10 = load_rgba8_lanes(src, i10);
            let s11 = load_rgba8_lanes(src, i11);
            let avg = ((s00 + s01 + s10 + s11) + two).shr(2_i32);
            store_rgba8_lanes(row, dx * 4, avg);
        }
    };
    if should_parallelize_downsample(dw, dst_h) {
        dst.par_chunks_exact_mut(row_bytes)
            .with_min_len(downsample_parallel_rows_per_chunk(dw))
            .enumerate()
            .for_each(process_row);
    } else {
        dst.chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(process_row);
    }
}

/// General-case downsample for non-2:1 mip ratios (rare: NPOT bases, intentional drops).
fn downsample_rgba8_box_general(
    src: &[u8],
    dst: &mut [u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) {
    let row_bytes = dw * 4;
    let process_row = |(dy, row): (usize, &mut [u8])| {
        let y0 = dy * sh / dh;
        let y1 = ((dy + 1) * sh).div_ceil(dh).max(y0 + 1).min(sh);
        for dx in 0..dw {
            let x0 = dx * sw / dw;
            let x1 = ((dx + 1) * sw).div_ceil(dw).max(x0 + 1).min(sw);
            let mut sum = u32x4::ZERO;
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let si = (sy * sw + sx) * 4;
                    sum += load_rgba8_lanes(src, si);
                    count += 1;
                }
            }
            let arr = sum.to_array();
            let di = dx * 4;
            row[di] = ((arr[0] + count / 2) / count) as u8;
            row[di + 1] = ((arr[1] + count / 2) / count) as u8;
            row[di + 2] = ((arr[2] + count / 2) / count) as u8;
            row[di + 3] = ((arr[3] + count / 2) / count) as u8;
        }
    };
    if should_parallelize_downsample(dw, dh) {
        dst.par_chunks_exact_mut(row_bytes)
            .with_min_len(downsample_parallel_rows_per_chunk(dw))
            .enumerate()
            .for_each(process_row);
    } else {
        dst.chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(process_row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference scalar implementation matching the pre-SIMD code.
    fn reference_downsample(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        let mut out = vec![0u8; dw * dh * 4];
        for dy in 0..dh {
            let y0 = dy * sh / dh;
            let y1 = ((dy + 1) * sh).div_ceil(dh).max(y0 + 1).min(sh);
            for dx in 0..dw {
                let x0 = dx * sw / dw;
                let x1 = ((dx + 1) * sw).div_ceil(dw).max(x0 + 1).min(sw);
                let mut sum = [0u32; 4];
                let mut count = 0u32;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        let si = (sy * sw + sx) * 4;
                        sum[0] += u32::from(src[si]);
                        sum[1] += u32::from(src[si + 1]);
                        sum[2] += u32::from(src[si + 2]);
                        sum[3] += u32::from(src[si + 3]);
                        count += 1;
                    }
                }
                let di = (dy * dw + dx) * 4;
                out[di] = ((sum[0] + count / 2) / count) as u8;
                out[di + 1] = ((sum[1] + count / 2) / count) as u8;
                out[di + 2] = ((sum[2] + count / 2) / count) as u8;
                out[di + 3] = ((sum[3] + count / 2) / count) as u8;
            }
        }
        out
    }

    #[test]
    fn downsample_parallel_gate_starts_at_medium_mips() {
        assert!(!should_parallelize_downsample(127, 64));
        assert!(should_parallelize_downsample(128, 64));
    }

    #[test]
    fn downsample_rejects_zero_sized_mips() {
        let err = downsample_rgba8_box(&[0u8; 4], 1, 1, 0, 1).expect_err("zero dst width");

        assert!(err.to_string().contains("zero-sized RGBA8 mip"));
    }

    #[test]
    fn downsample_rejects_source_length_mismatch() {
        let err = downsample_rgba8_box(&[0u8; 15], 2, 2, 1, 1).expect_err("short source");

        assert!(err.to_string().contains("RGBA8 mip len 15 != expected 16"));
    }

    #[test]
    fn downsample_2x2_matches_reference() {
        let sw = 8usize;
        let sh = 8usize;
        let dw = 4usize;
        let dh = 4usize;
        let src: Vec<u8> = (0..(sw * sh * 4))
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();
        let actual = downsample_rgba8_box(&src, sw as u32, sh as u32, dw as u32, dh as u32)
            .expect("downsample");
        let expected = reference_downsample(&src, sw, sh, dw, dh);
        assert_eq!(actual, expected);
    }

    #[test]
    fn downsample_single_pixel_to_single_pixel_is_identity() {
        let src = [7u8, 9, 11, 13];
        let actual = downsample_rgba8_box(&src, 1, 1, 1, 1).expect("downsample");

        assert_eq!(actual, src);
    }

    #[test]
    fn downsample_general_3to2_matches_reference() {
        // NPOT downsample exercises the general path.
        let sw = 6usize;
        let sh = 4usize;
        let dw = 4usize;
        let dh = 3usize;
        let src: Vec<u8> = (0..(sw * sh * 4))
            .map(|i| (i as u8).wrapping_mul(11))
            .collect();
        let actual = downsample_rgba8_box(&src, sw as u32, sh as u32, dw as u32, dh as u32)
            .expect("downsample");
        let expected = reference_downsample(&src, sw, sh, dw, dh);
        assert_eq!(actual, expected);
    }

    #[test]
    fn downsample_general_3x3_to_1x1_averages_all_source_pixels() {
        let mut src = Vec::new();
        for i in 0..9u8 {
            src.extend_from_slice(&[i, i.saturating_mul(2), 10, 255]);
        }

        let actual = downsample_rgba8_box(&src, 3, 3, 1, 1).expect("downsample");

        assert_eq!(actual, vec![4, 8, 10, 255]);
    }

    #[test]
    fn load_and_store_rgba8_lanes_roundtrip_channel_order() {
        let src = [1u8, 2, 3, 4, 9, 9, 9, 9];
        let lanes = load_rgba8_lanes(&src, 0);
        let mut dst = [0u8; 4];

        store_rgba8_lanes(&mut dst, 0, lanes);

        assert_eq!(dst, [1, 2, 3, 4]);
    }

    #[test]
    fn downsample_solid_color_preserves_color() {
        let sw = 16usize;
        let sh = 16usize;
        let src = [200u8, 100, 50, 255].repeat(sw * sh);
        let actual = downsample_rgba8_box(&src, sw as u32, sh as u32, 8, 8).expect("downsample");
        for px in actual.chunks_exact(4) {
            assert_eq!(px, [200, 100, 50, 255]);
        }
    }
}
