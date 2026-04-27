//! CPU-side conversion helpers for 2D mip-chain uploads.

use rayon::prelude::*;

use super::super::super::decode::{decode_mip_to_rgba8, flip_mip_rows};
use super::super::super::layout::{
    flip_compressed_mip_block_rows_y, host_format_is_compressed, mip_byte_len,
    mip_tight_bytes_per_texel,
};
use super::super::mip_write_common::{
    is_rgba8_family, mip_ctx_uses_storage_v_inversion, uncompressed_row_bytes, MipUploadFormatCtx,
    MipUploadPixels,
};
use super::super::TextureUploadError;

/// Converts a compressed mip that requested `flip_y` into bytes or a storage-orientation hint.
pub(super) fn compressed_mip_src_to_upload_pixels(
    ctx: MipUploadFormatCtx,
    gw: u32,
    gh: u32,
    mip_src: &[u8],
    mip_index: usize,
) -> Result<Vec<u8>, TextureUploadError> {
    let MipUploadFormatCtx {
        asset_id,
        fmt_format,
        ..
    } = ctx;
    let expected_len = mip_byte_len(fmt_format, gw, gh).ok_or_else(|| {
        TextureUploadError::from(format!(
            "texture {asset_id} mip {}: mip byte size unknown for {:?}",
            mip_index, fmt_format
        ))
    })? as usize;
    if mip_src.len() != expected_len {
        return Err(TextureUploadError::from(format!(
            "texture {asset_id} mip {}: mip len {} != expected {} for {:?}",
            mip_index,
            mip_src.len(),
            expected_len,
            fmt_format
        )));
    }
    if let Some(v) = flip_compressed_mip_block_rows_y(fmt_format, gw, gh, mip_src) {
        return Ok(v);
    }
    if mip_ctx_uses_storage_v_inversion(ctx, true) {
        return Ok(mip_src.to_vec());
    }
    Err(TextureUploadError::from(format!(
        "texture {asset_id} mip {mip_index}: flip_y unsupported for compressed {:?}; reject to avoid uploading inverted data under the engine's V-flip sampling convention",
        fmt_format
    )))
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
    let MipUploadFormatCtx {
        asset_id,
        fmt_format,
        wgpu_format,
        needs_rgba8_decode,
    } = ctx;
    if is_rgba8_family(wgpu_format) {
        if needs_rgba8_decode || host_format_is_compressed(fmt_format) {
            decode_mip_to_rgba8(fmt_format, gw, gh, flip, mip_src).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "RGBA decode failed for mip {} ({:?})",
                    mip_index, fmt_format
                ))
            })
        } else if flip {
            let mut v = mip_src.to_vec();
            let bpp = mip_tight_bytes_per_texel(v.len(), gw, gh).ok_or_else(|| {
                TextureUploadError::from(format!(
                    "mip {}: RGBA8 upload len {} not divisible by {}×{} texels",
                    mip_index,
                    v.len(),
                    gw,
                    gh
                ))
            })?;
            if bpp != 4 {
                return Err(TextureUploadError::from(format!(
                    "mip {}: RGBA8 family expects 4 bytes per texel, got {bpp}",
                    mip_index
                )));
            }
            flip_mip_rows(&mut v, gw, gh, bpp);
            Ok(v)
        } else {
            Ok(mip_src.to_vec())
        }
    } else if needs_rgba8_decode {
        Err(TextureUploadError::from(format!(
            "host {:?} must use RGBA decode but GPU format is {:?}",
            fmt_format, wgpu_format
        )))
    } else if flip && !host_format_is_compressed(fmt_format) {
        let mut v = mip_src.to_vec();
        let bpp_host = mip_tight_bytes_per_texel(v.len(), gw, gh).ok_or_else(|| {
            TextureUploadError::from(format!(
                "mip {}: len {} not divisible by {}×{} texels (cannot infer row stride for flip_y)",
                mip_index,
                v.len(),
                gw,
                gh
            ))
        })?;
        if let Ok(bpp_gpu) = uncompressed_row_bytes(wgpu_format) {
            if bpp_host != bpp_gpu {
                logger::warn!(
                    "texture {} mip {}: host texel stride {} B != GPU {:?} stride {} B; flip_y uses host packing",
                    asset_id,
                    mip_index,
                    bpp_host,
                    wgpu_format,
                    bpp_gpu
                );
            }
        }
        flip_mip_rows(&mut v, gw, gh, bpp_host);
        Ok(v)
    } else if flip && host_format_is_compressed(fmt_format) {
        compressed_mip_src_to_upload_pixels(ctx, gw, gh, mip_src, mip_index)
    } else {
        Ok(mip_src.to_vec())
    }
    .map(|bytes| {
        if mip_ctx_uses_storage_v_inversion(ctx, flip) {
            MipUploadPixels::storage_v_inverted(bytes)
        } else {
            MipUploadPixels::normal(bytes)
        }
    })
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

    out.par_chunks_mut(dw * 4)
        .enumerate()
        .for_each(|(dy, row)| {
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
                let di = dx * 4;
                row[di] = ((sum[0] + count / 2) / count) as u8;
                row[di + 1] = ((sum[1] + count / 2) / count) as u8;
                row[di + 2] = ((sum[2] + count / 2) / count) as u8;
                row[di + 3] = ((sum[3] + count / 2) / count) as u8;
            }
        });

    Ok(out)
}
