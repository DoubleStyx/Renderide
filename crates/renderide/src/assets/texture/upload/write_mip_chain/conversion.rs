//! CPU-side conversion helpers for 2D mip-chain uploads.

use rayon::prelude::*;

use super::super::mip_write_common::{
    mip_src_to_upload_pixels as shared_mip_src_to_upload_pixels, MipUploadFormatCtx,
    MipUploadLabel, MipUploadPixels,
};
use super::super::TextureUploadError;

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
