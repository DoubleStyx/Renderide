//! Partial mip0 upload when [`TextureUploadHint::has_region`](crate::shared::TextureUploadHint) is set (RGBA8 family, uncompressed).

use crate::shared::{SetTexture2DData, SetTexture2DFormat, TextureUploadHint};

use super::super::decode::needs_rgba8_decode_before_upload;
use super::super::layout::{host_format_is_compressed, mip_byte_len, mip_tight_bytes_per_texel};
use super::error::TextureUploadError;
use super::mip_write_common::{choose_mip_start_bias, copy_layout_for_mip, is_rgba8_family};
use super::write_mip_chain::Texture2dUploadContext;

/// Describes a sub-rectangle within a full mip for tight row-major extraction (uncompressed).
pub(super) struct MipSubrectCopy {
    /// Full mip width in texels.
    pub full_width: u32,
    /// Full mip height in texels.
    pub full_height: u32,
    /// Bytes per texel in the host mip slice.
    pub bpp: usize,
    /// Sub-rectangle min X in texels.
    pub x: u32,
    /// Sub-rectangle min Y in texels.
    pub y: u32,
    /// Sub-rectangle width in texels.
    pub w: u32,
    /// Sub-rectangle height in texels.
    pub h: u32,
}

/// Matches `TextureUploadHint.IsEmptyRegion` in `Renderite.Shared`.
pub(super) fn hint_region_is_empty(hint: &TextureUploadHint) -> bool {
    if hint.has_region == 0 {
        return false;
    }
    if hint.region_data.width != 0 {
        return hint.region_data.height == 0;
    }
    true
}

/// Packs a tight row-major buffer for `write_texture` from a rectangular sub-region of a full mip.
///
/// When `flip_y` is `true`, the source rows are read in bottom-to-top order so the produced
/// buffer matches the GPU bottom-up storage convention; the caller adjusts the GPU `origin_y`
/// to land at `full_height - r.y - r.h` (see [`try_write_texture2d_subregion`]).
pub(super) fn pack_subrect_tight(
    full: &[u8],
    r: &MipSubrectCopy,
    flip_y: bool,
) -> Result<Vec<u8>, TextureUploadError> {
    let row_stride = (r.full_width as usize)
        .checked_mul(r.bpp)
        .ok_or_else(|| TextureUploadError::from("row_stride overflow"))?;
    let row_len = (r.w as usize)
        .checked_mul(r.bpp)
        .ok_or_else(|| TextureUploadError::from("row_len overflow"))?;
    let total = row_len
        .checked_mul(r.h as usize)
        .ok_or_else(|| TextureUploadError::from("subrect total bytes overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(total).map_err(|e| e.to_string())?;
    for row in 0..r.h {
        let src_row = if flip_y { r.h - 1 - row } else { row };
        let y = r.y + src_row;
        if y >= r.full_height {
            return Err("subrect row out of bounds".into());
        }
        let row_start = (y as usize)
            .checked_mul(row_stride)
            .and_then(|b| b.checked_add((r.x as usize).checked_mul(r.bpp)?))
            .ok_or_else(|| TextureUploadError::from("row_start overflow"))?;
        let end = row_start
            .checked_add(row_len)
            .ok_or_else(|| TextureUploadError::from("row_end overflow"))?;
        if end > full.len() {
            return Err("subrect row extends past mip buffer".into());
        }
        out.extend_from_slice(&full[row_start..end]);
    }
    Ok(out)
}

/// Parameters for [`write_texture_subregion`] (partial [`wgpu::Queue::write_texture`]).
struct TextureWriteSubregion<'a> {
    queue: &'a wgpu::Queue,
    write_texture_submit_gate: &'a crate::gpu::WriteTextureSubmitGate,
    texture: &'a wgpu::Texture,
    mip_level: u32,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes: &'a [u8],
}

/// Writes a tight sub-rectangle of texels into `texture` at the given mip and origin.
fn write_texture_subregion(w: TextureWriteSubregion<'_>) -> Result<(), TextureUploadError> {
    let queue = w.queue;
    let write_texture_submit_gate = w.write_texture_submit_gate;
    let texture = w.texture;
    let mip_level = w.mip_level;
    let origin_x = w.origin_x;
    let origin_y = w.origin_y;
    let width = w.width;
    let height = w.height;
    let format = w.format;
    let bytes = w.bytes;
    let (bw, bh) = format.block_dimensions();
    let copy_width = if bw > 1 {
        width.div_ceil(bw) * bw
    } else {
        width
    };
    let copy_height = if bh > 1 {
        height.div_ceil(bh) * bh
    } else {
        height
    };
    let size = wgpu::Extent3d {
        width: copy_width,
        height: copy_height,
        depth_or_array_layers: 1,
    };
    let (layout, expected_len) = copy_layout_for_mip(format, width, height)?;
    if bytes.len() != expected_len {
        return Err(TextureUploadError::from(format!(
            "subregion mip data len {} != expected {} ({}x{} {:?})",
            bytes.len(),
            expected_len,
            width,
            height,
            format
        )));
    }

    // Gate against driver-thread `Queue::submit` to avoid the wgpu-core 29 ABBA.
    let _gate = write_texture_submit_gate.lock();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: origin_x,
                y: origin_y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        size,
    );
    Ok(())
}

/// Returns [`None`] when this path should defer to the full mip chain (criteria not met).
///
/// Checked: mip0-only descriptor, uncompressed host format, GPU RGBA8 family. `flip_y` is
/// supported here for the RGBA8 fast path; rows are reversed in [`pack_subrect_tight`].
fn subregion_fast_path_supported(
    device: &wgpu::Device,
    upload: &SetTexture2DData,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
) -> Option<()> {
    if upload.start_mip_level != 0 {
        return None;
    }
    if upload.mip_map_sizes.len() != 1 || upload.mip_starts.len() != 1 {
        return None;
    }
    if host_format_is_compressed(fmt.format) || needs_rgba8_decode_before_upload(device, fmt.format)
    {
        return None;
    }
    if !is_rgba8_family(wgpu_format) {
        return None;
    }
    Some(())
}

/// Resolves descriptor offset and mip0 host bytes for the subregion fast path (bias, starts, tight mip extent).
fn subregion_resolve_mip0_slice<'a>(
    fmt: &SetTexture2DFormat,
    upload: &SetTexture2DData,
    payload: &'a [u8],
) -> Result<(u32, u32, &'a [u8]), TextureUploadError> {
    let w = upload.mip_map_sizes[0].x.max(0) as u32;
    let h = upload.mip_map_sizes[0].y.max(0) as u32;
    if w == 0 || h == 0 {
        return Err("non-positive mip dimensions".into());
    }

    let (start_bias, _) = choose_mip_start_bias(fmt.format, upload, payload.len())?;

    let start_raw = upload.mip_starts[0];
    if start_raw < 0 {
        return Err("negative mip_starts".into());
    }
    let start_abs = start_raw as usize;
    if start_abs < start_bias {
        return Err(TextureUploadError::from(format!(
            "mip 0 start {} is before descriptor offset {}",
            start_abs, start_bias
        )));
    }
    let start = start_abs - start_bias;
    let host_len = mip_byte_len(fmt.format, w, h).ok_or_else(|| {
        TextureUploadError::from(format!("mip byte size unsupported for {:?}", fmt.format))
    })? as usize;
    let mip_src = payload
        .get(start..start + host_len)
        .ok_or_else(|| TextureUploadError::from("mip 0 slice out of range"))?;
    Ok((w, h, mip_src))
}

/// Interprets [`TextureUploadHint::region_data`] as a texel rectangle within a `w`×`h` mip.
fn subregion_rect_from_hint(
    hint: &TextureUploadHint,
    w: u32,
    h: u32,
) -> Result<(u32, u32, u32, u32), TextureUploadError> {
    let rx = hint.region_data.x.max(0) as u32;
    let ry = hint.region_data.y.max(0) as u32;
    let rw = hint.region_data.width.max(0) as u32;
    let rh = hint.region_data.height.max(0) as u32;
    if rw == 0 || rh == 0 {
        return Err("region width/height must be positive".into());
    }
    if rx.saturating_add(rw) > w || ry.saturating_add(rh) > h {
        return Err(TextureUploadError::from(format!(
            "region {rw}x{rh} at ({rx}, {ry}) out of bounds for mip {w}x{h}",
        )));
    }
    Ok((rx, ry, rw, rh))
}

/// Sub-rect upload for mip0 when the host sets [`TextureUploadHint::has_region`].
///
/// Returns [`None`] when the fast path does not apply (caller uses the full mip chain path).
pub(super) fn try_write_texture2d_subregion(
    ctx: &Texture2dUploadContext<'_>,
) -> Option<Result<u32, TextureUploadError>> {
    subregion_fast_path_supported(ctx.device, ctx.upload, ctx.fmt, ctx.wgpu_format)?;

    let want = ctx.upload.data.length.max(0) as usize;
    if ctx.raw.len() < want {
        return Some(Err(TextureUploadError::from(format!(
            "raw shorter than descriptor (need {want}, got {})",
            ctx.raw.len()
        ))));
    }
    let payload = &ctx.raw[..want];

    let tex_extent = ctx.texture.size();
    let w0 = ctx.upload.mip_map_sizes[0].x.max(0) as u32;
    let h0 = ctx.upload.mip_map_sizes[0].y.max(0) as u32;
    if tex_extent.width != w0 || tex_extent.height != h0 {
        return None;
    }

    let (w, h, mip_src) = match subregion_resolve_mip0_slice(ctx.fmt, ctx.upload, payload) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };

    let bpp = mip_tight_bytes_per_texel(mip_src.len(), w, h)?;
    if bpp != 4 {
        return None;
    }

    let (rx, ry, rw, rh) = match subregion_rect_from_hint(&ctx.upload.hint, w, h) {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
    };

    let packed = match pack_subrect_tight(
        mip_src,
        &MipSubrectCopy {
            full_width: w,
            full_height: h,
            bpp,
            x: rx,
            y: ry,
            w: rw,
            h: rh,
        },
        ctx.upload.flip_y,
    ) {
        Ok(p) => p,
        Err(e) => return Some(Err(e)),
    };

    // GPU storage is bottom-up (the engine's V-flip convention); when `flip_y` is set, the
    // host-space `(rx, ry)..(rx+rw, ry+rh)` rectangle maps to the GPU rectangle anchored at the
    // mirrored Y origin so the destination region corresponds to the same texels as the
    // (now-bottom-up) packed rows produced by [`pack_subrect_tight`].
    let origin_y = if ctx.upload.flip_y { h - ry - rh } else { ry };
    match write_texture_subregion(TextureWriteSubregion {
        queue: ctx.queue,
        write_texture_submit_gate: ctx.write_texture_submit_gate,
        texture: ctx.texture,
        mip_level: 0,
        origin_x: rx,
        origin_y,
        width: rw,
        height: rh,
        format: ctx.wgpu_format,
        bytes: &packed,
    }) {
        Ok(()) => Some(Ok(1)),
        Err(e) => Some(Err(e)),
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::TextureUploadHint;

    use super::{hint_region_is_empty, pack_subrect_tight, MipSubrectCopy};

    #[test]
    fn hint_region_empty_matches_shared_semantics() {
        let mut h = TextureUploadHint::default();
        assert!(!hint_region_is_empty(&h));
        h.has_region = 1;
        h.region_data.width = 0;
        h.region_data.height = 0;
        assert!(hint_region_is_empty(&h));
        h.region_data.width = 10;
        h.region_data.height = 0;
        assert!(hint_region_is_empty(&h));
        h.region_data.height = 10;
        assert!(!hint_region_is_empty(&h));
    }

    #[test]
    fn pack_subrect_tight_extracts_top_left() {
        let mut v = vec![0u8; 4 * 4 * 4];
        for y in 0..2 {
            for x in 0..2 {
                let i = (y * 4 + x) * 4;
                v[i..i + 4].fill(1);
            }
        }
        let out = pack_subrect_tight(
            &v,
            &MipSubrectCopy {
                full_width: 4,
                full_height: 4,
                bpp: 4,
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
            false,
        )
        .unwrap();
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&b| b == 1));
    }

    #[test]
    fn pack_subrect_tight_flips_rows_when_flip_y() {
        // 4×4 RGBA8 mip with row 0 = 0xAA, row 1 = 0xBB, row 2 = 0xCC, row 3 = 0xDD (per byte).
        let mut v = vec![0u8; 4 * 4 * 4];
        for y in 0..4 {
            let val = match y {
                0 => 0xAA,
                1 => 0xBB,
                2 => 0xCC,
                _ => 0xDD,
            };
            for byte in &mut v[y * 16..(y + 1) * 16] {
                *byte = val;
            }
        }
        // Pack rows 1..3 (inclusive of 1, exclusive of 3) flipped.
        let out = pack_subrect_tight(
            &v,
            &MipSubrectCopy {
                full_width: 4,
                full_height: 4,
                bpp: 4,
                x: 0,
                y: 1,
                w: 4,
                h: 2,
            },
            true,
        )
        .unwrap();
        assert_eq!(out.len(), 4 * 2 * 4);
        // Output row 0 should be source row (1 + (h-1-0)) = (1 + 1) = 2 → 0xCC.
        for byte in &out[0..16] {
            assert_eq!(*byte, 0xCC);
        }
        // Output row 1 should be source row (1 + (h-1-1)) = (1 + 0) = 1 → 0xBB.
        for byte in &out[16..32] {
            assert_eq!(*byte, 0xBB);
        }
    }
}
