//! Partial mip0 upload when [`TextureUploadHint::has_region`](crate::shared::TextureUploadHint) is set (RGBA8 family, uncompressed).

use crate::shared::{SetTexture2DData, SetTexture2DFormat, TextureUploadHint};

use super::super::decode::needs_rgba8_decode_before_upload;
use super::super::layout::{host_format_is_compressed, mip_byte_len, mip_tight_bytes_per_texel};
use super::error::TextureUploadError;
use super::mip_write_common::{
    TextureRegionWrite, choose_mip_start_bias, is_rgba8_family, write_texture_region,
};
use super::write_mip_chain::Texture2dUploadInputs;

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

/// Returns whether the host texture-upload hint encodes an empty region.
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
/// Rows are copied in source order. GPU storage matches host (Unity V=0 bottom) orientation, so
/// no row flip is performed regardless of the `_flip_y` hint.
pub(super) fn pack_subrect_tight(
    full: &[u8],
    r: &MipSubrectCopy,
    _flip_y: bool,
) -> Result<Vec<u8>, TextureUploadError> {
    profiling::scope!("asset::texture_pack_subregion");
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
        let y = r.y + row;
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
    /// Queue used for the texel copy.
    queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`].
    gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    queue_access_mode: crate::gpu::GpuQueueAccessMode,
    /// Destination texture.
    texture: &'a wgpu::Texture,
    /// Mip level index.
    mip_level: u32,
    /// Destination X origin in texels.
    origin_x: u32,
    /// Destination Y origin in texels.
    origin_y: u32,
    /// Region width in texels.
    width: u32,
    /// Region height in texels.
    height: u32,
    /// Texel format.
    format: wgpu::TextureFormat,
    /// Tightly packed region bytes.
    bytes: &'a [u8],
}

/// Writes a tight sub-rectangle of texels into `texture` at the given mip and origin.
fn write_texture_subregion(w: TextureWriteSubregion<'_>) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_subregion");
    write_texture_region(TextureRegionWrite {
        queue: w.queue,
        gpu_queue_access_gate: w.gpu_queue_access_gate,
        queue_access_mode: w.queue_access_mode,
        destination: wgpu::TexelCopyTextureInfo {
            texture: w.texture,
            mip_level: w.mip_level,
            origin: wgpu::Origin3d {
                x: w.origin_x,
                y: w.origin_y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        width: w.width,
        height: w.height,
        depth_or_array_layers: 1,
        format: w.format,
        bytes: w.bytes,
        label: "subregion mip",
    })
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
            "mip 0 start {start_abs} is before descriptor offset {start_bias}"
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

/// Interprets [`TextureUploadHint::region_data`] as a texel rectangle within a `w` x `h` mip.
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
    inputs: &Texture2dUploadInputs<'_>,
) -> Option<Result<u32, TextureUploadError>> {
    profiling::scope!("asset::texture_try_subregion");
    let queue = &inputs.queue;
    let target = &inputs.target;
    let payload = &inputs.payload;
    let upload = payload.upload;
    subregion_fast_path_supported(queue.device, upload, target.fmt, target.wgpu_format)?;

    let want = upload.data.length.max(0) as usize;
    if payload.raw.len() < want {
        return Some(Err(TextureUploadError::from(format!(
            "raw shorter than descriptor (need {want}, got {})",
            payload.raw.len()
        ))));
    }
    let raw = &payload.raw[..want];

    let tex_extent = target.texture.size();
    let w0 = upload.mip_map_sizes[0].x.max(0) as u32;
    let h0 = upload.mip_map_sizes[0].y.max(0) as u32;
    if tex_extent.width != w0 || tex_extent.height != h0 {
        return None;
    }

    let (w, h, mip_src) = match subregion_resolve_mip0_slice(target.fmt, upload, raw) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };

    let bpp = mip_tight_bytes_per_texel(mip_src.len(), w, h)?;
    if bpp != 4 {
        return None;
    }

    let (rx, ry, rw, rh) = match subregion_rect_from_hint(&upload.hint, w, h) {
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
        upload.flip_y,
    ) {
        Ok(p) => p,
        Err(e) => return Some(Err(e)),
    };

    match write_texture_subregion(TextureWriteSubregion {
        queue: queue.queue,
        gpu_queue_access_gate: queue.gpu_queue_access_gate,
        queue_access_mode: queue.queue_access_mode,
        texture: target.texture,
        mip_level: 0,
        origin_x: rx,
        origin_y: ry,
        width: rw,
        height: rh,
        format: target.wgpu_format,
        bytes: &packed,
    }) {
        Ok(()) => Some(Ok(1)),
        Err(e) => Some(Err(e)),
    }
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use crate::shared::{SetTexture2DData, SetTexture2DFormat, TextureFormat, TextureUploadHint};

    use super::{
        MipSubrectCopy, hint_region_is_empty, pack_subrect_tight, subregion_rect_from_hint,
        subregion_resolve_mip0_slice,
    };

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
        h.region_data.width = 0;
        h.region_data.height = 10;
        assert!(hint_region_is_empty(&h));
        h.region_data.height = 10;
        h.region_data.width = 10;
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
    fn pack_subrect_tight_extracts_offset_rectangle() {
        let mut v = vec![0u8; 4 * 3 * 4];
        for y in 0..3 {
            for x in 0..4 {
                let i = (y * 4 + x) * 4;
                v[i..i + 4].copy_from_slice(&[x as u8, y as u8, 0, 255]);
            }
        }

        let out = pack_subrect_tight(
            &v,
            &MipSubrectCopy {
                full_width: 4,
                full_height: 3,
                bpp: 4,
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
            false,
        )
        .expect("pack subrect");

        assert_eq!(
            out,
            vec![1, 1, 0, 255, 2, 1, 0, 255, 1, 2, 0, 255, 2, 2, 0, 255]
        );
    }

    #[test]
    fn pack_subrect_tight_rejects_rows_outside_source_extent() {
        let err = pack_subrect_tight(
            &[0u8; 4 * 4],
            &MipSubrectCopy {
                full_width: 2,
                full_height: 1,
                bpp: 4,
                x: 0,
                y: 1,
                w: 1,
                h: 1,
            },
            false,
        )
        .expect_err("row outside extent");

        assert!(err.to_string().contains("subrect row out of bounds"));
    }

    #[test]
    fn pack_subrect_tight_rejects_rows_past_buffer_len() {
        let err = pack_subrect_tight(
            &[0u8; 7],
            &MipSubrectCopy {
                full_width: 2,
                full_height: 1,
                bpp: 4,
                x: 0,
                y: 0,
                w: 2,
                h: 1,
            },
            false,
        )
        .expect_err("short source buffer");

        assert!(
            err.to_string()
                .contains("subrect row extends past mip buffer")
        );
    }

    #[test]
    fn pack_subrect_tight_keeps_source_row_order_regardless_of_flip_y() {
        // 4x4 RGBA8 mip with row 0 = 0xAA, row 1 = 0xBB, row 2 = 0xCC, row 3 = 0xDD (per byte).
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
        // Pack rows 1..3 (inclusive of 1, exclusive of 3); GPU storage uses Unity (V=0 bottom)
        // orientation, so the packed buffer matches host row order regardless of `flip_y`.
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
        // Output row 0 = source row 1 = 0xBB.
        for byte in &out[0..16] {
            assert_eq!(*byte, 0xBB);
        }
        // Output row 1 = source row 2 = 0xCC.
        for byte in &out[16..32] {
            assert_eq!(*byte, 0xCC);
        }
    }

    #[test]
    fn subregion_rect_from_hint_clamps_negative_origin_and_rejects_empty_size() {
        let mut hint = TextureUploadHint::default();
        hint.region_data.x = -7;
        hint.region_data.y = -3;
        hint.region_data.width = 2;
        hint.region_data.height = 3;
        assert_eq!(
            subregion_rect_from_hint(&hint, 4, 4).expect("clamped rect"),
            (0, 0, 2, 3)
        );

        hint.region_data.width = 0;
        let err = subregion_rect_from_hint(&hint, 4, 4).expect_err("empty rect");
        assert!(err.to_string().contains("region width/height"));
    }

    #[test]
    fn subregion_rect_from_hint_rejects_out_of_bounds_regions() {
        let mut hint = TextureUploadHint::default();
        hint.region_data.x = 3;
        hint.region_data.y = 0;
        hint.region_data.width = 2;
        hint.region_data.height = 1;

        let err = subregion_rect_from_hint(&hint, 4, 4).expect_err("out of bounds");

        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn subregion_resolve_mip0_slice_supports_descriptor_offset_rebase() {
        let fmt = SetTexture2DFormat {
            format: TextureFormat::RGBA32,
            ..Default::default()
        };
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 16;
        upload.mip_map_sizes = vec![IVec2::new(2, 2)];
        upload.mip_starts = vec![128];
        let payload: Vec<u8> = (0u8..16).collect();

        let (w, h, mip) = subregion_resolve_mip0_slice(&fmt, &upload, &payload).expect("mip0");

        assert_eq!((w, h), (2, 2));
        assert_eq!(mip, payload.as_slice());
    }

    #[test]
    fn subregion_resolve_mip0_slice_rejects_bad_dimensions_and_negative_start() {
        let fmt = SetTexture2DFormat {
            format: TextureFormat::RGBA32,
            ..Default::default()
        };
        let mut upload = SetTexture2DData::default();
        upload.data.length = 16;
        upload.mip_map_sizes = vec![IVec2::new(0, 2)];
        upload.mip_starts = vec![0];
        let err = subregion_resolve_mip0_slice(&fmt, &upload, &[0u8; 16])
            .expect_err("non-positive dimensions");
        assert!(err.to_string().contains("non-positive mip dimensions"));

        upload.mip_map_sizes = vec![IVec2::new(2, 2)];
        upload.mip_starts = vec![-1];
        let err =
            subregion_resolve_mip0_slice(&fmt, &upload, &[0u8; 16]).expect_err("negative start");
        assert!(
            err.to_string()
                .contains("mip region exceeds shared memory descriptor")
        );
    }
}
