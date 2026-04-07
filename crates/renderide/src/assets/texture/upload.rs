//! Applies [`SetTexture2DData`] into an existing [`wgpu::Texture`] using [`wgpu::Queue::write_texture`]
//! ([`wgpu::TexelCopyTextureInfo`] / [`wgpu::TexelCopyBufferLayout`]).
//!
//! The [`wgpu::TextureFormat`] must match the texture’s creation format (see [`resolve_texture2d_wgpu_format`]).

use crate::shared::{ColorProfile, SetTexture2DData, SetTexture2DFormat};

use super::decode::{decode_mip_to_rgba8, flip_mip_rows, needs_rgba8_decode_before_upload};
use super::format::pick_wgpu_storage_format;
use super::layout::{
    host_format_is_compressed, mip_byte_len, mip_dimensions_at_level, validate_mip_upload_layout,
};

/// Decides GPU storage format for a new 2D texture from host [`SetTexture2DFormat`].
///
/// Uses native compressed/uncompressed `wgpu` formats when supported; falls back to RGBA8 when
/// compression features are missing or the host layout needs swizzle ([`needs_rgba8_decode_before_upload`]).
pub fn resolve_texture2d_wgpu_format(
    device: &wgpu::Device,
    fmt: &SetTexture2DFormat,
) -> wgpu::TextureFormat {
    if needs_rgba8_decode_before_upload(fmt.format) {
        return rgba8_fallback_format(fmt.profile);
    }
    if let Some(f) = pick_wgpu_storage_format(device, fmt.format, fmt.profile) {
        return f;
    }
    rgba8_fallback_format(fmt.profile)
}

fn rgba8_fallback_format(profile: ColorProfile) -> wgpu::TextureFormat {
    match profile {
        ColorProfile::s_rgb | ColorProfile::s_rgb_alpha => wgpu::TextureFormat::Rgba8UnormSrgb,
        ColorProfile::linear => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Uploads mips from `raw` (exact [`SharedMemoryBufferDescriptor`] window) into `texture` using `wgpu_format`.
pub fn write_texture2d_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    fmt: &SetTexture2DFormat,
    wgpu_format: wgpu::TextureFormat,
    upload: &SetTexture2DData,
    raw: &[u8],
) -> Result<(), String> {
    if upload.hint.has_region != 0 {
        logger::trace!(
            "texture {}: TextureUploadHint.has_region ignored; full mips",
            upload.asset_id
        );
    }
    let want = upload.data.length.max(0) as usize;
    if raw.len() < want {
        return Err(format!(
            "raw shorter than descriptor (need {want}, got {})",
            raw.len()
        ));
    }
    let payload = &raw[..want];
    validate_mip_upload_layout(fmt.format, upload).map_err(|e| e.to_string())?;

    let start_base = upload.start_mip_level.max(0) as u32;
    let mipmap_count = fmt.mipmap_count.max(1) as u32;
    if start_base >= mipmap_count {
        return Err(format!(
            "start_mip_level {start_base} >= mipmap_count {mipmap_count}"
        ));
    }

    let flip = upload.flip_y;
    if flip && host_format_is_compressed(fmt.format) && !is_rgba8_family(wgpu_format) {
        logger::warn!(
            "texture {}: flip_y unsupported for compressed GPU texture {:?}; mips may look upside-down",
            upload.asset_id,
            wgpu_format
        );
    }

    let tex_extent = texture.size();
    let fmt_w = fmt.width.max(0) as u32;
    let fmt_h = fmt.height.max(0) as u32;
    if tex_extent.width != fmt_w || tex_extent.height != fmt_h {
        return Err(format!(
            "GPU texture {}x{} does not match SetTexture2DFormat {}x{} for asset {}",
            tex_extent.width, tex_extent.height, fmt_w, fmt_h, upload.asset_id
        ));
    }

    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        let w = sz.x.max(0) as u32;
        let h = sz.y.max(0) as u32;
        let mip_level = start_base + i as u32;
        if mip_level >= mipmap_count {
            return Err(format!(
                "upload mip {mip_level} exceeds texture mips {mipmap_count}"
            ));
        }

        let (gw, gh) = mip_dimensions_at_level(tex_extent.width, tex_extent.height, mip_level);
        if w != gw || h != gh {
            return Err(format!(
                "texture {} mip {mip_level}: upload says {w}x{h} but GPU mip is {gw}x{gh} (base {}x{} from format); fix host SetTexture2DFormat vs SetTexture2DData",
                upload.asset_id,
                tex_extent.width,
                tex_extent.height
            ));
        }

        let start = upload.mip_starts[i].max(0) as usize;
        let host_len = mip_byte_len(fmt.format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", fmt.format))?
            as usize;
        let mip_src = payload
            .get(start..start + host_len)
            .ok_or_else(|| format!("mip {i} slice out of range"))?;

        let pixels: std::borrow::Cow<'_, [u8]> = if is_rgba8_family(wgpu_format) {
            if needs_rgba8_decode_before_upload(fmt.format) || host_format_is_compressed(fmt.format)
            {
                std::borrow::Cow::Owned(
                    decode_mip_to_rgba8(fmt.format, w, h, flip, mip_src).ok_or_else(|| {
                        format!("RGBA decode failed for mip {i} ({:?})", fmt.format)
                    })?,
                )
            } else if flip {
                let mut v = mip_src.to_vec();
                flip_mip_rows(&mut v, w, h, 4);
                std::borrow::Cow::Owned(v)
            } else {
                std::borrow::Cow::Borrowed(mip_src)
            }
        } else {
            if needs_rgba8_decode_before_upload(fmt.format) {
                return Err(format!(
                    "host {:?} must use RGBA decode but GPU format is {:?}",
                    fmt.format, wgpu_format
                ));
            }
            if flip && !host_format_is_compressed(fmt.format) {
                let bpp = uncompressed_row_bytes(wgpu_format)?;
                let mut v = mip_src.to_vec();
                flip_mip_rows(&mut v, w, h, bpp);
                std::borrow::Cow::Owned(v)
            } else {
                if flip && host_format_is_compressed(fmt.format) {
                    logger::warn!(
                        "texture {} mip {i}: flip_y skipped for compressed {:?} GPU upload",
                        upload.asset_id,
                        wgpu_format
                    );
                }
                std::borrow::Cow::Borrowed(mip_src)
            }
        };

        write_one_mip(
            queue,
            texture,
            mip_level,
            w,
            h,
            wgpu_format,
            pixels.as_ref(),
        )?;
    }

    Ok(())
}

fn is_rgba8_family(gpu: wgpu::TextureFormat) -> bool {
    matches!(
        gpu,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    )
}

fn uncompressed_row_bytes(f: wgpu::TextureFormat) -> Result<usize, String> {
    let (bw, bh) = f.block_dimensions();
    if bw != 1 || bh != 1 {
        return Err("internal: expected uncompressed format".into());
    }
    let bsz = f
        .block_copy_size(None)
        .ok_or_else(|| format!("wgpu format {f:?} has no block size"))?;
    Ok(bsz as usize)
}

fn write_one_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let (layout, expected_len) = copy_layout_for_mip(format, width, height)?;
    if bytes.len() != expected_len {
        return Err(format!(
            "mip data len {} != expected {} ({}x{} {:?})",
            bytes.len(),
            expected_len,
            width,
            height,
            format
        ));
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        size,
    );
    Ok(())
}

fn copy_layout_for_mip(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<(wgpu::TexelCopyBufferLayout, usize), String> {
    let (bw, bh) = format.block_dimensions();
    let block_bytes = format
        .block_copy_size(None)
        .ok_or_else(|| format!("no block copy size for {:?}", format))?;
    if bw == 1 && bh == 1 {
        let bpp = block_bytes as usize;
        let bpr = bpp
            .checked_mul(width as usize)
            .ok_or("bytes_per_row overflow")?;
        let expected = bpr
            .checked_mul(height as usize)
            .ok_or("expected bytes overflow")?;
        let bpr_u32 = u32::try_from(bpr).map_err(|_| "bpr u32 overflow")?;
        return Ok((
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr_u32),
                rows_per_image: Some(height),
            },
            expected,
        ));
    }

    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let row_bytes_u = blocks_x
        .checked_mul(block_bytes)
        .ok_or("row bytes overflow")?;
    let expected_u = row_bytes_u
        .checked_mul(blocks_y)
        .ok_or("expected size overflow")?;
    let expected = expected_u as usize;
    Ok((
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes_u),
            rows_per_image: Some(blocks_y),
        },
        expected,
    ))
}
