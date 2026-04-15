//! Shared mip offset validation and [`wgpu::Queue::write_texture`] layout for full mip and subregion paths.

use crate::shared::SetTexture2DData;

use super::super::layout::{host_mip_payload_byte_offset, mip_byte_len};

/// Picks the descriptor offset bias that maximizes how many mips fit in the SHM payload.
pub(super) fn choose_mip_start_bias(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
) -> Result<(usize, usize), String> {
    let offset_bias = upload.data.offset.max(0) as usize;
    let candidates = if offset_bias > 0 {
        [0usize, offset_bias]
    } else {
        [0usize, 0usize]
    };
    let mut best_bias = 0usize;
    let mut best_prefix = 0usize;
    for bias in candidates {
        let prefix = valid_mip_prefix_len(format, upload, payload_len, bias)?;
        if prefix > best_prefix {
            best_prefix = prefix;
            best_bias = bias;
        }
    }
    if best_prefix == 0 {
        return Err(format!(
            "mip region exceeds shared memory descriptor (payload_len={}, descriptor_offset={})",
            payload_len, offset_bias
        ));
    }
    Ok((best_bias, best_prefix))
}

pub(super) fn valid_mip_prefix_len(
    format: crate::shared::TextureFormat,
    upload: &SetTexture2DData,
    payload_len: usize,
    bias: usize,
) -> Result<usize, String> {
    let mut count = 0usize;
    for (i, sz) in upload.mip_map_sizes.iter().enumerate() {
        if sz.x <= 0 || sz.y <= 0 {
            return Err("non-positive mip dimensions".into());
        }
        let w = sz.x as u32;
        let h = sz.y as u32;
        let host_len = mip_byte_len(format, w, h)
            .ok_or_else(|| format!("mip byte size unsupported for {:?}", format))?
            as usize;
        let start_raw = upload.mip_starts[i];
        if start_raw < 0 {
            return Err("negative mip_starts".into());
        }
        let start_abs = start_raw as usize;
        if start_abs < bias {
            break;
        }
        let start_rel = start_abs - bias;
        let start = match host_mip_payload_byte_offset(format, start_rel) {
            Some(b) => b,
            None => {
                return Err(format!(
                    "mip {i}: could not convert mip_starts offset to bytes for {:?}",
                    format
                ));
            }
        };
        if start
            .checked_add(host_len)
            .is_none_or(|end| end > payload_len)
        {
            break;
        }
        count += 1;
    }
    Ok(count)
}

pub(super) fn is_rgba8_family(gpu: wgpu::TextureFormat) -> bool {
    matches!(
        gpu,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
    )
}

pub(super) fn uncompressed_row_bytes(f: wgpu::TextureFormat) -> Result<usize, String> {
    let (bw, bh) = f.block_dimensions();
    if bw != 1 || bh != 1 {
        return Err("internal: expected uncompressed format".into());
    }
    let bsz = f
        .block_copy_size(None)
        .ok_or_else(|| format!("wgpu format {f:?} has no block size"))?;
    Ok(bsz as usize)
}

pub(super) fn write_one_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
    // For block-compressed formats wgpu requires the copy extent to be a multiple of the
    // block dimensions (the "physical" mip size).  The data produced by copy_layout_for_mip
    // already covers the padded block grid (via div_ceil), so only the Extent3d needs aligning.
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

/// Writes one mip level of a 3D texture (full `width`×`height`×`depth` volume).
#[allow(clippy::too_many_arguments)]
pub fn write_texture3d_volume_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    width: u32,
    height: u32,
    depth: u32,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
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
        depth_or_array_layers: depth,
    };
    let (layout, slice_len) = copy_layout_for_mip(format, width, height)?;
    let expected = slice_len
        .checked_mul(depth as usize)
        .ok_or_else(|| "3d mip expected bytes overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "3d mip data len {} != expected {} ({}x{}x{} {:?})",
            bytes.len(),
            expected,
            width,
            height,
            depth,
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

/// Writes one face × one mip of a cubemap (`D2` texture with six array layers).
#[allow(clippy::too_many_arguments)]
pub fn write_cubemap_face_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    face_layer: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> Result<(), String> {
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
        return Err(format!(
            "cubemap mip data len {} != expected {} ({}x{} {:?})",
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
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face_layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        size,
    );
    Ok(())
}

pub(super) fn copy_layout_for_mip(
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

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::{choose_mip_start_bias, valid_mip_prefix_len};
    use crate::shared::{SetTexture2DData, TextureFormat};

    #[test]
    fn relative_mip_starts_need_no_rebase() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, 64];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 0);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn absolute_mip_starts_rebase_to_descriptor_offset() {
        let mut upload = SetTexture2DData::default();
        upload.data.offset = 128;
        upload.data.length = 80;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![128, 192];

        let (bias, prefix) = choose_mip_start_bias(TextureFormat::RGBA32, &upload, 80).unwrap();
        assert_eq!(bias, 128);
        assert_eq!(prefix, 2);
    }

    #[test]
    fn valid_prefix_len_stops_when_later_mip_exceeds_payload() {
        let mut upload = SetTexture2DData::default();
        upload.data.length = 68;
        upload.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
        upload.mip_starts = vec![0, 64];

        let prefix = valid_mip_prefix_len(TextureFormat::RGBA32, &upload, 68, 0).unwrap();
        assert_eq!(prefix, 1);
    }
}
