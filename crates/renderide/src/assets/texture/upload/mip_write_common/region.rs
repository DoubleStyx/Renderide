//! Generic [`wgpu::Queue::write_texture`] region validation and submission.

use super::TextureUploadError;
use crate::gpu::GpuQueueAccessMode;

/// Descriptor for a generic texture region write.
pub(in crate::assets::texture::upload) struct TextureRegionWrite<'a> {
    /// Queue used for the texel copy.
    pub queue: &'a wgpu::Queue,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`].
    pub gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    /// Queue-gate acquisition policy for this write.
    pub queue_access_mode: GpuQueueAccessMode,
    /// Destination texture subresource.
    pub destination: wgpu::TexelCopyTextureInfo<'a>,
    /// Logical width in texels.
    pub width: u32,
    /// Logical height in texels.
    pub height: u32,
    /// Number of array layers or 3D depth slices to write.
    pub depth_or_array_layers: u32,
    /// Texel format.
    pub format: wgpu::TextureFormat,
    /// Tightly packed bytes.
    pub bytes: &'a [u8],
    /// Diagnostic label used in length mismatch errors.
    pub label: &'static str,
}

/// Validated region write data ready for queue submission.
struct PreparedTextureRegionWrite<'a> {
    queue: &'a wgpu::Queue,
    gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
    queue_access_mode: GpuQueueAccessMode,
    destination: wgpu::TexelCopyTextureInfo<'a>,
    bytes: &'a [u8],
    layout: wgpu::TexelCopyBufferLayout,
    size: wgpu::Extent3d,
}

/// Physical copy extent required by wgpu for a logical mip size.
pub(in crate::assets::texture::upload) fn copy_extent_for_mip(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
) -> wgpu::Extent3d {
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
    wgpu::Extent3d {
        width: copy_width,
        height: copy_height,
        depth_or_array_layers,
    }
}

/// Writes a texture subresource after shared layout, extent, and length validation.
pub(in crate::assets::texture::upload) fn write_texture_region(
    write: TextureRegionWrite<'_>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_region");
    let prepared = prepare_texture_region_write(write)?;
    let gpu_queue_access_gate = prepared.gpu_queue_access_gate;
    let queue_access_mode = prepared.queue_access_mode;
    let Some(_gate) = gpu_queue_access_gate.lock_for(queue_access_mode) else {
        return Err(TextureUploadError::QueueAccessBusy);
    };
    submit_prepared_texture_region_write(prepared);
    Ok(())
}

/// Writes a texture subresource while the caller already holds the queue gate.
pub(in crate::assets::texture::upload) fn write_texture_region_with_gate(
    write: TextureRegionWrite<'_>,
    _gate: &parking_lot::MutexGuard<'_, ()>,
) -> Result<(), TextureUploadError> {
    profiling::scope!("asset::texture_write_region_locked");
    let prepared = prepare_texture_region_write(write)?;
    submit_prepared_texture_region_write(prepared);
    Ok(())
}

fn prepare_texture_region_write(
    write: TextureRegionWrite<'_>,
) -> Result<PreparedTextureRegionWrite<'_>, TextureUploadError> {
    let size = copy_extent_for_mip(
        write.format,
        write.width,
        write.height,
        write.depth_or_array_layers,
    );
    let (layout, slice_len) = copy_layout_for_mip(write.format, write.width, write.height)?;
    let expected = slice_len
        .checked_mul(write.depth_or_array_layers as usize)
        .ok_or_else(|| {
            TextureUploadError::from(format!("{} expected bytes overflow", write.label))
        })?;
    if write.bytes.len() != expected {
        return Err(TextureUploadError::from(format!(
            "{} data len {} != expected {} ({}x{}x{} {:?})",
            write.label,
            write.bytes.len(),
            expected,
            write.width,
            write.height,
            write.depth_or_array_layers,
            write.format
        )));
    }

    Ok(PreparedTextureRegionWrite {
        queue: write.queue,
        gpu_queue_access_gate: write.gpu_queue_access_gate,
        queue_access_mode: write.queue_access_mode,
        destination: write.destination,
        bytes: write.bytes,
        layout,
        size,
    })
}

fn submit_prepared_texture_region_write(write: PreparedTextureRegionWrite<'_>) {
    write
        .queue
        .write_texture(write.destination, write.bytes, write.layout, write.size);
}

/// Builds a tight-copy layout and per-layer byte length for one mip.
pub(in crate::assets::texture::upload) fn copy_layout_for_mip(
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> Result<(wgpu::TexelCopyBufferLayout, usize), TextureUploadError> {
    let (bw, bh) = format.block_dimensions();
    let block_bytes = format
        .block_copy_size(None)
        .ok_or_else(|| TextureUploadError::from(format!("no block copy size for {format:?}")))?;
    if bw == 1 && bh == 1 {
        let bpp = block_bytes as usize;
        let bpr = bpp
            .checked_mul(width as usize)
            .ok_or_else(|| TextureUploadError::from("bytes_per_row overflow"))?;
        let expected = bpr
            .checked_mul(height as usize)
            .ok_or_else(|| TextureUploadError::from("expected bytes overflow"))?;
        #[expect(
            clippy::map_err_ignore,
            reason = "TryFromIntError adds no detail beyond the overflow label"
        )]
        let bpr_u32 =
            u32::try_from(bpr).map_err(|_| TextureUploadError::from("bpr u32 overflow"))?;
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
        .ok_or_else(|| TextureUploadError::from("row bytes overflow"))?;
    let expected_u = row_bytes_u
        .checked_mul(blocks_y)
        .ok_or_else(|| TextureUploadError::from("expected size overflow"))?;
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
