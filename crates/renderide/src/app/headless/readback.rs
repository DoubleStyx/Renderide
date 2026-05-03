//! Headless color-target readback and atomic PNG writing.

use std::path::Path;
use std::time::Duration;

use crate::gpu::GpuContext;

/// Copies the headless primary color texture to CPU memory and writes a PNG atomically.
pub(super) fn readback_and_write_png_atomically(
    gpu: &GpuContext,
    output_path: &Path,
) -> Result<(), HeadlessReadbackError> {
    let color_texture = gpu
        .headless_color_texture()
        .ok_or(HeadlessReadbackError::NoOffscreenTexture)?;
    let layout = compute_readback_layout(color_texture.size(), gpu.limits().max_buffer_size())?;
    let readback = create_readback_buffer(gpu, &layout);

    submit_texture_to_buffer_copy(gpu, color_texture, &layout, &readback);
    let slice = readback.slice(..);
    await_buffer_map(slice, gpu.device())?;
    let tight = {
        let view = slice.get_mapped_range();
        copy_padded_rows_to_tight(&view, &layout)
    };
    readback.unmap();

    write_png_atomically(&tight, layout.width, layout.height, output_path)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReadbackLayout {
    width: u32,
    height: u32,
    bytes_per_row_tight: u32,
    bytes_per_row_padded: u32,
    buffer_size: u64,
}

fn compute_readback_layout(
    extent: wgpu::Extent3d,
    max_buffer_size: u64,
) -> Result<ReadbackLayout, HeadlessReadbackError> {
    let width = extent.width;
    let height = extent.height;
    if width == 0 || height == 0 {
        return Err(HeadlessReadbackError::EmptyExtent);
    }

    let bytes_per_row_tight = width * 4;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_row_padded = bytes_per_row_tight.div_ceil(alignment) * alignment;
    let buffer_size = u64::from(bytes_per_row_padded) * u64::from(height);
    if buffer_size > max_buffer_size {
        return Err(HeadlessReadbackError::BufferSizeExceedsLimit {
            size: buffer_size,
            max: max_buffer_size,
        });
    }

    Ok(ReadbackLayout {
        width,
        height,
        bytes_per_row_tight,
        bytes_per_row_padded,
        buffer_size,
    })
}

fn create_readback_buffer(gpu: &GpuContext, layout: &ReadbackLayout) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("renderide-headless-readback"),
        size: layout.buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn submit_texture_to_buffer_copy(
    gpu: &GpuContext,
    color_texture: &wgpu::Texture,
    layout: &ReadbackLayout,
    readback: &wgpu::Buffer,
) {
    gpu.flush_driver();

    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("renderide-headless-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(layout.bytes_per_row_padded),
                rows_per_image: Some(layout.height),
            },
        },
        wgpu::Extent3d {
            width: layout.width,
            height: layout.height,
            depth_or_array_layers: 1,
        },
    );
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::headless_readback");
        encoder.finish()
    };
    gpu.queue().submit(std::iter::once(command_buffer));
}

fn await_buffer_map(
    slice: wgpu::BufferSlice<'_>,
    device: &wgpu::Device,
) -> Result<(), HeadlessReadbackError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| HeadlessReadbackError::DeviceLost(format!("{e:?}")))?;
    match receiver.recv_timeout(Duration::from_secs(5)) {
        Ok(result) => result.map_err(|e| HeadlessReadbackError::Map(format!("{e:?}"))),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            Err(HeadlessReadbackError::ReadbackTimeout)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(HeadlessReadbackError::Map(
            "map_async callback disconnected".to_owned(),
        )),
    }
}

fn copy_padded_rows_to_tight(bytes: &[u8], layout: &ReadbackLayout) -> Vec<u8> {
    let tight_len = (layout.bytes_per_row_tight as usize) * (layout.height as usize);
    let mut tight = vec![0u8; tight_len];
    for row in 0..(layout.height as usize) {
        let src_start = row * layout.bytes_per_row_padded as usize;
        let src_end = src_start + layout.bytes_per_row_tight as usize;
        let dst_start = row * layout.bytes_per_row_tight as usize;
        let dst_end = dst_start + layout.bytes_per_row_tight as usize;
        tight[dst_start..dst_end].copy_from_slice(&bytes[src_start..src_end]);
    }
    tight
}

fn write_png_atomically(
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), HeadlessReadbackError> {
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(HeadlessReadbackError::Io)?;
    }
    let buffer = image::RgbaImage::from_raw(width, height, rgba_bytes.to_vec())
        .ok_or(HeadlessReadbackError::EncodeBufferSize)?;
    let tmp_path = output_path.with_extension("png.tmp");
    buffer
        .save_with_format(&tmp_path, image::ImageFormat::Png)
        .map_err(|e| HeadlessReadbackError::Encode(format!("{e:?}")))?;
    std::fs::rename(&tmp_path, output_path).map_err(HeadlessReadbackError::Io)
}

/// Failures while copying the offscreen color texture back to CPU and writing the PNG.
#[derive(Debug, thiserror::Error)]
pub(super) enum HeadlessReadbackError {
    /// The headless offscreen color texture has not been allocated.
    #[error("no headless offscreen color texture allocated")]
    NoOffscreenTexture,
    /// The offscreen target had a zero-sized extent.
    #[error("headless offscreen target has empty extent")]
    EmptyExtent,
    /// `device.poll` reported device loss while waiting on `map_async`.
    #[error("device lost during readback poll: {0}")]
    DeviceLost(String),
    /// `map_async` callback never delivered a result before the timeout.
    #[error("buffer.map_async timed out")]
    ReadbackTimeout,
    /// `map_async` returned an error.
    #[error("map_async failed: {0}")]
    Map(String),
    /// The pixel buffer dimensions did not match the produced byte count.
    #[error("readback dimensions invalid for image::RgbaImage construction")]
    EncodeBufferSize,
    /// Encoding to PNG via the `image` crate failed.
    #[error("png encode: {0}")]
    Encode(String),
    /// Filesystem operation failed.
    #[error("io: {0}")]
    Io(#[source] std::io::Error),
    /// The readback buffer size would exceed the device max buffer size.
    #[error("readback buffer {size} bytes exceeds device max_buffer_size={max}")]
    BufferSizeExceedsLimit {
        /// Requested readback buffer size.
        size: u64,
        /// Device cap.
        max: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::{HeadlessReadbackError, compute_readback_layout, copy_padded_rows_to_tight};

    #[test]
    fn constant_variants_render_stable_messages() {
        assert_eq!(
            HeadlessReadbackError::NoOffscreenTexture.to_string(),
            "no headless offscreen color texture allocated"
        );
        assert_eq!(
            HeadlessReadbackError::EmptyExtent.to_string(),
            "headless offscreen target has empty extent"
        );
        assert_eq!(
            HeadlessReadbackError::ReadbackTimeout.to_string(),
            "buffer.map_async timed out"
        );
        assert_eq!(
            HeadlessReadbackError::EncodeBufferSize.to_string(),
            "readback dimensions invalid for image::RgbaImage construction"
        );
    }

    #[test]
    fn payload_variants_interpolate_inner_string() {
        let lost = HeadlessReadbackError::DeviceLost("adapter went away".into()).to_string();
        assert!(lost.contains("adapter went away"), "got {lost:?}");

        let mapped = HeadlessReadbackError::Map("OOM".into()).to_string();
        assert!(mapped.contains("OOM"), "got {mapped:?}");

        let encoded = HeadlessReadbackError::Encode("IO(BrokenPipe)".into()).to_string();
        assert!(encoded.contains("IO(BrokenPipe)"), "got {encoded:?}");
    }

    #[test]
    fn layout_rejects_empty_extent() {
        let error = compute_readback_layout(
            wgpu::Extent3d {
                width: 0,
                height: 1,
                depth_or_array_layers: 1,
            },
            1024,
        )
        .expect_err("zero width must fail");
        assert!(matches!(error, HeadlessReadbackError::EmptyExtent));
    }

    #[test]
    fn layout_pads_rows_to_wgpu_alignment() {
        let layout = compute_readback_layout(
            wgpu::Extent3d {
                width: 17,
                height: 3,
                depth_or_array_layers: 1,
            },
            4096,
        )
        .expect("layout fits");
        assert_eq!(layout.bytes_per_row_tight, 68);
        assert_eq!(
            layout.bytes_per_row_padded,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
        );
        assert_eq!(
            layout.buffer_size,
            u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * 3
        );
    }

    #[test]
    fn layout_rejects_buffers_over_limit() {
        let error = compute_readback_layout(
            wgpu::Extent3d {
                width: 64,
                height: 4,
                depth_or_array_layers: 1,
            },
            512,
        )
        .expect_err("buffer exceeds limit");
        assert!(matches!(
            error,
            HeadlessReadbackError::BufferSizeExceedsLimit { .. }
        ));
    }

    #[test]
    fn copy_padded_rows_to_tight_drops_padding() {
        let layout = compute_readback_layout(
            wgpu::Extent3d {
                width: 1,
                height: 2,
                depth_or_array_layers: 1,
            },
            1024,
        )
        .expect("layout fits");
        let mut padded = vec![0u8; layout.buffer_size as usize];
        padded[0..4].copy_from_slice(&[1, 2, 3, 4]);
        let second_row = layout.bytes_per_row_padded as usize;
        padded[second_row..second_row + 4].copy_from_slice(&[5, 6, 7, 8]);
        assert_eq!(
            copy_padded_rows_to_tight(&padded, &layout),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
    }
}
