//! GPU buffer-mapping plumbing shared across offscreen-render-task drains.
//!
//! These helpers are deliberately low-level: layout math, format conversion, and the per-domain
//! shared-memory write paths stay in [`super::camera`] and [`super::reflection_probe`] because
//! their shapes diverge (single image vs cubemap mip pyramid). The pieces below are the genuinely
//! identical bits: alignment math, the wait-for-map handshake, and the SIMD-fast zero fill used
//! when a failed task must produce a zero result buffer.

use std::time::Duration;

use rayon::prelude::*;

/// Failure modes for linear texture readback planning and row extraction.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(in crate::runtime) enum LinearTextureReadbackPlanError {
    /// The requested texture extent cannot produce a valid readback copy.
    #[error("linear texture readback extent {width}x{height} is invalid")]
    InvalidExtent {
        /// Texture width in pixels.
        width: u32,
        /// Texture height in pixels.
        height: u32,
    },
    /// The readback texel format has no bytes per pixel.
    #[error("linear texture readback bytes_per_pixel is zero")]
    InvalidBytesPerPixel,
    /// The output byte count overflowed while computing copy layout.
    #[error("linear texture readback byte count overflow")]
    OutputByteCountOverflow,
    /// The required readback buffer exceeds the device limit.
    #[error("linear texture readback buffer {size} bytes exceeds device max_buffer_size={max}")]
    ReadbackBufferTooLarge {
        /// Required readback buffer size.
        size: u64,
        /// Device `max_buffer_size`.
        max: u64,
    },
    /// The mapped readback buffer did not contain the planned padded rows.
    #[error("linear texture mapped readback is too small: need {required} bytes, got {actual}")]
    MappedReadbackTooSmall {
        /// Required mapped byte count.
        required: usize,
        /// Actual mapped byte count.
        actual: usize,
    },
}

/// Copy and packing layout for one tightly packed 2D texture readback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) struct LinearTextureReadbackPlan {
    width: u32,
    height: u32,
    bytes_per_row_tight: u32,
    bytes_per_row_padded: u32,
    buffer_size: u64,
}

impl LinearTextureReadbackPlan {
    /// Computes the padded GPU copy layout and validates the device readback-buffer limit.
    pub(in crate::runtime) fn new(
        extent: wgpu::Extent3d,
        bytes_per_pixel: u32,
        max_buffer_size: u64,
    ) -> Result<Self, LinearTextureReadbackPlanError> {
        let width = extent.width;
        let height = extent.height;
        if width == 0 || height == 0 {
            return Err(LinearTextureReadbackPlanError::InvalidExtent { width, height });
        }
        if bytes_per_pixel == 0 {
            return Err(LinearTextureReadbackPlanError::InvalidBytesPerPixel);
        }
        let bytes_per_row_tight = width
            .checked_mul(bytes_per_pixel)
            .ok_or(LinearTextureReadbackPlanError::OutputByteCountOverflow)?;
        let bytes_per_row_padded =
            align_u32_up(bytes_per_row_tight, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                .ok_or(LinearTextureReadbackPlanError::OutputByteCountOverflow)?;
        let buffer_size = u64::from(bytes_per_row_padded)
            .checked_mul(u64::from(height))
            .ok_or(LinearTextureReadbackPlanError::OutputByteCountOverflow)?;
        if buffer_size > max_buffer_size {
            return Err(LinearTextureReadbackPlanError::ReadbackBufferTooLarge {
                size: buffer_size,
                max: max_buffer_size,
            });
        }
        Ok(Self {
            width,
            height,
            bytes_per_row_tight,
            bytes_per_row_padded,
            buffer_size,
        })
    }

    /// Required GPU readback buffer byte size.
    pub(in crate::runtime) const fn buffer_size(self) -> u64 {
        self.buffer_size
    }

    /// Texture-to-buffer layout for the planned copy.
    pub(in crate::runtime) const fn copy_buffer_layout(self) -> wgpu::TexelCopyBufferLayout {
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(self.bytes_per_row_padded),
            rows_per_image: Some(self.height),
        }
    }

    /// Source texture extent copied by the plan.
    pub(in crate::runtime) const fn copy_extent(self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }

    /// Copies mapped padded GPU rows into a tightly packed CPU buffer.
    pub(in crate::runtime) fn copy_mapped_rows_to_tight(
        self,
        bytes: &[u8],
    ) -> Result<Vec<u8>, LinearTextureReadbackPlanError> {
        let required = usize::try_from(self.buffer_size)
            .map_err(|_err| LinearTextureReadbackPlanError::OutputByteCountOverflow)?;
        if bytes.len() < required {
            return Err(LinearTextureReadbackPlanError::MappedReadbackTooSmall {
                required,
                actual: bytes.len(),
            });
        }
        let tight_len =
            usize::try_from(u64::from(self.bytes_per_row_tight) * u64::from(self.height))
                .map_err(|_err| LinearTextureReadbackPlanError::OutputByteCountOverflow)?;
        let mut tight = vec![0u8; tight_len];
        let padded_stride = self.bytes_per_row_padded as usize;
        let tight_stride = self.bytes_per_row_tight as usize;
        for row in 0..(self.height as usize) {
            let src_start = row * padded_stride;
            let src_end = src_start + tight_stride;
            let dst_start = row * tight_stride;
            let dst_end = dst_start + tight_stride;
            tight[dst_start..dst_end].copy_from_slice(&bytes[src_start..src_end]);
        }
        Ok(tight)
    }
}

/// Failure modes for [`await_buffer_map`].
///
/// Domain error enums implement `From<AwaitBufferMapError>` so a `?` propagation maps each
/// variant onto the existing domain-specific variants without changing log strings.
#[derive(Debug, thiserror::Error)]
pub(in crate::runtime) enum AwaitBufferMapError {
    /// `wgpu::Device::poll` returned a device-lost error while pumping the map callback.
    #[error("device lost during readback poll: {0}")]
    DeviceLost(String),
    /// The map callback did not run within the supplied timeout.
    #[error("map_async timed out")]
    Timeout,
    /// `map_async` reported failure or the callback channel disconnected.
    #[error("map_async failed: {0}")]
    Map(String),
}

/// Waits for `slice.map_async(Read, ..)` to complete, polling `device` and timing out via
/// `timeout`.
///
/// Caller is responsible for wrapping the call in a [`profiling::scope!`] so Tracy traces
/// retain their domain-specific labels.
pub(in crate::runtime) fn await_buffer_map(
    slice: wgpu::BufferSlice<'_>,
    device: &wgpu::Device,
    timeout: Duration,
) -> Result<(), AwaitBufferMapError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| AwaitBufferMapError::DeviceLost(format!("{e:?}")))?;
    match receiver.recv_timeout(timeout) {
        Ok(result) => result.map_err(|e| AwaitBufferMapError::Map(format!("{e:?}"))),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(AwaitBufferMapError::Timeout),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(AwaitBufferMapError::Map(
            "map_async callback disconnected".to_owned(),
        )),
    }
}

/// Per-thread fill chunk for large shared-memory result buffers.
const PAR_FILL_CHUNK: usize = 32 * 1024;
/// Fill chunks assigned to one Rayon worker leaf.
const PAR_FILL_CHUNKS_PER_TASK: usize = 1;
/// Buffers at or above this size are zero-filled through rayon.
const PAR_FILL_THRESHOLD: usize = PAR_FILL_CHUNK * 2;

/// Zero-fills `bytes` using a parallel chunked path for large buffers and a single-threaded
/// `fill` for small ones.
pub(in crate::runtime) fn par_fill_zeros(bytes: &mut [u8]) {
    if bytes.len() >= PAR_FILL_THRESHOLD {
        bytes
            .par_chunks_mut(PAR_FILL_CHUNK)
            .with_min_len(PAR_FILL_CHUNKS_PER_TASK)
            .for_each(|chunk| chunk.fill(0));
    } else {
        bytes.fill(0);
    }
}

/// Rounds `value` up to the next multiple of `alignment`, returning `None` on overflow.
pub(in crate::runtime) fn align_u32_up(value: u32, alignment: u32) -> Option<u32> {
    value.div_ceil(alignment).checked_mul(alignment)
}

/// Rounds `value` up to the next multiple of `alignment`, returning `None` on overflow.
pub(in crate::runtime) fn align_u64_up(value: u64, alignment: u64) -> Option<u64> {
    let padded = value.checked_add(alignment.saturating_sub(1))?;
    let q = padded.checked_div(alignment)?;
    q.checked_mul(alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_u32_up_rounds_up_and_detects_overflow() {
        assert_eq!(align_u32_up(257, 256), Some(512));
        assert_eq!(align_u32_up(256, 256), Some(256));
        assert_eq!(align_u32_up(0, 256), Some(0));
        assert_eq!(align_u32_up(u32::MAX, 256), None);
    }

    #[test]
    fn align_u64_up_rounds_up_and_detects_overflow() {
        assert_eq!(align_u64_up(513, 256), Some(768));
        assert_eq!(align_u64_up(256, 256), Some(256));
        assert_eq!(align_u64_up(0, 256), Some(0));
        assert_eq!(align_u64_up(u64::MAX, 256), None);
    }

    #[test]
    fn par_fill_zeros_clears_small_and_large_buffers() {
        let mut small = vec![0xAAu8; 64];
        par_fill_zeros(&mut small);
        assert!(small.iter().all(|&b| b == 0));

        let mut large = vec![0xAAu8; PAR_FILL_THRESHOLD + 100];
        par_fill_zeros(&mut large);
        assert!(large.iter().all(|&b| b == 0));
    }

    #[test]
    fn linear_texture_plan_copies_padded_rows_to_tight() {
        let plan = LinearTextureReadbackPlan::new(
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            4,
            1024,
        )
        .expect("plan");
        assert_eq!(plan.copy_buffer_layout().bytes_per_row, Some(256));
        assert_eq!(plan.buffer_size(), 512);

        let mut mapped = vec![0u8; plan.buffer_size() as usize];
        mapped[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        mapped[256..264].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);

        let tight = plan.copy_mapped_rows_to_tight(&mapped).expect("tight");
        assert_eq!(
            tight,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn linear_texture_plan_rejects_invalid_inputs() {
        assert_eq!(
            LinearTextureReadbackPlan::new(
                wgpu::Extent3d {
                    width: 0,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                4,
                1024,
            )
            .expect_err("zero width"),
            LinearTextureReadbackPlanError::InvalidExtent {
                width: 0,
                height: 1,
            }
        );
        assert_eq!(
            LinearTextureReadbackPlan::new(
                wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                0,
                1024,
            )
            .expect_err("zero bpp"),
            LinearTextureReadbackPlanError::InvalidBytesPerPixel
        );
        assert_eq!(
            LinearTextureReadbackPlan::new(
                wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                4,
                511,
            )
            .expect_err("too large"),
            LinearTextureReadbackPlanError::ReadbackBufferTooLarge {
                size: 512,
                max: 511,
            }
        );
    }
}
