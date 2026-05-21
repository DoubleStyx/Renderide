//! GPU buffer-mapping plumbing shared across offscreen-render-task drains.
//!
//! These helpers are deliberately low-level: layout math, format conversion, and the per-domain
//! shared-memory write paths stay in [`super::camera`] and [`super::reflection_probe`] because
//! their shapes diverge (single image vs cubemap mip pyramid). The pieces below are the genuinely
//! identical bits: alignment math, the wait-for-map handshake, and the SIMD-fast zero fill used
//! when a failed task must produce a zero result buffer.

use std::time::Duration;

use rayon::prelude::*;

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
const PAR_FILL_CHUNK: usize = 64 * 1024;
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
}
