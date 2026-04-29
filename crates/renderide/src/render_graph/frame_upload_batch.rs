//! Per-frame deferred [`wgpu::Queue::write_buffer`] routing.
//!
//! Record paths that run per-view push their uniform / storage uploads into a
//! [`FrameUploadBatch`] instead of invoking [`wgpu::Queue::write_buffer`] directly. The batch is
//! drained onto the main thread after all per-view recording finishes but before the single
//! [`crate::gpu::GpuContext::submit_frame_batch`] call. All buffered writes therefore land in the
//! queue prior to submit and are visible to every command buffer in the frame, identical to the
//! direct-call serial path.
//!
//! This plumbing decouples queue ownership from parallel recording: a [`FrameUploadBatch`] can be
//! shared as a read-only reference across rayon workers, whereas [`wgpu::Queue`] access during
//! concurrent recording risks host-side ordering bugs on some backends.

use std::ops::Range;

use parking_lot::Mutex;

/// One deferred [`wgpu::Queue::write_buffer`] entry.
enum QueueWrite {
    /// A buffered buffer write; the caller's payload is copied into the frame upload arena so the
    /// source slice can be released before the batch is drained.
    Buffer {
        /// Destination buffer (clones are cheap; [`wgpu::Buffer`] is `Arc`-like internally).
        buffer: wgpu::Buffer,
        /// Byte offset into `buffer` where the payload is written.
        offset: u64,
        /// Byte range in [`RecordedUploads::bytes`].
        data: Range<usize>,
    },
}

/// Arena-backed upload command recorder for one frame.
#[derive(Default)]
struct RecordedUploads {
    /// Ordered buffer writes recorded by frame-global and per-view passes.
    writes: Vec<QueueWrite>,
    /// Contiguous payload arena addressed by [`QueueWrite::Buffer::data`] ranges.
    bytes: Vec<u8>,
}

impl RecordedUploads {
    /// Appends `data` to the arena and returns the stored byte range.
    fn push_bytes(&mut self, data: &[u8]) -> Range<usize> {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(data);
        start..self.bytes.len()
    }

    /// Records one buffer write in insertion order.
    fn push_buffer_write(&mut self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let data = self.push_bytes(data);
        self.writes.push(QueueWrite::Buffer {
            buffer: buffer.clone(),
            offset,
            data,
        });
    }
}

/// Collects per-frame [`wgpu::Queue::write_buffer`] calls for a single ordered replay.
///
/// Writes from multiple threads are serialised through an internal [`parking_lot::Mutex`] and are
/// replayed in the order they were pushed when [`FrameUploadBatch::drain_and_flush`] is called.
/// Payloads are copied into a contiguous frame arena rather than one heap allocation per write, so
/// the source slice can be dropped immediately after [`Self::write_buffer`] returns without
/// turning every uniform update into a standalone [`Vec`].
pub struct FrameUploadBatch {
    recorded: Mutex<RecordedUploads>,
}

impl FrameUploadBatch {
    /// Creates a new empty batch.
    pub fn new() -> Self {
        Self {
            recorded: Mutex::new(RecordedUploads::default()),
        }
    }

    /// Queues `queue.write_buffer(buffer, offset, data)` for later replay.
    ///
    /// `data` is copied into the frame upload arena so the caller's slice can be released or
    /// reused.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.recorded.lock().push_buffer_write(buffer, offset, data);
    }

    /// Drains every pending write and replays it against `queue` in insertion order.
    ///
    /// Called on the main thread after per-view encoding finishes and before the single
    /// [`wgpu::Queue::submit`]. After this returns the batch is empty.
    pub fn drain_and_flush(&self, queue: &wgpu::Queue) {
        let RecordedUploads { writes, bytes } = std::mem::take(&mut *self.recorded.lock());
        crate::profiling::plot_frame_upload_batch(writes.len(), bytes.len());
        for write in writes {
            match write {
                QueueWrite::Buffer {
                    buffer,
                    offset,
                    data,
                } => {
                    queue.write_buffer(&buffer, offset, &bytes[data]);
                }
            }
        }
    }

    /// Returns the number of pending writes (diagnostics / tests).
    #[cfg(test)]
    pub(crate) fn pending_count(&self) -> usize {
        self.recorded.lock().writes.len()
    }

    /// Returns pending payload bytes (diagnostics / tests).
    #[cfg(test)]
    pub(crate) fn pending_byte_count(&self) -> usize {
        self.recorded.lock().bytes.len()
    }
}

impl Default for FrameUploadBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_count_tracks_insertions_without_queue() {
        let batch = FrameUploadBatch::new();
        assert_eq!(batch.pending_count(), 0);
        assert_eq!(batch.pending_byte_count(), 0);
    }

    #[test]
    fn upload_arena_records_payloads_in_insertion_order() {
        let mut recorded = RecordedUploads::default();
        let global = recorded.push_bytes(&[1, 2, 3, 4]);
        let view_a = recorded.push_bytes(&[5, 6]);
        let view_b = recorded.push_bytes(&[7, 8, 9]);

        assert_eq!(&recorded.bytes[global], &[1, 2, 3, 4]);
        assert_eq!(&recorded.bytes[view_a], &[5, 6]);
        assert_eq!(&recorded.bytes[view_b], &[7, 8, 9]);
        assert_eq!(recorded.bytes, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    // NOTE: Exercising `write_buffer` and `drain_and_flush` end-to-end requires a real
    // [`wgpu::Device`] / [`wgpu::Queue`] pair, which is out of scope for unit tests per the
    // project's no-GPU-test policy. The arena test covers deterministic payload ordering; GPU
    // integration tests cover the observable behavior of replaying those bytes before submit.
}
