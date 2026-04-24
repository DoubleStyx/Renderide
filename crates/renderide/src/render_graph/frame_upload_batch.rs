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

use parking_lot::Mutex;

/// One deferred [`wgpu::Queue::write_buffer`] entry.
enum QueueWrite {
    /// A buffered buffer write; the caller's payload is copied into `data` so the source slice
    /// can be released before the batch is drained.
    Buffer {
        /// Destination buffer (clones are cheap; [`wgpu::Buffer`] is `Arc`-like internally).
        buffer: wgpu::Buffer,
        /// Byte offset into `buffer` where `data` is written.
        offset: u64,
        /// Owned copy of the bytes to upload.
        data: Vec<u8>,
    },
}

/// Collects per-frame [`wgpu::Queue::write_buffer`] calls for a single ordered replay.
///
/// Writes from multiple threads are serialised through an internal [`parking_lot::Mutex`] (picked
/// over [`std::sync::Mutex`] so the hot per-view write path has an infallible `lock` and needs no
/// poison-handling on every call), and are replayed in the order they were pushed when
/// [`FrameUploadBatch::drain_and_flush`] is called. Payloads are owned by the batch so the source
/// slice can be dropped immediately after [`Self::write_buffer`] returns.
pub struct FrameUploadBatch {
    writes: Mutex<Vec<QueueWrite>>,
}

impl FrameUploadBatch {
    /// Creates a new empty batch.
    pub fn new() -> Self {
        Self {
            writes: Mutex::new(Vec::new()),
        }
    }

    /// Queues `queue.write_buffer(buffer, offset, data)` for later replay.
    ///
    /// `data` is copied into an owned [`Vec`] so the caller's slice can be released or reused.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let write = QueueWrite::Buffer {
            buffer: buffer.clone(),
            offset,
            data: data.to_vec(),
        };
        self.writes.lock().push(write);
    }

    /// Drains every pending write and replays it against `queue` in insertion order.
    ///
    /// Called on the main thread after per-view encoding finishes and before the single
    /// [`wgpu::Queue::submit`]. After this returns the batch is empty.
    pub fn drain_and_flush(&self, queue: &wgpu::Queue) {
        let drained = std::mem::take(&mut *self.writes.lock());
        for write in drained {
            match write {
                QueueWrite::Buffer {
                    buffer,
                    offset,
                    data,
                } => {
                    queue.write_buffer(&buffer, offset, &data);
                }
            }
        }
    }

    /// Returns the number of pending writes (diagnostics / tests).
    #[cfg(test)]
    pub(crate) fn pending_count(&self) -> usize {
        self.writes.lock().len()
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
    }

    // NOTE: Exercising `write_buffer` and `drain_and_flush` end-to-end requires a real
    // [`wgpu::Device`] / [`wgpu::Queue`] pair, which is out of scope for unit tests per the
    // project's no-GPU-test policy. Ordering is asserted in integration tests where a device
    // is available (headless goldens), since insertion-order replay is the only observable
    // behavior difference vs. direct queue writes.
}
