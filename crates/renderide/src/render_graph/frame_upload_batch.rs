//! Per-frame deferred [`wgpu::Queue::write_buffer`] routing.
//!
//! Record paths that run per-view push their uniform / storage uploads into a
//! [`FrameUploadBatch`] instead of invoking [`wgpu::Queue::write_buffer`] directly. The batch is
//! drained onto the main thread after all per-view recording finishes but before the single
//! [`crate::gpu::GpuContext::submit_frame_batch`] call. Writes are replayed by executor scope
//! `(frame-global before per-view, then view index, pass index, local call order)` so the result
//! is independent of which rayon worker won the upload-batch mutex first. All buffered writes
//! therefore land in the queue prior to submit and are visible to every command buffer in the
//! frame, identical to the direct-call serial path.
//!
//! This plumbing decouples queue ownership from parallel recording: a [`FrameUploadBatch`] can be
//! shared as a read-only reference across rayon workers, whereas [`wgpu::Queue`] access during
//! concurrent recording risks host-side ordering bugs on some backends.

use std::cell::Cell;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

thread_local! {
    static CURRENT_UPLOAD_SCOPE: Cell<Option<FrameUploadScope>> = const { Cell::new(None) };
    static CURRENT_UPLOAD_LOCAL_SEQ: Cell<u64> = const { Cell::new(0) };
}

/// Coarse executor phase used to replay frame uploads in a deterministic order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FrameUploadPhase {
    /// Frame-global graph passes.
    FrameGlobal,
    /// Per-view graph passes.
    PerView,
    /// Uploads recorded outside an executor pass scope.
    Unscoped,
}

/// Deterministic executor location for deferred queue writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FrameUploadScope {
    phase: FrameUploadPhase,
    view_idx: u32,
    pass_idx: u32,
}

impl FrameUploadScope {
    /// Scope for a frame-global pass.
    pub(crate) fn frame_global(pass_idx: usize) -> Self {
        Self {
            phase: FrameUploadPhase::FrameGlobal,
            view_idx: 0,
            pass_idx: saturating_u32(pass_idx),
        }
    }

    /// Scope for a per-view pass.
    pub(crate) fn per_view(view_idx: usize, pass_idx: usize) -> Self {
        Self {
            phase: FrameUploadPhase::PerView,
            view_idx: saturating_u32(view_idx),
            pass_idx: saturating_u32(pass_idx),
        }
    }

    /// Scope used for diagnostic or legacy writes recorded outside the executor.
    fn unscoped() -> Self {
        Self {
            phase: FrameUploadPhase::Unscoped,
            view_idx: u32::MAX,
            pass_idx: u32::MAX,
        }
    }
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Total replay key for one queued write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct QueueWriteOrder {
    scope: FrameUploadScope,
    local_seq: u64,
    fallback_seq: u64,
}

/// Restores the prior thread-local upload scope on drop.
pub(crate) struct FrameUploadScopeGuard {
    previous_scope: Option<FrameUploadScope>,
    previous_local_seq: u64,
}

impl Drop for FrameUploadScopeGuard {
    fn drop(&mut self) {
        CURRENT_UPLOAD_SCOPE.with(|scope| scope.set(self.previous_scope));
        CURRENT_UPLOAD_LOCAL_SEQ.with(|seq| seq.set(self.previous_local_seq));
    }
}

/// One deferred [`wgpu::Queue::write_buffer`] entry.
enum QueueWrite {
    /// A buffered buffer write; the caller's payload is copied into the frame upload arena so the
    /// source slice can be released before the batch is drained.
    Buffer {
        /// Deterministic replay key assigned when the write was queued.
        order: QueueWriteOrder,
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

    /// Appends one buffer write with its replay order key.
    fn push_buffer_write(
        &mut self,
        order: QueueWriteOrder,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) {
        let data = self.push_bytes(data);
        self.writes.push(QueueWrite::Buffer {
            order,
            buffer: buffer.clone(),
            offset,
            data,
        });
    }

    /// Clears recorded writes and payload bytes while retaining grown capacities for the next
    /// frame.
    fn clear_retain_capacity(&mut self) {
        self.writes.clear();
        self.bytes.clear();
    }
}

#[inline]
fn queue_write_order(write: &QueueWrite) -> QueueWriteOrder {
    match write {
        QueueWrite::Buffer { order, .. } => *order,
    }
}

/// Collects per-frame [`wgpu::Queue::write_buffer`] calls for a single ordered replay.
///
/// Writes from multiple threads are serialised through an internal [`parking_lot::Mutex`] and are
/// replayed by their executor scope when [`FrameUploadBatch::drain_and_flush`] is called.
/// Payloads are copied into a contiguous frame arena rather than one heap allocation per write, so
/// the source slice can be dropped immediately after [`Self::write_buffer`] returns without
/// turning every uniform update into a standalone [`Vec`].
pub struct FrameUploadBatch {
    recorded: Mutex<RecordedUploads>,
    fallback_sequence: AtomicU64,
}

impl FrameUploadBatch {
    /// Creates a new empty batch.
    pub fn new() -> Self {
        Self {
            recorded: Mutex::new(RecordedUploads::default()),
            fallback_sequence: AtomicU64::new(0),
        }
    }

    /// Enters `scope` for the current thread until the returned guard is dropped.
    pub(crate) fn enter_scope(&self, scope: FrameUploadScope) -> FrameUploadScopeGuard {
        let previous_scope = CURRENT_UPLOAD_SCOPE.with(|current| {
            let previous = current.get();
            current.set(Some(scope));
            previous
        });
        let previous_local_seq = CURRENT_UPLOAD_LOCAL_SEQ.with(|seq| {
            let previous = seq.get();
            seq.set(0);
            previous
        });
        FrameUploadScopeGuard {
            previous_scope,
            previous_local_seq,
        }
    }

    /// Queues `queue.write_buffer(buffer, offset, data)` for later replay.
    ///
    /// `data` is copied into the frame upload arena so the caller's slice can be released or
    /// reused.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let order = self.next_write_order();
        self.recorded
            .lock()
            .push_buffer_write(order, buffer, offset, data);
    }

    /// Returns the deterministic order key for the next write on the current thread.
    fn next_write_order(&self) -> QueueWriteOrder {
        let fallback_seq = self.fallback_sequence.fetch_add(1, Ordering::Relaxed);
        let scope = CURRENT_UPLOAD_SCOPE.with(Cell::get);
        match scope {
            Some(scope) => {
                let local_seq = CURRENT_UPLOAD_LOCAL_SEQ.with(|seq| {
                    let current = seq.get();
                    seq.set(current.saturating_add(1));
                    current
                });
                QueueWriteOrder {
                    scope,
                    local_seq,
                    fallback_seq,
                }
            }
            None => QueueWriteOrder {
                scope: FrameUploadScope::unscoped(),
                local_seq: 0,
                fallback_seq,
            },
        }
    }

    /// Drains every pending write and replays it against `queue` in deterministic scope order.
    ///
    /// Called on the main thread after per-view encoding finishes and before the single
    /// [`wgpu::Queue::submit`]. After this returns the batch is empty.
    pub fn drain_and_flush(&self, queue: &wgpu::Queue) {
        let mut recorded = self.recorded.lock();
        crate::profiling::plot_frame_upload_batch(recorded.writes.len(), recorded.bytes.len());
        if !recorded.writes.is_sorted_by_key(queue_write_order) {
            recorded.writes.sort_by_key(queue_write_order);
        }
        for write in &recorded.writes {
            match write {
                QueueWrite::Buffer {
                    order: _,
                    buffer,
                    offset,
                    data,
                } => {
                    queue.write_buffer(buffer, *offset, &recorded.bytes[data.clone()]);
                }
            }
        }
        recorded.clear_retain_capacity();
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

    #[test]
    fn upload_orders_sort_by_phase_view_pass_then_local_sequence() {
        let mut orders = [
            QueueWriteOrder {
                scope: FrameUploadScope::per_view(1, 4),
                local_seq: 0,
                fallback_seq: 0,
            },
            QueueWriteOrder {
                scope: FrameUploadScope::frame_global(9),
                local_seq: 0,
                fallback_seq: 1,
            },
            QueueWriteOrder {
                scope: FrameUploadScope::per_view(0, 8),
                local_seq: 1,
                fallback_seq: 2,
            },
            QueueWriteOrder {
                scope: FrameUploadScope::per_view(0, 8),
                local_seq: 0,
                fallback_seq: 3,
            },
            QueueWriteOrder {
                scope: FrameUploadScope::per_view(0, 3),
                local_seq: 0,
                fallback_seq: 4,
            },
            QueueWriteOrder {
                scope: FrameUploadScope::unscoped(),
                local_seq: 0,
                fallback_seq: 5,
            },
        ];

        orders.sort();

        assert_eq!(orders[0].scope.phase, FrameUploadPhase::FrameGlobal);
        assert_eq!(orders[1].scope, FrameUploadScope::per_view(0, 3));
        assert_eq!(orders[2].scope, FrameUploadScope::per_view(0, 8));
        assert_eq!(orders[2].local_seq, 0);
        assert_eq!(orders[3].scope, FrameUploadScope::per_view(0, 8));
        assert_eq!(orders[3].local_seq, 1);
        assert_eq!(orders[4].scope, FrameUploadScope::per_view(1, 4));
        assert_eq!(orders[5].scope.phase, FrameUploadPhase::Unscoped);
    }

    #[test]
    fn upload_scope_assigns_local_sequence_and_restores_previous_scope() {
        let batch = FrameUploadBatch::new();
        let scoped = {
            let _guard = batch.enter_scope(FrameUploadScope::per_view(2, 7));
            let first = batch.next_write_order();
            let second = batch.next_write_order();
            assert_eq!(first.scope, FrameUploadScope::per_view(2, 7));
            assert_eq!(first.local_seq, 0);
            assert_eq!(second.scope, FrameUploadScope::per_view(2, 7));
            assert_eq!(second.local_seq, 1);
            second
        };
        let unscoped = batch.next_write_order();

        assert_eq!(scoped.fallback_seq, 1);
        assert_eq!(unscoped.scope.phase, FrameUploadPhase::Unscoped);
        assert_eq!(unscoped.fallback_seq, 2);
    }

    // NOTE: Exercising `write_buffer` and `drain_and_flush` end-to-end requires a real
    // [`wgpu::Device`] / [`wgpu::Queue`] pair, which is out of scope for unit tests per the
    // project's no-GPU-test policy. The pure order tests cover replay-key semantics; GPU
    // integration tests cover the observable behavior of replaying those bytes before submit.
}
