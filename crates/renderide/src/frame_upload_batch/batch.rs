//! `FrameUploadBatch` core: records deferred queue writes and drains them into a staging copy
//! command buffer on the main thread before submit.

use std::cell::Cell;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

use super::scope::{
    CURRENT_UPLOAD_LOCAL_SEQ, CURRENT_UPLOAD_SCOPE, FrameUploadScope, FrameUploadScopeGuard,
    QueueWriteOrder,
};
use super::stats::{FrameUploadBatchStats, FrameUploadFlush, force_queue_fallback_stats};
use crate::upload_arena::PersistentUploadArena;

/// Whether a recorded [`QueueWrite::Buffer`] entry can be served from the persistent staging
/// buffer (4-aligned offset and length per [`wgpu::COPY_BUFFER_ALIGNMENT`]) or has to fall back
/// to [`wgpu::Queue::write_buffer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WritePlan {
    /// Aligned write: payload is staged at `staging_offset` and copied via
    /// [`wgpu::CommandEncoder::copy_buffer_to_buffer`].
    Stage { staging_offset: u64, len: u64 },
    /// Unaligned write: served by [`wgpu::Queue::write_buffer`] as before.
    Fallback,
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
/// the source slice can be dropped immediately after [`Self::write_buffer_with_scope_fallback`]
/// returns without turning every uniform update into a standalone [`Vec`].
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

    /// Queues a buffer write with `fallback_scope` when no matching thread-local pass scope exists.
    pub(crate) fn write_buffer_with_scope_fallback(
        &self,
        fallback_scope: FrameUploadScope,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) {
        let order = self.next_write_order_with_scope_fallback(fallback_scope);
        self.recorded
            .lock()
            .push_buffer_write(order, buffer, offset, data);
    }

    /// Returns the next write order, using `fallback_scope` when this thread did not enter it.
    fn next_write_order_with_scope_fallback(
        &self,
        fallback_scope: FrameUploadScope,
    ) -> QueueWriteOrder {
        let fallback_seq = self.fallback_sequence.fetch_add(1, Ordering::Relaxed);
        let current_scope = CURRENT_UPLOAD_SCOPE.with(Cell::get);
        if current_scope == Some(fallback_scope) {
            let local_seq = CURRENT_UPLOAD_LOCAL_SEQ.with(|seq| {
                let current = seq.get();
                seq.set(current.saturating_add(1));
                current
            });
            return QueueWriteOrder {
                scope: fallback_scope,
                local_seq,
                fallback_seq,
            };
        }
        QueueWriteOrder {
            scope: fallback_scope,
            local_seq: fallback_seq,
            fallback_seq,
        }
    }

    /// Drains every pending write into a persistent staging slot plus per-write
    /// [`wgpu::CommandEncoder::copy_buffer_to_buffer`] operations, returning the recorded command
    /// buffer for inclusion at the head of the frame's submit batch.
    ///
    /// Replaces the previous "N x [`wgpu::Queue::write_buffer`]" replay: each `write_buffer` call
    /// internally allocates its own staging chunk and locks the queue, so a frame with dozens of
    /// per-view uniform writes paid that overhead per write. The arena path copies the whole frame
    /// payload into one mapped buffer, unmaps once, and emits one encoder op per write. Persistent
    /// slots are recycled only after the submitted GPU work completes; if every persistent slot is
    /// still busy, the frame falls back to a one-frame temporary staging buffer.
    ///
    /// Writes whose `offset` or `len` are not 4-byte aligned (the
    /// [`wgpu::COPY_BUFFER_ALIGNMENT`] requirement for `copy_buffer_to_buffer`) fall back to
    /// `queue.write_buffer`. In practice every renderer uniform/storage upload is 4-aligned, so
    /// the fast path covers the steady-state working set; the fallback is correctness insurance.
    ///
    /// Returns `None` when no writes were pending. After this returns the batch is empty.
    pub fn drain_and_flush(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_buffer_size: u64,
        upload_arena: &mut PersistentUploadArena,
        avoid_mapped_staging: bool,
        profiler: Option<&mut crate::profiling::GpuProfilerHandle>,
    ) -> Option<FrameUploadFlush> {
        crate::profiling::scope!("frame_upload::drain_and_flush");
        let (writes, payload_bytes, mut stats) = self.take_recorded_uploads()?;
        if avoid_mapped_staging {
            force_queue_fallback_stats(&mut stats);
            replay_all_writes_through_queue(queue, &writes, &payload_bytes);
            stats.finish_ms = 0.0;
            stats.apply_arena_pressure(upload_arena.pressure());
            self.restore_recorded_upload_capacity(writes, payload_bytes);
            return Some(FrameUploadFlush {
                command_buffer: None,
                on_submitted_work_done: None,
                stats,
            });
        }
        let (plans, staging_size) = plan_staging_writes(&writes, &mut stats);
        #[cfg(feature = "tracy")]
        tracy_client::plot!("frame_upload::staging_bytes", staging_size as f64);
        let staging = upload_arena.prepare_staging_buffer(
            device,
            max_buffer_size,
            staging_size,
            stats.staged_writes,
        );
        stats.apply_arena_acquire(staging.acquire_stats());
        if let Some(staging_buffer) = staging.buffer() {
            fill_staging_buffer(staging_buffer, &writes, &plans, &payload_bytes);
        }
        if staging.requires_queue_fallback() {
            stats.copy_ops = 0;
        }
        let staging = staging.finish(upload_arena);
        let (command_buffer, finish_ms) = record_upload_command_buffer(
            device,
            queue,
            &writes,
            &plans,
            &payload_bytes,
            staging.buffer.as_ref(),
            profiler,
        );
        stats.finish_ms = finish_ms;
        stats.apply_arena_pressure(upload_arena.pressure());
        self.restore_recorded_upload_capacity(writes, payload_bytes);
        Some(FrameUploadFlush {
            command_buffer,
            on_submitted_work_done: staging.on_submitted_work_done,
            stats,
        })
    }

    /// Takes pending writes and payload bytes while preserving reusable capacity for restore.
    fn take_recorded_uploads(&self) -> Option<(Vec<QueueWrite>, Vec<u8>, FrameUploadBatchStats)> {
        crate::profiling::scope!("frame_upload::take_recorded");
        let mut recorded = self.recorded.lock();
        crate::profiling::plot_frame_upload_batch(recorded.writes.len(), recorded.bytes.len());
        if recorded.writes.is_empty() {
            return None;
        }
        let stats = FrameUploadBatchStats {
            writes: recorded.writes.len(),
            bytes: recorded.bytes.len(),
            ..FrameUploadBatchStats::default()
        };
        if !recorded.writes.is_sorted_by_key(queue_write_order) {
            crate::profiling::scope!("frame_upload::sort_writes");
            recorded.writes.sort_by_key(queue_write_order);
        }
        Some((
            std::mem::take(&mut recorded.writes),
            std::mem::take(&mut recorded.bytes),
            stats,
        ))
    }

    /// Restores cleared scratch buffers so later frames reuse the grown allocations.
    fn restore_recorded_upload_capacity(
        &self,
        mut writes: Vec<QueueWrite>,
        mut payload_bytes: Vec<u8>,
    ) {
        crate::profiling::scope!("frame_upload::restore_capacity");
        writes.clear();
        payload_bytes.clear();
        let mut recorded = self.recorded.lock();
        recorded.writes = writes;
        recorded.bytes = payload_bytes;
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

fn replay_all_writes_through_queue(
    queue: &wgpu::Queue,
    writes: &[QueueWrite],
    payload_bytes: &[u8],
) {
    crate::profiling::scope!("frame_upload::recovery_queue_fallback");
    for write in writes {
        let QueueWrite::Buffer {
            buffer,
            offset,
            data,
            ..
        } = write;
        queue.write_buffer(buffer, *offset, &payload_bytes[data.clone()]);
    }
}

/// Assigns each aligned write a staging-buffer slot and marks unaligned writes for fallback.
fn plan_staging_writes(
    writes: &[QueueWrite],
    stats: &mut FrameUploadBatchStats,
) -> (Vec<WritePlan>, u64) {
    crate::profiling::scope!("frame_upload::plan_staging");
    let mut plans = Vec::with_capacity(writes.len());
    let mut staging_size: u64 = 0;
    for write in writes {
        let QueueWrite::Buffer { offset, data, .. } = write;
        let len = (data.end - data.start) as u64;
        let aligned = len > 0
            && (*offset).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            && len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if aligned {
            let aligned_off = staging_size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
            plans.push(WritePlan::Stage {
                staging_offset: aligned_off,
                len,
            });
            staging_size = aligned_off + len;
            stats.staged_writes = stats.staged_writes.saturating_add(1);
        } else {
            plans.push(WritePlan::Fallback);
            stats.fallback_writes = stats.fallback_writes.saturating_add(1);
        }
    }
    stats.staging_bytes = staging_size;
    stats.copy_ops = stats.staged_writes;
    (plans, staging_size)
}

/// Copies each staged payload slice into its mapped staging-buffer offset.
fn fill_staging_buffer(
    buf: &wgpu::Buffer,
    writes: &[QueueWrite],
    plans: &[WritePlan],
    payload_bytes: &[u8],
) {
    crate::profiling::scope!("frame_upload::copy_to_staging");
    let mut mapped = buf.slice(..).get_mapped_range_mut();
    for (write, plan) in writes.iter().zip(plans.iter()) {
        let (
            QueueWrite::Buffer { data, .. },
            WritePlan::Stage {
                staging_offset,
                len,
            },
        ) = (write, plan)
        else {
            continue;
        };
        let dst_start = *staging_offset as usize;
        let dst_end = dst_start + *len as usize;
        mapped
            .slice(dst_start..dst_end)
            .copy_from_slice(&payload_bytes[data.clone()]);
    }
}

/// Records copy commands for staged writes and replays unaligned writes through the queue.
fn record_upload_command_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    writes: &[QueueWrite],
    plans: &[WritePlan],
    payload_bytes: &[u8],
    staging: Option<&wgpu::Buffer>,
    profiler: Option<&mut crate::profiling::GpuProfilerHandle>,
) -> (Option<wgpu::CommandBuffer>, f64) {
    crate::profiling::scope!("frame_upload::record_encoder");
    let needs_copy_commands = staging.is_some()
        && plans
            .iter()
            .any(|plan| matches!(plan, WritePlan::Stage { .. }));
    let mut encoder = needs_copy_commands.then(|| {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_upload_staging_belt"),
        })
    });
    let upload_copy_query = encoder.as_mut().and_then(|encoder| {
        profiler
            .as_deref()
            .map(|p| p.begin_query("frame_upload::copy_buffer_batch", encoder))
    });
    for (write, plan) in writes.iter().zip(plans.iter()) {
        record_upload_write(encoder.as_mut(), queue, write, plan, payload_bytes, staging);
    }
    if let Some(mut encoder) = encoder {
        if let Some(query) = upload_copy_query
            && let Some(profiler) = profiler
        {
            profiler.end_query(&mut encoder, query);
            profiler.resolve_queries(&mut encoder);
        }
        crate::profiling::scope!("CommandEncoder::finish::frame_upload");
        let finish_start = Instant::now();
        let command_buffer = encoder.finish();
        let finish_ms = finish_start.elapsed().as_secs_f64() * 1000.0;
        (Some(command_buffer), finish_ms)
    } else {
        (None, 0.0)
    }
}

/// Records one staged copy or fallback queue write.
fn record_upload_write(
    encoder: Option<&mut wgpu::CommandEncoder>,
    queue: &wgpu::Queue,
    write: &QueueWrite,
    plan: &WritePlan,
    payload_bytes: &[u8],
    staging: Option<&wgpu::Buffer>,
) {
    let QueueWrite::Buffer {
        buffer,
        offset,
        data,
        ..
    } = write;
    match plan {
        WritePlan::Stage {
            staging_offset,
            len,
        } => {
            if let (Some(staging_buf), Some(encoder)) = (staging, encoder) {
                encoder.copy_buffer_to_buffer(staging_buf, *staging_offset, buffer, *offset, *len);
            } else {
                profiling::scope!("frame_upload::staged_queue_fallback_write_buffer");
                queue.write_buffer(buffer, *offset, &payload_bytes[data.clone()]);
            }
        }
        WritePlan::Fallback => {
            profiling::scope!("frame_upload::fallback_write_buffer");
            queue.write_buffer(buffer, *offset, &payload_bytes[data.clone()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::scope::FrameUploadPhase;
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
                scope: FrameUploadScope::pre_record(),
                local_seq: 0,
                fallback_seq: 6,
            },
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
        ];

        orders.sort();

        assert_eq!(orders[0].scope.phase, FrameUploadPhase::PreRecord);
        assert_eq!(orders[1].scope.phase, FrameUploadPhase::FrameGlobal);
        assert_eq!(orders[2].scope, FrameUploadScope::per_view(0, 3));
        assert_eq!(orders[3].scope, FrameUploadScope::per_view(0, 8));
        assert_eq!(orders[3].local_seq, 0);
        assert_eq!(orders[4].scope, FrameUploadScope::per_view(0, 8));
        assert_eq!(orders[4].local_seq, 1);
        assert_eq!(orders[5].scope, FrameUploadScope::per_view(1, 4));
    }

    #[test]
    fn upload_scope_assigns_local_sequence_and_restores_previous_scope() {
        let batch = FrameUploadBatch::new();
        let scoped = {
            let _guard = batch.enter_scope(FrameUploadScope::per_view(2, 7));
            let first =
                batch.next_write_order_with_scope_fallback(FrameUploadScope::per_view(2, 7));
            let second =
                batch.next_write_order_with_scope_fallback(FrameUploadScope::per_view(2, 7));
            assert_eq!(first.scope, FrameUploadScope::per_view(2, 7));
            assert_eq!(first.local_seq, 0);
            assert_eq!(second.scope, FrameUploadScope::per_view(2, 7));
            assert_eq!(second.local_seq, 1);
            second
        };
        let fallback = batch.next_write_order_with_scope_fallback(FrameUploadScope::pre_record());

        assert_eq!(scoped.fallback_seq, 1);
        assert_eq!(fallback.scope.phase, FrameUploadPhase::PreRecord);
        assert_eq!(fallback.fallback_seq, 2);
    }

    #[test]
    fn upload_sink_fallback_scope_does_not_require_thread_local_scope() {
        let batch = FrameUploadBatch::new();
        let order = batch.next_write_order_with_scope_fallback(FrameUploadScope::per_view(1, 5));

        assert_eq!(order.scope, FrameUploadScope::per_view(1, 5));
        assert_eq!(order.local_seq, order.fallback_seq);
    }

    // NOTE: Exercising `write_buffer` and `drain_and_flush` end-to-end requires a real
    // [`wgpu::Device`] / [`wgpu::Queue`] pair, which is out of scope for unit tests per the
    // project's no-GPU-test policy. The pure order tests cover replay-key semantics; GPU
    // integration tests cover the observable behavior of replaying those bytes before submit.
}
