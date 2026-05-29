//! Deferred mesh buffer upload batch for asset integration.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

use crate::assets::mesh::MeshBufferUploadSink;
use crate::render_graph::upload_arena::{
    PersistentUploadArena, UploadArenaAcquireStats, UploadArenaPressure,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WritePlan {
    Stage { staging_offset: u64, len: u64 },
    Fallback,
}

struct QueueWrite {
    order: u64,
    buffer: wgpu::Buffer,
    offset: u64,
    data: Range<usize>,
}

#[derive(Default)]
struct RecordedMeshUploads {
    writes: Vec<QueueWrite>,
    bytes: Vec<u8>,
}

impl RecordedMeshUploads {
    fn push_bytes(&mut self, data: &[u8]) -> Range<usize> {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(data);
        start..self.bytes.len()
    }

    fn push_buffer_write(&mut self, order: u64, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        let data = self.push_bytes(data);
        self.writes.push(QueueWrite {
            order,
            buffer: buffer.clone(),
            offset,
            data,
        });
    }
}

/// Counters describing one mesh upload batch drain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshUploadBatchStats {
    /// Number of queued buffer writes drained.
    pub(crate) writes: usize,
    /// Total payload bytes drained.
    pub(crate) bytes: usize,
    /// Writes served by staging-buffer copy commands.
    pub(crate) staged_writes: usize,
    /// Writes replayed through queue writes.
    pub(crate) fallback_writes: usize,
    /// Required staging bytes for aligned writes.
    pub(crate) staging_bytes: u64,
    /// Number of copy commands recorded.
    pub(crate) copy_ops: usize,
    /// Bytes staged through persistent arena slots.
    pub(crate) persistent_staging_bytes: u64,
    /// Persistent arena slot reuse count.
    pub(crate) persistent_slot_reuses: usize,
    /// Persistent arena allocation or growth count.
    pub(crate) persistent_slot_grows: usize,
    /// Bytes staged through temporary buffers.
    pub(crate) temporary_staging_bytes: u64,
    /// Temporary staging fallback count.
    pub(crate) temporary_staging_fallbacks: usize,
    /// Writes replayed because the staging payload exceeded device limits.
    pub(crate) oversized_queue_fallback_writes: usize,
    /// Writes replayed because the queue gate was busy.
    pub(crate) queue_gate_fallbacks: usize,
    /// Total persistent arena capacity.
    pub(crate) arena_capacity_bytes: u64,
    /// Persistent arena slots currently free.
    pub(crate) arena_free_slots: usize,
    /// Persistent arena slots referenced by submitted GPU work.
    pub(crate) arena_in_flight_slots: usize,
    /// Persistent arena slots waiting on remap completion.
    pub(crate) arena_remapping_slots: usize,
    /// CPU time spent finishing the upload command encoder.
    pub(crate) finish_ms: f64,
}

impl MeshUploadBatchStats {
    fn apply_arena_acquire(&mut self, stats: UploadArenaAcquireStats) {
        self.persistent_staging_bytes = stats.persistent_staging_bytes;
        self.persistent_slot_reuses = stats.persistent_slot_reuses;
        self.persistent_slot_grows = stats.persistent_slot_grows;
        self.temporary_staging_bytes = stats.temporary_staging_bytes;
        self.temporary_staging_fallbacks = stats.temporary_staging_fallbacks;
        self.oversized_queue_fallback_writes = stats.oversized_queue_fallback_writes;
    }

    fn apply_arena_pressure(&mut self, pressure: UploadArenaPressure) {
        self.arena_capacity_bytes = pressure.capacity_bytes;
        self.arena_free_slots = pressure.free_slots;
        self.arena_in_flight_slots = pressure.in_flight_slots;
        self.arena_remapping_slots = pressure.remapping_slots;
    }
}

/// Pending GPU submission and completion callback produced by a mesh upload batch drain.
pub(crate) struct MeshUploadFlush {
    /// Copy command buffer for staged writes.
    pub(crate) command_buffer: Option<wgpu::CommandBuffer>,
    /// Callback that recycles a persistent staging slot after GPU completion.
    pub(crate) on_submitted_work_done: Option<Box<dyn FnOnce() + Send + 'static>>,
    /// Batch traffic and staging stats.
    pub(crate) stats: MeshUploadBatchStats,
}

/// Ordered deferred upload batch for mesh buffer writes.
pub(crate) struct MeshUploadStagingBatch {
    recorded: Mutex<RecordedMeshUploads>,
    sequence: AtomicU64,
}

impl MeshUploadStagingBatch {
    /// Creates an empty mesh upload staging batch.
    pub(crate) fn new() -> Self {
        Self {
            recorded: Mutex::new(RecordedMeshUploads::default()),
            sequence: AtomicU64::new(0),
        }
    }

    /// Drains recorded writes and returns staged copy work or queue-write fallback results.
    pub(crate) fn drain_and_flush(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_buffer_size: u64,
        upload_arena: &mut PersistentUploadArena,
        force_queue_fallback: bool,
    ) -> Option<MeshUploadFlush> {
        profiling::scope!("mesh_upload::drain_and_flush");
        let (writes, payload_bytes, mut stats) = self.take_recorded_uploads()?;
        if force_queue_fallback {
            force_queue_fallback_stats(&mut stats);
            replay_all_writes_through_queue(queue, &writes, &payload_bytes);
            stats.queue_gate_fallbacks = stats.writes;
            stats.apply_arena_pressure(upload_arena.pressure());
            self.restore_recorded_upload_capacity(writes, payload_bytes);
            crate::profiling::plot_mesh_upload_batch(&stats);
            return Some(MeshUploadFlush {
                command_buffer: None,
                on_submitted_work_done: None,
                stats,
            });
        }

        let shapes = writes
            .iter()
            .map(|write| (write.offset, (write.data.end - write.data.start) as u64));
        let (plans, staging_size) = plan_staging_shapes(shapes, &mut stats);
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
        );
        stats.finish_ms = finish_ms;
        stats.apply_arena_pressure(upload_arena.pressure());
        self.restore_recorded_upload_capacity(writes, payload_bytes);
        crate::profiling::plot_mesh_upload_batch(&stats);
        Some(MeshUploadFlush {
            command_buffer,
            on_submitted_work_done: staging.on_submitted_work_done,
            stats,
        })
    }

    fn take_recorded_uploads(&self) -> Option<(Vec<QueueWrite>, Vec<u8>, MeshUploadBatchStats)> {
        profiling::scope!("mesh_upload::take_recorded");
        let mut recorded = self.recorded.lock();
        if recorded.writes.is_empty() {
            return None;
        }
        let stats = MeshUploadBatchStats {
            writes: recorded.writes.len(),
            bytes: recorded.bytes.len(),
            ..MeshUploadBatchStats::default()
        };
        if !recorded.writes.is_sorted_by_key(|write| write.order) {
            recorded.writes.sort_by_key(|write| write.order);
        }
        Some((
            std::mem::take(&mut recorded.writes),
            std::mem::take(&mut recorded.bytes),
            stats,
        ))
    }

    fn restore_recorded_upload_capacity(
        &self,
        mut writes: Vec<QueueWrite>,
        mut payload_bytes: Vec<u8>,
    ) {
        profiling::scope!("mesh_upload::restore_capacity");
        writes.clear();
        payload_bytes.clear();
        let mut recorded = self.recorded.lock();
        recorded.writes = writes;
        recorded.bytes = payload_bytes;
    }
}

impl Default for MeshUploadStagingBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshBufferUploadSink for MeshUploadStagingBatch {
    fn write_buffer(&self, buffer: &wgpu::Buffer, offset: wgpu::BufferAddress, contents: &[u8]) {
        if contents.is_empty() {
            return;
        }
        profiling::scope!("mesh_upload::record_write");
        let order = self.sequence.fetch_add(1, Ordering::Relaxed);
        self.recorded
            .lock()
            .push_buffer_write(order, buffer, offset, contents);
    }
}

fn force_queue_fallback_stats(stats: &mut MeshUploadBatchStats) {
    stats.fallback_writes = stats.writes;
    stats.staged_writes = 0;
    stats.staging_bytes = 0;
    stats.copy_ops = 0;
    stats.persistent_staging_bytes = 0;
    stats.persistent_slot_reuses = 0;
    stats.persistent_slot_grows = 0;
    stats.temporary_staging_bytes = 0;
    stats.temporary_staging_fallbacks = 0;
    stats.oversized_queue_fallback_writes = 0;
}

fn replay_all_writes_through_queue(
    queue: &wgpu::Queue,
    writes: &[QueueWrite],
    payload_bytes: &[u8],
) {
    profiling::scope!("mesh_upload::queue_fallback");
    for write in writes {
        queue.write_buffer(
            &write.buffer,
            write.offset,
            &payload_bytes[write.data.clone()],
        );
    }
}

fn plan_staging_shapes<I>(shapes: I, stats: &mut MeshUploadBatchStats) -> (Vec<WritePlan>, u64)
where
    I: IntoIterator<Item = (u64, u64)>,
{
    profiling::scope!("mesh_upload::plan_staging");
    let mut plans = Vec::new();
    let mut staging_size = 0u64;
    for (offset, len) in shapes {
        let aligned = len > 0
            && offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            && len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if aligned {
            let aligned_offset = staging_size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
            plans.push(WritePlan::Stage {
                staging_offset: aligned_offset,
                len,
            });
            staging_size = aligned_offset + len;
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

fn fill_staging_buffer(
    buffer: &wgpu::Buffer,
    writes: &[QueueWrite],
    plans: &[WritePlan],
    payload_bytes: &[u8],
) {
    profiling::scope!("mesh_upload::copy_to_staging");
    let mut mapped = buffer.slice(..).get_mapped_range_mut();
    for (write, plan) in writes.iter().zip(plans.iter()) {
        let WritePlan::Stage {
            staging_offset,
            len,
        } = plan
        else {
            continue;
        };
        let dst_start = *staging_offset as usize;
        let dst_end = dst_start + *len as usize;
        mapped
            .slice(dst_start..dst_end)
            .copy_from_slice(&payload_bytes[write.data.clone()]);
    }
}

fn record_upload_command_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    writes: &[QueueWrite],
    plans: &[WritePlan],
    payload_bytes: &[u8],
    staging: Option<&wgpu::Buffer>,
) -> (Option<wgpu::CommandBuffer>, f64) {
    profiling::scope!("mesh_upload::record_encoder");
    let needs_copy_commands = staging.is_some()
        && plans
            .iter()
            .any(|plan| matches!(plan, WritePlan::Stage { .. }));
    let mut encoder = needs_copy_commands.then(|| {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mesh_upload_staging_batch"),
        })
    });
    if let Some(encoder_ref) = encoder.as_mut() {
        let copy_scope = crate::profiling::GpuEncoderScope::begin(
            Option::<&crate::profiling::GpuProfilerHandle>::None,
            "asset::mesh_upload_staging_batch_copy",
            encoder_ref,
        );
        for (write, plan) in writes.iter().zip(plans.iter()) {
            record_upload_write(
                Some(&mut *encoder_ref),
                queue,
                write,
                plan,
                payload_bytes,
                staging,
            );
        }
        copy_scope.end(encoder_ref);
    } else {
        for (write, plan) in writes.iter().zip(plans.iter()) {
            record_upload_write(None, queue, write, plan, payload_bytes, staging);
        }
    }
    if let Some(encoder) = encoder {
        let finish_start = Instant::now();
        let command_buffer = encoder.finish();
        let finish_ms = finish_start.elapsed().as_secs_f64() * 1000.0;
        (Some(command_buffer), finish_ms)
    } else {
        (None, 0.0)
    }
}

fn record_upload_write(
    encoder: Option<&mut wgpu::CommandEncoder>,
    queue: &wgpu::Queue,
    write: &QueueWrite,
    plan: &WritePlan,
    payload_bytes: &[u8],
    staging: Option<&wgpu::Buffer>,
) {
    match plan {
        WritePlan::Stage {
            staging_offset,
            len,
        } => {
            if let (Some(staging), Some(encoder)) = (staging, encoder) {
                encoder.copy_buffer_to_buffer(
                    staging,
                    *staging_offset,
                    &write.buffer,
                    write.offset,
                    *len,
                );
            } else {
                queue.write_buffer(
                    &write.buffer,
                    write.offset,
                    &payload_bytes[write.data.clone()],
                );
            }
        }
        WritePlan::Fallback => {
            queue.write_buffer(
                &write.buffer,
                write.offset,
                &payload_bytes[write.data.clone()],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_plan_separates_aligned_and_unaligned_writes() {
        let mut stats = MeshUploadBatchStats::default();
        let (plans, staging_size) = plan_staging_shapes([(0, 4), (2, 4), (8, 3)], &mut stats);

        assert_eq!(
            plans,
            vec![
                WritePlan::Stage {
                    staging_offset: 0,
                    len: 4
                },
                WritePlan::Fallback,
                WritePlan::Fallback,
            ]
        );
        assert_eq!(staging_size, 4);
        assert_eq!(stats.staged_writes, 1);
        assert_eq!(stats.fallback_writes, 2);
    }

    #[test]
    fn staging_plan_preserves_order_while_aligning_staging_offsets() {
        let mut stats = MeshUploadBatchStats::default();
        let (plans, staging_size) = plan_staging_shapes([(4, 4), (0, 8)], &mut stats);

        assert_eq!(
            plans,
            vec![
                WritePlan::Stage {
                    staging_offset: 0,
                    len: 4
                },
                WritePlan::Stage {
                    staging_offset: 4,
                    len: 8
                },
            ]
        );
        assert_eq!(staging_size, 12);
        assert_eq!(stats.copy_ops, 2);
    }

    #[test]
    fn zero_length_writes_use_fallback_plan() {
        let mut stats = MeshUploadBatchStats::default();
        let (plans, staging_size) = plan_staging_shapes([(0, 0)], &mut stats);

        assert_eq!(plans, vec![WritePlan::Fallback]);
        assert_eq!(staging_size, 0);
        assert_eq!(stats.fallback_writes, 1);
    }

    #[test]
    fn forced_queue_fallback_clears_copy_stats() {
        let mut stats = MeshUploadBatchStats {
            writes: 3,
            staged_writes: 3,
            staging_bytes: 16,
            copy_ops: 3,
            persistent_staging_bytes: 16,
            persistent_slot_reuses: 1,
            ..MeshUploadBatchStats::default()
        };

        force_queue_fallback_stats(&mut stats);

        assert_eq!(stats.fallback_writes, 3);
        assert_eq!(stats.staged_writes, 0);
        assert_eq!(stats.copy_ops, 0);
        assert_eq!(stats.persistent_slot_reuses, 0);
    }
}
