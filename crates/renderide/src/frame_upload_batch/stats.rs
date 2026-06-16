//! Deferred-upload traffic statistics and the recorded command buffer produced by a drain.

use crate::upload_arena::{UploadArenaAcquireStats, UploadArenaPressure};

/// Deferred-upload traffic drained into the frame submit batch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameUploadBatchStats {
    /// Number of queued buffer writes drained.
    pub writes: usize,
    /// Total payload bytes drained.
    pub bytes: usize,
    /// Writes served by the staging-buffer copy path.
    pub staged_writes: usize,
    /// Writes replayed through [`wgpu::Queue::write_buffer`] because they were not copy-aligned.
    pub fallback_writes: usize,
    /// Size of the staging buffer allocated for aligned writes.
    pub staging_bytes: u64,
    /// Number of [`wgpu::CommandEncoder::copy_buffer_to_buffer`] operations recorded.
    pub copy_ops: usize,
    /// Bytes staged through a persistent upload arena slot.
    pub persistent_staging_bytes: u64,
    /// Persistent upload arena slot reuse count.
    pub persistent_slot_reuses: usize,
    /// Persistent upload arena allocation or growth count.
    pub persistent_slot_grows: usize,
    /// Bytes staged through a one-frame temporary fallback buffer.
    pub temporary_staging_bytes: u64,
    /// Count of temporary staging fallback buffers because all persistent slots were unavailable.
    pub temporary_staging_fallbacks: usize,
    /// Staged writes replayed through [`wgpu::Queue::write_buffer`] because no staging buffer fit.
    pub oversized_queue_fallback_writes: usize,
    /// Total bytes currently allocated across persistent upload arena slots.
    pub arena_capacity_bytes: u64,
    /// Persistent upload arena slots that are mapped and free after this drain.
    pub arena_free_slots: usize,
    /// Persistent upload arena slots referenced by submitted GPU work after this drain.
    pub arena_in_flight_slots: usize,
    /// Persistent upload arena slots waiting for `map_async` completion after this drain.
    pub arena_remapping_slots: usize,
    /// CPU time spent inside the upload encoder [`wgpu::CommandEncoder::finish`] call.
    pub finish_ms: f64,
}

impl FrameUploadBatchStats {
    pub(super) fn apply_arena_acquire(&mut self, stats: UploadArenaAcquireStats) {
        self.persistent_staging_bytes = stats.persistent_staging_bytes;
        self.persistent_slot_reuses = stats.persistent_slot_reuses;
        self.persistent_slot_grows = stats.persistent_slot_grows;
        self.temporary_staging_bytes = stats.temporary_staging_bytes;
        self.temporary_staging_fallbacks = stats.temporary_staging_fallbacks;
        self.oversized_queue_fallback_writes = stats.oversized_queue_fallback_writes;
    }

    pub(crate) fn apply_arena_pressure(&mut self, pressure: UploadArenaPressure) {
        self.arena_capacity_bytes = pressure.capacity_bytes;
        self.arena_free_slots = pressure.free_slots;
        self.arena_in_flight_slots = pressure.in_flight_slots;
        self.arena_remapping_slots = pressure.remapping_slots;
    }
}

/// Upload command buffer plus the traffic statistics that produced it.
pub struct FrameUploadFlush {
    /// Recorded copy command buffer for staged writes, or `None` when every write was replayed
    /// through the queue fallback path.
    pub command_buffer: Option<wgpu::CommandBuffer>,
    /// Callback installed after submit so a persistent upload slot is recycled only after GPU use.
    pub on_submitted_work_done: Option<Box<dyn FnOnce() + Send + 'static>>,
    /// Upload traffic and finish timing for diagnostics.
    pub stats: FrameUploadBatchStats,
}

pub(super) fn force_queue_fallback_stats(stats: &mut FrameUploadBatchStats) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_queue_fallback_clears_staging_stats() {
        let mut stats = FrameUploadBatchStats {
            writes: 3,
            bytes: 64,
            staged_writes: 3,
            staging_bytes: 64,
            copy_ops: 3,
            persistent_staging_bytes: 64,
            persistent_slot_reuses: 1,
            ..FrameUploadBatchStats::default()
        };

        force_queue_fallback_stats(&mut stats);

        assert_eq!(stats.fallback_writes, 3);
        assert_eq!(stats.staged_writes, 0);
        assert_eq!(stats.staging_bytes, 0);
        assert_eq!(stats.copy_ops, 0);
        assert_eq!(stats.persistent_staging_bytes, 0);
        assert_eq!(stats.persistent_slot_reuses, 0);
    }

    #[test]
    fn apply_arena_pressure_updates_slot_pressure_without_clearing_fallbacks() {
        let mut stats = FrameUploadBatchStats {
            fallback_writes: 2,
            temporary_staging_fallbacks: 3,
            oversized_queue_fallback_writes: 4,
            ..FrameUploadBatchStats::default()
        };

        stats.apply_arena_pressure(UploadArenaPressure {
            capacity_bytes: 1024,
            free_slots: 1,
            in_flight_slots: 2,
            remapping_slots: 3,
        });

        assert_eq!(stats.arena_capacity_bytes, 1024);
        assert_eq!(stats.arena_free_slots, 1);
        assert_eq!(stats.arena_in_flight_slots, 2);
        assert_eq!(stats.arena_remapping_slots, 3);
        assert_eq!(stats.fallback_writes, 2);
        assert_eq!(stats.temporary_staging_fallbacks, 3);
        assert_eq!(stats.oversized_queue_fallback_writes, 4);
    }
}
