//! Deferred-upload traffic statistics and the recorded command buffer produced by a drain.

use crate::upload_arena::{UploadArenaAcquireStats, UploadArenaPressure};
use crate::upload_stats::{UploadArenaStats, UploadTrafficStats};

/// Deferred-upload traffic drained into the frame submit batch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameUploadBatchStats {
    /// Queue-write and staging-copy traffic.
    pub traffic: UploadTrafficStats,
    /// Persistent upload arena acquire and pressure counters.
    pub arena: UploadArenaStats,
    /// CPU time spent inside the upload encoder [`wgpu::CommandEncoder::finish`] call.
    pub finish_ms: f64,
}

impl FrameUploadBatchStats {
    pub(super) fn apply_arena_acquire(&mut self, stats: UploadArenaAcquireStats) {
        self.arena.apply_acquire(stats);
    }

    pub(crate) fn apply_arena_pressure(&mut self, pressure: UploadArenaPressure) {
        self.arena.apply_pressure(pressure);
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
    stats.traffic.force_queue_fallback();
    stats.arena.clear_acquire();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_queue_fallback_clears_staging_stats() {
        let mut stats = FrameUploadBatchStats {
            traffic: UploadTrafficStats {
                writes: 3,
                bytes: 64,
                staged_writes: 3,
                staging_bytes: 64,
                copy_ops: 3,
                ..UploadTrafficStats::default()
            },
            arena: UploadArenaStats {
                acquire: UploadArenaAcquireStats {
                    persistent_staging_bytes: 64,
                    persistent_slot_reuses: 1,
                    ..UploadArenaAcquireStats::default()
                },
                ..UploadArenaStats::default()
            },
            ..FrameUploadBatchStats::default()
        };

        force_queue_fallback_stats(&mut stats);

        assert_eq!(stats.traffic.fallback_writes, 3);
        assert_eq!(stats.traffic.staged_writes, 0);
        assert_eq!(stats.traffic.staging_bytes, 0);
        assert_eq!(stats.traffic.copy_ops, 0);
        assert_eq!(stats.arena.acquire.persistent_staging_bytes, 0);
        assert_eq!(stats.arena.acquire.persistent_slot_reuses, 0);
    }

    #[test]
    fn apply_arena_pressure_updates_slot_pressure_without_clearing_fallbacks() {
        let mut stats = FrameUploadBatchStats {
            traffic: UploadTrafficStats {
                fallback_writes: 2,
                ..UploadTrafficStats::default()
            },
            arena: UploadArenaStats {
                acquire: UploadArenaAcquireStats {
                    temporary_staging_fallbacks: 3,
                    oversized_queue_fallback_writes: 4,
                    ..UploadArenaAcquireStats::default()
                },
                ..UploadArenaStats::default()
            },
            ..FrameUploadBatchStats::default()
        };

        stats.apply_arena_pressure(UploadArenaPressure {
            capacity_bytes: 1024,
            free_slots: 1,
            in_flight_slots: 2,
            remapping_slots: 3,
        });

        assert_eq!(stats.arena.pressure.capacity_bytes, 1024);
        assert_eq!(stats.arena.pressure.free_slots, 1);
        assert_eq!(stats.arena.pressure.in_flight_slots, 2);
        assert_eq!(stats.arena.pressure.remapping_slots, 3);
        assert_eq!(stats.traffic.fallback_writes, 2);
        assert_eq!(stats.arena.acquire.temporary_staging_fallbacks, 3);
        assert_eq!(stats.arena.acquire.oversized_queue_fallback_writes, 4);
    }
}
