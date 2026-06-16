//! Shared upload traffic and arena statistics.

/// Stats captured while acquiring staging storage for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UploadArenaAcquireStats {
    /// Bytes staged through a persistent slot this frame.
    pub(crate) persistent_staging_bytes: u64,
    /// Persistent slot reuse count.
    pub(crate) persistent_slot_reuses: usize,
    /// Persistent slot allocation or growth count.
    pub(crate) persistent_slot_grows: usize,
    /// Bytes staged through a one-frame temporary fallback buffer.
    pub(crate) temporary_staging_bytes: u64,
    /// Temporary fallback count caused by all persistent slots being unavailable.
    pub(crate) temporary_staging_fallbacks: usize,
    /// Staged writes replayed through `Queue::write_buffer` because no staging buffer could fit.
    pub(crate) oversized_queue_fallback_writes: usize,
}

/// Current persistent arena pressure after an upload drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UploadArenaPressure {
    /// Total bytes currently allocated across persistent slots.
    pub(crate) capacity_bytes: u64,
    /// Persistent slots that are mapped and free.
    pub(crate) free_slots: usize,
    /// Persistent slots referenced by submitted GPU work.
    pub(crate) in_flight_slots: usize,
    /// Persistent slots waiting for `map_async` completion.
    pub(crate) remapping_slots: usize,
}

/// Deferred upload traffic counters shared by frame and mesh upload drains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadTrafficStats {
    /// Number of queued buffer writes drained.
    pub writes: usize,
    /// Total payload bytes drained.
    pub bytes: usize,
    /// Writes served by staging-buffer copy commands.
    pub staged_writes: usize,
    /// Writes replayed through queue writes.
    pub fallback_writes: usize,
    /// Required staging bytes for aligned writes.
    pub staging_bytes: u64,
    /// Number of copy commands recorded.
    pub copy_ops: usize,
}

impl UploadTrafficStats {
    /// Marks every queued write as a queue fallback and clears copy-path counters.
    pub fn force_queue_fallback(&mut self) {
        self.fallback_writes = self.writes;
        self.staged_writes = 0;
        self.staging_bytes = 0;
        self.copy_ops = 0;
    }
}

/// Persistent upload arena acquire and pressure counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadArenaStats {
    /// Counters captured while acquiring staging storage for the drain.
    pub acquire: UploadArenaAcquireStats,
    /// Persistent arena pressure after the drain.
    pub pressure: UploadArenaPressure,
}

impl UploadArenaStats {
    /// Replaces acquire-path counters with the latest staging acquisition result.
    pub fn apply_acquire(&mut self, stats: UploadArenaAcquireStats) {
        self.acquire = stats;
    }

    /// Replaces pressure counters with the latest persistent arena state.
    pub fn apply_pressure(&mut self, pressure: UploadArenaPressure) {
        self.pressure = pressure;
    }

    /// Clears acquire-path counters when the drain did not use staging storage.
    pub fn clear_acquire(&mut self) {
        self.acquire = UploadArenaAcquireStats::default();
    }
}
