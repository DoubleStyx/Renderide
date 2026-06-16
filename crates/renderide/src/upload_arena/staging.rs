//! Staging-storage handles returned from `PersistentUploadArena::prepare_staging_buffer`.

use super::arena::PersistentUploadArena;

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

/// Origin of the staging storage handed out by the arena for one drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UploadStagingSource {
    None,
    Persistent { slot: usize, generation: u64 },
    Temporary,
    QueueFallbackOversized,
}

/// Staging storage prepared for one upload drain.
pub(crate) struct PreparedUploadStaging {
    pub(super) buffer: Option<wgpu::Buffer>,
    pub(super) source: UploadStagingSource,
    pub(super) size: u64,
    pub(super) acquire_stats: UploadArenaAcquireStats,
}

impl PreparedUploadStaging {
    /// Buffer to fill while it is mapped.
    pub(crate) fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Stats for the acquisition path that produced this staging storage.
    pub(crate) fn acquire_stats(&self) -> UploadArenaAcquireStats {
        self.acquire_stats
    }

    /// Whether staged writes must be replayed through `Queue::write_buffer`.
    pub(crate) fn requires_queue_fallback(&self) -> bool {
        self.size > 0 && self.buffer.is_none()
    }

    /// Unmaps staging storage and returns the buffer/callback pair required by submit.
    pub(crate) fn finish(self, arena: &mut PersistentUploadArena) -> FinishedUploadStaging {
        match self.source {
            UploadStagingSource::None | UploadStagingSource::QueueFallbackOversized => {
                FinishedUploadStaging {
                    buffer: None,
                    on_submitted_work_done: None,
                }
            }
            UploadStagingSource::Temporary => {
                if let Some(buffer) = self.buffer.as_ref() {
                    buffer.unmap();
                }
                FinishedUploadStaging {
                    buffer: self.buffer,
                    on_submitted_work_done: None,
                }
            }
            UploadStagingSource::Persistent { slot, generation } => {
                let on_submitted_work_done = arena.finish_persistent_write(slot, generation);
                FinishedUploadStaging {
                    buffer: self.buffer,
                    on_submitted_work_done,
                }
            }
        }
    }
}

/// Finished staging storage ready for copy-command recording and submit callbacks.
pub(crate) struct FinishedUploadStaging {
    /// Buffer used as `COPY_SRC` for staged writes.
    pub(crate) buffer: Option<wgpu::Buffer>,
    /// Callback that marks a persistent slot submitted after GPU completion.
    pub(crate) on_submitted_work_done: Option<Box<dyn FnOnce() + Send + 'static>>,
}
