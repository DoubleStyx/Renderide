//! Staging-storage handles returned from `PersistentUploadArena::prepare_staging_buffer`.

use super::arena::PersistentUploadArena;
use crate::upload_stats::UploadArenaAcquireStats;

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
