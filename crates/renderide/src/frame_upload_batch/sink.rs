//! Pass-facing `GraphUploadSink` that funnels deferred writes into the owning batch.

use super::batch::FrameUploadBatch;
use super::scope::FrameUploadScope;

/// Graph-time buffer upload recorder scoped to the current executor location.
///
/// The sink carries an explicit [`FrameUploadScope`] so uploads emitted by nested rayon workers
/// remain ordered inside the owning graph phase even though thread-local pass scope does not
/// propagate to those workers.
#[derive(Clone, Copy)]
pub struct GraphUploadSink<'a> {
    pub(super) batch: &'a FrameUploadBatch,
    pub(super) scope: FrameUploadScope,
}

impl<'a> GraphUploadSink<'a> {
    /// Creates a sink for `scope` backed by `batch`.
    pub(crate) fn new(batch: &'a FrameUploadBatch, scope: FrameUploadScope) -> Self {
        Self { batch, scope }
    }

    /// Creates a sink for pre-record resource preparation.
    pub(crate) fn pre_record(batch: &'a FrameUploadBatch) -> Self {
        Self::new(batch, FrameUploadScope::pre_record())
    }

    /// Creates a sink for pre-record resource preparation owned by one view index.
    pub(crate) fn pre_record_view(batch: &'a FrameUploadBatch, view_idx: usize) -> Self {
        Self::new(batch, FrameUploadScope::pre_record_view(view_idx))
    }

    /// Queues `queue.write_buffer(buffer, offset, data)` for ordered replay before submit.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.batch
            .write_buffer_with_scope_fallback(self.scope, buffer, offset, data);
    }
}
