//! Scope tracking for ordered replay of deferred queue writes.
//!
//! Scopes encode the executor location at which an upload was queued (pre-record, frame-global,
//! or per-view, with view and pass indices). A thread-local current-scope cell plus a per-scope
//! local sequence counter give every recorded write a deterministic sort key so the final replay
//! order is independent of which rayon worker won the upload-batch mutex first.

use std::cell::Cell;

thread_local! {
    pub(super) static CURRENT_UPLOAD_SCOPE: Cell<Option<FrameUploadScope>> = const { Cell::new(None) };
    pub(super) static CURRENT_UPLOAD_LOCAL_SEQ: Cell<u64> = const { Cell::new(0) };
}

/// Coarse executor phase used to replay frame uploads in a deterministic order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FrameUploadPhase {
    /// Pre-record resource preparation that runs before graph passes.
    PreRecord,
    /// Frame-global graph passes.
    FrameGlobal,
    /// Per-view graph passes.
    PerView,
}

/// Deterministic executor location for deferred queue writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FrameUploadScope {
    pub(super) phase: FrameUploadPhase,
    pub(super) view_idx: u32,
    pub(super) pass_idx: u32,
}

impl FrameUploadScope {
    /// Scope for pre-record resource preparation before any graph pass records.
    pub(crate) fn pre_record() -> Self {
        Self {
            phase: FrameUploadPhase::PreRecord,
            view_idx: 0,
            pass_idx: 0,
        }
    }

    /// Scope for pre-record resource preparation owned by one per-view worker.
    pub(crate) fn pre_record_view(view_idx: usize) -> Self {
        Self {
            phase: FrameUploadPhase::PreRecord,
            view_idx: saturating_u32(view_idx.saturating_add(1)),
            pass_idx: 0,
        }
    }

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
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Total replay key for one queued write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct QueueWriteOrder {
    pub(super) scope: FrameUploadScope,
    pub(super) local_seq: u64,
    pub(super) fallback_seq: u64,
}

/// Restores the prior thread-local upload scope on drop.
pub(crate) struct FrameUploadScopeGuard {
    pub(super) previous_scope: Option<FrameUploadScope>,
    pub(super) previous_local_seq: u64,
}

impl Drop for FrameUploadScopeGuard {
    fn drop(&mut self) {
        CURRENT_UPLOAD_SCOPE.with(|scope| scope.set(self.previous_scope));
        CURRENT_UPLOAD_LOCAL_SEQ.with(|seq| seq.set(self.previous_local_seq));
    }
}

#[cfg(test)]
mod tests {
    use super::FrameUploadScope;

    #[test]
    fn pre_record_view_scopes_sort_by_view_before_frame_global() {
        let mut scopes = [
            FrameUploadScope::frame_global(0),
            FrameUploadScope::pre_record_view(0),
            FrameUploadScope::pre_record_view(2),
            FrameUploadScope::pre_record(),
            FrameUploadScope::pre_record_view(1),
        ];

        scopes.sort_unstable();

        assert_eq!(scopes[0], FrameUploadScope::pre_record());
        assert_eq!(scopes[1], FrameUploadScope::pre_record_view(0));
        assert_eq!(scopes[2], FrameUploadScope::pre_record_view(1));
        assert_eq!(scopes[3], FrameUploadScope::pre_record_view(2));
        assert_eq!(scopes[4], FrameUploadScope::frame_global(0));
    }
}
