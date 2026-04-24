//! Error state shared between the driver thread and the main thread.
//!
//! The driver thread writes at most one error per failure; the main thread drains via
//! [`super::DriverThread::take_pending_error`] once per tick and routes the result
//! through the existing device-recovery path.

use std::sync::Mutex;

use thiserror::Error;

/// Kind of failure observed on the driver thread.
#[derive(Debug, Error)]
pub enum DriverErrorKind {
    /// `wgpu::Queue::submit` itself reported an error (reserved for future wgpu versions
    /// that return a `Result` from submit — current wgpu 29 submit is infallible).
    #[error("submit failed: {0}")]
    Submit(String),
    /// `SurfaceTexture::present` reported an error (reserved for future wgpu versions).
    #[error("present failed: {0}")]
    Present(String),
}

/// Full error payload including the frame sequence number so logs correlate cleanly.
#[derive(Debug, Error)]
#[error("driver error in frame {frame_seq}: {kind}")]
pub struct DriverError {
    /// Frame sequence the failure occurred on.
    pub frame_seq: u64,
    /// What went wrong on the driver thread.
    pub kind: DriverErrorKind,
}

/// Thread-safe slot for a pending [`DriverError`]. First-error-wins semantics: new errors
/// do not overwrite an existing unread entry. The main thread clears the slot via
/// [`Self::take`] once it has acted on the failure.
#[derive(Default)]
pub(super) struct DriverErrorState {
    slot: Mutex<Option<DriverError>>,
}

impl DriverErrorState {
    /// Records a failure unless one is already pending. Keeps the earliest failure so the
    /// main thread can correlate subsequent cascade failures without losing the root cause.
    #[expect(
        dead_code,
        reason = "wired once wgpu surfaces fallible submit/present results"
    )]
    pub(super) fn record(&self, err: DriverError) {
        let mut guard = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            *guard = Some(err);
        }
    }

    /// Drains and returns any pending error, leaving the slot empty.
    pub(super) fn take(&self) -> Option<DriverError> {
        let mut guard = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        guard.take()
    }
}
