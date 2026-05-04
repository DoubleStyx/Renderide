//! Producer/consumer counters for driver-thread submit backlog observability.
//!
//! Mirrors [`super::surface_counters::SurfaceCounters`] in shape but tracks **all** batches
//! pushed to the driver ring (not just surface-carrying ones). The instantaneous gap
//! `submits_pushed - submits_done` is the number of `Queue::submit` calls the driver
//! still owes the producer, surfaced via a Tracy plot so a regression in pipelining
//! shows up in the same trace as a regression in frame timing.
//!
//! Unlike `SurfaceCounters` this primitive does not gate any wait paths today; it is
//! purely diagnostic. The shared shape keeps the door open if a future caller wants to
//! wait on a specific submit token (e.g. an XR-finalize-only batch that needs ordering
//! relative to a render submit) without re-introducing the full-ring `flush_driver`
//! pattern that motivated this work.

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic token assigned to one batch pushed onto the renderer driver thread.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SubmitToken(u64);

impl SubmitToken {
    /// Builds a token from its raw monotonic value.
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw monotonic token value.
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Monotonic producer/consumer counters for driver-thread submit observability.
///
/// `submits_pushed` is incremented by [`Self::note_pushed`] on the main thread when a
/// batch (any kind, including the OpenXR-finalize-only and headless flush sentinels)
/// is enqueued. `submits_done` is incremented by [`Self::note_submit_done`] on the
/// driver thread immediately after [`wgpu::Queue::submit`] returns.
pub(super) struct SubmitCounters {
    /// Count of batches pushed to the driver ring by the main thread.
    submits_pushed: AtomicU64,
    /// Count of batches whose `Queue::submit` has returned on the driver thread.
    submits_done: AtomicU64,
}

impl Default for SubmitCounters {
    fn default() -> Self {
        Self {
            submits_pushed: AtomicU64::new(0),
            submits_done: AtomicU64::new(0),
        }
    }
}

impl SubmitCounters {
    /// Records that a batch has been pushed to the driver ring and returns its submit token.
    pub(super) fn note_pushed(&self) -> SubmitToken {
        SubmitToken::new(self.submits_pushed.fetch_add(1, Ordering::AcqRel) + 1)
    }

    /// Records that a batch's `Queue::submit` has returned. Driver-thread call site only.
    pub(super) fn note_submit_done(&self) {
        self.submits_done.fetch_add(1, Ordering::Release);
    }

    /// Snapshot of the (pushed, done) pair, suitable for computing a backlog plot.
    pub(super) fn snapshot(&self) -> (u64, u64) {
        let pushed = self.submits_pushed.load(Ordering::Acquire);
        let done = self.submits_done.load(Ordering::Acquire);
        (pushed, done)
    }

    /// Returns `true` once the driver thread has returned from `Queue::submit` for `token`.
    pub(super) fn is_submit_done(&self, token: SubmitToken) -> bool {
        self.submits_done.load(Ordering::Acquire) >= token.raw()
    }
}

#[cfg(test)]
mod tests {
    use super::SubmitCounters;

    #[test]
    fn snapshot_starts_at_zero() {
        let c = SubmitCounters::default();
        assert_eq!(c.snapshot(), (0, 0));
    }

    #[test]
    fn note_pushed_returns_monotonic_token_and_advances_snapshot() {
        let c = SubmitCounters::default();
        assert_eq!(c.note_pushed().raw(), 1);
        assert_eq!(c.note_pushed().raw(), 2);
        assert_eq!(c.snapshot(), (2, 0));
        c.note_submit_done();
        assert_eq!(c.snapshot(), (2, 1));
    }

    #[test]
    fn submit_done_tracks_tokens_individually() {
        let c = SubmitCounters::default();
        let first = c.note_pushed();
        let second = c.note_pushed();
        assert!(!c.is_submit_done(first));
        assert!(!c.is_submit_done(second));
        c.note_submit_done();
        assert!(c.is_submit_done(first));
        assert!(!c.is_submit_done(second));
        c.note_submit_done();
        assert!(c.is_submit_done(second));
    }

    #[test]
    fn backlog_returns_to_zero_when_consumer_catches_up() {
        let c = SubmitCounters::default();
        c.note_pushed();
        c.note_pushed();
        c.note_submit_done();
        c.note_submit_done();
        let (pushed, done) = c.snapshot();
        assert_eq!(pushed.saturating_sub(done), 0);
    }
}
