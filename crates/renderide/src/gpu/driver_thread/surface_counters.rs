//! Producer/consumer counters for surface-texture pipelining.
//!
//! The renderer relies on wgpu's invariant that only one [`wgpu::SurfaceTexture`] may be
//! outstanding at a time: the previous frame's [`wgpu::SurfaceTexture::present`] must run
//! before the next [`wgpu::Surface::get_current_texture`]. Phase 2 moved `present()` to
//! the driver thread, so the main thread needs a fine-grained wait -- not a full FIFO flush --
//! to block only on the specific event "previous surface texture has been presented".
//!
//! This module provides that primitive: [`SurfaceCounters`] tracks the number of
//! surface-carrying batches submitted by the main thread and the number actually presented
//! by the driver thread, plus a [`Condvar`] signaled on each present so the main thread
//! wakes as soon as the gap closes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

/// Monotonic producer/consumer counters with a condvar for present-gated wait.
///
/// `frames_submitted` is incremented by [`SurfaceCounters::note_submitted`] on the main
/// thread when a surface-carrying batch is pushed to the driver ring.
/// `frames_presented` is incremented by [`SurfaceCounters::note_presented`] on the driver
/// thread after [`wgpu::SurfaceTexture::present`] returns.
///
/// The gap `frames_submitted - frames_presented` equals the number of surface-carrying
/// batches currently in-flight on the driver (queued or being processed).
pub(super) struct SurfaceCounters {
    /// Count of surface-carrying batches pushed to the driver ring by the main thread.
    frames_submitted: AtomicU64,
    /// Count of surface-carrying batches for which the driver has completed `present()`.
    frames_presented: AtomicU64,
    /// Mutex guarding only the condvar wait slot; the counters themselves are atomics.
    present_mtx: Mutex<()>,
    /// Signaled by the driver thread after each successful present.
    present_cvar: Condvar,
}

impl Default for SurfaceCounters {
    fn default() -> Self {
        Self {
            frames_submitted: AtomicU64::new(0),
            frames_presented: AtomicU64::new(0),
            present_mtx: Mutex::new(()),
            present_cvar: Condvar::new(),
        }
    }
}

impl SurfaceCounters {
    /// Records that a surface-carrying batch has been pushed to the driver ring.
    pub(super) fn note_submitted(&self) {
        self.frames_submitted.fetch_add(1, Ordering::Release);
    }

    /// Records that the driver thread has completed `present()` on a surface texture.
    /// Wakes any main-thread waiter blocked in [`Self::wait_for_present_catchup`].
    pub(super) fn note_presented(&self) {
        self.frames_presented.fetch_add(1, Ordering::Release);
        // `notify_all` rather than `notify_one` keeps the primitive correct even if more
        // than one caller ever waits (not expected today, but cheap insurance).
        let _g = self
            .present_mtx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.present_cvar.notify_all();
    }

    /// Blocks until the number of in-flight surface-carrying batches is `<= max_in_flight`.
    ///
    /// With `max_in_flight == 0`, this waits until every previously-submitted surface
    /// texture has been presented -- the precise barrier needed by the main thread before
    /// it calls [`wgpu::Surface::get_current_texture`]. Using a larger value would require
    /// wgpu to support multiple outstanding surface textures, which it does not today.
    ///
    /// The predicate is checked **while holding** [`Self::present_mtx`]. Checking outside
    /// the mutex would race with [`Self::note_presented`]: a notifier can run the full
    /// `fetch_add` then `lock` then `notify_all` sequence in the gap between the reader's
    /// atomic load and its `wait` call, leaving the reader parked on a condvar no one
    /// will signal again (lost wakeup). Holding the mutex across both the load and the
    /// `wait` makes the two atomic with respect to the notifier -- [`std::sync::Condvar::wait`]
    /// releases the same mutex the notifier must acquire before `notify_all`. Do not
    /// hoist the lock back outside the loop.
    pub(super) fn wait_for_present_catchup(&self, max_in_flight: u64) {
        let mut guard = self
            .present_mtx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            let submitted = self.frames_submitted.load(Ordering::Acquire);
            let presented = self.frames_presented.load(Ordering::Acquire);
            if submitted.saturating_sub(presented) <= max_in_flight {
                return;
            }
            guard = self
                .present_cvar
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SurfaceCounters;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn wait_for_catchup_returns_immediately_when_no_work() {
        let counters = SurfaceCounters::default();
        // No submissions made; presented == submitted == 0, so wait returns at once.
        counters.wait_for_present_catchup(0);
    }

    #[test]
    fn wait_blocks_until_present_notifies() {
        let counters = Arc::new(SurfaceCounters::default());
        counters.note_submitted();
        counters.note_submitted();
        let c2 = counters.clone();
        let worker = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            c2.note_presented();
            c2.note_presented();
        });
        counters.wait_for_present_catchup(0);
        worker.join().expect("worker joined cleanly");
    }

    #[test]
    fn cap_of_one_allows_one_frame_in_flight() {
        let counters = SurfaceCounters::default();
        counters.note_submitted();
        // One in-flight; with a cap of 1, wait should return immediately.
        counters.wait_for_present_catchup(1);
    }
}
