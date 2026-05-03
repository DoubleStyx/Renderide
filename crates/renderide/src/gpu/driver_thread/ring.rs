//! Bounded FIFO ring shared between the main (producer) and driver (consumer) threads.
//!
//! A single [`std::sync::Mutex`] guards the deque. Two [`std::sync::Condvar`]s separate
//! wake-up reasons: `space_available` wakes the producer when the consumer drains a slot,
//! `message_ready` wakes the consumer when the producer pushes a slot. Capacity is set by
//! the caller (see [`RING_CAPACITY`] in [`super`]).
//!
//! The ring also tracks consumer liveness so [`BoundedRing::push`] cannot block forever
//! after the driver thread has exited (cleanly or via panic). The consumer-side guard
//! pattern in [`super::worker`] flips the flag and broadcasts on the producer condvar from
//! its `Drop`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// How often [`BoundedRing::push`] re-checks consumer liveness while waiting on a full
/// ring. Short enough that a dead consumer is detected within one frame at typical 60 Hz,
/// long enough that a live consumer is not woken needlessly.
const PUSH_LIVENESS_POLL: Duration = Duration::from_millis(16);

/// Bounded blocking FIFO queue tuned for one producer (main thread) and one consumer
/// (driver thread). Pushes block when full; pops block when empty.
pub(super) struct BoundedRing<T> {
    inner: Mutex<VecDeque<T>>,
    space_available: Condvar,
    message_ready: Condvar,
    capacity: usize,
    consumer_alive: AtomicBool,
}

impl<T> BoundedRing<T> {
    /// Creates an empty ring with the given maximum in-flight item count.
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity >= 1, "BoundedRing capacity must be at least 1");
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            space_available: Condvar::new(),
            message_ready: Condvar::new(),
            capacity,
            consumer_alive: AtomicBool::new(true),
        }
    }

    /// Marks the consumer side as no longer running and wakes any blocked producer.
    ///
    /// Called from the driver-thread guard in [`super::worker`] so that a clean exit or a
    /// panic both release the producer instead of letting it block on `space_available`.
    pub(super) fn mark_consumer_dead(&self) {
        self.consumer_alive.store(false, Ordering::Release);
        self.space_available.notify_all();
    }

    /// `true` while the consumer side is still draining the ring.
    pub(super) fn is_consumer_alive(&self) -> bool {
        self.consumer_alive.load(Ordering::Acquire)
    }

    /// Snapshot of how many items are currently queued behind the consumer.
    ///
    /// Used by [`super::DriverThread::submit`] to plot the ring depth into tracy so saturation
    /// (depth at capacity) versus idle catches the eye next to other gpu metrics. Acquires the
    /// inner mutex; this is fine because the function runs once per submit, not per per-frame
    /// hot-path operation. Gated on the `tracy` feature because the only call site is the
    /// matching tracy plot in [`super::DriverThread::submit`]; without it the method is dead
    /// code.
    #[cfg(feature = "tracy")]
    pub(super) fn depth(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Pushes `item` into the ring, blocking the caller while the ring is full.
    ///
    /// Returns `Err(item)` if the consumer side has been marked dead while we were
    /// waiting for space -- the caller can then surface this through the existing driver
    /// error state instead of blocking the main render thread forever. While the ring has
    /// space the call always succeeds, so callers under normal operation never observe
    /// `Err`.
    ///
    /// On a poisoned mutex the contents are recovered via `into_inner`, matching the rest
    /// of the project's "log and keep going" policy -- driver failures surface via the
    /// separate error state rather than thread poisoning.
    pub(super) fn push(&self, item: T) -> Result<(), T> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while guard.len() >= self.capacity {
            if !self.is_consumer_alive() {
                return Err(item);
            }
            let (g, _timeout) = self
                .space_available
                .wait_timeout(guard, PUSH_LIVENESS_POLL)
                .unwrap_or_else(|p| {
                    let inner = p.into_inner();
                    (inner.0, inner.1)
                });
            guard = g;
        }
        guard.push_back(item);
        drop(guard);
        self.message_ready.notify_one();
        Ok(())
    }

    /// Pops the next item, blocking the caller while the ring is empty.
    pub(super) fn pop(&self) -> T {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(item) = guard.pop_front() {
                drop(guard);
                self.space_available.notify_one();
                return item;
            }
            guard = self
                .message_ready
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}
