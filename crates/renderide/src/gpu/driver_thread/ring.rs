//! Bounded FIFO ring shared between the main (producer) and driver (consumer) threads.
//!
//! A single [`std::sync::Mutex`] guards the deque. Two [`std::sync::Condvar`]s separate
//! wake-up reasons: `space_available` wakes the producer when the consumer drains a slot,
//! `message_ready` wakes the consumer when the producer pushes a slot. Capacity is set by
//! the caller (see [`RING_CAPACITY`] in [`super`]).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// Bounded blocking FIFO queue tuned for one producer (main thread) and one consumer
/// (driver thread). Pushes block when full; pops block when empty.
pub(super) struct BoundedRing<T> {
    inner: Mutex<VecDeque<T>>,
    space_available: Condvar,
    message_ready: Condvar,
    capacity: usize,
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
        }
    }

    /// Pushes `item` into the ring, blocking the caller while the ring is full.
    ///
    /// On a poisoned mutex the contents are recovered via `into_inner`, matching the rest
    /// of the project's "log and keep going" policy — driver failures surface via the
    /// separate error state rather than thread poisoning.
    pub(super) fn push(&self, item: T) {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        while guard.len() >= self.capacity {
            guard = self
                .space_available
                .wait(guard)
                .unwrap_or_else(|p| p.into_inner());
        }
        guard.push_back(item);
        drop(guard);
        self.message_ready.notify_one();
    }

    /// Pops the next item, blocking the caller while the ring is empty.
    pub(super) fn pop(&self) -> T {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(item) = guard.pop_front() {
                drop(guard);
                self.space_available.notify_one();
                return item;
            }
            guard = self
                .message_ready
                .wait(guard)
                .unwrap_or_else(|p| p.into_inner());
        }
    }
}
