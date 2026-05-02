//! Heartbeat slots and the [`Heartbeat`] / [`WatchdogPause`] handles handed to watched threads.
//!
//! Pet is a single relaxed atomic store of the calling thread's monotonic offset (~5 ns); pause
//! is one atomic increment / decrement on a counter the watchdog thread checks before reporting
//! hitches or hangs.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Sentinel for "we have not yet reported a hitch/hang for the current pet value".
///
/// `last_pet_ns` is monotonically increasing nanoseconds since the slot's epoch (created at
/// registration), so a real pet can never produce `0`. Using `0` as the "never reported"
/// marker means the very first stall on a freshly registered slot will be reportable.
const NEVER_REPORTED: u64 = 0;

/// Per-thread watchdog state. Owned by an [`Arc`] shared between the watched thread (via
/// [`Heartbeat`] / [`WatchdogPause`]) and the watchdog poll thread (via the registry snapshot).
pub(super) struct HeartbeatSlot {
    /// Static name for log output (e.g. `"main"`). Lifetime-`'static` keeps the slot trivially
    /// `Send + Sync` without owning a `String`.
    pub(super) name: &'static str,
    /// Platform-erased OS thread identity used by the signal handler's "is this me?" check on
    /// POSIX. On Linux this is the `gettid()` value; on macOS it is the `pthread_threadid_np`
    /// value. Stored as `i64` so both fit in the same field. Captured on Windows too (currently
    /// `0`) but unread there until the Windows stack-capture path lands.
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_os = "macos")),
        expect(
            dead_code,
            reason = "read by the POSIX signal/stack-capture path; unused on e.g. Windows until that path exists"
        )
    )]
    pub(super) os_tid: i64,
    /// macOS `pthread_t` encoded as `usize` so the watchdog thread can call `pthread_kill` to
    /// deliver `SIGUSR2`. `0` on platforms where `pthread_kill` capture is not used.
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_os = "macos")),
        expect(
            dead_code,
            reason = "read by the POSIX `pthread_kill` stack-capture path; unused on e.g. Windows until that path exists"
        )
    )]
    pub(super) pthread_handle: usize,
    /// Wall-clock anchor for [`Self::now_ns`]. Never changes after registration.
    epoch: Instant,
    /// Nanoseconds since [`Self::epoch`] of the most recent [`Heartbeat::pet`]. Initialized to
    /// `1` (not `0`) so the watchdog's first poll sees a fresh pet rather than treating the
    /// slot as instantly stale; `1 ns` is well below any realistic poll interval.
    pub(super) last_pet_ns: AtomicU64,
    /// Hitch threshold for this slot, in nanoseconds. `0` disables hitch reporting.
    pub(super) hitch_ns: u64,
    /// Hang threshold for this slot, in nanoseconds. `0` disables hang reporting.
    pub(super) hang_ns: u64,
    /// When `> 0`, the watchdog ignores this slot. Incremented by [`Heartbeat::pause`] / the
    /// [`WatchdogPause`] guard so legitimate long stalls (XR `wait_frame`, shader compile,
    /// swapchain reconfigure) do not produce false-positive reports.
    pub(super) suspend_count: AtomicU32,
    /// Snapshot of [`Self::last_pet_ns`] at the time the most recent hang was reported. The
    /// watchdog suppresses repeat hang reports until [`Self::last_pet_ns`] advances past this
    /// value (i.e. a fresh pet has happened), so one stuck event produces exactly one report.
    pub(super) hang_reported_at_pet: AtomicU64,
    /// Same dedup as [`Self::hang_reported_at_pet`] but for hitch reports.
    pub(super) hitch_reported_at_pet: AtomicU64,
}

impl HeartbeatSlot {
    /// Returns nanoseconds since [`Self::epoch`].
    pub(super) fn now_ns(&self) -> u64 {
        // u64 nanoseconds wraps after ~584 years of process uptime; ignored.
        self.epoch.elapsed().as_nanos() as u64
    }
}

/// Process-wide registry of heartbeat slots, owned by the watchdog and shared with watched
/// threads through [`Arc<HeartbeatSlot>`].
pub(super) struct HeartbeatRegistry {
    slots: RwLock<Vec<Arc<HeartbeatSlot>>>,
}

impl HeartbeatRegistry {
    /// New empty registry.
    pub(super) fn new() -> Self {
        Self {
            slots: RwLock::new(Vec::new()),
        }
    }

    /// Register a new slot for the given thread identity and thresholds.
    pub(super) fn register(
        &self,
        name: &'static str,
        os_tid: i64,
        pthread_handle: usize,
        hitch: Duration,
        hang: Duration,
    ) -> Arc<HeartbeatSlot> {
        let slot = Arc::new(HeartbeatSlot {
            name,
            os_tid,
            pthread_handle,
            epoch: Instant::now(),
            last_pet_ns: AtomicU64::new(1),
            hitch_ns: hitch.as_nanos() as u64,
            hang_ns: hang.as_nanos() as u64,
            suspend_count: AtomicU32::new(0),
            hang_reported_at_pet: AtomicU64::new(NEVER_REPORTED),
            hitch_reported_at_pet: AtomicU64::new(NEVER_REPORTED),
        });
        self.slots.write().push(Arc::clone(&slot));
        slot
    }

    /// Snapshot the current set of registered slots for one watchdog poll iteration.
    pub(super) fn snapshot(&self) -> Vec<Arc<HeartbeatSlot>> {
        self.slots.read().iter().map(Arc::clone).collect()
    }
}

/// Handle returned to a watched thread. Cheap [`Self::pet`] (one relaxed atomic store).
///
/// Heartbeats are not [`Clone`]: each watched thread owns its own handle. To gate a region as a
/// legitimate long stall, call [`Self::pause`] and hold the resulting [`WatchdogPause`] guard.
pub struct Heartbeat {
    slot: Arc<HeartbeatSlot>,
}

impl Heartbeat {
    /// Wrap an [`Arc<HeartbeatSlot>`] in a public handle. Crate-internal: only the watchdog
    /// constructs heartbeats.
    pub(super) fn from_slot(slot: Arc<HeartbeatSlot>) -> Self {
        Self { slot }
    }

    /// Update the liveness signal. Call once per loop iteration on the watched thread; ~5 ns,
    /// safe to invoke every frame.
    #[inline]
    pub fn pet(&self) {
        let now = self.slot.now_ns();
        // `Release` so the watchdog's `Acquire` load on the same atomic sees a fresh value
        // ordered after any work the watched thread did before petting.
        self.slot.last_pet_ns.store(now, Ordering::Release);
    }

    /// Slot name (debug / log).
    pub fn name(&self) -> &'static str {
        self.slot.name
    }

    /// Begin a region in which the watchdog should ignore this slot. Returns a guard whose
    /// [`Drop`] re-enables reporting. Nest freely -- the suspend count is reference-counted.
    pub fn pause(&self) -> WatchdogPause {
        self.slot.suspend_count.fetch_add(1, Ordering::AcqRel);
        WatchdogPause {
            slot: Arc::clone(&self.slot),
        }
    }
}

/// RAII guard suppressing watchdog hitch / hang reports on a single heartbeat slot for its
/// lifetime. Wrap legitimate long stalls (initial pipeline compile, `xrWaitFrame`, bulk asset
/// upload bursts) so they do not trip the deadline.
#[must_use = "WatchdogPause re-enables reporting when dropped -- bind it to a `_pause` local"]
pub struct WatchdogPause {
    slot: Arc<HeartbeatSlot>,
}

impl Drop for WatchdogPause {
    fn drop(&mut self) {
        self.slot.suspend_count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Outcome of inspecting one slot during a watchdog poll iteration.
///
/// Pure data, returned from [`evaluate_slot`] so the policy logic can be unit-tested without
/// spinning up a real watchdog thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SlotEvaluation {
    /// Slot is healthy or paused -- emit nothing.
    Quiet,
    /// Heartbeat is older than the hitch threshold but younger than the hang threshold; this
    /// is the first time we've observed this stall (dedup against [`HeartbeatSlot::hitch_reported_at_pet`]).
    /// Caller should log a `warn` line and call [`record_hitch_reported`] to dedup.
    Hitch {
        /// Elapsed nanoseconds since the last pet at observation time.
        elapsed_ns: u64,
        /// Pet value at observation time; pass back to [`record_hitch_reported`].
        pet_value: u64,
    },
    /// Heartbeat is older than the hang threshold; first observation. Caller should capture
    /// stacks, emit an `error` line, then [`record_hang_reported`].
    Hang {
        /// Elapsed nanoseconds since the last pet at observation time.
        elapsed_ns: u64,
        /// Pet value at observation time; pass back to [`record_hang_reported`].
        pet_value: u64,
    },
}

/// Decide what (if anything) to report for `slot` at the given absolute time `now_ns` on the
/// slot's epoch. Pure logic -- no I/O -- so the dedup behaviour is unit-testable.
pub(super) fn evaluate_slot(slot: &HeartbeatSlot, now_ns: u64) -> SlotEvaluation {
    if slot.suspend_count.load(Ordering::Acquire) > 0 {
        return SlotEvaluation::Quiet;
    }
    let last_pet = slot.last_pet_ns.load(Ordering::Acquire);
    let elapsed = now_ns.saturating_sub(last_pet);

    if slot.hang_ns > 0 && elapsed >= slot.hang_ns {
        let already = slot.hang_reported_at_pet.load(Ordering::Acquire);
        if already != last_pet {
            return SlotEvaluation::Hang {
                elapsed_ns: elapsed,
                pet_value: last_pet,
            };
        }
    } else if slot.hitch_ns > 0 && elapsed >= slot.hitch_ns {
        let already = slot.hitch_reported_at_pet.load(Ordering::Acquire);
        if already != last_pet {
            return SlotEvaluation::Hitch {
                elapsed_ns: elapsed,
                pet_value: last_pet,
            };
        }
    }
    SlotEvaluation::Quiet
}

/// Mark a hitch as reported for `pet_value` so [`evaluate_slot`] returns [`SlotEvaluation::Quiet`]
/// until the watched thread pets the heartbeat again.
pub(super) fn record_hitch_reported(slot: &HeartbeatSlot, pet_value: u64) {
    slot.hitch_reported_at_pet
        .store(pet_value, Ordering::Release);
}

/// Mark a hang as reported for `pet_value`. Same dedup rationale as [`record_hitch_reported`].
pub(super) fn record_hang_reported(slot: &HeartbeatSlot, pet_value: u64) {
    slot.hang_reported_at_pet
        .store(pet_value, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slot(hitch: Duration, hang: Duration) -> Arc<HeartbeatSlot> {
        let registry = HeartbeatRegistry::new();
        registry.register("test", 0, 0, hitch, hang)
    }

    #[test]
    fn pet_advances_last_pet_ns_strictly() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        let h = Heartbeat::from_slot(Arc::clone(&slot));
        let before = slot.last_pet_ns.load(Ordering::Acquire);
        std::thread::sleep(Duration::from_millis(2));
        h.pet();
        let after = slot.last_pet_ns.load(Ordering::Acquire);
        assert!(
            after > before,
            "pet must move last_pet_ns forward (before={before}, after={after})"
        );
    }

    #[test]
    fn pause_increments_and_drop_decrements_suspend_count() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        let h = Heartbeat::from_slot(Arc::clone(&slot));
        assert_eq!(slot.suspend_count.load(Ordering::Acquire), 0);
        let p1 = h.pause();
        assert_eq!(slot.suspend_count.load(Ordering::Acquire), 1);
        let p2 = h.pause();
        assert_eq!(slot.suspend_count.load(Ordering::Acquire), 2);
        drop(p1);
        assert_eq!(slot.suspend_count.load(Ordering::Acquire), 1);
        drop(p2);
        assert_eq!(slot.suspend_count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn registry_snapshot_returns_all_registered_slots() {
        let r = HeartbeatRegistry::new();
        let _a = r.register("a", 1, 0, Duration::from_millis(50), Duration::from_secs(5));
        let _b = r.register("b", 2, 0, Duration::from_millis(50), Duration::from_secs(5));
        let _c = r.register("c", 3, 0, Duration::from_millis(50), Duration::from_secs(5));
        let snap = r.snapshot();
        assert_eq!(snap.len(), 3);
        let names: Vec<&str> = snap.iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn evaluate_quiet_when_pet_is_fresh() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        // Pet at time 0; observe at time 1ms -- well under hitch threshold.
        slot.last_pet_ns.store(1_000_000, Ordering::Release);
        let eval = evaluate_slot(&slot, 2_000_000);
        assert_eq!(eval, SlotEvaluation::Quiet);
    }

    #[test]
    fn evaluate_hitch_when_elapsed_crosses_hitch_threshold() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        slot.last_pet_ns.store(1_000_000, Ordering::Release);
        // Now is 1ms + 150ms = 151ms -- hitch territory but well under 10s hang threshold.
        let now = 1_000_000 + 150 * 1_000_000;
        let eval = evaluate_slot(&slot, now);
        match eval {
            SlotEvaluation::Hitch { pet_value, .. } => assert_eq!(pet_value, 1_000_000),
            other => panic!("expected Hitch, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_hang_when_elapsed_crosses_hang_threshold() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        slot.last_pet_ns.store(1_000_000, Ordering::Release);
        let now = 1_000_000 + 11 * 1_000_000_000;
        let eval = evaluate_slot(&slot, now);
        match eval {
            SlotEvaluation::Hang { pet_value, .. } => assert_eq!(pet_value, 1_000_000),
            other => panic!("expected Hang, got {other:?}"),
        }
    }

    #[test]
    fn evaluate_quiet_when_paused_even_past_hang_threshold() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        let h = Heartbeat::from_slot(Arc::clone(&slot));
        let _pause = h.pause();
        slot.last_pet_ns.store(1, Ordering::Release);
        let eval = evaluate_slot(&slot, 30 * 1_000_000_000);
        assert_eq!(eval, SlotEvaluation::Quiet);
    }

    #[test]
    fn hitch_dedup_suppresses_repeat_until_fresh_pet() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        slot.last_pet_ns.store(1_000_000, Ordering::Release);
        let now = 1_000_000 + 150 * 1_000_000;
        let first = evaluate_slot(&slot, now);
        match first {
            SlotEvaluation::Hitch { pet_value, .. } => record_hitch_reported(&slot, pet_value),
            other => panic!("expected Hitch on first poll, got {other:?}"),
        }
        // Second observation at the same pet value: deduped.
        let second = evaluate_slot(&slot, now + 5 * 1_000_000);
        assert_eq!(second, SlotEvaluation::Quiet);

        // Fresh pet -> the next stall is reportable again.
        slot.last_pet_ns
            .store(now + 6 * 1_000_000, Ordering::Release);
        let third = evaluate_slot(&slot, now + 6 * 1_000_000 + 150 * 1_000_000);
        assert!(
            matches!(third, SlotEvaluation::Hitch { .. }),
            "expected re-armed Hitch after fresh pet, got {third:?}"
        );
    }

    #[test]
    fn hang_dedup_suppresses_repeat_until_fresh_pet() {
        let slot = make_slot(Duration::from_millis(100), Duration::from_secs(10));
        slot.last_pet_ns.store(1_000_000, Ordering::Release);
        let now = 1_000_000 + 11 * 1_000_000_000;
        let first = evaluate_slot(&slot, now);
        match first {
            SlotEvaluation::Hang { pet_value, .. } => record_hang_reported(&slot, pet_value),
            other => panic!("expected Hang, got {other:?}"),
        }
        let second = evaluate_slot(&slot, now + 5 * 1_000_000_000);
        assert_eq!(second, SlotEvaluation::Quiet);
    }

    #[test]
    fn evaluate_quiet_when_thresholds_zero() {
        let slot = make_slot(Duration::ZERO, Duration::ZERO);
        slot.last_pet_ns.store(1_000_000, Ordering::Release);
        let eval = evaluate_slot(&slot, 1_000_000 + 60 * 1_000_000_000);
        assert_eq!(eval, SlotEvaluation::Quiet);
    }
}
