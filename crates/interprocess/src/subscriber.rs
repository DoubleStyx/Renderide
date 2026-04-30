//! Consumer side of the shared-memory queue.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::layout::QueueHeader;
use crate::layout::{
    MESSAGE_BODY_OFFSET, STATE_LOCKED, STATE_READY, TICKS_FOR_TEN_SECONDS, padded_message_length,
};
use crate::options::QueueOptions;
use crate::queue_resources::QueueResources;

/// `DateTime.UtcNow.Ticks` value at the Unix epoch (100 ns ticks since 0001-01-01 UTC).
const DOTNET_TICKS_AT_UNIX_EPOCH: i64 = 621_355_968_000_000_000;

/// Starting value for the contention counter in blocking [`Subscriber::dequeue`] (managed client parity).
const DEQUEUE_BACKOFF_COUNTER_INITIAL: i32 = -5;

/// After this many backoff steps, use a fixed long semaphore wait instead of ramping wait milliseconds from the counter.
const DEQUEUE_BACKOFF_HEAVY_PHASE_AFTER: i32 = 10;

/// Milliseconds for the steady semaphore wait once past [`DEQUEUE_BACKOFF_HEAVY_PHASE_AFTER`].
const DEQUEUE_BACKOFF_HEAVY_WAIT_MS: u64 = 10;

/// Pure-spin iterations on a `STATE_WRITING` slot before [`Subscriber::try_extract_message`]
/// graduates to [`std::thread::yield_now`].
const EXTRACT_SPIN_ITERATIONS: u32 = 1024;

/// Yield iterations after [`EXTRACT_SPIN_ITERATIONS`] before sleeping briefly between checks.
const EXTRACT_YIELD_ITERATIONS: u32 = 64;

/// Sleep duration between checks once both spin and yield phases are exhausted.
const EXTRACT_PARK_INTERVAL: Duration = Duration::from_micros(200);

/// Current instant in the same 100 ns tick domain as .NET `DateTime.UtcNow.Ticks`.
fn utc_now_ticks() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let since_unix_100ns = (d.as_nanos() / 100) as i64;
            since_unix_100ns.saturating_add(DOTNET_TICKS_AT_UNIX_EPOCH)
        }
        Err(_) => DOTNET_TICKS_AT_UNIX_EPOCH,
    }
}

/// Clears [`QueueHeader::read_lock_timestamp`] when a dequeue attempt completes.
struct ReadLockGuard<'a> {
    /// Header whose lock field must be released.
    header: &'a QueueHeader,
}

impl Drop for ReadLockGuard<'_> {
    fn drop(&mut self) {
        self.header.read_lock_timestamp.store(0, Ordering::SeqCst);
    }
}

/// Semaphore-backed backoff matching the managed dequeue loop.
struct DequeueBackoff {
    /// Counter carried across idle iterations (starts negative for yield-only phase).
    counter: i32,
}

impl DequeueBackoff {
    /// Builds a backoff state machine starting in the yield-heavy phase.
    const fn new() -> Self {
        Self {
            counter: DEQUEUE_BACKOFF_COUNTER_INITIAL,
        }
    }

    /// Performs one wait or yield step using `resources`' semaphore.
    fn step(&mut self, resources: &QueueResources) {
        if self.counter > DEQUEUE_BACKOFF_HEAVY_PHASE_AFTER {
            resources.wait_semaphore_timeout(Duration::from_millis(DEQUEUE_BACKOFF_HEAVY_WAIT_MS));
            return;
        }
        let old = self.counter;
        self.counter = self.counter.saturating_add(1);
        if old > 0 {
            resources.wait_semaphore_timeout(Duration::from_millis(self.counter as u64));
        } else {
            std::thread::yield_now();
        }
    }
}

/// Receives messages from the queue using the same contention and backoff pattern as the managed client.
pub struct Subscriber {
    /// Mapping, ring capacity, paired semaphore, and optional Unix file cleanup.
    res: QueueResources,
}

impl Subscriber {
    /// Opens the backing mapping and semaphore.
    ///
    /// On attach, clears [`QueueHeader::read_lock_timestamp`] to zero. The wire protocol is
    /// single-subscriber by design (the bootstrap handshake creates exactly one subscriber per
    /// queue), so a non-zero value at attach time can only be a stale lock left by a prior
    /// crashed subscriber on the same shared mapping. Without this reset, the new subscriber
    /// would wait up to [`TICKS_FOR_TEN_SECONDS`] before reclaiming on the first dequeue.
    pub fn new(options: QueueOptions) -> Result<Self, crate::OpenError> {
        let res = QueueResources::open(options)?;
        res.header().read_lock_timestamp.store(0, Ordering::SeqCst);
        Ok(Self { res })
    }

    /// Blocks until a message arrives or `cancel` is set, using semaphore-backed backoff.
    pub fn dequeue(&mut self, cancel: &AtomicBool) -> Vec<u8> {
        let mut backoff = DequeueBackoff::new();
        loop {
            if let Some(msg) = self.try_dequeue() {
                return msg;
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            backoff.step(&self.res);
        }
        vec![]
    }

    /// Returns the next message if one is ready; non-blocking aside from contender spin windows.
    pub fn try_dequeue(&mut self) -> Option<Vec<u8>> {
        let spin_start_ticks = self.try_acquire_read_lock()?;
        let header = self.res.header();
        let _lock = ReadLockGuard { header };
        self.try_extract_message(spin_start_ticks)
    }

    /// Attempts to claim the subscriber read lock when the queue is non-empty and the lock is stale.
    fn try_acquire_read_lock(&self) -> Option<i64> {
        let header = self.res.header();
        if header.is_empty() {
            return None;
        }
        let ticks = utc_now_ticks();
        let read_lock = header.read_lock_timestamp.load(Ordering::SeqCst);
        if ticks - read_lock < TICKS_FOR_TEN_SECONDS {
            return None;
        }
        header
            .read_lock_timestamp
            .compare_exchange(read_lock, ticks, Ordering::SeqCst, Ordering::SeqCst)
            .ok()?;
        Some(ticks)
    }

    /// Consumes one ready message after the read lock is held; caller supplies the tick value used for CAS spin limits.
    fn try_extract_message(&self, spin_start_ticks: i64) -> Option<Vec<u8>> {
        let header = self.res.header();
        if header.is_empty() {
            return None;
        }
        let read_offset = header.read_offset.load(Ordering::SeqCst);
        let write_offset = header.write_offset.load(Ordering::SeqCst);
        let ring = self.res.ring();
        // SAFETY: `read_offset` is produced by the publisher after a space check and the wire
        // protocol guarantees a contiguous eight-byte `MessageHeader` at this slot. A `None`
        // return means the slot is misaligned (corrupted publisher); drain past every queued
        // slot rather than dereference a misaligned pointer.
        let Some(msg) = (unsafe { ring.message_header_at(read_offset) }) else {
            header.read_offset.store(write_offset, Ordering::SeqCst);
            return None;
        };
        let mut backoff_iter: u32 = 0;
        loop {
            if msg
                .state
                .compare_exchange(
                    STATE_READY,
                    STATE_LOCKED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                break;
            }
            if utc_now_ticks() - spin_start_ticks > TICKS_FOR_TEN_SECONDS {
                header.read_offset.store(write_offset, Ordering::SeqCst);
                return None;
            }
            if backoff_iter < EXTRACT_SPIN_ITERATIONS {
                std::hint::spin_loop();
            } else if backoff_iter
                < EXTRACT_SPIN_ITERATIONS.saturating_add(EXTRACT_YIELD_ITERATIONS)
            {
                std::thread::yield_now();
            } else {
                std::thread::sleep(EXTRACT_PARK_INTERVAL);
            }
            backoff_iter = backoff_iter.saturating_add(1);
        }
        let body_len = i64::from(msg.body_length);
        let capacity = self.res.capacity;
        if body_len < 0 || body_len > capacity {
            // Corrupted slot: a sane publisher only writes message bodies whose padded length
            // fits in the ring. Drain past every queued slot rather than reading a giant or
            // negative body length into `Vec::with_capacity`.
            header.read_offset.store(write_offset, Ordering::SeqCst);
            return None;
        }
        let padded = padded_message_length(body_len);
        if padded > capacity {
            header.read_offset.store(write_offset, Ordering::SeqCst);
            return None;
        }
        let body_offset = read_offset + MESSAGE_BODY_OFFSET;
        let body_len_usize = body_len as usize;
        let msg_result = ring.read(body_offset, body_len_usize);
        ring.clear(read_offset, padded as usize);
        let new_read = (read_offset + padded) % (capacity * 2);
        header.read_offset.store(new_read, Ordering::SeqCst);
        Some(msg_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::QueueOptions;
    use crate::publisher::Publisher;

    #[test]
    fn try_dequeue_empty_returns_none() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_sub_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("sub_empty", &dir, 4096).expect("valid");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        assert!(subscriber.try_dequeue().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dequeue_respects_cancel_when_idle() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_sub_cancel_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("sub_cancel", &dir, 4096).expect("valid");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        let cancel = AtomicBool::new(true);
        assert!(subscriber.dequeue(&cancel).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dequeue_after_message_then_cancel() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_sub_cancel2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("sub_cancel2", &dir, 4096).expect("valid");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        assert!(publisher.try_enqueue(b"ping"));
        assert_eq!(
            subscriber.try_dequeue().as_deref(),
            Some(b"ping".as_slice())
        );
        let cancel = AtomicBool::new(true);
        assert!(subscriber.dequeue(&cancel).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fifo_across_many_messages() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_sub_fifo_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = QueueOptions::with_path("sub_fifo", &dir, 4096).expect("valid");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        for i in 0u32..30 {
            assert!(publisher.try_enqueue(format!("n{i}").as_bytes()));
        }
        for i in 0u32..30 {
            let expected = format!("n{i}");
            assert_eq!(
                subscriber.try_dequeue().as_deref(),
                Some(expected.as_bytes())
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn utc_now_ticks_is_after_dotnet_unix_epoch() {
        let now = utc_now_ticks();
        assert!(
            now > DOTNET_TICKS_AT_UNIX_EPOCH,
            "utc_now_ticks() = {now} should be past the .NET Unix epoch"
        );
        // Year 3000 in .NET ticks: ~9.4e17. A loose upper bound that fails loudly on a
        // saturating-add or sign-flip regression while leaving room for clock drift.
        const YEAR_3000_DOTNET_TICKS: i64 = 946_708_416_000_000_000;
        assert!(
            now < YEAR_3000_DOTNET_TICKS,
            "utc_now_ticks() = {now} unexpectedly past year 3000 — saturating-add bug?"
        );
    }

    /// Verifies `Subscriber::new` surfaces an `OpenError` when the backing directory cannot
    /// accept a new file. Unix-only because the equivalent permission denial on Windows is
    /// brittle to set up reliably from a unit test.
    #[cfg(unix)]
    #[test]
    fn subscriber_new_returns_err_for_unwritable_path() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        struct PermsGuard {
            path: std::path::PathBuf,
        }

        impl Drop for PermsGuard {
            fn drop(&mut self) {
                let _ = fs::set_permissions(&self.path, fs::Permissions::from_mode(0o755));
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o500)).expect("chmod 0o500");
        let _guard = PermsGuard { path: path.clone() };

        let opts =
            QueueOptions::with_path("sub_no_perm", &path, 4096).expect("options should validate");
        assert!(
            Subscriber::new(opts).is_err(),
            "expected Subscriber::new to fail when backing dir is read-only"
        );
    }
}
