//! Named semaphore paired with the queue mapping for wakeup hints.

#[cfg(unix)]
mod posix;
#[cfg(windows)]
mod win;

use std::io;
use std::time::Duration;

/// Longest interval the POSIX wait helper sleeps for in a single `sem_timedwait` call.
///
/// Clamping the requested timeout keeps `clock_gettime` arithmetic far below `i128::MAX`, so the
/// nanosecond conversion to `i128` is exact without a defensive fallback.
#[cfg(unix)]
pub const MAX_WAIT_DURATION: Duration = Duration::from_secs(60 * 60 * 24 * 365);

/// Threshold above which the Windows wait helper switches to `WaitForSingleObject(INFINITE)`
/// instead of converting the timeout to milliseconds.
#[cfg(windows)]
pub(super) const WIN_WAIT_INFINITE_THRESHOLD: Duration = Duration::from_secs(60 * 60 * 24 * 7);

/// Cross-process wakeup primitive paired with the queue mapping (post on enqueue, wait while idle).
///
/// On Unix this is a POSIX named semaphore; on Windows, a global semaphore under `Global\CT.IP.{name}`.
///
/// # Threading
///
/// The OS primitives are designed for cross-thread and cross-process use; this wrapper carries no
/// additional mutable Rust state between calls.
pub struct Semaphore {
    /// Platform semaphore implementation.
    #[cfg(unix)]
    inner: posix::PosixSemaphore,
    /// Platform semaphore implementation.
    #[cfg(windows)]
    inner: win::WinSemaphore,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "OS-managed semaphore; Send is enforced by the kernel, not Rust field types"
)]
// SAFETY: the inner handle is a process-global kernel object; all operations delegate to the OS,
// which enforces its own thread-safety. No Rust-level aliasing invariants are at stake.
unsafe impl Send for Semaphore {}

// SAFETY: concurrent `post` / `wait` calls are defined by the OS semaphore semantics; `&Semaphore`
// never yields mutable Rust state.
unsafe impl Sync for Semaphore {}

impl Semaphore {
    /// Opens or creates the semaphore for the given queue name (same logical name as the mapping).
    pub(crate) fn open(memory_view_name: &str) -> io::Result<Self> {
        #[cfg(unix)]
        {
            Ok(Self {
                inner: posix::PosixSemaphore::open(memory_view_name)?,
            })
        }
        #[cfg(windows)]
        {
            Ok(Self {
                inner: win::WinSemaphore::open(memory_view_name)?,
            })
        }
    }

    /// Signals waiters that new data may be available (called after a successful enqueue).
    pub(crate) fn post(&self) {
        self.inner.post();
    }

    /// Blocks up to `timeout` waiting for a signal; returns `true` if a token was acquired.
    pub(crate) fn wait_timeout(&self, timeout: Duration) -> bool {
        self.inner.wait_timeout(timeout)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::Semaphore;

    /// Counter combined with the test process id to keep semaphore names unique across tests.
    ///
    /// POSIX named semaphores live in a kernel-global namespace (`/ct.ip.{name}`), and Windows
    /// global semaphores live under `Global\CT.IP.{name}`. Reusing a name across tests would let
    /// stale tokens from one test leak into another, so each test grabs a fresh value here.
    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// Returns a queue-style name guaranteed to be unique within this test process.
    fn unique_name() -> String {
        format!(
            "semtest_facade_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        )
    }

    #[test]
    fn wait_timeout_zero_returns_false_when_no_post() {
        let sem = Semaphore::open(&unique_name()).expect("open");
        assert!(!sem.wait_timeout(Duration::from_millis(0)));
    }

    #[test]
    fn post_then_wait_returns_true() {
        let sem = Semaphore::open(&unique_name()).expect("open");
        sem.post();
        assert!(sem.wait_timeout(Duration::from_millis(50)));
    }

    #[test]
    fn multiple_posts_release_multiple_waits() {
        let sem = Semaphore::open(&unique_name()).expect("open");
        sem.post();
        sem.post();
        sem.post();
        assert!(sem.wait_timeout(Duration::from_millis(50)));
        assert!(sem.wait_timeout(Duration::from_millis(50)));
        assert!(sem.wait_timeout(Duration::from_millis(50)));
        assert!(!sem.wait_timeout(Duration::from_millis(0)));
    }

    #[test]
    fn wait_timeout_returns_false_within_budget_when_idle() {
        let sem = Semaphore::open(&unique_name()).expect("open");
        let start = Instant::now();
        let acquired = sem.wait_timeout(Duration::from_millis(20));
        let elapsed = start.elapsed();
        assert!(!acquired);
        assert!(
            elapsed < Duration::from_millis(500),
            "wait_timeout(20ms) idle should return well under 500ms, got {elapsed:?}"
        );
    }

    #[test]
    fn concurrent_post_wakes_blocked_waiter() {
        let name = unique_name();
        let waiter = {
            let name = name.clone();
            thread::spawn(move || {
                let sem = Semaphore::open(&name).expect("open in waiter");
                let start = Instant::now();
                let acquired = sem.wait_timeout(Duration::from_secs(2));
                (acquired, start.elapsed())
            })
        };

        thread::sleep(Duration::from_millis(20));
        let poster = Semaphore::open(&name).expect("open in poster");
        poster.post();

        let (acquired, elapsed) = waiter.join().expect("waiter join");
        assert!(acquired, "waiter should have acquired after post");
        assert!(
            elapsed < Duration::from_millis(500),
            "waiter should return promptly after post, got {elapsed:?}"
        );
    }
}
