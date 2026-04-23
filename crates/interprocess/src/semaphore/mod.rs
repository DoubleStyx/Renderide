//! Named semaphore paired with the queue mapping for wakeup hints.

#[cfg(unix)]
mod posix;
#[cfg(windows)]
mod win;

use std::io;
use std::time::Duration;

/// Cross-process wakeup primitive paired with the queue mapping (post on enqueue, wait while idle).
///
/// On Unix this is a POSIX named semaphore; on Windows, a global semaphore under `Global\CT.IP.{name}`.
///
/// # Threading
///
/// The OS primitives are designed for cross-thread and cross-process use; this wrapper carries no
/// additional mutable Rust state between calls.
pub(crate) struct Semaphore {
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
