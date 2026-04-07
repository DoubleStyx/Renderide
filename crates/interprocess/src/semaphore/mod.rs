//! Named semaphore paired with the queue mapping for wakeup hints.

#[cfg(unix)]
mod posix;
#[cfg(windows)]
mod win;

use std::io;
use std::time::Duration;

/// Cross-process wakeup primitive matching the managed queue pairing.
pub(crate) struct Semaphore {
    #[cfg(unix)]
    inner: posix::PosixSemaphore,
    #[cfg(windows)]
    inner: win::WinSemaphore,
}

impl Semaphore {
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

    pub(crate) fn post(&self) {
        self.inner.post();
    }

    /// Blocks up to `timeout` for a post; returns `true` if a token was acquired.
    pub(crate) fn wait_timeout(&self, timeout: Duration) -> bool {
        self.inner.wait_timeout(timeout)
    }
}
