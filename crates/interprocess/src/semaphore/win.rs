//! Windows named semaphore (`Global\CT.IP.{name}`).

use std::ffi::OsStr;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null_mut;
use std::time::Duration;

use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{
    CreateSemaphoreW, ReleaseSemaphore, WaitForSingleObject, INFINITE,
};

use super::WIN_WAIT_INFINITE_THRESHOLD;
use crate::naming;

/// Win32 semaphore handle from `CreateSemaphoreW` (`Global\CT.IP.{name}`).
pub(super) struct WinSemaphore {
    /// Raw semaphore handle; closed on drop.
    handle: windows_sys::Win32::Foundation::HANDLE,
    /// Logical queue name (matches the mapping name); used in diagnostic log lines.
    queue_name: String,
}

impl WinSemaphore {
    /// Creates or opens the named global semaphore (initial count `0`, max `i32::MAX`).
    pub(super) fn open(memory_view_name: &str) -> io::Result<Self> {
        let full_name = naming::windows_semaphore_wide_name(memory_view_name);
        let name_wide: Vec<u16> = OsStr::new(&full_name)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `name_wide` is NUL-terminated wide string; security attrs arg is null (default ACL).
        let handle = unsafe { CreateSemaphoreW(null_mut(), 0, i32::MAX, name_wide.as_ptr()) };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            handle,
            queue_name: memory_view_name.to_string(),
        })
    }

    /// Releases one semaphore count (`ReleaseSemaphore`).
    pub(super) fn post(&self) {
        // SAFETY: `self.handle` is a live semaphore handle owned by `self`; `lpPreviousCount` null
        // is permitted by the Win32 API.
        let rc = unsafe { ReleaseSemaphore(self.handle, 1, null_mut()) };
        if rc == 0 {
            let err = io::Error::last_os_error();
            logger::warn!(
                "interprocess: ReleaseSemaphore failed for queue '{}': {}",
                self.queue_name,
                err
            );
            debug_assert!(false, "ReleaseSemaphore failed: {:?}", err);
        }
    }

    /// Waits on the semaphore with a timeout in milliseconds (capped; very long waits use `INFINITE`).
    pub(super) fn wait_timeout(&self, timeout: Duration) -> bool {
        let ms = if timeout.is_zero() {
            0u32
        } else if timeout >= WIN_WAIT_INFINITE_THRESHOLD {
            INFINITE
        } else {
            timeout.as_millis().min(u32::MAX as u128) as u32
        };
        // SAFETY: `self.handle` is a live semaphore handle owned by `self`.
        let r = unsafe { WaitForSingleObject(self.handle, ms) };
        r == WAIT_OBJECT_0
    }
}

impl Drop for WinSemaphore {
    fn drop(&mut self) {
        if !self.handle.is_null() && self.handle != INVALID_HANDLE_VALUE {
            // SAFETY: `self.handle` is the semaphore handle created in `open`, still live (non-null
            // and not sentinel); closed exactly once here.
            let rc = unsafe { CloseHandle(self.handle) };
            if rc == 0 {
                logger::warn!(
                    "interprocess: CloseHandle on semaphore failed for queue '{}': {}",
                    self.queue_name,
                    io::Error::last_os_error()
                );
            }
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use super::WinSemaphore;

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_queue_name() -> String {
        format!(
            "wsem_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        )
    }

    #[test]
    fn post_then_try_wait_zero_timeout() {
        let s = WinSemaphore::open(&unique_queue_name()).expect("open");
        s.post();
        assert!(s.wait_timeout(Duration::ZERO));
    }

    #[test]
    fn zero_timeout_without_post_returns_false() {
        let s = WinSemaphore::open(&unique_queue_name()).expect("open");
        assert!(!s.wait_timeout(Duration::ZERO));
    }

    #[test]
    fn post_then_short_wait_acquires() {
        let s = WinSemaphore::open(&unique_queue_name()).expect("open");
        s.post();
        assert!(s.wait_timeout(Duration::from_millis(500)));
    }

    #[test]
    fn multiple_posts_drain() {
        let s = WinSemaphore::open(&unique_queue_name()).expect("open");
        s.post();
        s.post();
        assert!(s.wait_timeout(Duration::from_millis(500)));
        assert!(s.wait_timeout(Duration::from_millis(500)));
        assert!(!s.wait_timeout(Duration::ZERO));
    }
}
