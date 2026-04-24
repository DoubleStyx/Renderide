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

use crate::naming;

/// Win32 semaphore handle from `CreateSemaphoreW` (`Global\CT.IP.{name}`).
pub(super) struct WinSemaphore(
    /// Raw semaphore handle; closed on drop.
    windows_sys::Win32::Foundation::HANDLE,
);

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
        Ok(Self(handle))
    }

    /// Releases one semaphore count (`ReleaseSemaphore`).
    pub(super) fn post(&self) {
        // SAFETY: `self.0` is a live semaphore handle owned by `self`; `lpPreviousCount` null is
        // permitted by the Win32 API.
        let rc = unsafe { ReleaseSemaphore(self.0, 1, null_mut()) };
        if rc == 0 {
            debug_assert!(
                false,
                "ReleaseSemaphore failed: {:?}",
                io::Error::last_os_error()
            );
        }
    }

    /// Waits on the semaphore with a timeout in milliseconds (capped; very long waits use `INFINITE`).
    pub(super) fn wait_timeout(&self, timeout: Duration) -> bool {
        let ms = if timeout.is_zero() {
            0u32
        } else if timeout.as_secs() > 60 * 60 * 24 * 7 {
            INFINITE
        } else {
            timeout.as_millis().min(u32::MAX as u128) as u32
        };
        // SAFETY: `self.0` is a live semaphore handle owned by `self`.
        let r = unsafe { WaitForSingleObject(self.0, ms) };
        r == WAIT_OBJECT_0
    }
}

impl Drop for WinSemaphore {
    fn drop(&mut self) {
        if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
            // SAFETY: `self.0` is the semaphore handle created in `open`, still live (non-null and
            // not sentinel); closed exactly once here.
            unsafe {
                CloseHandle(self.0);
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
