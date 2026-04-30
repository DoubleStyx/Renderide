//! POSIX named semaphores opened with `sem_open`.
//!
//! - **Linux and non-Apple Unix:** name `"/ct.ip.{memory_view_name}"`.
//! - **macOS:** a shorter `"/sem_{prefix}"` derived from a SHA-256 hash (POSIX named-semaphore length limits).

use std::ffi::CString;
use std::io;
use std::time::Duration;
#[cfg(target_vendor = "apple")]
use std::time::Instant;

#[cfg(target_os = "macos")]
use base64::prelude::*;
#[cfg(target_os = "macos")]
use sha2::{Digest, Sha256};

use super::MAX_WAIT_DURATION;

/// Handle to a POSIX named semaphore created with [`PosixSemaphore::open`].
pub(super) struct PosixSemaphore {
    /// Opaque `sem_t` pointer returned by `sem_open`.
    handle: *mut libc::sem_t,
    /// Logical queue name (matches the mapping name); used in diagnostic log lines.
    queue_name: String,
}

impl PosixSemaphore {
    /// Opens or creates the semaphore with mode `0o777` and initial value `0`.
    pub(super) fn open(memory_view_name: &str) -> io::Result<Self> {
        let full_name;
        #[cfg(not(target_os = "macos"))]
        {
            full_name = format!("/ct.ip.{memory_view_name}");
        }
        #[cfg(target_os = "macos")]
        {
            let path_for_hash = format!("/ct.ip.{memory_view_name}");
            let digest = Sha256::digest(path_for_hash.as_bytes());
            let encoded = BASE64_URL_SAFE.encode(digest);
            let prefix = encoded.get(..24).map_or(encoded.as_str(), |s| s);
            full_name = format!("/sem_{prefix}");
        }
        let c_name = CString::new(full_name).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "semaphore name contains NUL at position {}",
                    e.nul_position()
                ),
            )
        })?;
        // SAFETY: `c_name` is a NUL-terminated C string; remaining args are constants.
        let h = unsafe { libc::sem_open(c_name.as_ptr(), libc::O_CREAT, 0o777, 0) };
        if h == libc::SEM_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            handle: h,
            queue_name: memory_view_name.to_string(),
        })
    }

    /// Increments the semaphore (wake one waiter).
    pub(super) fn post(&self) {
        // SAFETY: `self.handle` is a non-`SEM_FAILED` pointer returned by `sem_open` in `open`.
        let rc = unsafe { libc::sem_post(self.handle) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            logger::warn!(
                "interprocess: sem_post failed for queue '{}': {}",
                self.queue_name,
                err
            );
            debug_assert!(false, "sem_post: {:?}", err);
        }
    }

    /// Waits for a post, using `sem_timedwait` on non-Apple Unix and polling on Apple platforms.
    pub(super) fn wait_timeout(&self, timeout: Duration) -> bool {
        if timeout.is_zero() {
            return self.try_wait();
        }
        #[cfg(target_vendor = "apple")]
        {
            self.wait_poll(timeout)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            self.wait_timed(timeout)
        }
    }

    /// Non-blocking wait; returns `true` if the semaphore was acquired.
    fn try_wait(&self) -> bool {
        loop {
            // SAFETY: `self.handle` is a live `sem_open` handle owned by `self`.
            let rc = unsafe { libc::sem_trywait(self.handle) };
            if rc == 0 {
                return true;
            }
            let err = io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if err == libc::EINTR {
                continue;
            }
            if err == libc::EAGAIN || err == libc::EBUSY {
                return false;
            }
            logger::warn!(
                "interprocess: sem_trywait unexpected errno {} for queue '{}'",
                err,
                self.queue_name
            );
            return false;
        }
    }

    /// Linux and other non-Apple Unix: absolute deadline via `sem_timedwait`.
    ///
    /// Restarts the syscall when it returns `EINTR`. Uses Euclidean division so negative clock
    /// edge cases remain well-defined.
    #[cfg(not(target_vendor = "apple"))]
    fn wait_timed(&self, timeout: Duration) -> bool {
        // SAFETY: `timespec` is a POD struct of integers; all-zero is a valid bit pattern.
        let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
        // SAFETY: `&mut ts` is a valid out-pointer; clockid is a constant.
        if unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, core::ptr::addr_of_mut!(ts)) } != 0 {
            return false;
        }
        let clamped = timeout.min(MAX_WAIT_DURATION);
        let add_ns = clamped.as_nanos() as i128;
        let cur_ns = ts.tv_sec as i128 * 1_000_000_000i128 + ts.tv_nsec as i128;
        let deadline_ns = cur_ns.saturating_add(add_ns);
        let d_sec = deadline_ns.div_euclid(1_000_000_000);
        let d_nsec = deadline_ns.rem_euclid(1_000_000_000);
        if d_sec > i64::MAX as i128 || d_sec < i64::MIN as i128 {
            return false;
        }
        ts.tv_sec = d_sec as libc::time_t;
        ts.tv_nsec = d_nsec as libc::c_long;
        loop {
            // SAFETY: `self.handle` is a live sem handle; `&ts` is a valid absolute timespec.
            let rc = unsafe { libc::sem_timedwait(self.handle, core::ptr::addr_of!(ts)) };
            if rc == 0 {
                return true;
            }
            let err = io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if err == libc::EINTR {
                continue;
            }
            if err == libc::ETIMEDOUT {
                return false;
            }
            logger::warn!(
                "interprocess: sem_timedwait unexpected errno {} for queue '{}'",
                err,
                self.queue_name
            );
            return false;
        }
    }

    /// macOS / iOS: no `sem_timedwait`; poll with `sem_trywait` and short yields.
    #[cfg(target_vendor = "apple")]
    fn wait_poll(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if Instant::now() >= deadline {
                return false;
            }
            match self.try_wait() {
                true => return true,
                false => std::thread::yield_now(),
            }
        }
    }
}

impl Drop for PosixSemaphore {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is a live sem handle owned by `self`; dropped exactly once.
        let rc = unsafe { libc::sem_close(self.handle) };
        if rc != 0 {
            logger::warn!(
                "interprocess: sem_close failed for queue '{}': {}",
                self.queue_name,
                io::Error::last_os_error()
            );
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use super::PosixSemaphore;

    /// Unique logical queue names for isolated POSIX semaphore tests.
    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_queue_name() -> String {
        format!(
            "semtest_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        )
    }

    #[test]
    fn post_then_zero_timeout_wait_acquires() {
        let s = PosixSemaphore::open(&unique_queue_name()).expect("open");
        s.post();
        assert!(s.wait_timeout(Duration::ZERO));
    }

    #[test]
    fn zero_timeout_without_post_returns_false() {
        let s = PosixSemaphore::open(&unique_queue_name()).expect("open");
        assert!(!s.wait_timeout(Duration::ZERO));
    }

    #[test]
    fn post_then_short_wait_acquires() {
        let s = PosixSemaphore::open(&unique_queue_name()).expect("open");
        s.post();
        assert!(s.wait_timeout(Duration::from_millis(500)));
    }

    #[test]
    fn multiple_posts_drain_with_waits() {
        let s = PosixSemaphore::open(&unique_queue_name()).expect("open");
        s.post();
        s.post();
        assert!(s.wait_timeout(Duration::from_millis(500)));
        assert!(s.wait_timeout(Duration::from_millis(500)));
        assert!(!s.wait_timeout(Duration::ZERO));
    }
}
