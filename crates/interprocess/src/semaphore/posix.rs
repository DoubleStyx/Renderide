//! POSIX named semaphores opened with `sem_open`.
//!
//! - **Linux and non-Apple Unix:** name `"/ct.ip.{memory_view_name}"`.
//! - **macOS:** name `"/ct.ip.{memory_view_name}"`, plus a shorter `"/sem_{prefix}"`
//!   fallback derived from a SHA-256 hash when the full name cannot be opened.

use std::ffi::CString;
use std::io;
use std::time::Duration;
#[cfg(target_vendor = "apple")]
use std::time::Instant;

#[cfg(target_os = "macos")]
use base64::prelude::*;
#[cfg(target_os = "macos")]
use sha2::{Digest, Sha256};

#[cfg(not(target_vendor = "apple"))]
use super::MAX_WAIT_DURATION;

/// Handle to a POSIX named semaphore created with [`PosixSemaphore::open`].
pub(super) struct PosixSemaphore {
    /// Opaque `sem_t` pointers returned by `sem_open`.
    handles: Vec<PosixSemaphoreHandle>,
    /// Logical queue name (matches the mapping name); used in diagnostic log lines.
    queue_name: String,
}

/// One opened POSIX semaphore name.
struct PosixSemaphoreHandle {
    /// Opaque `sem_t` pointer returned by `sem_open`.
    handle: *mut libc::sem_t,
    /// Full POSIX semaphore name passed to `sem_open`.
    name: String,
}

impl PosixSemaphore {
    /// Opens or creates the semaphore with owner-only access and initial value `0`.
    pub(super) fn open(memory_view_name: &str) -> io::Result<Self> {
        let mut handles = Vec::new();
        let mut first_error = None;
        for full_name in candidate_semaphore_names(memory_view_name) {
            match open_one(full_name) {
                Ok(handle) => handles.push(handle),
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        if handles.is_empty() {
            if let Some(err) = first_error {
                return Err(err);
            }
            return Err(io::Error::other("no POSIX semaphore names were available"));
        }

        Ok(Self {
            handles,
            queue_name: memory_view_name.to_string(),
        })
    }

    /// Increments the semaphore (wake one waiter).
    pub(super) fn post(&self) {
        for handle in &self.handles {
            // SAFETY: `handle.handle` is a non-`SEM_FAILED` pointer returned by `sem_open`.
            let rc = unsafe { libc::sem_post(handle.handle) };
            if rc != 0 {
                let err = io::Error::last_os_error();
                logger::warn!(
                    "interprocess: sem_post failed for queue '{}' semaphore '{}': {}",
                    self.queue_name,
                    handle.name,
                    err
                );
                debug_assert!(false, "sem_post: {err:?}");
            }
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
        let mut acquired = false;
        for handle in &self.handles {
            loop {
                // SAFETY: `handle.handle` is a live `sem_open` handle owned by `self`.
                let rc = unsafe { libc::sem_trywait(handle.handle) };
                if rc == 0 {
                    acquired = true;
                    break;
                }
                let err = io::Error::last_os_error().raw_os_error().unwrap_or(0);
                if err == libc::EINTR {
                    continue;
                }
                if err == libc::EAGAIN || err == libc::EBUSY {
                    break;
                }
                logger::warn!(
                    "interprocess: sem_trywait unexpected errno {} for queue '{}' semaphore '{}'",
                    err,
                    self.queue_name,
                    handle.name
                );
                break;
            }
        }
        acquired
    }

    /// Linux and other non-Apple Unix: absolute deadline via `sem_timedwait`.
    ///
    /// Restarts the syscall when it returns `EINTR`. Uses Euclidean division so negative clock
    /// edge cases remain well-defined.
    #[cfg(not(target_vendor = "apple"))]
    fn wait_timed(&self, timeout: Duration) -> bool {
        let Some(handle) = self.handles.first() else {
            return false;
        };
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `&mut ts` is a valid out-pointer; clockid is a constant.
        if unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, core::ptr::addr_of_mut!(ts)) } != 0 {
            return false;
        }
        let clamped = timeout.min(MAX_WAIT_DURATION);
        let add_ns = clamped.as_nanos() as i128;
        let cur_ns = i128::from(ts.tv_sec) * 1_000_000_000i128 + i128::from(ts.tv_nsec);
        let deadline_ns = cur_ns.saturating_add(add_ns);
        let d_sec = deadline_ns.div_euclid(1_000_000_000);
        let d_nsec = deadline_ns.rem_euclid(1_000_000_000);
        if d_sec > i128::from(i64::MAX) || d_sec < i128::from(i64::MIN) {
            return false;
        }
        ts.tv_sec = d_sec as libc::time_t;
        ts.tv_nsec = d_nsec as libc::c_long;
        loop {
            // SAFETY: `handle.handle` is a live sem handle; `&ts` is a valid absolute timespec.
            let rc = unsafe { libc::sem_timedwait(handle.handle, core::ptr::addr_of!(ts)) };
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
                "interprocess: sem_timedwait unexpected errno {} for queue '{}' semaphore '{}'",
                err,
                self.queue_name,
                handle.name
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
            if self.try_wait() {
                return true;
            }
            std::thread::yield_now();
        }
    }
}

impl Drop for PosixSemaphore {
    fn drop(&mut self) {
        for handle in &self.handles {
            // SAFETY: `handle.handle` is a live sem handle owned by `self`; dropped exactly once.
            let rc = unsafe { libc::sem_close(handle.handle) };
            if rc != 0 {
                logger::warn!(
                    "interprocess: sem_close failed for queue '{}' semaphore '{}': {}",
                    self.queue_name,
                    handle.name,
                    io::Error::last_os_error()
                );
            }
        }
    }
}

fn open_one(full_name: String) -> io::Result<PosixSemaphoreHandle> {
    let c_name = CString::new(full_name.as_str()).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "semaphore name contains NUL at position {}",
                e.nul_position()
            ),
        )
    })?;
    // Mode `0o600` so the named semaphore is reachable only by the owning user. With
    // the historical `0o777`, any local process could `sem_open` the same name and call
    // `sem_wait` (draining wakeups) or `sem_post` (spurious wakeups) -- a DoS vector on
    // multi-user systems. Single-user systems see no behavioural change.
    // SAFETY: `c_name` is a NUL-terminated C string; remaining args are constants.
    let handle = unsafe { libc::sem_open(c_name.as_ptr(), libc::O_CREAT, 0o600, 0) };
    if handle == libc::SEM_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(PosixSemaphoreHandle {
        handle,
        name: full_name,
    })
}

fn candidate_semaphore_names(memory_view_name: &str) -> Vec<String> {
    let managed_name = format!("/ct.ip.{memory_view_name}");
    #[cfg(not(target_os = "macos"))]
    {
        vec![managed_name]
    }
    #[cfg(target_os = "macos")]
    {
        let hashed_name = hashed_macos_semaphore_name(&managed_name);
        if hashed_name == managed_name {
            vec![managed_name]
        } else {
            vec![managed_name, hashed_name]
        }
    }
}

#[cfg(target_os = "macos")]
fn hashed_macos_semaphore_name(managed_name: &str) -> String {
    let digest = Sha256::digest(managed_name.as_bytes());
    let encoded = BASE64_URL_SAFE.encode(digest);
    let prefix = encoded.get(..24).map_or(encoded.as_str(), |s| s);
    format!("/sem_{prefix}")
}

#[cfg(all(test, unix))]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use super::{PosixSemaphore, candidate_semaphore_names};

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

    #[test]
    fn candidate_names_include_managed_name_first() {
        let names = candidate_semaphore_names("queue");
        assert_eq!(names.first().map(String::as_str), Some("/ct.ip.queue"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_candidate_names_include_hashed_fallback() {
        let names = candidate_semaphore_names("queue");
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "/ct.ip.queue");
        assert!(names[1].starts_with("/sem_"));
        assert_ne!(names[1], names[0]);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_candidate_names_use_single_managed_name() {
        let names = candidate_semaphore_names("queue");
        assert_eq!(names, vec!["/ct.ip.queue".to_string()]);
    }
}
