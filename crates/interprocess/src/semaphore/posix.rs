//! POSIX named semaphore (`/ct.ip.{name}`).

use std::ffi::CString;
use std::io;
use std::time::Duration;
#[cfg(target_vendor = "apple")]
use std::time::Instant;

/// Opens `/ct.ip.{memory_view_name}`.
pub(super) struct PosixSemaphore(*mut libc::sem_t);

impl PosixSemaphore {
    pub(super) fn open(memory_view_name: &str) -> io::Result<Self> {
        let full_name = format!("/ct.ip.{memory_view_name}");
        let c_name = CString::new(full_name).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "semaphore name contains NUL")
        })?;
        let h = unsafe { libc::sem_open(c_name.as_ptr(), libc::O_CREAT, 0o777, 0) };
        if h == libc::SEM_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(h))
    }

    pub(super) fn post(&self) {
        let rc = unsafe { libc::sem_post(self.0) };
        if rc != 0 {
            debug_assert!(false, "sem_post: {:?}", io::Error::last_os_error());
        }
    }

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

    fn try_wait(&self) -> bool {
        loop {
            let rc = unsafe { libc::sem_trywait(self.0) };
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
            // Unexpected failure — treat as not acquired.
            return false;
        }
    }

    /// Linux and other non-Apple Unix: absolute deadline via `sem_timedwait`.
    #[cfg(not(target_vendor = "apple"))]
    fn wait_timed(&self, timeout: Duration) -> bool {
        let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
        if unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) } != 0 {
            return false;
        }
        let add_nanos = timeout.as_nanos().min(i128::MAX as u128) as i128;
        let deadline_ns = ts.tv_sec as i128 * 1_000_000_000i128 + ts.tv_nsec as i128 + add_nanos;
        ts.tv_sec = (deadline_ns / 1_000_000_000) as libc::time_t;
        ts.tv_nsec = (deadline_ns % 1_000_000_000) as libc::c_long;
        loop {
            let rc = unsafe { libc::sem_timedwait(self.0, &ts) };
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
            return false;
        }
    }

    /// macOS / iOS: no `sem_timedwait`; spin on `sem_trywait` like the reference implementation.
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
        let _ = unsafe { libc::sem_close(self.0) };
    }
}
