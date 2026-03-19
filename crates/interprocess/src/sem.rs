//! Semaphore abstraction: POSIX on Unix, Windows named semaphore on Windows.
//! Naming matches Cloudtoid.Interprocess: /ct.ip.{name} on Unix, Global\CT.IP.{name} on Windows.

#[cfg(unix)]
mod imp {
    use std::ffi::CString;
    use std::io;

    pub type Handle = *mut libc::sem_t;

    /// Opens or creates a named semaphore. Returns an error on failure (e.g. permissions, stale semaphores).
    pub fn open(name: &str) -> io::Result<Handle> {
        let full_name = format!("/ct.ip.{}", name);
        let c_name = CString::new(full_name).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "semaphore name contains null")
        })?;
        let handle = unsafe { libc::sem_open(c_name.as_ptr(), libc::O_CREAT, 0o777, 0) };
        if handle == libc::SEM_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(handle)
    }

    pub fn post(handle: &Handle) {
        unsafe { libc::sem_post(*handle) };
    }

    pub fn close(handle: &Handle) {
        unsafe { libc::sem_close(*handle) };
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::OsStr;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;

    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{CreateSemaphoreW, ReleaseSemaphore};

    const HANDLE_NAME_PREFIX: &str = "Global\\CT.IP.";
    const INVALID_HANDLE: windows_sys::Win32::Foundation::HANDLE = -1i32 as _;

    pub struct Handle(windows_sys::Win32::Foundation::HANDLE);

    fn to_wide(s: &str) -> Vec<u16> {
        OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    /// Opens or creates a named semaphore. Returns an error on failure.
    pub fn open(name: &str) -> io::Result<Handle> {
        let full_name = format!("{}{}", HANDLE_NAME_PREFIX, name);
        let name_wide = to_wide(&full_name);
        let handle = unsafe { CreateSemaphoreW(null_mut(), 0, i32::MAX, name_wide.as_ptr()) };
        if handle == 0 || handle == INVALID_HANDLE {
            return Err(io::Error::last_os_error());
        }
        Ok(Handle(handle))
    }

    pub fn post(handle: &Handle) {
        unsafe {
            ReleaseSemaphore(handle.0, 1, null_mut());
        }
    }

    pub fn close(handle: &Handle) {
        if handle.0 != 0 && handle.0 != -1 {
            unsafe {
                CloseHandle(handle.0);
            }
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    use std::io;

    #[derive(Clone, Copy)]
    pub struct Handle;

    pub fn open(_name: &str) -> io::Result<Handle> {
        Ok(Handle)
    }

    pub fn post(_handle: &Handle) {}

    pub fn close(_handle: &Handle) {}
}

pub(super) use imp::{close, open, post};

#[cfg(unix)]
pub(super) type SemHandle = imp::Handle;

#[cfg(windows)]
pub(super) type SemHandle = imp::Handle;

#[cfg(not(any(unix, windows)))]
pub(super) type SemHandle = imp::Handle;
