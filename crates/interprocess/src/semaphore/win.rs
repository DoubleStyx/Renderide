//! Windows named semaphore (`Global\CT.IP.{name}`).

use std::ffi::OsStr;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null_mut;
use std::time::Duration;

use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{
    CreateSemaphoreW, ReleaseSemaphore, WaitForSingleObject, INFINITE,
};

use crate::naming;

/// Win32 semaphore handle.
pub(super) struct WinSemaphore(windows_sys::Win32::Foundation::HANDLE);

impl WinSemaphore {
    pub(super) fn open(memory_view_name: &str) -> io::Result<Self> {
        let full_name = naming::windows_semaphore_wide_name(memory_view_name);
        let name_wide: Vec<u16> = OsStr::new(&full_name)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let handle = unsafe { CreateSemaphoreW(null_mut(), 0, i32::MAX, name_wide.as_ptr()) };
        if handle == 0 || handle == (-1_isize as _) {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(handle))
    }

    pub(super) fn post(&self) {
        let rc = unsafe { ReleaseSemaphore(self.0, 1, null_mut()) };
        if rc == 0 {
            debug_assert!(
                false,
                "ReleaseSemaphore failed: {:?}",
                io::Error::last_os_error()
            );
        }
    }

    pub(super) fn wait_timeout(&self, timeout: Duration) -> bool {
        let ms = if timeout.is_zero() {
            0u32
        } else if timeout.as_secs() > 60 * 60 * 24 * 7 {
            INFINITE
        } else {
            timeout.as_millis().min(u32::MAX as u128) as u32
        };
        let r = unsafe { WaitForSingleObject(self.0, ms) };
        r == WAIT_OBJECT_0
    }
}

impl Drop for WinSemaphore {
    fn drop(&mut self) {
        if self.0 != 0 && self.0 != (-1_isize as _) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}
