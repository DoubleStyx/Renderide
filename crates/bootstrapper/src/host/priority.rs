//! Host process priority adjustments.

use std::process::Child;

/// Raises Host process priority on Windows (ResoBoot `AboveNormal`).
#[cfg(windows)]
pub fn set_host_above_normal_priority(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Threading::{ABOVE_NORMAL_PRIORITY_CLASS, SetPriorityClass};

    let handle = child.as_raw_handle();
    // SAFETY: `handle` is a valid process handle from `Child` until the child is reaped.
    let rc = unsafe { SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS) };
    if rc == 0 {
        logger::warn!(
            "SetPriorityClass failed: {}",
            std::io::Error::last_os_error()
        );
    } else {
        logger::info!("Host process priority set to AboveNormal");
    }
}

/// Non-Windows hosts keep the OS-default child process priority.
#[cfg(not(windows))]
pub const fn set_host_above_normal_priority(_child: &Child) {}
