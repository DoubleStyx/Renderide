//! Windows: job object with kill-on-close so child processes exit with the bootstrapper.

use std::io;
use std::os::windows::io::AsRawHandle;
use std::process::{Child, Command};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_BASIC_LIMIT_INFORMATION,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// Owns a Windows job object handle; [`Drop`] closes it.
struct JobHandle(HANDLE);

impl Drop for JobHandle {
    fn drop(&mut self) {
        if !self.0.is_null() && self.0 != -1_isize as HANDLE {
            // SAFETY: Win32 `CloseHandle` on a valid job object handle owned by this struct.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

/// Job object assigned to every spawned child.
pub struct PlatformGroup {
    job: JobHandle,
}

impl PlatformGroup {
    /// Allocates a job object with kill-on-close semantics.
    pub fn new() -> io::Result<Self> {
        // SAFETY: Win32 API; `NULL` names create an unnamed job object local to this process.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() || job == -1_isize as HANDLE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: Win32 JOB structs are POD; zeroed fields are valid defaults aside from `LimitFlags`.
        let basic = JOBOBJECT_BASIC_LIMIT_INFORMATION {
            LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            ..unsafe { std::mem::zeroed() }
        };
        let info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
            BasicLimitInformation: basic,
            ..unsafe { std::mem::zeroed() }
        };

        // SAFETY: `SetInformationJobObject` expects a pointer to `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`
        // with the accompanying byte length.
        let r = unsafe {
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if r == 0 {
            // SAFETY: `job` is a valid handle when creation succeeded but configuration failed.
            unsafe {
                CloseHandle(job);
            }
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            job: JobHandle(job),
        })
    }

    /// No extra Win32 flags on the [`Command`] beyond job assignment after spawn.
    pub fn prepare_command(&self, _cmd: &mut Command) {}

    /// Assigns the child process to this job object.
    pub fn register_spawned(&self, child: &Child) -> io::Result<()> {
        let raw = child.as_raw_handle();
        // SAFETY: `raw` is a live process handle for `child` for the duration of this call.
        let ok = unsafe { AssignProcessToJobObject(self.job.0, raw as HANDLE) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}
