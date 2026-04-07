//! Ties child processes to bootstrapper lifetime (job object on Windows, `PR_SET_PDEATHSIG` on Linux,
//! tracked PIDs + `SIGTERM`/`SIGKILL` on macOS).
//!
//! On **macOS** there is no `PR_SET_PDEATHSIG` or job object. [`ChildLifetimeGroup`] records each
//! direct child PID from [`Self::register_spawned`] and [`Self::shutdown_tracked_children`] sends
//! `SIGTERM`, then `SIGKILL` after a short grace period. [`std::sync::Arc`]-shared [`Drop`] runs the
//! same cleanup so panics or abrupt exits still attempt to tear down children (does not cover
//! `SIGKILL` of the bootstrapper itself).

use std::io;
use std::process::{Child, Command};
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};
#[cfg(target_os = "macos")]
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_BASIC_LIMIT_INFORMATION,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// Holds OS resources so direct children are terminated when the bootstrapper exits unexpectedly.
pub struct ChildLifetimeGroup {
    #[cfg(windows)]
    job: JobHandle,
    /// Direct child PIDS (`dotnet` / Wine `start`, renderer) for coordinated shutdown on Darwin.
    #[cfg(target_os = "macos")]
    macos_tracked_pids: Arc<Mutex<Vec<u32>>>,
}

#[cfg(windows)]
struct JobHandle(HANDLE);

#[cfg(windows)]
impl Drop for JobHandle {
    fn drop(&mut self) {
        if self.0 != 0 && self.0 != -1_isize as HANDLE {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

impl ChildLifetimeGroup {
    /// Creates a lifetime group. On Windows, allocates a job object; elsewhere succeeds with no resources.
    pub fn new() -> io::Result<Self> {
        #[cfg(windows)]
        {
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job == 0 || job == -1_isize as HANDLE {
                return Err(io::Error::last_os_error());
            }
            let basic = JOBOBJECT_BASIC_LIMIT_INFORMATION {
                LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                ..unsafe { std::mem::zeroed() }
            };
            let info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
                BasicLimitInformation: basic,
                ..unsafe { std::mem::zeroed() }
            };

            let r = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if r == 0 {
                unsafe {
                    CloseHandle(job);
                }
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                job: JobHandle(job),
            })
        }
        #[cfg(target_os = "macos")]
        {
            Ok(Self {
                macos_tracked_pids: Arc::new(Mutex::new(Vec::new())),
            })
        }
        #[cfg(all(not(windows), not(target_os = "macos")))]
        {
            Ok(Self {})
        }
    }

    /// Applies platform-specific options so the child exits when the bootstrapper dies (where supported).
    pub fn prepare_command(&self, cmd: &mut Command) {
        let _ = self;
        #[cfg(target_os = "linux")]
        linux::apply_parent_death_signal(cmd);
    }

    /// Registers a spawned direct child (required on Windows for job assignment; tracks PIDs on macOS).
    pub fn register_spawned(&self, child: &Child) -> io::Result<()> {
        #[cfg(windows)]
        {
            let raw = child.as_raw_handle();
            let ok = unsafe { AssignProcessToJobObject(self.job.0, raw as HANDLE) };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(target_os = "macos")]
        {
            let pid = child.id();
            if pid != 0 {
                self.macos_tracked_pids
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(pid);
            }
        }
        #[cfg(all(not(windows), not(target_os = "macos")))]
        {
            let _ = child;
        }
        Ok(())
    }

    /// Sends `SIGTERM`, then `SIGKILL` after a grace period, to every PID registered via [`Self::register_spawned`].
    ///
    /// Idempotent: clears the tracking list on first run; later calls are no-ops until new children register.
    #[cfg(target_os = "macos")]
    pub fn shutdown_tracked_children(&self) {
        let pids: Vec<u32> = {
            let mut guard = self
                .macos_tracked_pids
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };
        if pids.is_empty() {
            return;
        }
        logger::info!(
            "macOS: stopping {} direct child process(es) (SIGTERM, then SIGKILL if needed)",
            pids.len()
        );
        for &pid in &pids {
            macos_kill(pid, libc::SIGTERM);
        }
        std::thread::sleep(Duration::from_millis(800));
        for &pid in &pids {
            macos_kill(pid, libc::SIGKILL);
        }
    }
}

#[cfg(target_os = "macos")]
fn macos_kill(pid: u32, signal: libc::c_int) {
    if pid == 0 {
        return;
    }
    // SAFETY: libc kill; ESRCH means process already gone.
    let rc = unsafe { libc::kill(pid as libc::pid_t, signal) };
    if rc == 0 {
        return;
    }
    let err = io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::ESRCH) {
        return;
    }
    logger::debug!("macOS: kill(pid={}, sig={}) — {}", pid, signal, err);
}

#[cfg(target_os = "macos")]
impl Drop for ChildLifetimeGroup {
    fn drop(&mut self) {
        self.shutdown_tracked_children();
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    /// Requests `SIGTERM` when the parent (bootstrapper) exits.
    pub(super) fn apply_parent_death_signal(cmd: &mut Command) {
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_lifetime_group_new_succeeds() {
        let g = ChildLifetimeGroup::new();
        assert!(g.is_ok(), "{:?}", g.err());
    }
}
