//! macOS: track direct child PIDs and escalate shutdown `SIGINT` -> `SIGTERM` -> `SIGKILL`.

use std::io;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

use crate::constants::{
    macos_shutdown_sigint_to_sigterm_delay, macos_shutdown_sigterm_to_sigkill_delay,
};

/// Records PIDs from [`Self::register_spawned`] for coordinated teardown.
pub struct PlatformGroup {
    tracked_pids: Arc<Mutex<Vec<u32>>>,
}

impl PlatformGroup {
    /// Creates an empty PID list.
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            tracked_pids: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// No `PR_SET_PDEATHSIG` equivalent; children rely on signal escalation from [`Self::shutdown`].
    pub fn prepare_command(&self, _cmd: &mut Command) {}

    /// Records non-zero child PIDs for [`Self::shutdown`].
    pub fn register_spawned(&self, child: &Child) -> io::Result<()> {
        let pid = child.id();
        if pid != 0 {
            self.tracked_pids
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(pid);
        }
        Ok(())
    }

    /// Sends `SIGINT`, then `SIGTERM`, then `SIGKILL` after grace periods, to every registered PID.
    ///
    /// Idempotent: clears the tracking list on first run.
    pub fn shutdown(&self) {
        let pids: Vec<u32> = {
            let mut guard = self.tracked_pids.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };
        if pids.is_empty() {
            return;
        }
        logger::info!(
            "macOS: stopping {} direct child process(es) (SIGINT, then SIGTERM, then SIGKILL if needed)",
            pids.len()
        );
        for &pid in &pids {
            macos_kill(pid, libc::SIGINT);
        }
        std::thread::sleep(macos_shutdown_sigint_to_sigterm_delay());
        for &pid in &pids {
            macos_kill(pid, libc::SIGTERM);
        }
        std::thread::sleep(macos_shutdown_sigterm_to_sigkill_delay());
        for &pid in &pids {
            macos_kill(pid, libc::SIGKILL);
        }
    }
}

impl Drop for PlatformGroup {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn macos_kill(pid: u32, signal: libc::c_int) {
    if pid == 0 {
        return;
    }
    // SAFETY: libc `kill`; `ESRCH` means the process is already gone.
    let rc = unsafe { libc::kill(pid as libc::pid_t, signal) };
    if rc == 0 {
        return;
    }
    let err = io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::ESRCH) {
        return;
    }
    logger::debug!("macOS: kill(pid={}, sig={}) -- {}", pid, signal, err);
}
