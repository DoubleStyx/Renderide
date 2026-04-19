//! Linux: `PR_SET_PDEATHSIG` so children receive `SIGTERM` when the bootstrapper exits.

use std::io;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};

/// No kernel objects; applies parent-death signal in [`Self::prepare_command`].
pub struct PlatformGroup;

impl PlatformGroup {
    /// Always succeeds (no resources).
    pub fn new() -> io::Result<Self> {
        Ok(Self)
    }

    /// Requests `SIGTERM` when the parent (bootstrapper) exits.
    pub fn prepare_command(&self, cmd: &mut Command) {
        apply_parent_death_signal(cmd);
    }

    /// No PID tracking on Linux (kernel handles parent death).
    pub fn register_spawned(&self, _child: &Child) -> io::Result<()> {
        Ok(())
    }
}

fn apply_parent_death_signal(cmd: &mut Command) {
    // SAFETY: `pre_exec` runs in the child after `fork` and before `exec`; `prctl` only affects the child.
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}
