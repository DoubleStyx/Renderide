//! Non-Linux Unix (e.g. *BSD): no parent-death signal; children are not auto-signalled on bootstrapper exit.

use std::io;
use std::process::{Child, Command};

/// Empty platform state (no job object or PID list).
pub struct PlatformGroup;

impl PlatformGroup {
    /// Always succeeds.
    pub fn new() -> io::Result<Self> {
        Ok(Self)
    }

    /// No-op: no `PR_SET_PDEATHSIG` on this platform in the bootstrapper build.
    pub fn prepare_command(&self, _cmd: &mut Command) {}

    /// No-op registration.
    pub fn register_spawned(&self, _child: &Child) -> io::Result<()> {
        Ok(())
    }
}
