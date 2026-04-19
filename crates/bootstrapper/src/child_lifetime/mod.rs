//! Ties child processes to bootstrapper lifetime (job object on Windows, `PR_SET_PDEATHSIG` on Linux,
//! tracked PIDs + `SIGINT`/`SIGTERM`/`SIGKILL` on macOS).
//!
//! On **macOS** there is no `PR_SET_PDEATHSIG` or job object. [`ChildLifetimeGroup`] records each
//! direct child PID from [`Self::register_spawned`] and [`Self::shutdown_tracked_children`] sends
//! `SIGINT` first (so the engine can treat shutdown like Ctrl+C and flush logs), then `SIGTERM`,
//! then `SIGKILL` after a grace period. [`std::sync::Arc`]-shared [`Drop`] on the macOS backend runs the
//! same cleanup so panics or abrupt exits still attempt to tear down children (does not cover
//! `SIGKILL` of the bootstrapper itself).

#[cfg(all(unix, not(windows), not(target_os = "macos"), target_os = "linux"))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(all(unix, not(windows), not(target_os = "macos"), not(target_os = "linux")))]
mod other;
#[cfg(windows)]
mod windows;

#[cfg(windows)]
type Inner = windows::PlatformGroup;
#[cfg(target_os = "macos")]
type Inner = macos::PlatformGroup;
#[cfg(all(unix, not(windows), not(target_os = "macos"), target_os = "linux"))]
type Inner = linux::PlatformGroup;
#[cfg(all(unix, not(windows), not(target_os = "macos"), not(target_os = "linux")))]
type Inner = other::PlatformGroup;

use std::io;
use std::process::{Child, Command};

/// Holds OS resources so direct children are terminated when the bootstrapper exits unexpectedly.
pub struct ChildLifetimeGroup(Inner);

impl ChildLifetimeGroup {
    /// Creates a lifetime group (job object on Windows, PID list on macOS, otherwise empty).
    pub fn new() -> io::Result<Self> {
        Ok(Self(Inner::new()?))
    }

    /// Applies platform-specific options so the child exits when the bootstrapper dies (where supported).
    pub fn prepare_command(&self, cmd: &mut Command) {
        self.0.prepare_command(cmd);
    }

    /// Registers a spawned direct child (required on Windows for job assignment; tracks PIDs on macOS).
    pub fn register_spawned(&self, child: &Child) -> io::Result<()> {
        self.0.register_spawned(child)
    }

    /// Sends `SIGINT`, then `SIGTERM`, then `SIGKILL` after grace periods, to every PID registered via [`Self::register_spawned`].
    ///
    /// `SIGINT` is sent first so children can run the same path as interactive Ctrl+C (e.g. flush
    /// logging); `SIGTERM` follows if the process is still running.
    ///
    /// Idempotent: clears the tracking list on first run; later calls are no-ops until new children register.
    #[cfg(target_os = "macos")]
    pub fn shutdown_tracked_children(&self) {
        self.0.shutdown();
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

    #[cfg(unix)]
    #[test]
    fn prepare_command_true_round_trip() {
        let g = ChildLifetimeGroup::new().expect("group");
        let mut cmd = std::process::Command::new("true");
        g.prepare_command(&mut cmd);
        assert!(cmd.status().expect("status").success());
    }

    #[cfg(windows)]
    #[test]
    fn prepare_command_cmd_round_trip() {
        let g = ChildLifetimeGroup::new().expect("group");
        let mut cmd = std::process::Command::new("cmd.exe");
        cmd.arg("/c").arg("exit").arg("0");
        g.prepare_command(&mut cmd);
        assert!(cmd.status().expect("status").success());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shutdown_tracked_children_empty_no_panic() {
        let g = ChildLifetimeGroup::new().expect("group");
        g.shutdown_tracked_children();
    }

    #[cfg(unix)]
    #[test]
    fn register_spawned_with_exiting_child() {
        let g = ChildLifetimeGroup::new().expect("group");
        let mut cmd = std::process::Command::new("true");
        g.prepare_command(&mut cmd);
        let mut child = cmd.spawn().expect("spawn");
        let _ = g.register_spawned(&child);
        let _ = child.wait();
    }

    #[cfg(windows)]
    #[test]
    fn register_spawned_with_exiting_child_windows() {
        let g = ChildLifetimeGroup::new().expect("group");
        let mut cmd = std::process::Command::new("cmd.exe");
        cmd.arg("/c").arg("exit").arg("0");
        g.prepare_command(&mut cmd);
        let mut child = cmd.spawn().expect("spawn");
        let _ = g.register_spawned(&child);
        let _ = child.wait();
    }
}
