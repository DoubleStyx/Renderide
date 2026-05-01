//! Integration: renderer error logs reach both the log file and preserved terminal stderr.

#![cfg(unix)]

use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::time::SystemTime;

struct StderrRedirect {
    saved: OwnedFd,
}

impl StderrRedirect {
    fn to_file(file: &File) -> Self {
        // SAFETY: `dup` only reads the process stderr fd and returns a new owned fd on success.
        let saved_raw = unsafe { libc::dup(libc::STDERR_FILENO) };
        assert!(
            saved_raw >= 0,
            "dup(stderr): {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: `saved_raw` was just returned by `dup`, is open, and is owned by this guard.
        let saved = unsafe { OwnedFd::from_raw_fd(saved_raw) };
        // SAFETY: `file` is open for writing and fd 2 is the process stderr descriptor.
        let rc = unsafe { libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) };
        assert!(
            rc >= 0,
            "dup2(file -> stderr): {}",
            std::io::Error::last_os_error()
        );
        Self { saved }
    }
}

impl Drop for StderrRedirect {
    fn drop(&mut self) {
        // SAFETY: `self.saved` is an open duplicate of the original stderr descriptor.
        let _ = unsafe { libc::dup2(self.saved.as_raw_fd(), libc::STDERR_FILENO) };
    }
}

#[test]
fn renderer_error_logs_are_mirrored_to_preserved_stderr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let terminal_path = dir.path().join("terminal-stderr.log");
    let terminal_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&terminal_path)
        .expect("open terminal file");
    let _redirect = StderrRedirect::to_file(&terminal_file);

    // SAFETY: env mutation is local to this integration-test process and happens before logger init.
    unsafe {
        std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());
        std::env::set_var("RENDERIDE_LOG_TEE_TERMINAL", "1");
    }

    let ts = format!(
        "error_terminal_mirror_{}",
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let log_path = logger::init_for(
        logger::LogComponent::Renderer,
        &ts,
        logger::LogLevel::Debug,
        false,
    )
    .expect("logger init");

    renderide::ensure_native_stdio_forwarded_to_logger();

    const ERROR_MARKER: &str = "RENDERIDE_ERROR_TERMINAL_MIRROR_MARKER";
    const WARN_MARKER: &str = "RENDERIDE_WARN_NOT_TERMINAL_MIRROR_MARKER";
    logger::warn!("{WARN_MARKER}");
    logger::error!("{ERROR_MARKER}");
    logger::flush();

    let log_contents = std::fs::read_to_string(&log_path).expect("read log");
    assert!(log_contents.contains(WARN_MARKER));
    assert!(log_contents.contains(ERROR_MARKER));

    let terminal_contents = std::fs::read_to_string(&terminal_path).expect("read terminal file");
    assert!(!terminal_contents.contains(WARN_MARKER));
    assert!(terminal_contents.contains(ERROR_MARKER));
}
