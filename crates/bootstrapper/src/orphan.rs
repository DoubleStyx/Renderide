//! Orphan process cleanup from previous crashed runs.
//! Uses the PID file written by the bootstrapper.
//! On Windows, also kills processes by name (renderide.exe, bootstrapper.exe, Renderite.Host.exe)
//! in case the PID file was lost or never written.

use std::fs;
use std::io::Write;
use std::process::Command;

use crate::paths;

/// Process names to kill on Windows when cleaning orphans (exact image names).
/// Excludes bootstrapper.exe since that would kill the current process.
#[cfg(windows)]
const ORPHAN_PROCESS_NAMES: &[&str] = &[
    "renderide.exe",       // Renderide renderer
    "Renderite.Host.exe",  // Host when run as self-contained exe
];

/// Kills orphaned Host/renderer processes from a previous crashed run.
/// Call before spawning anything. Reads PIDs from the PID file; on Unix uses SIGTERM,
/// on Windows uses TerminateProcess. On Windows, also kills by process name.
pub fn kill_orphans() {
    #[cfg(windows)]
    kill_orphans_by_name();

    let path = paths::pid_file_path();
    let Ok(contents) = fs::read_to_string(&path) else {
        return;
    };
    let _ = fs::remove_file(&path);

    for line in contents.lines() {
        let pid = match line
            .strip_prefix("host:")
            .or_else(|| line.strip_prefix("renderer:"))
        {
            Some(rest) => rest.trim().parse::<u32>().ok(),
            None => continue,
        };
        let Some(pid) = pid else {
            continue;
        };
        if process_exists(pid) {
            logger::info!("Killing orphan process {} from previous run", pid);
            kill_process(pid);
        }
    }
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(unix)]
fn kill_process(pid: u32) {
    let _ = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle != 0 && handle != -1 {
        unsafe { CloseHandle(handle) };
        true
    } else {
        false
    }
}

#[cfg(windows)]
fn kill_process(pid: u32) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};

    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
    if handle != 0 && handle != -1 {
        unsafe {
            TerminateProcess(handle, 1);
            CloseHandle(handle);
        }
    }
}

/// On Windows, kills known orphan process names via taskkill.
/// Used when the PID file was lost or never written (e.g. crash before spawn).
#[cfg(windows)]
fn kill_orphans_by_name() {
    for name in ORPHAN_PROCESS_NAMES {
        if let Ok(output) = Command::new("taskkill")
            .args(["/IM", name, "/F"])
            .output()
        {
            if output.status.success() {
                logger::info!("Killed orphan process(es) by name: {}", name);
            }
        }
    }
}

/// Appends a PID entry to the PID file.
pub fn write_pid_file(pid: u32, kind: &str) {
    let path = paths::pid_file_path();
    match fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{}:{}", kind, pid) {
                logger::error!("Failed to write PID file: {}", e);
            }
            let _ = f.flush();
        }
        Err(e) => {
            logger::error!("Failed to open PID file: {}", e);
        }
    }
}

/// Removes the PID file at shutdown.
pub fn remove_pid_file() {
    let _ = fs::remove_file(paths::pid_file_path());
}
