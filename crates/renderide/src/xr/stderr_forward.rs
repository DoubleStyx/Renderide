//! Redirect native **stderr** into the Renderide file logger on Unix and Windows.
//!
//! OpenXR runtimes often log with `fprintf(stderr, ...)` from C/C++. That bypasses
//! [`XR_EXT_debug_utils`] and Rust’s [`std::io::stderr`]. A pipe plus a background reader sends
//! those messages to [`logger`] instead of the terminal.
//!
//! - **Unix:** `pipe` + `dup2` so fd 2 is the pipe write end.
//! - **Windows:** Win32 `CreatePipe` plus `SetStdHandle(STD_ERROR_HANDLE, …)` so the standard error
//!   handle is the pipe write end.
//!
//! The reader uses [`logger::try_log`] (non-blocking lock + append fallback) so it cannot deadlock
//! with the main thread on the global logger mutex, and reads the pipe in chunks so a missing
//! newline cannot fill the pipe and block all writers.
//!
//! On other targets this module is a no-op.
//!
//! Avoid enabling the logger’s **mirror-to-stderr** option together with this redirect: mirrored
//! lines would be written back into the pipe and re-logged.

use std::sync::Once;

use logger::LogLevel;

static INSTALL: Once = Once::new();

/// Ensures process stderr is forwarded to [`logger`] and no longer writes to the original terminal
/// stream. Idempotent.
pub(crate) fn ensure_stderr_forwarded_to_logger() {
    INSTALL.call_once(|| {
        #[cfg(unix)]
        {
            if let Err(e) = try_redirect_unix_stderr() {
                logger::warn!("Native stderr could not be redirected to log file: {e}");
            }
        }
        #[cfg(windows)]
        {
            if let Err(e) = try_redirect_windows_stderr() {
                logger::warn!("Native stderr could not be redirected to log file: {e}");
            }
        }
    });
}

#[cfg(unix)]
fn try_redirect_unix_stderr() -> Result<(), String> {
    use std::thread;

    unsafe {
        let mut fds = [0i32; 2];
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return Err(format!("pipe: {}", std::io::Error::last_os_error()));
        }
        let rfd = fds[0];
        let wfd = fds[1];

        for fd in [rfd, wfd] {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags >= 0 {
                libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
            }
        }

        let saved_tty = libc::dup(libc::STDERR_FILENO);
        if saved_tty < 0 {
            libc::close(rfd);
            libc::close(wfd);
            return Err(format!(
                "dup(STDERR_FILENO): {}",
                std::io::Error::last_os_error()
            ));
        }

        if libc::dup2(wfd, libc::STDERR_FILENO) < 0 {
            let e = std::io::Error::last_os_error();
            libc::close(rfd);
            libc::close(wfd);
            libc::close(saved_tty);
            return Err(format!("dup2(pipe -> stderr): {e}"));
        }
        libc::close(wfd);

        let spawn = thread::Builder::new()
            .name("renderide-stderr".into())
            .spawn(move || forward_pipe_lines_to_logger_unix(rfd));

        match spawn {
            Ok(_) => {
                libc::close(saved_tty);
                Ok(())
            }
            Err(e) => {
                let _ = libc::dup2(saved_tty, libc::STDERR_FILENO);
                libc::close(rfd);
                libc::close(saved_tty);
                Err(format!("thread spawn: {e}"))
            }
        }
    }
}

#[cfg(windows)]
fn try_redirect_windows_stderr() -> Result<(), String> {
    use std::ptr;
    use std::thread;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle, STD_ERROR_HANDLE};
    use windows_sys::Win32::System::Pipes::CreatePipe;

    unsafe {
        let mut read_h: HANDLE = INVALID_HANDLE_VALUE;
        let mut write_h: HANDLE = INVALID_HANDLE_VALUE;
        if CreatePipe(&mut read_h, &mut write_h, ptr::null(), 0) == 0 {
            return Err(format!("CreatePipe: {}", std::io::Error::last_os_error()));
        }

        let old_stderr = GetStdHandle(STD_ERROR_HANDLE);
        if old_stderr == 0 || old_stderr == INVALID_HANDLE_VALUE {
            CloseHandle(read_h);
            CloseHandle(write_h);
            return Err(format!(
                "GetStdHandle(STD_ERROR_HANDLE): {}",
                std::io::Error::last_os_error()
            ));
        }

        if SetStdHandle(STD_ERROR_HANDLE, write_h) == 0 {
            CloseHandle(read_h);
            CloseHandle(write_h);
            return Err(format!("SetStdHandle: {}", std::io::Error::last_os_error()));
        }

        let spawn = thread::Builder::new()
            .name("renderide-stderr".into())
            .spawn(move || forward_pipe_lines_to_logger_windows(read_h));

        match spawn {
            Ok(_) => {
                let _ = CloseHandle(old_stderr);
                Ok(())
            }
            Err(e) => {
                let _ = SetStdHandle(STD_ERROR_HANDLE, old_stderr);
                CloseHandle(read_h);
                CloseHandle(write_h);
                Err(format!("thread spawn: {e}"))
            }
        }
    }
}

#[cfg(any(unix, windows))]
fn forward_pipe_lines_to_logger_impl<R: std::io::Read>(mut reader: R) {
    let mut pending = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => {
                if !pending.is_empty() {
                    emit_stderr_line(&pending, LogLevel::Info);
                }
                break;
            }
            Ok(n) => {
                pending.extend_from_slice(&chunk[..n]);
                while let Some(pos) = pending.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = pending.drain(..pos).collect();
                    if !pending.is_empty() && pending[0] == b'\n' {
                        pending.remove(0);
                    }
                    emit_stderr_line(&line, LogLevel::Info);
                }
            }
            Err(e) => {
                let _ = logger::try_log(
                    LogLevel::Debug,
                    format_args!("stderr forward read ended: {e}"),
                );
                break;
            }
        }
    }
}

#[cfg(unix)]
fn forward_pipe_lines_to_logger_unix(rfd: i32) {
    use std::fs::File;
    use std::os::unix::io::FromRawFd;

    let f = unsafe { File::from_raw_fd(rfd) };
    forward_pipe_lines_to_logger_impl(f);
}

#[cfg(windows)]
fn forward_pipe_lines_to_logger_windows(read_h: windows_sys::Win32::Foundation::HANDLE) {
    use std::fs::File;
    use std::os::windows::io::FromRawHandle;

    let f = unsafe { File::from_raw_handle(read_h) };
    forward_pipe_lines_to_logger_impl(f);
}

#[cfg(any(unix, windows))]
fn emit_stderr_line(line: &[u8], level: LogLevel) {
    let t = String::from_utf8_lossy(line).trim().to_string();
    if t.is_empty() {
        return;
    }
    let _ = logger::try_log(level, format_args!("{t}"));
}
