//! Redirect native **stdout** and **stderr** into the Renderide file logger on Unix and Windows.
//!
//! Vulkan validation layers and **spirv-val** often emit via **`printf`** (stdout) and/or
//! **`fprintf(stderr, …)`**. WGPU’s instance flags do not control whether users enable layers via
//! `VK_INSTANCE_LAYERS`, so the renderer installs forwarding **unconditionally** after file logging
//! starts (see [`crate::app::run`]).
//!
//! OpenXR runtimes use the same native paths; [`crate::xr::bootstrap::init_wgpu_openxr`] also calls
//! [`ensure_stdio_forwarded_to_logger`] for entry points that skip `run` (idempotent via [`Once`]).
//!
//! - **Unix:** `pipe` + `dup2` per stream (`STDOUT_FILENO` / `STDERR_FILENO`).
//! - **Windows:** `CreatePipe` + `SetStdHandle(STD_OUTPUT_HANDLE / STD_ERROR_HANDLE, …)`.
//!
//! The readers use [`logger::try_log`] (non-blocking lock + append fallback) so they cannot deadlock
//! with the main thread on the global logger mutex, and read in chunks so a missing newline cannot
//! fill the pipe and block writers.
//!
//! **Terminal tee:** Before redirecting, the original console file descriptors / handles are kept.
//! Each forwarded line is also written to that original stream so Vulkan validation and similar
//! output appears on the launching terminal as well as in the log file. Disable with
//! **`RENDERIDE_LOG_TEE_TERMINAL=0`** (or `false` / `no`) for CI or headless runs.
//!
//! On other targets this module is a no-op.
//!
//! Avoid enabling the logger’s **mirror-to-stderr** option together with this redirect: mirrored
//! lines would be written back into the pipe and re-logged. Tee uses the **preserved** handles, not
//! [`std::io::stderr`].

use std::io::Write;
use std::sync::{Mutex, Once, OnceLock};

use logger::LogLevel;

static INSTALL: Once = Once::new();

#[cfg(unix)]
static PRESERVED_STDERR: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

#[cfg(unix)]
static PRESERVED_STDOUT: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

#[cfg(windows)]
static PRESERVED_STDERR: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

#[cfg(windows)]
static PRESERVED_STDOUT: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

/// Which standard stream was redirected; used to tee to the matching preserved handle.
#[derive(Clone, Copy, Debug)]
enum StdioStream {
    Stdout,
    Stderr,
}

/// When `false`, forwarded native lines and panic terminal output are not copied to the original
/// console (log file only). Default: enabled unless `RENDERIDE_LOG_TEE_TERMINAL` is `0`, `false`,
/// or `no` (case-insensitive).
fn tee_terminal_enabled() -> bool {
    match std::env::var("RENDERIDE_LOG_TEE_TERMINAL") {
        Ok(v) => {
            let v = v.trim();
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        }
        Err(_) => true,
    }
}

/// Ensures process **stdout** and **stderr** are forwarded to [`logger`] and no longer write to the
/// original terminal streams. Idempotent.
pub(crate) fn ensure_stdio_forwarded_to_logger() {
    INSTALL.call_once(|| {
        #[cfg(unix)]
        {
            if let Err(e) = try_redirect_unix_stream(
                libc::STDERR_FILENO,
                "renderide-stderr",
                StdioStream::Stderr,
            ) {
                logger::warn!("Native stderr could not be redirected to log file: {e}");
            }
            if let Err(e) = try_redirect_unix_stream(
                libc::STDOUT_FILENO,
                "renderide-stdout",
                StdioStream::Stdout,
            ) {
                logger::warn!("Native stdout could not be redirected to log file: {e}");
            }
        }
        #[cfg(windows)]
        {
            if let Err(e) = try_redirect_windows_stream(
                windows_sys::Win32::System::Console::STD_ERROR_HANDLE,
                "renderide-stderr",
                StdioStream::Stderr,
            ) {
                logger::warn!("Native stderr could not be redirected to log file: {e}");
            }
            if let Err(e) = try_redirect_windows_stream(
                windows_sys::Win32::System::Console::STD_OUTPUT_HANDLE,
                "renderide-stdout",
                StdioStream::Stdout,
            ) {
                logger::warn!("Native stdout could not be redirected to log file: {e}");
            }
        }
    });
}

/// Writes `data` to the process’s **original** stderr (before [`ensure_stdio_forwarded_to_logger`]),
/// for panic reporting. No-op if redirect did not run, tee is disabled, or the platform is
/// unsupported.
pub(crate) fn try_write_preserved_stderr(data: &[u8]) {
    if !tee_terminal_enabled() {
        return;
    }
    #[cfg(any(unix, windows))]
    if let Some(m) = PRESERVED_STDERR.get() {
        if let Ok(mut f) = m.lock() {
            let _ = f.write_all(data);
            let _ = f.flush();
        }
    }
}

/// Duplicates the **preserved** stderr stream (the launching terminal) for async-signal-safe
/// `write` from a fatal crash handler. Call only after [`ensure_stdio_forwarded_to_logger`].
///
/// Returns [`None`] when tee is disabled, stderr was not redirected, or duplication fails.
#[cfg(unix)]
pub(crate) fn duplicate_preserved_stderr_raw_fd() -> Option<std::os::fd::OwnedFd> {
    if !tee_terminal_enabled() {
        return None;
    }
    let m = PRESERVED_STDERR.get()?;
    let guard = m.lock().ok()?;
    use std::os::fd::AsFd;
    guard.as_fd().try_clone_to_owned().ok()
}

/// See [`duplicate_preserved_stderr_raw_fd`]. Windows uses a duplicated [`std::fs::File`].
#[cfg(windows)]
pub(crate) fn duplicate_preserved_stderr_file_for_crash_log() -> Option<std::fs::File> {
    if !tee_terminal_enabled() {
        return None;
    }
    let m = PRESERVED_STDERR.get()?;
    let guard = m.lock().ok()?;
    guard.try_clone().ok()
}

#[cfg(any(unix, windows))]
fn try_write_preserved_stdout(data: &[u8]) {
    if !tee_terminal_enabled() {
        return;
    }
    if let Some(m) = PRESERVED_STDOUT.get() {
        if let Ok(mut f) = m.lock() {
            let _ = f.write_all(data);
            let _ = f.flush();
        }
    }
}

#[cfg(unix)]
fn store_preserved_unix(stream: StdioStream, saved: i32) {
    use std::fs::File;
    use std::os::fd::FromRawFd;
    use std::os::fd::OwnedFd;

    // SAFETY: `saved` was just produced by `libc::dup`, is open, owned by this process, and has
    // not been handed to another `OwnedFd`/`File`. Transferring ownership to `OwnedFd` is sound.
    let owned = unsafe { OwnedFd::from_raw_fd(saved) };
    let file = File::from(owned);
    let cell = match stream {
        StdioStream::Stderr => &PRESERVED_STDERR,
        StdioStream::Stdout => &PRESERVED_STDOUT,
    };
    let _ = cell.set(Mutex::new(file));
}

#[cfg(unix)]
fn try_redirect_unix_stream(
    target_fd: i32,
    thread_name: &'static str,
    stream: StdioStream,
) -> Result<(), String> {
    use std::thread;

    // SAFETY: all libc calls below operate on file descriptors that this function either just
    // created (via `pipe`/`dup`) or received from the caller (`target_fd` is always a valid
    // stdio fd). Ownership is tracked manually: each branch that errors out closes every fd it
    // created; the success path transfers ownership into `OwnedFd` via `store_preserved_unix`.
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

        let saved = libc::dup(target_fd);
        if saved < 0 {
            libc::close(rfd);
            libc::close(wfd);
            return Err(format!(
                "dup({target_fd}): {}",
                std::io::Error::last_os_error()
            ));
        }

        if libc::dup2(wfd, target_fd) < 0 {
            let e = std::io::Error::last_os_error();
            libc::close(rfd);
            libc::close(wfd);
            libc::close(saved);
            return Err(format!("dup2(pipe -> fd {target_fd}): {e}"));
        }
        libc::close(wfd);

        let spawn = thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || forward_pipe_lines_to_logger_unix(rfd, stream));

        match spawn {
            Ok(_) => {
                store_preserved_unix(stream, saved);
                Ok(())
            }
            Err(e) => {
                let _ = libc::dup2(saved, target_fd);
                libc::close(rfd);
                libc::close(saved);
                Err(format!("thread spawn: {e}"))
            }
        }
    }
}

#[cfg(windows)]
fn store_preserved_windows(stream: StdioStream, file: std::fs::File) {
    let cell = match stream {
        StdioStream::Stderr => &PRESERVED_STDERR,
        StdioStream::Stdout => &PRESERVED_STDOUT,
    };
    let _ = cell.set(Mutex::new(file));
}

#[cfg(windows)]
fn try_redirect_windows_stream(
    std_handle: u32,
    thread_name: &'static str,
    stream: StdioStream,
) -> Result<(), String> {
    use std::fs::File;
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use std::ptr;
    use std::thread;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle};
    use windows_sys::Win32::System::Pipes::CreatePipe;

    // SAFETY: Win32 API calls on handles this function owns; each error path closes every
    // handle it created, and the success path transfers handles into `OwnedHandle`/`File`.
    unsafe {
        let mut read_h: HANDLE = INVALID_HANDLE_VALUE;
        let mut write_h: HANDLE = INVALID_HANDLE_VALUE;
        if CreatePipe(&mut read_h, &mut write_h, ptr::null(), 0) == 0 {
            return Err(format!("CreatePipe: {}", std::io::Error::last_os_error()));
        }

        let old = GetStdHandle(std_handle);
        if old.is_null() || old == INVALID_HANDLE_VALUE {
            CloseHandle(read_h);
            CloseHandle(write_h);
            return Err(format!(
                "GetStdHandle({std_handle}): {}",
                std::io::Error::last_os_error()
            ));
        }

        if SetStdHandle(std_handle, write_h) == 0 {
            CloseHandle(read_h);
            CloseHandle(write_h);
            return Err(format!("SetStdHandle: {}", std::io::Error::last_os_error()));
        }

        let read_owned = OwnedHandle::from_raw_handle(read_h);

        let spawn = thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || {
                let f = File::from(read_owned);
                forward_pipe_lines_to_logger_impl(f, stream);
            });

        match spawn {
            Ok(_) => {
                let old_owned = OwnedHandle::from_raw_handle(old);
                let preserved_file = File::from(old_owned);
                store_preserved_windows(stream, preserved_file);
                Ok(())
            }
            Err(e) => {
                let _ = SetStdHandle(std_handle, old);
                CloseHandle(write_h);
                Err(format!("thread spawn: {e}"))
            }
        }
    }
}

#[cfg(any(unix, windows))]
fn forward_pipe_lines_to_logger_impl<R: std::io::Read>(mut reader: R, stream: StdioStream) {
    let mut pending = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => {
                if !pending.is_empty() {
                    emit_stdio_line(&pending, LogLevel::Info, stream);
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
                    emit_stdio_line(&line, LogLevel::Info, stream);
                }
            }
            Err(e) => {
                let _ = logger::try_log(
                    LogLevel::Debug,
                    format_args!("stdio forward read ended: {e}"),
                );
                break;
            }
        }
    }
}

#[cfg(unix)]
fn forward_pipe_lines_to_logger_unix(rfd: i32, stream: StdioStream) {
    use std::fs::File;
    use std::os::unix::io::FromRawFd;

    // SAFETY: `rfd` is the read end of the pipe created in `try_redirect_unix_stream`; ownership
    // is transferred exclusively to the spawned thread via this call and has no other owner.
    let f = unsafe { File::from_raw_fd(rfd) };
    forward_pipe_lines_to_logger_impl(f, stream);
}

#[cfg(any(unix, windows))]
fn emit_stdio_line(line: &[u8], level: LogLevel, stream: StdioStream) {
    let t = String::from_utf8_lossy(line).trim().to_string();
    if t.is_empty() {
        return;
    }
    let _ = logger::try_log(level, format_args!("{t}"));
    let mut out = t;
    out.push('\n');
    let bytes = out.as_bytes();
    match stream {
        StdioStream::Stderr => try_write_preserved_stderr(bytes),
        StdioStream::Stdout => try_write_preserved_stdout(bytes),
    }
}

#[cfg(test)]
mod tee_env_tests {
    use super::tee_terminal_enabled;

    /// Environment variable consulted by [`tee_terminal_enabled`].
    const VAR: &str = "RENDERIDE_LOG_TEE_TERMINAL";

    /// RAII guard that restores the original value of [`VAR`] (or unsets it) on drop so tests do
    /// not leak process-global state.
    struct EnvGuard(Option<String>);

    impl EnvGuard {
        /// Captures the current value for later restoration.
        fn capture() -> Self {
            Self(std::env::var(VAR).ok())
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => std::env::set_var(VAR, v),
                None => std::env::remove_var(VAR),
            }
        }
    }

    /// All env-var parsing cases are exercised in a single serialized test because
    /// [`std::env::set_var`] mutates process-global state.
    #[test]
    fn tee_terminal_enabled_parses_env_var() {
        let _guard = EnvGuard::capture();

        std::env::remove_var(VAR);
        assert!(tee_terminal_enabled(), "unset should default to enabled");

        for disabled in ["0", "false", "no", "off", "FALSE", "  No  "] {
            std::env::set_var(VAR, disabled);
            assert!(
                !tee_terminal_enabled(),
                "value {disabled:?} should disable tee"
            );
        }

        for enabled in ["1", "true", "yes", "", "anything else"] {
            std::env::set_var(VAR, enabled);
            assert!(
                tee_terminal_enabled(),
                "value {enabled:?} should keep tee enabled"
            );
        }
    }
}
