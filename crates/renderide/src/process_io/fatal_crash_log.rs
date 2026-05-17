//! Fatal process faults (POSIX signals, Windows structured exceptions, macOS Mach exceptions) do
//! not invoke Rust's panic hook. This module registers [`crash_handler::CrashHandler`] so a short
//! line is appended to the **same** log file as [`logger::init_for`], using only pre-opened fds and
//! stack buffers in [`crash_handler::CrashEvent::on_crash`]. On Unix, writes use [`libc::write`]
//! only (async-signal-safe). After [`crate::native_stdio::ensure_stdio_forwarded_to_logger`], fd 2
//! is a pipe; a **duplicate** of the preserved terminal stderr is used for console output when tee
//! is enabled.
//!
//! **Linux `write(2)`:** A failed `write` may set `errno` to **`EINTR`**; the handler must **retry**
//! the same buffer without advancing (POSIX async-signal-safe pattern). Otherwise the first fd
//! (log file) can fail while the second (terminal duplicate) still succeeds.
//!
//! If the dedicated append **log fd** still has **unwritten bytes** after retries, the remainder is
//! written to **fd 2** (the stderr **pipe** to the logger forwarder), so the line can still appear
//! in the log file without using [`logger::log`] (mutex).
//!
//! **macOS:** `crash-handler` uses Mach exception ports, which can interact with other signal
//! machinery (see upstream docs). **Manual testing:** `kill -BUS <pid>` on Linux; Windows fault
//! injection is environment-specific.
//!
//! **Stack traces (Linux + Windows):** after the signal-info line, two additional passes run.
//! Phase 1 walks frames via [`backtrace::trace_unsynchronized`] into a stack array and formats
//! hex instruction pointers into a 2 KB stack buffer (signal-safe, allocation-free). Phase 2
//! best-effort symbolicates through [`backtrace::resolve`] (heap-allocating); both are guarded
//! by a reentry flag plus [`std::panic::catch_unwind`] so a fault inside resolution cannot
//! recurse. Stripped release binaries produce hex only from Phase 2. macOS keeps the
//! signal-info line alone -- the Mach exception callback runs on a dedicated thread, so a plain
//! `trace` walks the wrong stack; proper macOS support requires unwinding from
//! `thread_get_state` and is tracked as follow-up work.
//!
//! **Linux alt signal stack:** libstd's per-thread altstack (~8 KB) is too small for the
//! gimli DWARF parser inside [`backtrace::resolve`] -- a fatal-signal handler running on it
//! aborts Phase 2 partway through with no diagnostic. The Unix install path installs a
//! 512 KB altstack on the main thread before [`crash_handler::CrashHandler::attach`] so
//! Phase 2 has room to complete. Crashes on worker threads still use libstd's small altstack
//! and may lose Phase 2 silently; Phase 1 (hex IPs) remains durable on every thread.

use std::path::Path;

mod format;
#[cfg(any(target_os = "linux", target_os = "android", windows))]
mod stack_trace;
#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

/// Prepares the final log-directory footer for crash-handler output.
///
/// The native crash callback may run in a signal or structured-exception context, so the footer is
/// allocated before the handler is attached and then reused as immutable bytes.
#[cfg(any(unix, windows))]
pub(super) fn log_directory_footer_bytes() -> Box<[u8]> {
    logger::log_directory_footer(logger::logs_root())
        .into_bytes()
        .into_boxed_slice()
}

/// Installs the crash handler after logging and stdio forwarding are initialized.
///
/// Failures are logged and ignored so startup continues without fatal-crash logging.
pub(crate) fn install(log_path: &Path) {
    #[cfg(unix)]
    if let Err(e) = unix::install_impl(log_path) {
        logger::warn!("Failed to install fatal crash log handler: {e}");
    }
    #[cfg(windows)]
    if let Err(e) = windows::install_impl(log_path) {
        logger::warn!("Failed to install fatal crash log handler: {e}");
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = log_path;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(any(unix, windows))]
    fn log_directory_footer_bytes_are_single_line() {
        let footer = super::log_directory_footer_bytes();
        let text = std::str::from_utf8(&footer).expect("utf8 footer");

        assert!(text.starts_with(logger::LOG_DIRECTORY_FOOTER_PREFIX));
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), 1);
    }
}
