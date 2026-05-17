//! Windows install path: opens the log file, duplicates the preserved terminal stderr file,
//! and attaches the crash handler with a vectored-exception-safe callback.

use std::path::Path;
use std::sync::OnceLock;

use crash_handler::{CrashEventResult, CrashHandler};
use parking_lot::Mutex;

use super::format::format_fatal_line_windows;
use super::stack_trace::write_stack_trace;

/// Output sinks routed by the Windows fatal-crash callback.
struct WindowsCrashFds {
    /// [`parking_lot::Mutex`] allows writing through [`OnceLock::get`] (`&` only). The crash
    /// path is not async-signal-safe like Linux; Windows structured exceptions follow
    /// different rules, and `parking_lot` primitives are sound in vectored-exception
    /// callbacks.
    log: Mutex<std::fs::File>,
    /// Optional duplicate of the launching terminal stderr, used for the dual-output tee.
    term: Option<Mutex<std::fs::File>>,
    /// Preformatted final line pointing at the shared logs root.
    log_directory_footer: Box<[u8]>,
}

static WINDOWS_CRASH_FDS: OnceLock<WindowsCrashFds> = OnceLock::new();

/// Opens the log file, duplicates the preserved terminal stderr file for tee output, and
/// attaches [`crash_handler::CrashHandler`] with a Windows-structured-exception callback.
pub(super) fn install_impl(log_path: &Path) -> Result<(), String> {
    use std::fs::OpenOptions;

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| e.to_string())?;
    let term = crate::native_stdio::duplicate_preserved_stderr_file_for_crash_log();

    WINDOWS_CRASH_FDS
        .set(WindowsCrashFds {
            log: Mutex::new(log),
            term: term.map(Mutex::new),
            log_directory_footer: super::log_directory_footer_bytes(),
        })
        .map_err(|_e| "fatal crash log fds already installed".to_string())?;

    // SAFETY: installs a process-wide vectored exception handler; the closure only performs
    // synchronous writes to files guarded by mutexes. Called once at startup; handler is
    // leaked below so the installation persists for process lifetime.
    let handler = unsafe {
        CrashHandler::attach(crash_handler::make_crash_event(|ctx| {
            let mut buf = [0u8; 224];
            let n = format_fatal_line_windows(ctx, &mut buf);
            let data = &buf[..n];
            if let Some(fds) = WINDOWS_CRASH_FDS.get() {
                fds.write_all(data);
                let mut context_buf = [0u8; 512];
                let context_n =
                    crate::diagnostics::crash_context::write_minimal_snapshot(&mut context_buf);
                fds.write_all(&context_buf[..context_n]);
                const NO_POSIX_SIGNAL: i32 = 0;
                write_stack_trace(NO_POSIX_SIGNAL, |chunk| fds.write_all(chunk));
                fds.write_all(&fds.log_directory_footer);
            }
            CrashEventResult::from(false)
        }))
        .map_err(|e| e.to_string())?
    };
    #[expect(
        clippy::mem_forget,
        reason = "CrashHandler must not be dropped after attach; the handler is process-global until exit"
    )]
    std::mem::forget(handler);
    Ok(())
}

impl WindowsCrashFds {
    /// Writes `data` to the log file and (if configured) the terminal duplicate, matching the
    /// dual-output routing of the existing Unix path. Individual write errors are swallowed --
    /// the crash handler has no meaningful recovery path.
    fn write_all(&self, data: &[u8]) {
        use std::io::Write;
        {
            let mut g = self.log.lock();
            let _ = g.write_all(data);
            let _ = g.flush();
        }
        if let Some(t) = &self.term {
            let mut g = t.lock();
            let _ = g.write_all(data);
            let _ = g.flush();
        }
    }
}
