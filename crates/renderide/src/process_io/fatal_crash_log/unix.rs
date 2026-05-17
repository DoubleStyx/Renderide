//! Unix install path: opens the log fd, duplicates the preserved terminal stderr fd, installs
//! an enlarged alt signal stack on Linux/Android, and attaches the crash handler with a
//! signal-safe callback.

use std::path::Path;
use std::sync::OnceLock;

use crash_handler::{CrashEventResult, CrashHandler};

use super::format::format_fatal_line_unix;
#[cfg(any(target_os = "linux", target_os = "android"))]
use super::stack_trace::write_stack_trace;

/// Global state for raw [`libc::write`] targets (log file + optional terminal duplicate).
struct UnixCrashFds {
    /// Dedicated append fd for the renderer log file.
    log_fd: std::os::unix::io::RawFd,
    /// Optional duplicate of the launching terminal stderr, used for tee output.
    term_fd: Option<std::os::unix::io::RawFd>,
    /// Preformatted final line pointing at the shared logs root.
    log_directory_footer: Box<[u8]>,
}

static UNIX_CRASH_FDS: OnceLock<UnixCrashFds> = OnceLock::new();

/// Size of the alternate signal stack installed for the main thread before attaching the
/// crash handler.
///
/// libstd installs a small per-thread alt signal stack (~`SIGSTKSZ`, typically 8 KB) for
/// stack-overflow detection. `crash-handler` reuses whatever altstack is in place, but the
/// gimli DWARF parser inside [`backtrace::resolve`] consumes more than that and would
/// silently abort Phase 2 partway through. 512 KB is well above the resolver's worst case
/// while staying small enough that the leaked allocation is negligible.
#[cfg(any(target_os = "linux", target_os = "android"))]
const ALT_SIGNAL_STACK_SIZE: usize = 512 * 1024;

/// One-shot flag tracking whether [`ensure_alt_signal_stack`] has installed its larger
/// altstack on the calling thread (the main thread, in normal startup).
#[cfg(any(target_os = "linux", target_os = "android"))]
static ALT_STACK_INSTALLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Opens the log fd, duplicates the preserved terminal stderr fd for tee output, installs the
/// enlarged altstack on Linux/Android, and attaches [`crash_handler::CrashHandler`].
pub(super) fn install_impl(log_path: &Path) -> Result<(), String> {
    use std::fs::OpenOptions;
    use std::os::unix::io::IntoRawFd;

    let log_f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| e.to_string())?;
    let log_fd = log_f.into_raw_fd();
    let term_fd =
        crate::native_stdio::duplicate_preserved_stderr_raw_fd().map(IntoRawFd::into_raw_fd);
    let log_directory_footer = super::log_directory_footer_bytes();

    UNIX_CRASH_FDS
        .set(UnixCrashFds {
            log_fd,
            term_fd,
            log_directory_footer,
        })
        .map_err(|_e| "fatal crash log fds already installed".to_string())?;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Err(e) = ensure_alt_signal_stack() {
        logger::warn!(
            "failed to install enlarged alt signal stack ({e}); fatal-crash symbolication may abort partway"
        );
    }

    // SAFETY: `CrashHandler::attach` installs a process-wide signal handler; the closure only
    // calls async-signal-safe operations (`libc::write`, `__errno_location`) and touches global
    // state via a `OnceLock`. Invoked once during process startup; the handle is `mem::forget`ed
    // below so the handler stays installed for the process lifetime.
    let handler = unsafe {
        CrashHandler::attach(crash_handler::make_crash_event(|ctx| {
            let mut buf = [0u8; 224];
            let n = format_fatal_line_unix(ctx, &mut buf);
            let data = &buf[..n];
            if let Some(fds) = UNIX_CRASH_FDS.get() {
                fds.write_all(data);
                let mut context_buf = [0u8; 512];
                let context_n =
                    crate::diagnostics::crash_context::write_minimal_snapshot(&mut context_buf);
                fds.write_all(&context_buf[..context_n]);
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    // `ssi_signo` is `u32`; the stack-trace writer wants an `i32` so it can
                    // compare against `libc::SIGABRT` and skip Phase 2 on allocator-originated
                    // aborts. The cast is lossless for valid signal numbers.
                    let signal = ctx.siginfo.ssi_signo as i32;
                    write_stack_trace(signal, |chunk| fds.write_all(chunk));
                }
                fds.write_all(&fds.log_directory_footer);
            }
            CrashEventResult::from(false)
        }))
        .map_err(|e| e.to_string())?
    };
    #[expect(
        clippy::mem_forget,
        reason = "handler must stay installed for the process lifetime; see SAFETY comment above"
    )]
    std::mem::forget(handler);
    Ok(())
}

impl UnixCrashFds {
    fn write_all(&self, data: &[u8]) {
        // SAFETY: called from inside the signal handler; only uses async-signal-safe `libc::write`.
        // The `RawFd` values were obtained from `File::into_raw_fd` / `OwnedFd::into_raw_fd` at
        // install time and remain valid for the process lifetime (never closed).
        unsafe {
            let remainder = write_loop_fd(self.log_fd, data);
            if !remainder.is_empty() {
                let _pipe_remainder = write_loop_fd(libc::STDERR_FILENO, remainder);
                let _ = _pipe_remainder;
            }
            if let Some(fd) = self.term_fd {
                let _ = write_loop_fd(fd, data);
            }
        }
    }
}

/// Writes as much as possible to `fd`. Returns the **suffix of `data` that was not written** (empty
/// on full success). Retries on **`EINTR`** only.
///
/// # Safety
///
/// `fd` must be an open file descriptor valid for `write(2)` for the duration of the call. Uses
/// only async-signal-safe operations so it is callable from a crash signal handler.
unsafe fn write_loop_fd(fd: std::os::unix::io::RawFd, mut data: &[u8]) -> &[u8] {
    while !data.is_empty() {
        // SAFETY: see the function contract above -- `fd` is a valid open descriptor for write(2);
        // `errno_value()` only reads the thread-local errno pointer.
        let n = unsafe { libc::write(fd, data.as_ptr().cast(), data.len()) };
        if n < 0 {
            // SAFETY: reads the thread-local errno pointer per `errno_value`'s contract.
            if unsafe { errno_value() } == libc::EINTR {
                continue;
            }
            return data;
        }
        if n == 0 {
            return data;
        }
        data = &data[n as usize..];
    }
    &[]
}

/// Reads `errno` after a failed libc call (async-signal-safe on POSIX).
///
/// # Safety
///
/// Dereferences the thread-local errno pointer returned by libc. The pointer is guaranteed by
/// POSIX to be valid for the lifetime of the thread; callers must not retain the reference.
#[inline]
unsafe fn errno_value() -> libc::c_int {
    // SAFETY: see the function contract above -- the thread-local errno pointer is always valid.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe {
        *libc::__errno_location()
    }
    #[cfg(target_os = "macos")]
    // SAFETY: `__error()` is the per-thread errno pointer on macOS; valid for the calling thread's lifetime.
    unsafe {
        *libc::__error()
    }
    #[cfg(all(
        unix,
        not(any(target_os = "linux", target_os = "android", target_os = "macos"))
    ))]
    // SAFETY: same contract as the Linux `__errno_location` branch; valid per-thread storage.
    unsafe {
        *libc::__errno_location()
    }
}

/// Installs a [`ALT_SIGNAL_STACK_SIZE`]-byte alternate signal stack on the calling thread,
/// replacing libstd's smaller default so [`backtrace::resolve`] has room to run inside the
/// crash handler without silently aborting Phase 2.
///
/// Idempotent: subsequent calls are no-ops once the flag is set. The stack memory is leaked
/// for the process lifetime -- freeing it would invite use-after-free from the next signal.
/// Affects only the thread that invokes this; worker threads keep libstd's default altstack
/// and may lose Phase 2 if they crash, but Phase 1 (hex IPs) remains durable everywhere.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn ensure_alt_signal_stack() -> Result<(), String> {
    use std::sync::atomic::Ordering;

    if ALT_STACK_INSTALLED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let buf = vec![0u8; ALT_SIGNAL_STACK_SIZE].into_boxed_slice();
    let stack_ptr = Box::leak(buf).as_mut_ptr();

    // SAFETY: `stack_t` is a plain integer/pointer aggregate; all-zero is a valid bit pattern.
    let mut ss: libc::stack_t = unsafe { core::mem::zeroed() };
    ss.ss_sp = stack_ptr.cast();
    ss.ss_size = ALT_SIGNAL_STACK_SIZE;
    ss.ss_flags = 0;

    // SAFETY: `ss` is fully initialized above with a pointer/length to a leaked
    // `Box<[u8]>` that lives for the process lifetime. Passing null for `oss` discards the
    // previous altstack pointer -- the previous backing memory leaks, but it was libstd's
    // own per-thread allocation and dropping our reference to it does not invalidate it.
    let rc = unsafe { libc::sigaltstack(core::ptr::addr_of!(ss), core::ptr::null_mut()) };
    if rc != 0 {
        // Reset the flag so a future caller could retry, though in practice this never
        // happens -- if `sigaltstack` rejected our parameters once it will reject them again.
        ALT_STACK_INSTALLED.store(false, Ordering::Release);
        return Err(format!("sigaltstack failed (rc={rc})"));
    }
    Ok(())
}
