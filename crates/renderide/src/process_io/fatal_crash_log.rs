//! Fatal process faults (POSIX signals, Windows structured exceptions, macOS Mach exceptions) do
//! not invoke Rust’s panic hook. This module registers [`crash_handler::CrashHandler`] so a short
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
//! **Stack traces (Linux + Windows):** after the signal-info line, two additional passes run:
//! Phase 1 walks frames via [`backtrace::trace_unsynchronized`] into a stack array and formats
//! hex instruction pointers into a 2 KB stack buffer (signal-safe, allocation-free). Phase 2
//! best-effort symbolicates through [`backtrace::resolve`] (heap-allocating); both are guarded
//! by a reentry flag plus [`std::panic::catch_unwind`] so a fault inside resolution cannot
//! recurse. Stripped release binaries produce hex only from Phase 2. macOS keeps the
//! signal-info line alone — the Mach exception callback runs on a dedicated thread, so a plain
//! `trace` walks the wrong stack; proper macOS support requires unwinding from
//! `thread_get_state` and is tracked as follow-up work.
//!
//! **Linux alt signal stack:** libstd's per-thread altstack (~8 KB) is too small for the
//! gimli DWARF parser inside [`backtrace::resolve`] — a fatal-signal handler running on it
//! aborts Phase 2 partway through with no diagnostic. [`ensure_alt_signal_stack`] installs a
//! 512 KB altstack on the main thread before [`crash_handler::CrashHandler::attach`] so
//! Phase 2 has room to complete. Crashes on worker threads still use libstd's small altstack
//! and may lose Phase 2 silently; Phase 1 (hex IPs) remains durable on every thread.

use core::ffi::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use crash_handler::{CrashContext, CrashEventResult, CrashHandler};

/// Upper bound on captured stack frames per fatal crash.
///
/// Sized to cover realistic render-graph depths while keeping the Phase 1 hex buffer under
/// 2 KB and the capture itself on the alternate signal stack.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
const MAX_FRAMES: usize = 64;

/// Byte capacity of the stack-allocated buffer used by [`format_frames_hex`].
///
/// Holds [`MAX_FRAMES`] lines of `"  0x"` + 16 hex digits + newline (≈1.3 KB) plus the
/// `"STACK (N frames):\n"` header with headroom.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
const HEX_BUF_LEN: usize = 2048;

/// Reentry guard for fatal-crash stack-trace collection.
///
/// [`write_stack_trace`] `compare_exchange`s this to `true` before doing any trace work; a
/// secondary fault inside [`symbolicate_frames`] would find it already set and fall through
/// without attempting a nested capture. The flag is never cleared — by the time it is set,
/// the process is about to terminate.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
static CRASH_REENTRY: AtomicBool = AtomicBool::new(false);

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
static ALT_STACK_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Global state for raw [`libc::write`] targets (log file + optional terminal duplicate).
#[cfg(unix)]
struct UnixCrashFds {
    log_fd: std::os::unix::io::RawFd,
    term_fd: Option<std::os::unix::io::RawFd>,
}

#[cfg(unix)]
static UNIX_CRASH_FDS: OnceLock<UnixCrashFds> = OnceLock::new();

#[cfg(windows)]
struct WindowsCrashFds {
    /// [`std::sync::Mutex`] allows writing through [`OnceLock::get`] (`&` only). The crash path is
    /// not async-signal-safe like Linux; Windows structured exceptions follow different rules.
    log: std::sync::Mutex<std::fs::File>,
    term: Option<std::sync::Mutex<std::fs::File>>,
}

#[cfg(windows)]
static WINDOWS_CRASH_FDS: OnceLock<WindowsCrashFds> = OnceLock::new();

/// Installs the crash handler after logging and stdio forwarding are initialized.
///
/// Failures are logged and ignored so startup continues without fatal-crash logging.
pub(crate) fn install(log_path: &Path) {
    #[cfg(any(unix, windows))]
    {
        if let Err(e) = install_impl(log_path) {
            logger::warn!("Failed to install fatal crash log handler: {e}");
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = log_path;
    }
}

#[cfg(unix)]
fn install_impl(log_path: &Path) -> Result<(), String> {
    use std::fs::OpenOptions;
    use std::os::unix::io::IntoRawFd;

    let log_f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| e.to_string())?;
    let log_fd = log_f.into_raw_fd();
    let term_fd = crate::native_stdio::duplicate_preserved_stderr_raw_fd().map(|o| o.into_raw_fd());

    UNIX_CRASH_FDS
        .set(UnixCrashFds { log_fd, term_fd })
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
                #[cfg(any(target_os = "linux", target_os = "android"))]
                write_stack_trace(|chunk| fds.write_all(chunk));
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

#[cfg(unix)]
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
#[cfg(unix)]
unsafe fn write_loop_fd(fd: std::os::unix::io::RawFd, mut data: &[u8]) -> &[u8] {
    while !data.is_empty() {
        // SAFETY: see the function contract above — `fd` is a valid open descriptor for write(2);
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
#[cfg(unix)]
#[inline]
unsafe fn errno_value() -> libc::c_int {
    // SAFETY: see the function contract above — the thread-local errno pointer is always valid.
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
/// for the process lifetime — freeing it would invite use-after-free from the next signal.
/// Affects only the thread that invokes this; worker threads keep libstd's default altstack
/// and may lose Phase 2 if they crash, but Phase 1 (hex IPs) remains durable everywhere.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn ensure_alt_signal_stack() -> Result<(), String> {
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
    // previous altstack pointer — the previous backing memory leaks, but it was libstd's
    // own per-thread allocation and dropping our reference to it does not invalidate it.
    let rc = unsafe { libc::sigaltstack(&ss, core::ptr::null_mut()) };
    if rc != 0 {
        // Reset the flag so a future caller could retry, though in practice this never
        // happens — if `sigaltstack` rejected our parameters once it will reject them again.
        ALT_STACK_INSTALLED.store(false, Ordering::Release);
        return Err(format!("sigaltstack failed (rc={rc})"));
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn format_fatal_line_unix(ctx: &CrashContext, buf: &mut [u8; 224]) -> usize {
    format_linux_signal(ctx, buf)
}

#[cfg(target_os = "macos")]
fn format_fatal_line_unix(ctx: &CrashContext, buf: &mut [u8; 224]) -> usize {
    format_macos_exception(ctx, buf)
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_os = "macos"))
))]
fn format_fatal_line_unix(_ctx: &CrashContext, buf: &mut [u8; 224]) -> usize {
    const MSG: &[u8] = b"FATAL: unix crash (fatal fault; see crash-handler)\n";
    let n = MSG.len().min(buf.len());
    buf[..n].copy_from_slice(&MSG[..n]);
    n
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn format_linux_signal(ctx: &CrashContext, buf: &mut [u8; 224]) -> usize {
    const PREFIX: &[u8] = b"FATAL: fatal signal (ssi_signo=";
    let sig = ctx.siginfo.ssi_signo;
    write_prefix_u32_newline(buf, PREFIX, sig)
}

#[cfg(target_os = "macos")]
fn format_macos_exception(ctx: &CrashContext, buf: &mut [u8; 224]) -> usize {
    match ctx.exception {
        Some(ex) => {
            const P1: &[u8] = b"FATAL: macOS exception (kind=";
            const P2: &[u8] = b", code=";
            const SUF: &[u8] = b")\n";
            let mut w = 0usize;
            buf[w..w + P1.len()].copy_from_slice(P1);
            w += P1.len();
            w += write_u32_decimal(ex.kind, &mut buf[w..]);
            buf[w..w + P2.len()].copy_from_slice(P2);
            w += P2.len();
            w += write_u64_decimal(ex.code, &mut buf[w..]);
            buf[w..w + SUF.len()].copy_from_slice(SUF);
            w += SUF.len();
            w
        }
        None => {
            const MSG: &[u8] = b"FATAL: macOS crash (no exception details)\n";
            let n = MSG.len().min(buf.len());
            buf[..n].copy_from_slice(&MSG[..n]);
            n
        }
    }
}

#[cfg(windows)]
fn install_impl(log_path: &Path) -> Result<(), String> {
    use std::fs::OpenOptions;

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| e.to_string())?;
    let term = crate::native_stdio::duplicate_preserved_stderr_file_for_crash_log();

    WINDOWS_CRASH_FDS
        .set(WindowsCrashFds {
            log: std::sync::Mutex::new(log),
            term: term.map(std::sync::Mutex::new),
        })
        .map_err(|_e| "fatal crash log fds already installed".to_string())?;

    // SAFETY: installs a process-wide vectored exception handler; the closure only performs
    // synchronous writes to files guarded by poisoning-tolerant mutexes. Called once at startup;
    // handler is leaked below so the installation persists for process lifetime.
    let handler = unsafe {
        CrashHandler::attach(crash_handler::make_crash_event(|ctx| {
            let mut buf = [0u8; 224];
            let n = format_fatal_line_windows(ctx, &mut buf);
            let data = &buf[..n];
            if let Some(fds) = WINDOWS_CRASH_FDS.get() {
                fds.write_all(data);
                write_stack_trace(|chunk| fds.write_all(chunk));
            }
            CrashEventResult::from(false)
        }))
        .map_err(|e| e.to_string())?
    };
    // Leak the handler for process lifetime: dropping would uninstall the crash handler.
    #[expect(
        clippy::mem_forget,
        reason = "CrashHandler must not be dropped after attach; the handler is process-global until exit"
    )]
    std::mem::forget(handler);
    Ok(())
}

#[cfg(windows)]
impl WindowsCrashFds {
    /// Writes `data` to the log file and (if configured) the terminal duplicate, matching the
    /// dual-output routing of the existing Unix path. Individual write errors are swallowed —
    /// the crash handler has no meaningful recovery path.
    fn write_all(&self, data: &[u8]) {
        use std::io::Write;
        if let Ok(mut g) = self.log.lock() {
            let _ = g.write_all(data);
            let _ = g.flush();
        }
        if let Some(t) = &self.term {
            if let Ok(mut g) = t.lock() {
                let _ = g.write_all(data);
                let _ = g.flush();
            }
        }
    }
}

#[cfg(windows)]
fn format_fatal_line_windows(ctx: &CrashContext, buf: &mut [u8; 224]) -> usize {
    const PREFIX: &[u8] = b"FATAL: Windows exception (code=0x";
    const SUFFIX: &[u8] = b")\n";
    let code = ctx.exception_code as u32;
    let mut w = 0usize;
    buf[w..w + PREFIX.len()].copy_from_slice(PREFIX);
    w += PREFIX.len();
    w += write_hex_u32(code, &mut buf[w..]);
    buf[w..w + SUFFIX.len()].copy_from_slice(SUFFIX);
    w + SUFFIX.len()
}

/// Writes `prefix`, decimal `n`, then `)\n`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn write_prefix_u32_newline(buf: &mut [u8; 224], prefix: &[u8], n: u32) -> usize {
    const SUFFIX: &[u8] = b")\n";
    let mut w = 0usize;
    if w + prefix.len() > buf.len() {
        return 0;
    }
    buf[w..w + prefix.len()].copy_from_slice(prefix);
    w += prefix.len();
    w += write_u32_decimal(n, &mut buf[w..]);
    if w + SUFFIX.len() <= buf.len() {
        buf[w..w + SUFFIX.len()].copy_from_slice(SUFFIX);
        w += SUFFIX.len();
    }
    w
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
))]
fn write_u32_decimal(mut n: u32, out: &mut [u8]) -> usize {
    if n == 0 {
        if out.is_empty() {
            return 0;
        }
        out[0] = b'0';
        return 1;
    }
    let mut tmp = [0u8; 10];
    let mut i = 0usize;
    while n > 0 {
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    let mut w = 0usize;
    while i > 0 {
        i -= 1;
        if w >= out.len() {
            break;
        }
        out[w] = tmp[i];
        w += 1;
    }
    w
}

#[cfg(target_os = "macos")]
fn write_u64_decimal(mut n: u64, out: &mut [u8]) -> usize {
    if n == 0 {
        if out.is_empty() {
            return 0;
        }
        out[0] = b'0';
        return 1;
    }
    let mut tmp = [0u8; 20];
    let mut i = 0usize;
    while n > 0 {
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    let mut w = 0usize;
    while i > 0 {
        i -= 1;
        if w >= out.len() {
            break;
        }
        out[w] = tmp[i];
        w += 1;
    }
    w
}

#[cfg(windows)]
fn write_hex_u32(n: u32, out: &mut [u8]) -> usize {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let need = 8;
    if out.len() < need {
        return 0;
    }
    let mut v = n;
    for i in (0..8).rev() {
        out[i] = HEX[(v & 0xf) as usize];
        v >>= 4;
    }
    need
}

/// Writes `n` as 16 uppercase hex digits (64-bit instruction pointer) into `out`.
///
/// Returns the byte count written, or `0` if `out` is smaller than 16 bytes.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn write_hex_u64(n: u64, out: &mut [u8]) -> usize {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let need = 16;
    if out.len() < need {
        return 0;
    }
    let mut v = n;
    for i in (0..16).rev() {
        out[i] = HEX[(v & 0xf) as usize];
        v >>= 4;
    }
    need
}

/// Walks the current thread's stack and stores up to [`MAX_FRAMES`] instruction pointers
/// into `out`, returning the count captured.
///
/// Uses [`backtrace::trace_unsynchronized`] so no locks are taken. No heap allocation; safe
/// to invoke from a crash handler when paired with [`CRASH_REENTRY`].
///
/// # Safety
///
/// Caller must ensure no other thread is unwinding concurrently through the `backtrace`
/// crate's globals. In the crash-handler path, [`CRASH_REENTRY`] enforces this for the
/// process lifetime.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
unsafe fn capture_frame_ips(out: &mut [*mut c_void; MAX_FRAMES]) -> usize {
    let mut n = 0usize;
    // SAFETY: see function contract; the caller guarantees no concurrent unwinding.
    unsafe {
        backtrace::trace_unsynchronized(|frame| {
            if n < out.len() {
                out[n] = frame.ip();
                n += 1;
                true
            } else {
                false
            }
        });
    }
    n
}

/// Formats captured instruction pointers as `STACK (<n> frames):\n  0x…\n…` into a
/// caller-provided stack buffer. Returns bytes written; silently stops if the buffer would
/// overflow.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn format_frames_hex(ips: &[*mut c_void], out: &mut [u8; HEX_BUF_LEN]) -> usize {
    const HDR_PREFIX: &[u8] = b"STACK (";
    const HDR_SUFFIX: &[u8] = b" frames):\n";
    const LINE_PREFIX: &[u8] = b"  0x";
    const LINE_SUFFIX: &[u8] = b"\n";
    const HEX_DIGITS: usize = 16;

    let mut w = 0usize;
    if w + HDR_PREFIX.len() > out.len() {
        return 0;
    }
    out[w..w + HDR_PREFIX.len()].copy_from_slice(HDR_PREFIX);
    w += HDR_PREFIX.len();
    w += write_u32_decimal(ips.len() as u32, &mut out[w..]);
    if w + HDR_SUFFIX.len() > out.len() {
        return w;
    }
    out[w..w + HDR_SUFFIX.len()].copy_from_slice(HDR_SUFFIX);
    w += HDR_SUFFIX.len();

    for ip in ips {
        if w + LINE_PREFIX.len() + HEX_DIGITS + LINE_SUFFIX.len() > out.len() {
            break;
        }
        out[w..w + LINE_PREFIX.len()].copy_from_slice(LINE_PREFIX);
        w += LINE_PREFIX.len();
        w += write_hex_u64(*ip as u64, &mut out[w..]);
        out[w..w + LINE_SUFFIX.len()].copy_from_slice(LINE_SUFFIX);
        w += LINE_SUFFIX.len();
    }
    w
}

/// Best-effort symbolicated trace as `SYMBOLS:\n  #NN <name> at <file>:<line>\n…`.
///
/// Allocates freely through [`backtrace::resolve`]. The caller must wrap this in
/// [`std::panic::catch_unwind`] and guard with [`CRASH_REENTRY`]; a fault inside
/// `backtrace::resolve` (corrupt heap, exhausted stack) must not recurse into the crash
/// handler. If symbolication finds no name for a frame, the line records `<no symbol>` so
/// the indices still line up with the hex output above.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn symbolicate_frames(ips: &[*mut c_void]) -> String {
    use std::fmt::Write;

    let mut out = String::with_capacity(ips.len().saturating_mul(128));
    out.push_str("SYMBOLS:\n");
    for (idx, ip) in ips.iter().enumerate() {
        let mut any_sym = false;
        backtrace::resolve(*ip, |sym| {
            any_sym = true;
            let _ = write!(out, "  #{idx:02} ");
            match sym.name() {
                Some(name) => {
                    let _ = write!(out, "{name}");
                }
                None => {
                    out.push_str("???");
                }
            }
            if let Some(file) = sym.filename() {
                let _ = write!(out, " at {}", file.display());
                if let Some(line) = sym.lineno() {
                    let _ = write!(out, ":{line}");
                }
            }
            out.push('\n');
        });
        if !any_sym {
            let _ = writeln!(out, "  #{idx:02} <no symbol>");
        }
    }
    out
}

/// Captures and emits a stack trace through the caller-supplied writer.
///
/// Phase 1 writes a signal-safe hex instruction-pointer list from a fixed stack buffer.
/// Phase 2 best-effort symbolicates through [`backtrace::resolve`]; it is allocation-heavy
/// and wrapped in [`std::panic::catch_unwind`] so a poisoned allocator or corrupt heap
/// cannot propagate back into the crash handler. Both phases route through the same
/// `write_all` closure, preserving the "crashes appear in both log and terminal" invariant
/// of the existing Unix/Windows writers.
///
/// On reentry (another fault inside Phase 2), the guard short-circuits and the function
/// returns immediately.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn write_stack_trace<F>(write_all: F)
where
    F: Fn(&[u8]),
{
    if CRASH_REENTRY
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let mut ips: [*mut c_void; MAX_FRAMES] = [core::ptr::null_mut(); MAX_FRAMES];
    // SAFETY: `CRASH_REENTRY` was just set to `true`; no other thread can drive
    // `backtrace::trace_unsynchronized` concurrently via this handler for the lifetime of
    // the process.
    let n = unsafe { capture_frame_ips(&mut ips) };
    let mut hex_buf = [0u8; HEX_BUF_LEN];
    let hex_n = format_frames_hex(&ips[..n], &mut hex_buf);
    write_all(&hex_buf[..hex_n]);

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sym = symbolicate_frames(&ips[..n]);
        write_all(sym.as_bytes());
    }));
}

#[cfg(test)]
mod formatter_tests {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn write_u32_decimal_formats() {
        let mut out = [0u8; 16];
        let n = super::write_u32_decimal(12345, &mut out);
        assert_eq!(&out[..n], b"12345");
        let n0 = super::write_u32_decimal(0, &mut out);
        assert_eq!(&out[..n0], b"0");
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn linux_fatal_signal_line_contains_signo() {
        use crash_handler::CrashContext;
        // SAFETY: `CrashContext` on Linux is a plain aggregate of integer fields (siginfo_t-like);
        // all-zero is a valid bit pattern. Test-only construction; never observed by the kernel.
        let mut ctx: CrashContext = unsafe { std::mem::zeroed() };
        ctx.siginfo.ssi_signo = 11;
        let mut buf = [0u8; 224];
        let n = super::format_linux_signal(&ctx, &mut buf);
        let line = std::str::from_utf8(&buf[..n]).expect("utf8");
        assert!(line.starts_with("FATAL: fatal signal (ssi_signo="));
        assert!(line.contains("11"));
        assert!(line.ends_with(")\n"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_fatal_line_contains_exception_code() {
        use crash_handler::CrashContext;
        // SAFETY: Windows `CrashContext` is an integer/pointer aggregate; all-zero is a valid
        // bit pattern. Test-only value that never traverses the real crash path.
        let mut ctx: CrashContext = unsafe { std::mem::zeroed() };
        ctx.exception_code = 0xC000_0005_u32 as i32;
        let mut buf = [0u8; 224];
        let n = super::format_fatal_line_windows(&ctx, &mut buf);
        let line = std::str::from_utf8(&buf[..n]).expect("utf8");
        assert!(line.starts_with("FATAL: Windows exception (code=0x"));
        assert!(line.contains("C0000005"));
        assert!(line.ends_with(")\n"));
    }

    #[cfg(windows)]
    #[test]
    fn write_hex_u32_uppercase() {
        let mut out = [0u8; 8];
        let n = super::write_hex_u32(0xDEAD_BEEF, &mut out);
        assert_eq!(n, 8);
        assert_eq!(&out[..n], b"DEADBEEF");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_fatal_line_no_exception() {
        use crash_handler::CrashContext;
        // SAFETY: macOS `CrashContext` fields are integers/`Option<ExceptionInfo>`; all-zero is a
        // valid bit pattern (`None` discriminant). Test-only value.
        let ctx: CrashContext = unsafe { std::mem::zeroed() };
        let mut buf = [0u8; 224];
        let n = super::format_macos_exception(&ctx, &mut buf);
        let line = std::str::from_utf8(&buf[..n]).expect("utf8");
        assert!(line.contains("FATAL:"));
        assert!(line.ends_with('\n'));
    }

    #[cfg(any(target_os = "linux", target_os = "android", windows))]
    #[test]
    fn write_hex_u64_formats() {
        let mut out = [0u8; 16];
        let n = super::write_hex_u64(0xDEAD_BEEF_CAFE_BABE, &mut out);
        assert_eq!(n, 16);
        assert_eq!(&out[..n], b"DEADBEEFCAFEBABE");

        let mut small = [0u8; 8];
        let n_small = super::write_hex_u64(0x1, &mut small);
        assert_eq!(n_small, 0, "buffer shorter than 16 bytes must return 0");
    }

    #[cfg(any(target_os = "linux", target_os = "android", windows))]
    #[test]
    fn format_frames_hex_shape() {
        let ips: [*mut core::ffi::c_void; 3] = [
            0xDEAD_BEEF_CAFE_BABE_u64 as *mut _,
            0x0123_4567_89AB_CDEF_u64 as *mut _,
            0xFFFF_FFFF_FFFF_FFFF_u64 as *mut _,
        ];
        let mut out = [0u8; super::HEX_BUF_LEN];
        let n = super::format_frames_hex(&ips, &mut out);
        let s = std::str::from_utf8(&out[..n]).expect("utf8");
        assert!(s.starts_with("STACK (3 frames):\n"), "header: {s:?}");
        assert!(s.contains("  0xDEADBEEFCAFEBABE\n"));
        assert!(s.contains("  0x0123456789ABCDEF\n"));
        assert!(s.contains("  0xFFFFFFFFFFFFFFFF\n"));
        assert!(s.ends_with('\n'));
    }

    #[cfg(any(target_os = "linux", target_os = "android", windows))]
    #[test]
    fn reentry_guard_blocks_second_entry() {
        use std::cell::Cell;
        use std::sync::atomic::Ordering;

        // Reset in case a prior test in the same process left the guard set.
        super::CRASH_REENTRY.store(false, Ordering::Release);

        let count = Cell::new(0usize);
        super::write_stack_trace(|_chunk| {
            count.set(count.get() + 1);
        });
        let first = count.get();
        super::write_stack_trace(|_chunk| {
            count.set(count.get() + 1);
        });
        let second = count.get();

        assert!(
            first >= 1,
            "first call should have written at least the Phase 1 hex block"
        );
        assert_eq!(
            first, second,
            "second call should be blocked by the reentry guard"
        );
    }
}
