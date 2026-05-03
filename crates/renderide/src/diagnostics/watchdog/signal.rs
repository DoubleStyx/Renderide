//! POSIX stack capture for the watchdog (`SIGUSR2` + `pthread_kill`).
//!
//! # Mechanism
//!
//! 1. [`install`] registers a `SIGUSR2` handler process-wide. The handler walks the calling
//!    thread's own stack via [`backtrace::trace_unsynchronized`] into a fixed-size static buffer,
//!    sets a "done" flag, and returns. Only async-signal-safe operations are used (atomic stores,
//!    no `malloc`, no mutex acquisitions).
//! 2. The watchdog calls [`request_capture`] from its own thread, which acquires a serialization
//!    mutex (only the watchdog ever calls into here, so the mutex is uncontended), arms the
//!    target identity, sends `SIGUSR2` via `pthread_kill` (or `tgkill` on Linux for fewer
//!    indirections), and busy-waits with short sleeps for the handler to flip the "done" flag.
//! 3. The watchdog reads the captured instruction pointers out of the static buffer and then
//!    symbolicates them -- heap allocation is fine here because we are no longer in signal context.
//!
//! Modeled after the existing two-phase pattern in [`crate::fatal_crash_log`] (signal-safe IP
//! capture, deferred symbolication via [`backtrace::resolve`]).

use core::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Maximum number of stack frames captured per request. Sized to cover realistic render-graph
/// depths while keeping the static buffer a few hundred bytes.
const MAX_FRAMES: usize = 64;

/// Per-frame instruction pointer storage. Written by the signal handler (one atomic store per
/// frame), read by the watchdog after [`CAPTURE_DONE`] is set.
static CAPTURE_FRAME_IPS: [AtomicUsize; MAX_FRAMES] = [const { AtomicUsize::new(0) }; MAX_FRAMES];

/// Number of valid frames in [`CAPTURE_FRAME_IPS`] after the handler returns.
static CAPTURE_FRAME_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Set to `true` by the handler when it has finished writing into the static buffer.
static CAPTURE_DONE: AtomicBool = AtomicBool::new(false);

/// OS thread identity (Linux `gettid()` / macOS `pthread_threadid_np`) the handler should match
/// before it walks. Defends against spurious `SIGUSR2` deliveries to other threads.
static CAPTURE_TARGET_TID: AtomicI64 = AtomicI64::new(0);

/// Serializes [`request_capture`] calls -- the static buffers above are process-wide.
static CAPTURE_LOCK: Mutex<()> = Mutex::new(());

/// One-shot guard so [`install`] is idempotent.
static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Returns the calling thread's OS identity (the value stored by `os_tid` on a heartbeat slot).
///
/// On Linux this is the kernel `pid_t` from `gettid(2)`. On macOS it is the 64-bit thread id
/// from `pthread_threadid_np`. Stored as `i64` so both fit; `0` on platforms where this isn't
/// available.
pub(super) fn current_os_tid() -> i64 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        // SAFETY: `gettid()` (via SYS_gettid) is always safe; returns the calling thread's TID
        // and never touches user memory.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) };
        #[cfg(target_pointer_width = "64")]
        {
            tid
        }
        #[cfg(target_pointer_width = "32")]
        {
            i64::from(tid)
        }
    }
    #[cfg(target_os = "macos")]
    {
        let mut tid: u64 = 0;
        // SAFETY: `pthread_threadid_np(0, ..)` queries the calling thread's id; pointer must be
        // a valid writable u64, which `addr_of_mut!(tid)` provides without creating a borrow.
        unsafe {
            libc::pthread_threadid_np(0, core::ptr::addr_of_mut!(tid));
        }
        tid as i64
    }
}

/// Returns the calling thread's macOS `pthread_t` encoded as `usize`.
///
/// Linux and Android targets capture through `tgkill` using [`current_os_tid`], so they do not
/// need to preserve the opaque `pthread_t` value.
pub(super) fn current_pthread_handle() -> usize {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `pthread_self()` returns the calling thread's pthread; safe in any context.
        unsafe { libc::pthread_self() }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        0
    }
}

/// Signal handler installed for `SIGUSR2`. Walks the current thread's stack into the static
/// buffers and signals completion.
///
/// # Async-signal safety
///
/// Only async-signal-safe operations: atomic stores, [`backtrace::trace_unsynchronized`] (the
/// gimli/libunwind walk uses no allocation when the closure does only atomic stores), and
/// [`current_os_tid`] (a syscall wrapper). No `malloc`, no mutex acquisitions, no `printf`.
extern "C" fn capture_signal_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    // Defensive: only respond if this is the thread the watchdog asked for.
    let me = current_os_tid();
    if CAPTURE_TARGET_TID.load(Ordering::Acquire) != me {
        return;
    }

    let mut n: usize = 0;
    // SAFETY: `CAPTURE_LOCK` (held by `request_capture`) ensures no other watchdog capture is
    // in flight, and the renderer's other use of `backtrace` (the fatal-crash handler) is gated
    // by its own `CRASH_REENTRY` guard. The closure performs only atomic stores into a static
    // array, which is async-signal-safe.
    unsafe {
        backtrace::trace_unsynchronized(|frame| {
            if n < MAX_FRAMES {
                CAPTURE_FRAME_IPS[n].store(frame.ip() as usize, Ordering::Release);
                n += 1;
                true
            } else {
                false
            }
        });
    };
    CAPTURE_FRAME_COUNT.store(n, Ordering::Release);
    CAPTURE_DONE.store(true, Ordering::Release);
}

/// Install the `SIGUSR2` handler process-wide. Idempotent.
pub(super) fn install() -> Result<(), String> {
    if HANDLER_INSTALLED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    // SAFETY: `sigaction` is a plain integer/pointer aggregate; all-zero is a valid bit pattern.
    let mut sa: libc::sigaction = unsafe { core::mem::zeroed() };
    sa.sa_sigaction = capture_signal_handler as *const () as usize;
    sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
    // SAFETY: `sigemptyset` initializes the signal set in place; the pointer is non-null and
    // points to writable memory inside `sa`.
    unsafe { libc::sigemptyset(core::ptr::addr_of_mut!(sa.sa_mask)) };
    // SAFETY: registers a process-wide handler for `SIGUSR2`; the handler is async-signal-safe
    // (see its doc-comment). Old handler is discarded -- `SIGUSR2` is otherwise unused in the
    // renderer, so there is no preexisting handler to chain to.
    let rc = unsafe {
        libc::sigaction(
            libc::SIGUSR2,
            core::ptr::addr_of!(sa),
            core::ptr::null_mut(),
        )
    };
    if rc != 0 {
        HANDLER_INSTALLED.store(false, Ordering::Release);
        return Err(format!("sigaction(SIGUSR2) failed (rc={rc})"));
    }
    Ok(())
}

/// Send `SIGUSR2` to the thread identified by `os_tid` / `pthread_handle` and wait up to
/// `timeout` for the handler to capture frames into the static buffer.
///
/// Returns the captured instruction pointers on success, or `None` if the signal could not be
/// delivered or the handler did not respond before the timeout.
pub(super) fn request_capture(
    os_tid: i64,
    pthread_handle: usize,
    timeout: Duration,
) -> Option<Vec<usize>> {
    // The static capture buffer is process-wide; only one capture in flight at a time.
    let _g = match CAPTURE_LOCK.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    CAPTURE_DONE.store(false, Ordering::Release);
    CAPTURE_FRAME_COUNT.store(0, Ordering::Release);
    CAPTURE_TARGET_TID.store(os_tid, Ordering::Release);

    let rc = send_capture_signal(os_tid, pthread_handle);
    if rc != 0 {
        return None;
    }

    let deadline = Instant::now() + timeout;
    while !CAPTURE_DONE.load(Ordering::Acquire) {
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_micros(200));
    }

    let n = CAPTURE_FRAME_COUNT.load(Ordering::Acquire);
    let mut frames = Vec::with_capacity(n);
    for slot in CAPTURE_FRAME_IPS.iter().take(n) {
        frames.push(slot.load(Ordering::Acquire));
    }
    Some(frames)
}

/// Send `SIGUSR2` to the target thread. Linux uses `tgkill(getpid(), tid, SIGUSR2)` directly;
/// macOS uses `pthread_kill(pthread, SIGUSR2)`.
fn send_capture_signal(os_tid: i64, pthread_handle: usize) -> i32 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let _ = pthread_handle;
        // SAFETY: `tgkill` requires no user-memory pointers; arguments are integers. `getpid()`
        // is always safe.
        let pid = unsafe { libc::getpid() };
        // SAFETY: SYS_tgkill takes (tgid, tid, sig). Sending SIGUSR2 to a non-existent tid
        // returns -ESRCH which we surface as a non-zero rc; no memory is dereferenced.
        unsafe { libc::syscall(libc::SYS_tgkill, pid, os_tid as libc::pid_t, libc::SIGUSR2) as i32 }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = os_tid;
        if pthread_handle == 0 {
            return -1;
        }
        let thread: libc::pthread_t = pthread_handle;
        // SAFETY: `pthread_kill(thread, sig)` takes a pthread_t identifier and a signal number.
        // `pthread_handle` was captured at heartbeat registration via `pthread_self()` and the
        // value stays valid for the lifetime of that thread; sending SIGUSR2 to a thread that
        // exited returns ESRCH, surfaced here as a non-zero rc.
        unsafe { libc::pthread_kill(thread, libc::SIGUSR2) }
    }
}

/// Best-effort symbolicate `frames` as a multi-line `"  #NN <name> at <file>:<line>"` string.
///
/// Allocates freely -- only call from the watchdog thread, never from a signal handler. Mirrors
/// the Phase-2 symbolication in [`crate::fatal_crash_log`].
pub(super) fn symbolicate(frames: &[usize]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(frames.len().saturating_mul(96));
    for (idx, ip) in frames.iter().enumerate() {
        let mut any_sym = false;
        backtrace::resolve(*ip as *mut core::ffi::c_void, |sym| {
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
            let _ = writeln!(out, "  #{idx:02} 0x{ip:016X} <no symbol>");
        }
    }
    out
}
