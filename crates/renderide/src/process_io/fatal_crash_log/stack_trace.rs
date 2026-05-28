//! Two-phase stack-trace capture for fatal crashes on Linux and Windows.
//!
//! Phase 1 walks frames via [`backtrace::trace_unsynchronized`] and formats hex instruction
//! pointers into a stack-local buffer (signal-safe, allocation-free). Phase 2 best-effort
//! symbolicates through [`backtrace::resolve`] (heap-allocating) under
//! [`std::panic::catch_unwind`] so a fault during resolution cannot recurse into the crash
//! handler. macOS is excluded: the Mach exception callback runs on a dedicated thread, so a
//! plain `trace` would walk the wrong stack.

use core::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use super::format::{write_decimal, write_hex_fixed};

/// Upper bound on captured stack frames per fatal crash.
///
/// Sized to cover realistic render-graph depths while keeping the Phase 1 hex buffer under
/// 2 KB and the capture itself on the alternate signal stack.
const MAX_FRAMES: usize = 64;

/// Byte capacity of the stack-allocated buffer used by [`format_frames_hex`].
///
/// Holds [`MAX_FRAMES`] lines of `"  0x"` + 16 hex digits + newline (~=1.3 KB) plus the
/// `"STACK (N frames):\n"` header with headroom.
const HEX_BUF_LEN: usize = 2048;

/// Reentry guard for fatal-crash stack-trace collection.
///
/// [`write_stack_trace`] `compare_exchange`s this to `true` before doing any trace work; a
/// secondary fault inside [`symbolicate_frames`] would find it already set and fall through
/// without attempting a nested capture. The flag is never cleared -- by the time it is set,
/// the process is about to terminate.
static CRASH_REENTRY: AtomicBool = AtomicBool::new(false);

/// Captures and emits a stack trace through the caller-supplied writer.
///
/// Phase 1 writes a signal-safe hex instruction-pointer list from a fixed stack buffer.
/// Phase 2 best-effort symbolicates through [`backtrace::resolve`]; it is allocation-heavy
/// and wrapped in [`std::panic::catch_unwind`] so an unrelated fault inside symbolication
/// cannot propagate back into the crash handler. Both phases route through the same
/// `write_all` closure, preserving the "crashes appear in both log and terminal" invariant
/// of the existing Unix/Windows writers.
///
/// When `signal` is [`libc::SIGABRT`] (Linux/Android), Phase 2 is **skipped**. SIGABRT
/// almost always comes from glibc raising `abort()` from inside an allocator critical
/// section (e.g. `double free or corruption`); the malloc arena lock is held by the
/// faulting thread, so Phase 2's `String::with_capacity` would park on
/// `__lll_lock_wait_private` forever and `catch_unwind` cannot rescue a thread blocked
/// on a mutex. Phase 1's hex IPs are already sufficient to recover the call site
/// offline via `addr2line`.
///
/// On reentry (another fault inside Phase 2), the guard short-circuits and the function
/// returns immediately.
pub(super) fn write_stack_trace<F>(signal: i32, write_all: F)
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

    #[cfg(any(target_os = "linux", target_os = "android"))]
    if signal == libc::SIGABRT {
        const SKIP_MSG: &[u8] =
            b"SYMBOLS: skipped (SIGABRT; symbolicator would deadlock on malloc lock)\n";
        write_all(SKIP_MSG);
        return;
    }
    // Silence unused-variable on non-Linux/Android targets where the SIGABRT skip is gated out.
    let _ = signal;

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sym = symbolicate_frames(&ips[..n]);
        write_all(sym.as_bytes());
    }));
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
    };
    n
}

/// Formats captured instruction pointers as `STACK (<n> frames):\n  0x...\n...` into a
/// caller-provided stack buffer. Returns bytes written; silently stops if the buffer would
/// overflow.
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
    w += write_decimal(ips.len() as u64, &mut out[w..]);
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
        w += write_hex_fixed::<16>(*ip as u64, &mut out[w..]);
        out[w..w + LINE_SUFFIX.len()].copy_from_slice(LINE_SUFFIX);
        w += LINE_SUFFIX.len();
    }
    w
}

/// Best-effort symbolicated trace as `SYMBOLS:\n  #NN <name> at <file>:<line>\n...`.
///
/// Allocates freely through [`backtrace::resolve`]. The caller must wrap this in
/// [`std::panic::catch_unwind`] and guard with [`CRASH_REENTRY`]; a fault inside
/// `backtrace::resolve` (corrupt heap, exhausted stack) must not recurse into the crash
/// handler. If symbolication finds no name for a frame, the line records `<no symbol>` so
/// the indices still line up with the hex output above.
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

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::{Mutex, MutexGuard};

    /// Serializes tests that mutate the fatal-crash reentry guard.
    static CRASH_REENTRY_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Acquires exclusive access to [`CRASH_REENTRY`] for tests.
    fn lock_crash_reentry_test() -> MutexGuard<'static, ()> {
        CRASH_REENTRY_TEST_LOCK.lock()
    }

    #[test]
    fn format_frames_hex_shape() {
        let ips: [*mut c_void; 3] = [
            0xDEAD_BEEF_CAFE_BABE_u64 as *mut _,
            0x0123_4567_89AB_CDEF_u64 as *mut _,
            0xFFFF_FFFF_FFFF_FFFF_u64 as *mut _,
        ];
        let mut out = [0u8; HEX_BUF_LEN];
        let n = format_frames_hex(&ips, &mut out);
        let s = std::str::from_utf8(&out[..n]).expect("utf8");
        assert!(s.starts_with("STACK (3 frames):\n"), "header: {s:?}");
        assert!(s.contains("  0xDEADBEEFCAFEBABE\n"));
        assert!(s.contains("  0x0123456789ABCDEF\n"));
        assert!(s.contains("  0xFFFFFFFFFFFFFFFF\n"));
        assert!(s.ends_with('\n'));
    }

    #[test]
    fn reentry_guard_blocks_second_entry() {
        use std::cell::Cell;

        let _guard = lock_crash_reentry_test();

        // Reset in case a prior test in the same process left the guard set.
        CRASH_REENTRY.store(false, Ordering::Release);

        // SIGSEGV (11) is a non-SIGABRT signal so the test still exercises Phase 2.
        let count = Cell::new(0usize);
        write_stack_trace(11, |_chunk| {
            count.set(count.get() + 1);
        });
        let first = count.get();
        write_stack_trace(11, |_chunk| {
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

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn sigabrt_skips_phase_two_symbols() {
        use std::cell::RefCell;

        let _guard = lock_crash_reentry_test();

        CRASH_REENTRY.store(false, Ordering::Release);

        let captured: RefCell<Vec<u8>> = RefCell::new(Vec::new());
        write_stack_trace(libc::SIGABRT, |chunk| {
            captured.borrow_mut().extend_from_slice(chunk);
        });
        let text = String::from_utf8(captured.into_inner()).expect("utf8");
        assert!(
            text.contains("STACK ("),
            "Phase 1 hex block should still be emitted on SIGABRT"
        );
        assert!(
            text.contains("SYMBOLS: skipped"),
            "SIGABRT skip notice should be emitted in place of Phase 2"
        );
        assert!(
            !text.contains("SYMBOLS:\n"),
            "Phase 2 symbol header must not appear on SIGABRT"
        );
    }
}
