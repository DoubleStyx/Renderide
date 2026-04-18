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

use std::path::Path;
use std::sync::OnceLock;

use crash_handler::{CrashContext, CrashEventResult, CrashHandler};

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
        .map_err(|_| "fatal crash log fds already installed".to_string())?;

    let handler = unsafe {
        CrashHandler::attach(crash_handler::make_crash_event(|ctx| {
            let mut buf = [0u8; 224];
            let n = format_fatal_line_unix(ctx, &mut buf);
            let data = &buf[..n];
            if let Some(fds) = UNIX_CRASH_FDS.get() {
                fds.write_all(data);
            }
            CrashEventResult::from(false)
        }))
        .map_err(|e| e.to_string())?
    };
    std::mem::forget(handler);
    Ok(())
}

#[cfg(unix)]
impl UnixCrashFds {
    fn write_all(&self, data: &[u8]) {
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
#[cfg(unix)]
unsafe fn write_loop_fd(fd: std::os::unix::io::RawFd, mut data: &[u8]) -> &[u8] {
    while !data.is_empty() {
        let n = libc::write(fd, data.as_ptr().cast(), data.len());
        if n < 0 {
            if errno_value() == libc::EINTR {
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
#[cfg(unix)]
#[inline]
unsafe fn errno_value() -> libc::c_int {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        *libc::__errno_location()
    }
    #[cfg(target_os = "macos")]
    {
        *libc::__error()
    }
    #[cfg(all(
        unix,
        not(any(target_os = "linux", target_os = "android", target_os = "macos"))
    ))]
    {
        *libc::__errno_location()
    }
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
    use std::io::Write;

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
        .map_err(|_| "fatal crash log fds already installed".to_string())?;

    let handler = unsafe {
        CrashHandler::attach(crash_handler::make_crash_event(|ctx| {
            let mut buf = [0u8; 224];
            let n = format_fatal_line_windows(ctx, &mut buf);
            let data = &buf[..n];
            if let Some(fds) = WINDOWS_CRASH_FDS.get() {
                if let Ok(mut g) = fds.log.lock() {
                    let _ = g.write_all(data);
                    let _ = g.flush();
                }
                if let Some(t) = &fds.term {
                    if let Ok(mut g) = t.lock() {
                        let _ = g.write_all(data);
                        let _ = g.flush();
                    }
                }
            }
            CrashEventResult::from(false)
        }))
        .map_err(|e| e.to_string())?
    };
    std::mem::forget(handler);
    Ok(())
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

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
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
        let mut ctx: CrashContext = unsafe { std::mem::zeroed() };
        ctx.exception_code = 0xC0000005u32 as i32;
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
        let n = super::write_hex_u32(0xDEADBEEF, &mut out);
        assert_eq!(n, 8);
        assert_eq!(&out[..n], b"DEADBEEF");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_fatal_line_no_exception() {
        use crash_handler::CrashContext;
        let ctx: CrashContext = unsafe { std::mem::zeroed() };
        let mut buf = [0u8; 224];
        let n = super::format_macos_exception(&ctx, &mut buf);
        let line = std::str::from_utf8(&buf[..n]).expect("utf8");
        assert!(line.contains("FATAL:"));
        assert!(line.ends_with('\n'));
    }
}
