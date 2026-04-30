//! Global file logger: one [`std::sync::OnceLock`] sink, optional stderr mirroring, and atomic
//! max-level filtering without reopening the log file.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::level::{LogLevel, tag_to_level};
use crate::timestamp::write_line_timestamp;

/// Default capacity reserved on a thread's reusable line buffer so that steady-state log calls
/// avoid reallocation.
const LINE_BUF_INITIAL_CAPACITY: usize = 256;

thread_local! {
    /// Per-thread reusable buffer for log line formatting. Cleared on every successful borrow so a
    /// panic mid-format leaves no observable corruption for the next caller.
    static LINE_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Global logger state: mutex-protected file sink, optional stderr mirror, and atomic max level.
struct Logger {
    /// Active log file path (used by [`try_log`] when the primary mutex is busy).
    path: PathBuf,
    /// File output. Mutex for thread-safe writes.
    file: Mutex<std::fs::File>,
    /// When true, each log line is also written to stderr.
    mirror_stderr: bool,
    /// Maximum level to log. Messages at or below this level are written (see [`LogLevel`] ordering).
    ///
    /// Atomic so [`set_max_level`] can change filtering after [`init_with_mirror`] without re-init.
    max_level: AtomicU8,
}

/// Global logger instance. Set by [`init`] or [`init_with_mirror`].
static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Returns whether [`init`] or [`init_with_mirror`] has successfully installed the global logger.
///
/// A second successful call to [`init`] still returns [`Ok`], but it does **not** replace the
/// existing logger; [`is_initialized`] remains `true` from the first install.
pub fn is_initialized() -> bool {
    LOGGER.get().is_some()
}

/// Initializes logging. Creates parent directory if needed, opens file.
///
/// Call once at startup before installing a panic hook. Mirror to stderr is disabled; use
/// [`init_with_mirror`] to enable it.
///
/// If the global logger is already installed, this function returns [`Ok`] after opening the
/// requested path and then **drops** that handle without replacing the active logger. Prefer
/// [`is_initialized`] if you need to detect a duplicate init attempt.
///
/// # Errors
///
/// Returns [`Err`] if the log file cannot be opened (for example permission denied or an invalid path).
/// Callers should fail fast on error rather than continuing without logging.
pub fn init(path: impl AsRef<Path>, max_level: LogLevel, append: bool) -> std::io::Result<()> {
    init_with_mirror(path, max_level, append, false)
}

/// Like [`init`], but when `mirror_stderr` is true each log line is also written to stderr.
///
/// # Errors
///
/// Same as [`init`].
pub fn init_with_mirror(
    path: impl AsRef<Path>,
    max_level: LogLevel,
    append: bool,
    mirror_stderr: bool,
) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut opts = OpenOptions::new();
    opts.create(true).write(true);
    if append {
        opts.append(true);
    } else {
        opts.truncate(true);
    }
    let file = opts.open(path)?;
    let logger = Logger {
        path: path.to_path_buf(),
        file: Mutex::new(file),
        mirror_stderr,
        max_level: AtomicU8::new(max_level as u8),
    };
    let _ = LOGGER.set(logger);
    Ok(())
}

/// Sets the maximum log level for the initialized global logger.
///
/// Has no effect if [`init`] / [`init_with_mirror`] has not succeeded. Safe to call from any thread;
/// takes effect immediately for subsequent [`log`] / macro calls.
pub fn set_max_level(level: LogLevel) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    logger.max_level.store(level as u8, Ordering::Relaxed);
}

/// Returns the effective max level from `logger`'s atomic tag.
#[inline]
fn current_max_level(logger: &Logger) -> LogLevel {
    tag_to_level(logger.max_level.load(Ordering::Relaxed))
}

/// Returns whether a message at `level` would be written given the current max level and an
/// initialized logger.
///
/// Use to avoid expensive formatting when logging is filtered out. Returns `false` when the logger
/// has not been initialized.
pub fn enabled(level: LogLevel) -> bool {
    LOGGER
        .get()
        .is_some_and(|logger| level <= current_max_level(logger))
}

/// Flushes any buffered log output. Call periodically if desired for API consistency.
///
/// Does nothing when the logger is not initialized.
///
/// Do not call from a panic hook: if the panic occurred while holding the logger mutex
/// (for example inside a log macro), this would deadlock.
pub fn flush() {
    if let Some(logger) = LOGGER.get()
        && let Ok(mut file) = logger.file.lock()
    {
        let _ = file.flush();
    }
}

/// Writes a full log line into `out` in the canonical `[HH:MM:SS.mmm] LEVEL message\n` shape.
///
/// `out` is cleared first so the buffer can be reused across calls without observable carry-over.
fn format_log_line_into(out: &mut String, level: LogLevel, args: std::fmt::Arguments<'_>) {
    out.clear();
    if out.capacity() < LINE_BUF_INITIAL_CAPACITY {
        out.reserve(LINE_BUF_INITIAL_CAPACITY - out.capacity());
    }
    out.push('[');
    write_line_timestamp(out);
    out.push_str("] ");
    out.push_str(level.as_label());
    out.push(' ');
    let _ = out.write_fmt(args);
    out.push('\n');
}

/// Writes the formatted line in `bytes` to the global logger's file (locking the mutex) and to
/// stderr if mirroring is enabled.
fn write_line_locked(logger: &Logger, bytes: &[u8]) {
    if let Ok(mut file) = logger.file.lock() {
        let _ = file.write_all(bytes);
        let _ = file.flush();
    }
    if logger.mirror_stderr {
        let _ = std::io::stderr().write_all(bytes);
        let _ = std::io::stderr().flush();
    }
}

/// Calls `f` with a thread-local reusable line buffer when available, otherwise with a
/// stack-managed fallback so a [`std::fmt::Display`] impl that recursively logs cannot panic on a
/// borrow conflict. If the thread-local has been destroyed (only possible during thread teardown),
/// returns `R::default()` so the logger remains a no-op rather than panicking.
fn with_line_buf<R: Default>(f: impl FnOnce(&mut String) -> R) -> R {
    LINE_BUF
        .try_with(|cell| {
            if let Ok(mut buf) = cell.try_borrow_mut() {
                f(&mut buf)
            } else {
                let mut fallback = String::with_capacity(LINE_BUF_INITIAL_CAPACITY);
                f(&mut fallback)
            }
        })
        .unwrap_or_default()
}

/// Internal log writer. Called by the log macros.
///
/// Does nothing when the logger is not initialized or when `level` is above the current max level.
#[doc(hidden)]
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    let max = current_max_level(logger);
    if level > max {
        return;
    }
    with_line_buf(|buf| {
        format_log_line_into(buf, level, args);
        write_line_locked(logger, buf.as_bytes());
    });
}

/// Like [`log`], but uses [`Mutex::try_lock`] on the file handle. If the mutex is busy, appends the
/// same formatted line via a separate open of the log file path recorded at init when available.
///
/// Intended for **background threads** (such as a stderr pipe reader) that must not block on the
/// global logger mutex while other code may be writing to the same log or to stderr.
///
/// Returns `true` if the line was written (primary or fallback), `false` if the logger is not
/// initialized, the line is filtered by max level, or the fallback open fails.
pub fn try_log(level: LogLevel, args: std::fmt::Arguments<'_>) -> bool {
    let Some(logger) = LOGGER.get() else {
        return false;
    };
    let max = current_max_level(logger);
    if level > max {
        return false;
    }
    with_line_buf(|buf| {
        format_log_line_into(buf, level, args);
        let bytes = buf.as_bytes();
        if let Ok(mut file) = logger.file.try_lock() {
            let _ = file.write_all(bytes);
            let _ = file.flush();
            return true;
        }
        let mut opts = OpenOptions::new();
        opts.create(true).append(true);
        if let Ok(mut file) = opts.open(&logger.path) {
            let _ = file.write_all(bytes);
            let _ = file.flush();
            return true;
        }
        false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panic::log_panic_payload;
    use std::fs;

    #[test]
    fn global_logger_full_smoke() {
        let path =
            std::env::temp_dir().join(format!("logger_output_smoke_{}.log", std::process::id()));
        let _ = fs::remove_file(&path);

        init(&path, LogLevel::Trace, false).expect("init");
        assert!(is_initialized());
        assert!(enabled(LogLevel::Info));
        assert!(enabled(LogLevel::Trace));

        log(LogLevel::Info, format_args!("smoke_line_marker"));
        crate::info!("info_macro_marker");
        crate::warn!("warn_macro_marker");
        crate::error!("error_macro_marker");
        crate::debug!("debug_macro_marker");
        crate::trace!("trace_macro_marker");
        flush();

        set_max_level(LogLevel::Warn);
        assert!(!enabled(LogLevel::Info));
        assert!(enabled(LogLevel::Warn));
        crate::info!("hidden_info_should_not_appear");

        assert!(try_log(LogLevel::Warn, format_args!("try_log_line_marker")));

        let other_path =
            std::env::temp_dir().join(format!("logger_second_init_{}.log", std::process::id()));
        let _ = fs::remove_file(&other_path);
        init(&other_path, LogLevel::Trace, false).expect("second init returns Ok");
        assert!(is_initialized());

        log_panic_payload(Box::new("boom".to_string()), "ctx_payload_string");
        log_panic_payload(Box::new("static boom"), "ctx_payload_static");
        log_panic_payload(Box::new(7_i32), "ctx_payload_other");

        set_max_level(LogLevel::Trace);

        let contents = fs::read_to_string(&path).expect("read log");
        assert!(contents.contains("smoke_line_marker"));
        assert!(contents.contains("info_macro_marker"));
        assert!(contents.contains("warn_macro_marker"));
        assert!(contents.contains("error_macro_marker"));
        assert!(contents.contains("debug_macro_marker"));
        assert!(contents.contains("trace_macro_marker"));
        assert!(contents.contains("try_log_line_marker"));
        assert!(
            !contents.contains("hidden_info_should_not_appear"),
            "filtered info should not be written: {contents}"
        );
        assert!(contents.contains("ctx_payload_string"));
        assert!(contents.contains("boom"));
        assert!(contents.contains("ctx_payload_static"));
        assert!(contents.contains("ctx_payload_other"));
        assert!(contents.contains("panic (payload type not string)"));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&other_path);
    }

    #[test]
    fn format_log_line_into_clears_existing_buffer_content() {
        let mut buf = String::from("stale_should_be_overwritten");
        format_log_line_into(&mut buf, LogLevel::Info, format_args!("fresh_line"));
        assert!(!buf.contains("stale_should_be_overwritten"));
        assert!(buf.contains(" INFO fresh_line\n"));
        assert!(buf.starts_with('['));
    }

    #[test]
    fn format_log_line_into_grows_capacity_to_initial() {
        let mut buf = String::new();
        format_log_line_into(&mut buf, LogLevel::Trace, format_args!("x"));
        assert!(buf.capacity() >= LINE_BUF_INITIAL_CAPACITY);
    }

    /// Verifies the canonical `[ts] LEVEL message\n` shape: bracketed timestamp prefix, level
    /// label surrounded by spaces, original message body, and a trailing newline.
    #[test]
    fn format_log_line_into_emits_bracketed_timestamp_level_and_newline() {
        let mut buf = String::new();
        format_log_line_into(&mut buf, LogLevel::Info, format_args!("hello_world"));
        assert!(buf.starts_with('['), "expected leading '[' in {buf:?}");
        assert!(buf.contains("] "), "expected '] ' separator in {buf:?}");
        assert!(buf.contains(" INFO "), "expected ' INFO ' token in {buf:?}");
        assert!(buf.contains("hello_world"), "expected message in {buf:?}");
        assert!(buf.ends_with('\n'), "expected trailing newline in {buf:?}");
        assert_eq!(
            buf.matches('\n').count(),
            1,
            "expected exactly one newline in {buf:?}"
        );
    }

    /// Verifies every [`LogLevel`] variant emits its [`LogLevel::as_label`] token surrounded by
    /// spaces. Guards against accidental label drift in the formatter.
    #[test]
    fn format_log_line_into_uses_correct_label_for_each_level() {
        let mut buf = String::new();
        for level in LogLevel::all() {
            format_log_line_into(&mut buf, level, format_args!("body"));
            let token = format!(" {} body", level.as_label());
            assert!(
                buf.contains(&token),
                "expected token {token:?} in line for {level:?}: {buf:?}"
            );
        }
    }

    /// Verifies an empty message still produces a well-formed line ending in `LEVEL \n` so
    /// downstream parsers can rely on the level-then-space invariant.
    #[test]
    fn format_log_line_into_handles_empty_message() {
        let mut buf = String::new();
        format_log_line_into(&mut buf, LogLevel::Warn, format_args!(""));
        assert!(buf.starts_with('['));
        assert!(buf.ends_with(" WARN \n"), "got {buf:?}");
        assert_eq!(buf.matches('\n').count(), 1);
    }

    /// Pins the documented behavior that newlines inside the formatted message are written
    /// verbatim. If a future change introduces sanitization this test will fail and force the
    /// decision to be deliberate.
    #[test]
    fn format_log_line_into_preserves_embedded_newlines() {
        let mut buf = String::new();
        format_log_line_into(&mut buf, LogLevel::Error, format_args!("first\nsecond"));
        assert!(buf.contains("first\nsecond"), "got {buf:?}");
        assert!(buf.ends_with('\n'));
        assert_eq!(
            buf.matches('\n').count(),
            2,
            "expected embedded + trailing newline: {buf:?}"
        );
    }
}
