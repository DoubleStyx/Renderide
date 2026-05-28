//! Global file sink, log facade integration, and mirror routing.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::level::{LogLevel, tag_to_level};

use super::line::{format_log_line_into, with_line_buf};

/// Default target used by programmatic logging helpers that do not receive a Rust module path.
const DEFAULT_TARGET: &str = "renderide";

/// Global logger state: mutex-protected file sink, optional stderr mirror, and atomic max level.
struct Logger {
    /// Active log file path (used by [`try_log`] when the primary mutex is busy).
    path: PathBuf,
    /// File output. Mutex for thread-safe writes.
    file: Mutex<File>,
    /// When true, each log line is also written to stderr.
    mirror_stderr: bool,
    /// Optional process-specific sink for already-formatted lines that should also reach a
    /// preserved terminal or supervisor-visible stream.
    mirror_writer: Mutex<Option<MirrorWriter>>,
    /// Maximum level to log. Messages at or below this level are written (see [`LogLevel`]
    /// ordering).
    ///
    /// Atomic so [`set_max_level`] can change filtering after [`init_with_mirror`] without re-init.
    max_level: AtomicU8,
}

impl Logger {
    /// Creates logger state around an already-opened file handle.
    fn new(path: PathBuf, file: File, max_level: LogLevel, mirror_stderr: bool) -> Self {
        Self {
            path,
            file: Mutex::new(file),
            mirror_stderr,
            mirror_writer: Mutex::new(None),
            max_level: AtomicU8::new(max_level as u8),
        }
    }
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        let level: LogLevel = metadata.level().into();
        let mut max_level = current_max_level(self);
        if max_level == LogLevel::Debug && metadata.target().starts_with("naga") {
            // Keep Naga's high-volume diagnostics out of regular debug logs.
            max_level = LogLevel::Info;
        }
        level <= max_level
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let level: LogLevel = record.level().into();
            with_line_buf(|buf| {
                format_log_line_into(buf, record.target(), level, *record.args());
                write_line_locked(self, level, buf.as_bytes());
            });
        }
    }

    fn flush(&self) {
        flush_file(self);
    }
}

/// Severity-filtered callback for mirroring already-formatted log lines.
#[derive(Clone, Copy)]
struct MirrorWriter {
    /// Most verbose level accepted by this mirror.
    max_level: LogLevel,
    /// Callback that receives the full formatted line bytes.
    write: fn(&[u8]),
}

/// Global logger instance. Set by [`init`] or [`init_with_mirror`].
static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Tracks whether this crate owns the global `log` facade sink.
static LOG_FACADE_INSTALLED: AtomicBool = AtomicBool::new(false);

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
/// Returns [`Err`] if the log file cannot be opened (for example permission denied or an invalid
/// path). Callers should fail fast on error rather than continuing without logging.
pub fn init(path: impl AsRef<Path>, max_level: LogLevel, append: bool) -> io::Result<()> {
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
) -> io::Result<()> {
    let path = path.as_ref();
    let file = open_log_file(path, append)?;
    let logger = Logger::new(path.to_path_buf(), file, max_level, mirror_stderr);
    if LOGGER.set(logger).is_ok()
        && let Some(logger) = LOGGER.get()
        && matches!(log::set_logger(logger), Ok(()))
    {
        LOG_FACADE_INSTALLED.store(true, Ordering::Relaxed);
        log::set_max_level(max_level.into());
    }
    Ok(())
}

/// Opens a log file for initialization, creating the parent directory first when needed.
fn open_log_file(path: &Path, append: bool) -> io::Result<File> {
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
    opts.open(path)
}

/// Installs or replaces a severity-filtered mirror for already-formatted log lines.
///
/// The writer is invoked after the primary file write and only for lines at or above
/// `max_level`. For example, `LogLevel::Error` mirrors only error-level lines. Writer failures
/// cannot be observed because the callback returns `()`, keeping terminal visibility best-effort
/// and never blocking file logging policy.
///
/// This is intentionally separate from [`init_with_mirror`]: renderide redirects current stderr
/// into the file logger, then uses this hook to write error lines to the preserved original
/// terminal handle without feeding them back into the redirected stderr pipe.
///
/// Calling this before [`init`] / [`init_with_mirror`] succeeds has no effect.
pub fn set_mirror_writer(max_level: LogLevel, writer: fn(&[u8])) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    if let Ok(mut mirror) = logger.mirror_writer.lock() {
        *mirror = Some(MirrorWriter {
            max_level,
            write: writer,
        });
    }
}

/// Sets the maximum log level for the initialized global logger.
///
/// Has no effect if [`init`] / [`init_with_mirror`] has not succeeded. Safe to call from any
/// thread; takes effect immediately for subsequent [`log`] / macro calls.
pub fn set_max_level(level: LogLevel) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    logger.max_level.store(level as u8, Ordering::Relaxed);
    if LOG_FACADE_INSTALLED.load(Ordering::Relaxed) {
        log::set_max_level(level.into());
    }
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
    if let Some(logger) = LOGGER.get() {
        flush_file(logger);
    }
}

/// Flushes the primary log file when the file mutex is available.
fn flush_file(logger: &Logger) {
    if let Ok(mut file) = logger.file.lock() {
        let _ = file.flush();
    }
}

/// Returns the effective max level from `logger`'s atomic tag.
#[inline]
fn current_max_level(logger: &Logger) -> LogLevel {
    tag_to_level(logger.max_level.load(Ordering::Relaxed))
}

/// Writes the formatted line in `bytes` to the global logger's file and configured mirrors.
fn write_line_locked(logger: &Logger, level: LogLevel, bytes: &[u8]) {
    write_primary_file(logger, bytes);
    write_stderr_mirror(logger, bytes);
    write_callback_mirror(logger, level, bytes);
}

/// Writes `bytes` to the primary log file and flushes immediately.
fn write_primary_file(logger: &Logger, bytes: &[u8]) {
    if let Ok(mut file) = logger.file.lock() {
        write_and_flush(&mut *file, bytes);
    }
}

/// Writes `bytes` to stderr when stderr mirroring is enabled.
fn write_stderr_mirror(logger: &Logger, bytes: &[u8]) {
    if logger.mirror_stderr {
        write_and_flush(&mut io::stderr(), bytes);
    }
}

/// Writes `bytes` to the optional callback mirror when the mirror's level filter accepts the line.
fn write_callback_mirror(logger: &Logger, level: LogLevel, bytes: &[u8]) {
    let mirror = logger.mirror_writer.lock().ok().and_then(|guard| *guard);
    if let Some(mirror) = mirror
        && level <= mirror.max_level
    {
        let _ = std::panic::catch_unwind(|| (mirror.write)(bytes));
    }
}

/// Best-effort write followed by a best-effort flush.
fn write_and_flush(writer: &mut impl Write, bytes: &[u8]) {
    let _ = writer.write_all(bytes);
    let _ = writer.flush();
}

/// Internal log writer. Called by the log macros.
///
/// Does nothing when the logger is not initialized or when `level` is above the current max level.
#[doc(hidden)]
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    log_with_target(DEFAULT_TARGET, level, args);
}

/// Internal target-aware log writer. Called by the log macros.
///
/// Does nothing when the logger is not initialized or when `level` is above the current max level.
#[doc(hidden)]
pub fn log_with_target(target: &'static str, level: LogLevel, args: std::fmt::Arguments<'_>) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    let max = current_max_level(logger);
    if level > max {
        return;
    }
    with_line_buf(|buf| {
        format_log_line_into(buf, target, level, args);
        write_line_locked(logger, level, buf.as_bytes());
    });
}

/// Like [`log`], but uses [`Mutex::try_lock`] on the file handle. If the mutex is busy, appends the
/// same formatted line via a separate open of the log file path recorded at init when available.
///
/// Intended for **background threads** (such as a stderr pipe reader) that must not block on the
/// global logger mutex while other code may be writing to the same log or to stderr.
///
/// This fallback path is file-only and deliberately does not invoke configured mirror writers.
/// Native stdio forwarders use it after stdout/stderr redirection; mirroring here would duplicate
/// terminal output and could feed logs back into the redirected pipe.
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
        format_log_line_into(buf, DEFAULT_TARGET, level, args);
        let bytes = buf.as_bytes();
        if let Ok(mut file) = logger.file.try_lock() {
            write_and_flush(&mut *file, bytes);
            return true;
        }
        let mut opts = OpenOptions::new();
        opts.create(true).append(true);
        if let Ok(mut file) = opts.open(&logger.path) {
            write_and_flush(&mut file, bytes);
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
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// Creates an isolated `Logger` for tests that should not touch the global singleton.
    fn test_logger(path: &Path, max_level: LogLevel) -> Logger {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .expect("open log file");
        Logger::new(path.to_path_buf(), file, max_level, false)
    }

    /// End-to-end smoke test for singleton initialization, level filtering, duplicate init, panic
    /// payload logging, and final file contents.
    #[test]
    #[expect(
        clippy::cognitive_complexity,
        reason = "end-to-end smoke test that intentionally exercises init, level filtering, re-init, panic payloads, and file content in a single sequence"
    )]
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

    /// Verifies the `log` facade target filter suppresses Naga debug records only at debug max
    /// level and still accepts regular debug and Naga info records.
    #[test]
    fn logger_enabled_filters_naga_debug_when_max_level_is_debug() {
        let path = std::env::temp_dir().join(format!("logger_enabled_{}.log", std::process::id()));
        let _ = fs::remove_file(&path);
        let logger = test_logger(&path, LogLevel::Debug);
        let regular_debug = log::Metadata::builder()
            .level(log::Level::Debug)
            .target("renderide")
            .build();
        let naga_debug = log::Metadata::builder()
            .level(log::Level::Debug)
            .target("naga::front")
            .build();
        let naga_info = log::Metadata::builder()
            .level(log::Level::Info)
            .target("naga::front")
            .build();

        assert!(log::Log::enabled(&logger, &regular_debug));
        assert!(!log::Log::enabled(&logger, &naga_debug));
        assert!(log::Log::enabled(&logger, &naga_info));
        let _ = fs::remove_file(&path);
    }

    /// Verifies callback mirroring obeys its level filter while file output receives every line
    /// accepted by the primary logger.
    #[test]
    fn write_line_locked_respects_mirror_writer_level_filter() {
        static MIRRORED: AtomicUsize = AtomicUsize::new(0);

        /// Counts mirrored bytes for the mirror-writer level-filter test.
        fn mirror(bytes: &[u8]) {
            MIRRORED.fetch_add(bytes.len(), AtomicOrdering::SeqCst);
        }

        MIRRORED.store(0, AtomicOrdering::SeqCst);
        let path = std::env::temp_dir().join(format!("logger_mirror_{}.log", std::process::id()));
        let _ = fs::remove_file(&path);
        let logger = test_logger(&path, LogLevel::Trace);
        if let Ok(mut mirror_writer) = logger.mirror_writer.lock() {
            *mirror_writer = Some(MirrorWriter {
                max_level: LogLevel::Warn,
                write: mirror,
            });
        }

        write_line_locked(&logger, LogLevel::Info, b"info\n");
        assert_eq!(MIRRORED.load(AtomicOrdering::SeqCst), 0);
        write_line_locked(&logger, LogLevel::Error, b"error\n");
        assert_eq!(MIRRORED.load(AtomicOrdering::SeqCst), 6);
        let contents = fs::read_to_string(&path).expect("read log");
        assert!(contents.contains("info"));
        assert!(contents.contains("error"));
        let _ = fs::remove_file(&path);
    }
}
