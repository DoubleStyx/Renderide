//! Global file logger: one [`std::sync::OnceLock`] sink, optional stderr mirroring, and atomic
//! max-level filtering without reopening the log file.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::level::{level_to_tag, tag_to_level, LogLevel};
use crate::timestamp::format_line_timestamp;

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
        max_level: AtomicU8::new(level_to_tag(max_level)),
    };
    let _ = LOGGER.set(logger);
    crate::log_adapter::install(max_level);
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
    logger
        .max_level
        .store(level_to_tag(level), Ordering::Relaxed);
    crate::log_adapter::set_log_crate_max_level(level);
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
    if let Some(logger) = LOGGER.get() {
        if let Ok(mut file) = logger.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Full log line with UTC prefix timestamp, for file (and optional stderr) output.
fn format_log_line(level: LogLevel, args: std::fmt::Arguments<'_>) -> String {
    let mut line = String::with_capacity(64);
    let timestamp = format_line_timestamp();
    let _ = write!(line, "[{timestamp}] {level} ");
    let _ = line.write_fmt(args);
    line.push('\n');
    line
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
    let line = format_log_line(level, args);
    if let Ok(mut file) = logger.file.lock() {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
    if logger.mirror_stderr {
        let _ = std::io::stderr().write_all(line.as_bytes());
        let _ = std::io::stderr().flush();
    }
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
    let line = format_log_line(level, args);
    if let Ok(mut file) = logger.file.try_lock() {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
        return true;
    }
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    if let Ok(mut file) = opts.open(&logger.path) {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
        return true;
    }
    false
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
}
