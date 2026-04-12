//! Global file logger and `init` / [`crate::log`] implementation.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::level::{level_to_tag, tag_to_level, LogLevel};
use crate::timestamp::format_line_timestamp;

/// Path of the active log file, set by [`init_with_mirror`]. Used for non-blocking append when the
/// primary mutex is held (for example a stderr forwarder thread).
static LOG_FILE_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Logger that writes to file and optionally mirrors each line to stderr.
struct Logger {
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

/// Initializes logging. Creates parent directory if needed, opens file.
///
/// Call once at startup before installing a panic hook. Mirror to stderr is disabled; use
/// [`init_with_mirror`] to enable it.
///
/// # Errors
///
/// Returns [`Err`] if the log file cannot be opened (for example permission denied or an invalid path).
/// Callers should fail fast on error rather than continuing without logging.
pub fn init(path: impl AsRef<Path>, max_level: LogLevel, append: bool) -> std::io::Result<()> {
    init_with_mirror(path, max_level, append, false)
}

/// Like [`init`], but when `mirror_stderr` is true each log line is also written to stderr.
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
        file: Mutex::new(file),
        mirror_stderr,
        max_level: AtomicU8::new(level_to_tag(max_level)),
    };
    let _ = LOGGER.set(logger);
    let _ = LOG_FILE_PATH.set(path.to_path_buf());
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
}

#[inline]
fn current_max_level(logger: &Logger) -> LogLevel {
    tag_to_level(logger.max_level.load(Ordering::Relaxed))
}

/// Returns whether a message at `level` would be written given the current max level.
///
/// Use to avoid expensive formatting when logging is filtered out.
pub fn enabled(level: LogLevel) -> bool {
    LOGGER
        .get()
        .is_some_and(|logger| level <= current_max_level(logger))
}

/// Flushes any buffered log output. Call periodically if desired for API consistency.
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

/// Used by macros to skip argument evaluation when the level is disabled.
#[doc(hidden)]
#[inline(always)]
pub fn is_level_enabled(level: LogLevel) -> bool {
    LOGGER.get().is_some_and(|l| level <= current_max_level(l))
}

/// Internal log writer. Called by the log macros.
#[doc(hidden)]
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    let max = current_max_level(logger);
    if level > max {
        return;
    }
    let msg = args.to_string();
    let timestamp = format_line_timestamp();
    let line = format!("[{timestamp}] {level:?} {msg}\n");
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
/// Returns `true` if the line was written (primary or fallback), `false` if nothing was written.
pub fn try_log(level: LogLevel, args: std::fmt::Arguments<'_>) -> bool {
    let Some(logger) = LOGGER.get() else {
        return false;
    };
    let max = current_max_level(logger);
    if level > max {
        return false;
    }
    let msg = args.to_string();
    let timestamp = format_line_timestamp();
    let line = format!("[{timestamp}] {level:?} {msg}\n");
    if let Ok(mut file) = logger.file.try_lock() {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
        return true;
    }
    let Some(path) = LOG_FILE_PATH.get() else {
        return false;
    };
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    if let Ok(mut file) = opts.open(path) {
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
        return true;
    }
    false
}
