//! Global file logger and `init` / [`crate::log`] implementation.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use crate::level::LogLevel;
use crate::timestamp::format_line_timestamp;

/// Logger that writes to file and optionally mirrors each line to stderr.
struct Logger {
    /// File output. Mutex for thread-safe writes.
    file: Mutex<std::fs::File>,
    /// When true, each log line is also written to stderr.
    mirror_stderr: bool,
    /// Maximum level to log. Messages at or below this level are written.
    max_level: LogLevel,
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
        max_level,
    };
    let _ = LOGGER.set(logger);
    Ok(())
}

/// Returns whether a message at `level` would be written given the current max level.
///
/// Use to avoid expensive formatting when logging is filtered out.
pub fn enabled(level: LogLevel) -> bool {
    LOGGER.get().is_some_and(|logger| level <= logger.max_level)
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
    LOGGER.get().is_some_and(|l| level <= l.max_level)
}

/// Internal log writer. Called by the log macros.
#[doc(hidden)]
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    if level > logger.max_level {
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
