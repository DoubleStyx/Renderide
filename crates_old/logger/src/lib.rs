//! Tiered file logging for Renderide and bootstrapper.
//!
//! Provides `LogLevel`, `init()`, `log_panic()`, and macros `error!`, `warn!`, `info!`, `debug!`, `trace!`.
//! Call `init(path, max_level, append)` once at startup; use `log_panic(path, info)` in panic hooks.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Log level for filtering. Lower ordinal = higher priority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Critical errors.
    Error,
    /// Warnings.
    Warn,
    /// Informational messages.
    Info,
    /// Debug diagnostics.
    Debug,
    /// Verbose trace.
    Trace,
}

impl LogLevel {
    /// Parses a level string (case-insensitive). Returns None for invalid values.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" | "e" => Some(Self::Error),
            "warn" | "warning" | "w" => Some(Self::Warn),
            "info" | "i" => Some(Self::Info),
            "debug" | "d" => Some(Self::Debug),
            "trace" | "t" => Some(Self::Trace),
            _ => None,
        }
    }

    /// Returns the string to pass as `-LogLevel` value.
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warn => write!(f, "WARN"),
            Self::Info => write!(f, "INFO"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Trace => write!(f, "TRACE"),
        }
    }
}

/// Logger that writes to file and optionally to stderr.
struct Logger {
    /// File output. Mutex for thread-safe writes.
    file: Mutex<std::fs::File>,
    /// Whether to also write to stderr.
    console: bool,
    /// Maximum level to log. Messages at or below this level are written.
    max_level: LogLevel,
}

/// Global logger instance. Set by `init()`.
static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Parses `-LogLevel` from command line args (case-insensitive).
/// Returns `None` if not present or invalid; otherwise the parsed level.
pub fn parse_log_level_from_args() -> Option<LogLevel> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg.eq_ignore_ascii_case("-LogLevel") && i + 1 < args.len() {
            return LogLevel::parse(&args[i + 1]);
        }
        i += 1;
    }
    None
}

/// Initializes logging. Creates parent directory if needed, opens file.
/// Call once at startup before panic hook.
///
/// # Arguments
/// * `path` - Path to the log file.
/// * `max_level` - Maximum level to log (messages at or below this level are written).
/// * `append` - If true, append to file; if false, truncate.
///
/// # Errors
/// Returns `Err` if the log file cannot be opened (e.g. permission denied, path invalid).
/// Callers should fail fast on error rather than continuing without logging.
pub fn init(
    path: impl AsRef<Path>,
    max_level: LogLevel,
    append: bool,
) -> Result<(), std::io::Error> {
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
        console: false,
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

/// Flushes any buffered log output. For `std::fs::File`, `flush()` is a no-op (data goes
/// to the kernel on write). Call periodically for API consistency.
///
/// Do not call from a panic hook: if the panic occurred while holding the logger mutex
/// (e.g. inside a log macro), this would deadlock.
pub fn flush() {
    if let Some(logger) = LOGGER.get()
        && let Ok(mut file) = logger.file.lock()
    {
        let _ = file.flush();
    }
}

/// Logs a panic payload from `catch_unwind`. Extracts `String` or `&'static str` if possible.
/// Use when handling `Err(e)` from `std::panic::catch_unwind` to surface the panic message.
pub fn log_panic_payload(payload: Box<dyn std::any::Any + Send>, context: &str) {
    let msg = match payload.downcast::<String>() {
        Ok(s) => format!("{}: {}", context, *s),
        Err(p) => match p.downcast::<&'static str>() {
            Ok(s) => format!("{}: {}", context, *s),
            Err(_) => format!("{}: panic (payload type not string)", context),
        },
    };
    log(LogLevel::Error, format_args!("{}", msg));
}

/// Writes panic info and backtrace to the log file. Flushes immediately so the panic
/// is visible on disk. Does not acquire the logger mutex (safe from panic handler).
/// Uses `force_capture()` so backtraces are always recorded, regardless of `RUST_BACKTRACE`.
pub fn log_panic(path: impl AsRef<Path>, info: &dyn std::fmt::Display) {
    let path = path.as_ref();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "PANIC: {}", info);
        let _ = writeln!(
            f,
            "Backtrace:\n{:#?}",
            std::backtrace::Backtrace::force_capture()
        );
        let _ = f.flush();
        let _ = f.sync_all();
    }
}

/// Returns true if messages at `level` would be written. Used by macros to skip
/// argument evaluation when the level is disabled.
#[doc(hidden)]
#[inline(always)]
pub fn is_level_enabled(level: LogLevel) -> bool {
    LOGGER.get().is_some_and(|l| level <= l.max_level)
}

/// Internal log writer. Called by the log macros.
#[doc(hidden)]
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    if let Some(logger) = LOGGER.get()
        && level <= logger.max_level
    {
        let msg = args.to_string();
        let timestamp = format_timestamp();
        let line = format!("[{}] {:?} {}\n", timestamp, level, msg);
        if let Ok(mut file) = logger.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
        if logger.console {
            let _ = std::io::stderr().write_all(line.as_bytes());
            let _ = std::io::stderr().flush();
        }
    }
}

/// Returns a filename-safe UTC timestamp: `YYYY-MM-DD_HH-MM-SS`. Used for log file names.
pub fn log_filename_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "unknown".to_string();
    };
    let secs = dur.as_secs();
    let day_secs = secs % 86400;
    let h = day_secs / 3600;
    let m = (day_secs / 60) % 60;
    let s = day_secs % 60;
    let (y, mo, d) = days_since_epoch_to_ymd(secs / 86400);
    format!("{:04}-{:02}-{:02}_{:02}-{:02}-{:02}", y, mo, d, h, m, s)
}

/// Converts days since Unix epoch (1970-01-01) to (year, month, day).
/// Algorithm: <http://howardhinnant.github.io/date_algorithms.html> `civil_from_days`.
fn days_since_epoch_to_ymd(days: u64) -> (u32, u32, u32) {
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe.min(146_096) / 146_096) / 365;
    let y = (yoe as i64 + era * 400) as u32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Returns a simple timestamp string. Uses std::time for a minimal format.
fn format_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "?".to_string();
    };
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let mins = (secs / 60) % 60;
    let hours = (secs / 3600) % 24;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}

/// Logs at error level.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        if $crate::is_level_enabled($crate::LogLevel::Error) {
            $crate::log($crate::LogLevel::Error, format_args!($($arg)*))
        }
    };
}

/// Logs at warn level.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        if $crate::is_level_enabled($crate::LogLevel::Warn) {
            $crate::log($crate::LogLevel::Warn, format_args!($($arg)*))
        }
    };
}

/// Logs at info level.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        if $crate::is_level_enabled($crate::LogLevel::Info) {
            $crate::log($crate::LogLevel::Info, format_args!($($arg)*))
        }
    };
}

/// Logs at debug level.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        if $crate::is_level_enabled($crate::LogLevel::Debug) {
            $crate::log($crate::LogLevel::Debug, format_args!($($arg)*))
        }
    };
}

/// Logs at trace level.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        if $crate::is_level_enabled($crate::LogLevel::Trace) {
            $crate::log($crate::LogLevel::Trace, format_args!($($arg)*))
        }
    };
}
