//! File-first logging for the Renderide workspace (bootstrapper, captured host output, renderer).
//!
//! ## API split
//!
//! - **Macros** ([`error!`], [`warn!`], [`info!`], [`debug!`], [`trace!`]) call [`enabled`] before
//!   evaluating format arguments, then [`log`] when the level passes the filter.
//! - **Functions** such as [`init`], [`flush`], and [`try_log`] are for programmatic control,
//!   background threads, and panic / crash paths that must not touch the logger mutex.
//!
//! Install the global sink **once** with [`init`] or [`init_for`]. A second successful [`init`]
//! returns [`Ok`] but does not replace the first logger; use [`is_initialized`] to detect ordering
//! issues.
//!
//! # Layout
//!
//! Logs default to **`Renderide/logs/<component>/<UTC-date>_<UTC-time-to-the-second>.log`**, where
//! `<component>` is one of [`LogComponent`]. The repository root is inferred from this crate’s
//! manifest path: `.../Renderide/crates/logger` → two levels up → `.../Renderide/logs`.
//!
//! Override the root directory with the **`RENDERIDE_LOGS_ROOT`** environment variable; the value
//! is used as-is as the logs root for all components.
//!
//! # Usage
//!
//! - Call [`init`] or [`init_for`] once at startup, then install a panic hook that calls
//!   [`log_panic`] with the same file path, or compose [`panic_report`] and
//!   [`append_panic_report_to_file`] if you also mirror the report to a preserved terminal handle
//!   (see the renderer’s `native_stdio` module).
//! - Use [`parse_log_level_from_args`] for `-LogLevel` (case-insensitive). After init, use
//!   [`set_max_level`] to change filtering without reopening the log file.
//! - Prefer [`init_for`] when using the standard layout; use [`init`] with a custom path when needed.
//!
//! # Panics and flushing
//!
//! Do not call [`flush`] from a panic handler if the panic might have occurred while holding the
//! logger’s internal mutex (for example inside a log macro), or you risk deadlock.

mod level;
mod log_adapter;
mod output;
mod panic;
mod paths;
mod timestamp;

pub use level::{parse_log_level_from_args, LogLevel};
pub use output::{
    enabled, flush, init, init_with_mirror, is_initialized, log, set_max_level, try_log,
};
pub use panic::{append_panic_report_to_file, log_panic, log_panic_payload, panic_report};
pub use paths::{
    ensure_log_dir, init_for, log_dir_for, log_file_path, logs_root, logs_root_with, LogComponent,
    LogsRootError,
};
pub use timestamp::log_filename_timestamp;

/// Hidden helper used by [`error!`], [`warn!`], [`info!`], [`debug!`], and [`trace!`] to share the
/// `enabled` + [`log`] pattern.
#[doc(hidden)]
#[macro_export]
macro_rules! __log_at {
    ($level:expr, $($arg:tt)*) => {
        if $crate::enabled($level) {
            $crate::log($level, format_args!($($arg)*))
        }
    };
}

/// Writes an error-level line through [`crate::log`] when [`LogLevel::Error`] is enabled.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::__log_at!($crate::LogLevel::Error, $($arg)*)
    };
}

/// Writes a warn-level line through [`crate::log`] when [`LogLevel::Warn`] is enabled.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        $crate::__log_at!($crate::LogLevel::Warn, $($arg)*)
    };
}

/// Writes an info-level line through [`crate::log`] when [`LogLevel::Info`] is enabled.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        $crate::__log_at!($crate::LogLevel::Info, $($arg)*)
    };
}

/// Writes a debug-level line through [`crate::log`] when [`LogLevel::Debug`] is enabled.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        $crate::__log_at!($crate::LogLevel::Debug, $($arg)*)
    };
}

/// Writes a trace-level line through [`crate::log`] when [`LogLevel::Trace`] is enabled.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        $crate::__log_at!($crate::LogLevel::Trace, $($arg)*)
    };
}
