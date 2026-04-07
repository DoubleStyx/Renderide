//! File-first logging for the Renderide workspace (bootstrapper, captured host output, renderer).
//!
//! # Layout
//!
//! Logs default to **`Renderide/logs/<component>/<UTC-date>_<UTC-time-to-the-second>.log`**, where
//! `<component>` is one of [`LogComponent`]. The repository root is inferred from this crate’s
//! manifest path: `.../Renderide/crates/logger` → two levels up → `.../Renderide/logs`.
//!
//! Override the root directory with the **`RENDERIDE_LOGS_ROOT`** environment variable (the value
//! is used as-is as the logs root for all components).
//!
//! # Usage
//!
//! - Call [`init`] or [`init_for`] once at startup, then install a panic hook that calls
//!   [`log_panic`] with the same file path.
//! - Use [`parse_log_level_from_args`] for `-LogLevel` (case-insensitive).
//! - Prefer [`init_for`] when using the standard layout; use [`init`] with a custom path when needed.
//!
//! # Panics and flushing
//!
//! Do not call [`flush`] from a panic handler if the panic might have occurred while holding the
//! logger’s internal mutex (for example inside a log macro), or you risk deadlock.

mod level;
mod output;
mod panic;
mod paths;
mod timestamp;

pub use level::{parse_log_level_from_args, LogLevel};
pub use output::{enabled, flush, init, init_with_mirror, log};
pub use panic::{log_panic, log_panic_payload};
pub use paths::{
    ensure_log_dir, init_for, log_dir_for, log_file_path, logs_root, logs_root_with, LogComponent,
};
pub use timestamp::log_filename_timestamp;

/// Returns true if messages at `level` would be written. Used by macros to skip
/// argument evaluation when the level is disabled.
#[doc(hidden)]
#[inline(always)]
pub fn is_level_enabled(level: LogLevel) -> bool {
    output::is_level_enabled(level)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn log_level_parse_aliases() {
        assert_eq!(LogLevel::parse("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("E"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("WARN"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::parse("bogus"), None);
    }

    #[test]
    fn log_level_as_arg_roundtrip() {
        for level in [
            LogLevel::Error,
            LogLevel::Warn,
            LogLevel::Info,
            LogLevel::Debug,
            LogLevel::Trace,
        ] {
            assert_eq!(LogLevel::parse(level.as_arg()), Some(level));
        }
    }

    #[test]
    fn logs_root_from_manifest_path() {
        let manifest = Path::new("/workspace/Renderide/crates/logger");
        let root = logs_root_with(manifest, None);
        assert_eq!(root, PathBuf::from("/workspace/Renderide/logs"));
    }

    #[test]
    fn logs_root_env_override_wins() {
        let manifest = Path::new("/workspace/Renderide/crates/logger");
        let root = logs_root_with(manifest, Some(std::ffi::OsStr::new("/tmp/custom_logs")));
        assert_eq!(root, PathBuf::from("/tmp/custom_logs"));
    }

    #[test]
    fn log_component_subdirs() {
        assert_eq!(LogComponent::Bootstrapper.subdir(), "bootstrapper");
        assert_eq!(LogComponent::Host.subdir(), "host");
        assert_eq!(LogComponent::Renderer.subdir(), "renderer");
    }

    #[test]
    fn log_file_path_layout() {
        let manifest = Path::new("/r/Renderide/crates/logger");
        let root = logs_root_with(manifest, None);
        let expected = root.join("renderer").join("2026-04-05_12-00-00.log");
        let got = logs_root_with(manifest, None)
            .join(LogComponent::Renderer.subdir())
            .join("2026-04-05_12-00-00.log");
        assert_eq!(got, expected);
    }

    #[test]
    fn log_filename_timestamp_format() {
        let s = log_filename_timestamp();
        assert_eq!(s.len(), 19);
        assert!(s.contains('_'));
        let (date, time) = s.split_once('_').expect("timestamp contains underscore");
        assert_eq!(date.len(), 10);
        assert_eq!(time.len(), 8);
        assert!(date.chars().filter(|c| *c == '-').count() == 2);
        assert!(time.chars().filter(|c| *c == '-').count() == 2);
    }
}
