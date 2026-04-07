//! Append-only panic logging so panic hooks never block on the global logger mutex.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::level::LogLevel;
use crate::output;

/// Logs a panic payload from `catch_unwind`. Extracts [`String`] or `&'static str` if possible.
///
/// Use when handling [`Err`] from [`std::panic::catch_unwind`] to surface the panic message through
/// the normal logger (requires an initialized global logger).
pub fn log_panic_payload(payload: Box<dyn std::any::Any + Send>, context: &str) {
    let msg = match payload.downcast::<String>() {
        Ok(s) => format!("{context}: {}", *s),
        Err(p) => match p.downcast::<&'static str>() {
            Ok(s) => format!("{context}: {}", *s),
            Err(_) => format!("{context}: panic (payload type not string)"),
        },
    };
    output::log(LogLevel::Error, format_args!("{msg}"));
}

/// Writes panic info and backtrace to the given log file. Flushes immediately so the panic
/// is visible on disk. Does not acquire the logger mutex (safe from a panic handler).
///
/// Uses [`std::backtrace::Backtrace::force_capture`] so backtraces are recorded regardless of
/// `RUST_BACKTRACE`.
pub fn log_panic(path: impl AsRef<Path>, info: &dyn std::fmt::Display) {
    let path = path.as_ref();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "PANIC: {info}");
        let _ = writeln!(
            f,
            "Backtrace:\n{:#?}",
            std::backtrace::Backtrace::force_capture()
        );
        let _ = f.flush();
        let _ = f.sync_all();
    }
}
