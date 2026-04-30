//! Append-only panic logging so panic hooks never block on the global logger mutex.
//!
//! Prefer [`append_panic_report_to_file`] and [`log_panic`] from panic handlers; use [`log_panic_payload`]
//! only when you already have a `catch_unwind` payload and an initialized global logger.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::level::LogLevel;
use crate::output;

/// Logs a panic payload from `catch_unwind`. Extracts [`String`] or `&'static str` if possible.
///
/// Use when handling [`Err`] from [`std::panic::catch_unwind`] to surface the panic message through
/// the normal logger (requires an initialized global logger).
///
/// Mis-typed payloads fall back to a generic message instead of unwinding.
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

/// Formats a panic line and full backtrace for logging and optional terminal output.
///
/// Uses [`std::backtrace::Backtrace::force_capture`] so backtraces are recorded regardless of
/// `RUST_BACKTRACE`.
pub fn panic_report(info: &dyn std::fmt::Display) -> String {
    format!(
        "PANIC: {info}\nBacktrace:\n{:#?}\n",
        std::backtrace::Backtrace::force_capture()
    )
}

/// Appends a pre-formatted panic report to the log file without acquiring the global logger mutex.
///
/// Safe to call from a panic hook alongside [`panic_report`].
pub fn append_panic_report_to_file(path: impl AsRef<Path>, report: &str) {
    let path = path.as_ref();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(report.as_bytes());
        let _ = f.flush();
        let _ = f.sync_all();
    }
}

/// Writes panic info and backtrace to the given log file. Flushes immediately so the panic
/// is visible on disk. Does not acquire the logger mutex (safe from a panic handler).
///
/// Uses [`panic_report`] internally.
pub fn log_panic(path: impl AsRef<Path>, info: &dyn std::fmt::Display) {
    let report = panic_report(info);
    append_panic_report_to_file(path, &report);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    /// Minimal [`std::fmt::Display`] for [`panic_report`] tests.
    struct Dummy;

    impl fmt::Display for Dummy {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test panic display")
        }
    }

    #[test]
    fn panic_report_contains_panic_and_backtrace_labels() {
        let s = panic_report(&Dummy);
        assert!(s.starts_with("PANIC: "));
        assert!(s.contains("test panic display"));
        assert!(s.contains("Backtrace:"));
    }

    #[test]
    fn panic_report_includes_provided_display() {
        let s = panic_report(&Dummy);
        assert!(
            s.contains("PANIC: test panic display"),
            "expected full display in report: {s}"
        );
    }

    #[test]
    fn append_panic_report_to_file_appends_bytes() {
        let path = std::env::temp_dir().join(format!("logger_panic_append_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        append_panic_report_to_file(&path, "hello panic\n");
        let got = std::fs::read_to_string(&path).unwrap();
        assert_eq!(got, "hello panic\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_panic_writes_panic_prefix_and_backtrace_label() {
        let path = std::env::temp_dir().join(format!("logger_log_panic_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        log_panic(&path, &Dummy);
        let got = std::fs::read_to_string(&path).expect("read");
        assert!(got.starts_with("PANIC: "));
        assert!(got.contains("Backtrace:"));
        let _ = std::fs::remove_file(&path);
    }

    /// Verifies two appended reports by matching the full panic line and the backtrace header
    /// boundary. A bare `"PANIC:"` count is unreliable on Windows because `Backtrace` `Debug`
    /// output can contain that substring in paths or symbols.
    #[test]
    fn log_panic_appends_to_existing_file() {
        let path =
            std::env::temp_dir().join(format!("logger_log_panic_append_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        log_panic(&path, &Dummy);
        log_panic(&path, &Dummy);
        let got = std::fs::read_to_string(&path).expect("read");
        assert_eq!(got.matches("PANIC: test panic display").count(), 2);
        assert_eq!(got.matches("\nBacktrace:\n").count(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_panic_payload_string_branch() {
        log_panic_payload(Box::new("boom".to_string()), "ctx");
    }

    #[test]
    fn log_panic_payload_static_str_branch() {
        log_panic_payload(Box::new("static boom"), "ctx");
    }

    #[test]
    fn log_panic_payload_other_type() {
        log_panic_payload(Box::new(42_i32), "ctx");
    }

    /// Drives [`log_panic`] through the same path real panic hooks use:
    /// [`std::panic::catch_unwind`] returns the panic payload, and the helper writes a complete
    /// report to disk. Confirms the helper survives the unwinding context and produces the same
    /// `PANIC: ` / `Backtrace:` shape as direct invocation.
    #[test]
    fn log_panic_writes_report_when_called_from_catch_unwind() {
        let path = std::env::temp_dir().join(format!(
            "logger_log_panic_catch_unwind_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(|| {
            panic!("catch_unwind panic message");
        });
        std::panic::set_hook(prev);

        let payload = result.expect_err("closure should have panicked");
        let message = payload
            .downcast::<&'static str>()
            .map(|s| (*s).to_string())
            .unwrap_or_else(|_| "non-string payload".to_string());

        log_panic(&path, &message);

        let got = std::fs::read_to_string(&path).expect("read");
        assert!(got.starts_with("PANIC: "), "got {got:?}");
        assert!(
            got.contains("catch_unwind panic message"),
            "expected payload in report: {got:?}"
        );
        assert!(got.contains("Backtrace:"), "got {got:?}");

        let _ = std::fs::remove_file(&path);
    }
}
