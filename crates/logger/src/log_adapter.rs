//! Bridges the `log` crate facade onto this crate's file sink so records emitted by upstream
//! dependencies (`rfd`, `wgpu`, `winit`, `naga`, …) reach `logs/<component>/*.log` instead of
//! disappearing into an uninstalled `log::set_logger` hole.
//!
//! ## Why this exists
//!
//! Most ecosystem crates emit diagnostics through `log::error!` / `log::warn!` / etc. Without a
//! registered global logger those calls are dropped. The bootstrapper's desktop/VR dialog is the
//! original motivating case: `rfd`'s `xdg_desktop_portal` backend reports zenity-spawn failures
//! via `log::error!` only, so a silently broken zenity made the bootstrapper exit with no usable
//! diagnostic. Installing a thin proxy here keeps every future "what did the dependency think
//! went wrong?" question answerable from the same log file we already write to.
//!
//! ## Filtering
//!
//! [`install`] mirrors the level passed to [`crate::init_for`] / [`crate::init_with_mirror`] into
//! `log::set_max_level`, so dependency records honor the same `-LogLevel` budget as in-crate
//! macros. [`crate::set_max_level`] also forwards through this module after install.
//!
//! ## Safety
//!
//! `log::set_logger` succeeds at most once per process; a second call is silently ignored here so
//! repeated `init_with_mirror` calls (the existing crate contract) stay no-op-on-second-call.

use crate::level::LogLevel;
use crate::output;

/// Stateless `log::Log` proxy that forwards every accepted record through [`crate::output::log`].
///
/// Holds no per-instance state; one zero-sized static value backs the global `log::set_logger`
/// registration so the proxy can be `'static` without heap allocation.
struct LogProxy;

impl log::Log for LogProxy {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        output::enabled(log_level_to_local(metadata.level()))
    }

    fn log(&self, record: &log::Record<'_>) {
        let level = log_level_to_local(record.level());
        if !output::enabled(level) {
            return;
        }
        let target = record.target();
        if target.is_empty() {
            output::log(level, format_args!("{}", record.args()));
        } else {
            output::log(level, format_args!("[{target}] {}", record.args()));
        }
    }

    fn flush(&self) {
        output::flush();
    }
}

/// Singleton proxy registered with the `log` crate.
///
/// `log::set_logger` requires `&'static dyn log::Log`; using a zero-sized type here means the
/// reference is trivially static without `Box::leak` or `OnceLock` ceremony.
static PROXY: LogProxy = LogProxy;

/// Maps a `log` crate severity onto the local [`LogLevel`].
///
/// The two enumerations have the same five variants in the same order, so this is a total
/// straight-through mapping with no fallback.
fn log_level_to_local(level: log::Level) -> LogLevel {
    match level {
        log::Level::Error => LogLevel::Error,
        log::Level::Warn => LogLevel::Warn,
        log::Level::Info => LogLevel::Info,
        log::Level::Debug => LogLevel::Debug,
        log::Level::Trace => LogLevel::Trace,
    }
}

/// Maps the local [`LogLevel`] onto a `log::LevelFilter` for `log::set_max_level`.
///
/// Used by [`install`] and [`set_log_crate_max_level`] so dependency records share the local
/// max-level budget exactly.
fn local_to_log_filter(level: LogLevel) -> log::LevelFilter {
    match level {
        LogLevel::Error => log::LevelFilter::Error,
        LogLevel::Warn => log::LevelFilter::Warn,
        LogLevel::Info => log::LevelFilter::Info,
        LogLevel::Debug => log::LevelFilter::Debug,
        LogLevel::Trace => log::LevelFilter::Trace,
    }
}

/// Registers the [`PROXY`] as the process-wide `log` crate logger and aligns its max level.
///
/// Idempotent: a second call (typical when both bootstrapper and an in-process subsystem call
/// [`crate::init_with_mirror`]) leaves the existing registration in place but still updates the
/// `log` crate's max level, so a later init that raised the budget takes effect for upstream
/// records too.
pub(crate) fn install(max_level: LogLevel) {
    let _ = log::set_logger(&PROXY);
    log::set_max_level(local_to_log_filter(max_level));
}

/// Updates only the `log` crate's max level filter, leaving registration untouched.
///
/// Called from [`crate::set_max_level`] so a runtime level change after init also affects
/// dependency records routed through the proxy.
pub(crate) fn set_log_crate_max_level(level: LogLevel) {
    log::set_max_level(local_to_log_filter(level));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LogLevel;

    /// `log::Level` → [`LogLevel`] mapping is a total straight-through mapping with no fallback.
    ///
    /// File-level integration (a record sent through `log::logger().log()` reaching the global
    /// file sink) is exercised by [`crate::output::tests::global_logger_full_smoke`] now that
    /// `init_with_mirror` calls [`install`]; reproducing that here would race the smoke test on
    /// the process-wide `LOGGER` `OnceLock`.
    #[test]
    fn log_level_mapping_is_total() {
        assert_eq!(log_level_to_local(log::Level::Error), LogLevel::Error);
        assert_eq!(log_level_to_local(log::Level::Warn), LogLevel::Warn);
        assert_eq!(log_level_to_local(log::Level::Info), LogLevel::Info);
        assert_eq!(log_level_to_local(log::Level::Debug), LogLevel::Debug);
        assert_eq!(log_level_to_local(log::Level::Trace), LogLevel::Trace);
    }

    /// [`LogLevel`] → `log::LevelFilter` mapping covers every variant.
    #[test]
    fn log_filter_mapping_is_total() {
        assert_eq!(
            local_to_log_filter(LogLevel::Error),
            log::LevelFilter::Error
        );
        assert_eq!(local_to_log_filter(LogLevel::Warn), log::LevelFilter::Warn);
        assert_eq!(local_to_log_filter(LogLevel::Info), log::LevelFilter::Info);
        assert_eq!(
            local_to_log_filter(LogLevel::Debug),
            log::LevelFilter::Debug
        );
        assert_eq!(
            local_to_log_filter(LogLevel::Trace),
            log::LevelFilter::Trace
        );
    }

    /// `LogProxy::enabled` honors the local logger's filter once the global sink is installed,
    /// driven by the ambient state established by other tests (smoke test installs the file
    /// sink at `LogLevel::Trace`, so every level maps to "enabled").
    ///
    /// This test only runs assertions when the ambient logger is initialized; if it runs first
    /// in the harness order it falls through silently rather than racing init.
    #[test]
    fn proxy_enabled_tracks_local_filter_when_initialized() {
        if !crate::is_initialized() {
            return;
        }
        let proxy = LogProxy;
        let metadata = log::Metadata::builder()
            .level(log::Level::Error)
            .target("")
            .build();
        assert!(log::Log::enabled(&proxy, &metadata));
    }
}
