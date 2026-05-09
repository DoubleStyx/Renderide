//! File logging, native stdio forwarding, fatal-crash logging, and panic reporting.

use std::path::Path;

use logger::{LogComponent, LogLevel};

use crate::run_error::RunError;

/// Logging state produced during app bootstrap.
pub(crate) struct LoggingBootstrap {
    /// Parsed `-LogLevel` command-line override.
    pub(crate) log_level_cli: Option<LogLevel>,
}

/// Initializes file logging and crash/panic visibility.
pub(crate) fn init_logging() -> Result<LoggingBootstrap, RunError> {
    let timestamp = logger::log_filename_timestamp();
    let log_level_cli = logger::parse_log_level_from_args();
    let initial_log_level = log_level_cli.unwrap_or(LogLevel::Info);
    let log_path = logger::init_for(LogComponent::Renderer, &timestamp, initial_log_level, false)
        .map_err(RunError::logging_init)?;

    logger::info!(
        "Logging to {} at max level {:?}",
        log_path.display(),
        initial_log_level
    );

    crate::native_stdio::ensure_stdio_forwarded_to_logger();
    crate::fatal_crash_log::install(&log_path);
    install_panic_hook(&log_path);

    Ok(LoggingBootstrap { log_level_cli })
}

fn install_panic_hook(log_path: &Path) {
    let log_path_hook = log_path.to_path_buf();
    std::panic::set_hook(Box::new(move |info| {
        let report = logger::panic_report(info);
        logger::append_panic_report_to_file(&log_path_hook, &report);
        crate::native_stdio::try_write_preserved_stderr(report.as_bytes());
    }));
}
