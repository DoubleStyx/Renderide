//! File logging, native stdio forwarding, and renderer-startup banner.

use std::env;
use std::path::Path;

use logger::{LogComponent, LogLevel};

use crate::diagnostics::crash_context::{self, TickPhase};
use crate::run_error::RunError;

/// Logging state produced during app bootstrap.
pub(crate) struct LoggingBootstrap {
    /// Parsed `-LogLevel` command-line override.
    pub(crate) log_level_cli: Option<LogLevel>,
}

/// Initializes file logging and crash/panic visibility.
pub(crate) fn init_logging() -> Result<LoggingBootstrap, RunError> {
    crash_context::init_process_context();
    crash_context::set_tick_phase(TickPhase::Startup);

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
    log_renderer_startup_context(&log_path, initial_log_level);

    crate::native_stdio::ensure_stdio_forwarded_to_logger();
    crate::fatal_crash_log::install(&log_path);
    super::panic::install_panic_hook(&log_path);

    Ok(LoggingBootstrap { log_level_cli })
}

fn log_renderer_startup_context(log_path: &Path, initial_log_level: LogLevel) {
    logger::info!(
        "Renderer process: version={} pid={} target={} {} arch={} exe={} cwd={} cli_mode={} log_path={} log_level={:?}",
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
        env::consts::OS,
        env::consts::FAMILY,
        env::consts::ARCH,
        env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| format!("<unavailable: {e}>")),
        env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| format!("<unavailable: {e}>")),
        sanitized_cli_mode(),
        log_path.display(),
        initial_log_level,
    );
    for key in [
        "RENDERIDE_CONFIG",
        "RENDERIDE_LOGS_ROOT",
        "RENDERIDE_INTERPROCESS_DIR",
        "RENDERIDE_GPU_VALIDATION",
        "WGPU_BACKEND",
    ] {
        if let Ok(value) = env::var(key)
            && !value.trim().is_empty()
        {
            logger::info!("Renderer env override: {key}={value}");
        }
    }
}

fn sanitized_cli_mode() -> String {
    let args: Vec<String> = env::args().collect();
    sanitized_cli_mode_for_args(&args)
}

/// Returns a redacted command-line mode summary for startup logging.
fn sanitized_cli_mode_for_args(args: &[impl AsRef<str>]) -> String {
    let has_queue = args
        .iter()
        .any(|arg| arg_has_ascii_suffix(arg.as_ref(), "queuename"));
    let has_queue_capacity = args
        .iter()
        .any(|arg| arg_has_ascii_suffix(arg.as_ref(), "queuecapacity"));
    let has_log_level = args
        .iter()
        .any(|arg| arg_has_ascii_suffix(arg.as_ref(), "loglevel"));
    let headless = args.iter().any(|arg| {
        let arg = arg.as_ref();
        arg.eq_ignore_ascii_case("--headless") || arg.eq_ignore_ascii_case("-headless")
    });
    let ignore_config = args.iter().any(|arg| {
        let arg = arg.as_ref();
        arg.eq_ignore_ascii_case("--ignore-config") || arg.eq_ignore_ascii_case("-ignore-config")
    });
    let attach_renderer = args
        .iter()
        .any(|arg| arg_has_ascii_suffix(arg.as_ref(), "attachrenderer"));

    let mut parts = Vec::new();
    if has_queue {
        parts.push("ipc-queue");
    }
    if has_queue_capacity {
        parts.push("queue-capacity");
    }
    if attach_renderer {
        parts.push("attach-renderer");
    }
    if headless {
        parts.push("headless");
    }
    if ignore_config {
        parts.push("ignore-config");
    }
    if has_log_level {
        parts.push("log-level");
    }
    if parts.is_empty() {
        "standalone".to_string()
    } else {
        parts.join("+")
    }
}

/// Returns true when `arg` ends with `suffix`, ignoring ASCII case.
fn arg_has_ascii_suffix(arg: &str, suffix: &str) -> bool {
    let arg = arg.as_bytes();
    let suffix = suffix.as_bytes();
    arg.len() >= suffix.len() && arg[arg.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanitized_cli_mode_empty_defaults_to_standalone_shape() {
        let mode = super::sanitized_cli_mode();
        assert!(!mode.contains("QueueName"));
    }

    #[test]
    fn sanitized_cli_mode_reports_attach_renderer() {
        assert_eq!(
            super::sanitized_cli_mode_for_args(&["renderide", "-AttachRenderer"]),
            "attach-renderer"
        );
        assert_eq!(
            super::sanitized_cli_mode_for_args(&["renderide", "--renderide-attachrenderer"]),
            "attach-renderer"
        );
    }

    #[test]
    fn sanitized_cli_mode_keeps_stable_redacted_order() {
        assert_eq!(
            super::sanitized_cli_mode_for_args(&[
                "renderide",
                "-QueueName",
                "secret",
                "-QueueCapacity",
                "8388608",
                "-AttachRenderer",
                "--headless",
                "-ignore-config",
                "-LogLevel",
                "debug",
            ]),
            "ipc-queue+queue-capacity+attach-renderer+headless+ignore-config+log-level"
        );
    }
}
