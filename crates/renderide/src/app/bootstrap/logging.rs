//! File logging, native stdio forwarding, and renderer-startup banner.

use std::env;
use std::path::Path;

use logger::{LogComponent, LogLevel};
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System,
};

use crate::build_info::{renderer_commit_sha8_label, renderer_commit_source, renderer_identifier};
use crate::crash_context::{self, TickPhase};
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
    let bootstrap_log_level = log_level_cli.unwrap_or(LogLevel::Debug);
    let log_path = logger::init_for(
        LogComponent::Renderer,
        &timestamp,
        bootstrap_log_level,
        false,
    )
    .map_err(RunError::logging_init)?;

    logger::info!("Logging to {}", log_path.display());
    log_renderer_startup_context(&log_path);

    crate::native_stdio::ensure_stdio_forwarded_to_logger();
    crate::fatal_crash_log::install(&log_path);
    super::panic::install_panic_hook(&log_path);

    Ok(LoggingBootstrap { log_level_cli })
}

fn log_renderer_startup_context(log_path: &Path) {
    logger::info!(
        "Renderer process: identifier={} version={} commit_sha8={} commit_source={} pid={} target={} {} arch={} exe={} cwd={} cli_mode={} log_path={}",
        renderer_identifier(),
        env!("CARGO_PKG_VERSION"),
        renderer_commit_sha8_label(),
        renderer_commit_source(),
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
    );
    logger::info!("{}", StartupSystemDiagnostics::capture().log_line());
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct StartupSystemDiagnostics {
    os: String,
    kernel: String,
    cpu_model: String,
    logical_cpus: usize,
    physical_cpus: Option<usize>,
    total_ram_bytes: u64,
    process_ram_bytes: Option<u64>,
}

impl StartupSystemDiagnostics {
    fn capture() -> Self {
        if !sysinfo::IS_SUPPORTED_SYSTEM {
            return Self {
                os: "unsupported platform".to_string(),
                kernel: unavailable().to_string(),
                cpu_model: unavailable().to_string(),
                logical_cpus: 0,
                physical_cpus: None,
                total_ram_bytes: 0,
                process_ram_bytes: None,
            };
        }

        let mut system = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        system.refresh_memory();
        let process_ram_bytes = sysinfo::get_current_pid().ok().and_then(|pid| {
            system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                true,
                ProcessRefreshKind::nothing().with_memory(),
            );
            system.process(pid).map(sysinfo::Process::memory)
        });

        Self {
            os: System::long_os_version()
                .or_else(System::name)
                .unwrap_or_else(|| unavailable().to_string()),
            kernel: System::kernel_long_version(),
            cpu_model: system
                .cpus()
                .first()
                .map(|cpu| cpu.brand().trim().to_string())
                .filter(|brand| !brand.is_empty())
                .unwrap_or_else(|| unavailable().to_string()),
            logical_cpus: system.cpus().len(),
            physical_cpus: System::physical_core_count(),
            total_ram_bytes: system.total_memory(),
            process_ram_bytes,
        }
    }

    fn log_line(&self) -> String {
        format!(
            "Renderer host system: os={} kernel={} cpu={} logical_cpus={} physical_cpus={} ram_total={} process_ram={}",
            self.os,
            self.kernel,
            self.cpu_model,
            self.logical_cpus,
            optional_usize(self.physical_cpus),
            format_bytes(self.total_ram_bytes),
            self.process_ram_bytes
                .map(format_bytes)
                .unwrap_or_else(|| unavailable().to_string()),
        )
    }
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| unavailable().to_string())
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f64 = bytes as f64;
    if bytes_f64 >= GIB {
        format!("{bytes} bytes ({:.2} GiB)", bytes_f64 / GIB)
    } else if bytes_f64 >= MIB {
        format!("{bytes} bytes ({:.2} MiB)", bytes_f64 / MIB)
    } else if bytes_f64 >= KIB {
        format!("{bytes} bytes ({:.2} KiB)", bytes_f64 / KIB)
    } else {
        format!("{bytes} bytes")
    }
}

fn unavailable() -> &'static str {
    "<unavailable>"
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
    use super::StartupSystemDiagnostics;

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

    #[test]
    fn system_diagnostics_log_line_formats_missing_values() {
        let diagnostics = StartupSystemDiagnostics {
            os: "TestOS 1.0".to_string(),
            kernel: "TestKernel 9.9".to_string(),
            cpu_model: "<unavailable>".to_string(),
            logical_cpus: 8,
            physical_cpus: None,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            process_ram_bytes: None,
        };

        assert_eq!(
            diagnostics.log_line(),
            "Renderer host system: os=TestOS 1.0 kernel=TestKernel 9.9 cpu=<unavailable> logical_cpus=8 physical_cpus=<unavailable> ram_total=17179869184 bytes (16.00 GiB) process_ram=<unavailable>"
        );
    }

    #[test]
    fn format_bytes_uses_binary_units() {
        assert_eq!(super::format_bytes(512), "512 bytes");
        assert_eq!(super::format_bytes(1536), "1536 bytes (1.50 KiB)");
        assert_eq!(
            super::format_bytes(2 * 1024 * 1024),
            "2097152 bytes (2.00 MiB)"
        );
    }
}
