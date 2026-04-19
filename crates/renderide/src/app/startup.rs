//! Process bootstrap before the winit loop: logging, panic hook, config, IPC connection, and
//! [`super::renderide_app::RenderideApp`] construction.
//!
//! # Process exit visibility (crashes, panics, graceful signals)
//!
//! Three separate mechanisms apply; they are **not** merged into one implementation because each has
//! different safety constraints:
//!
//! 1. **Fatal faults** (e.g. `SIGSEGV`, Windows SEH, macOS Mach exceptions): [`crate::fatal_crash_log::install`]
//!    registers [`crash_handler::CrashHandler`] so a short line is appended to the log file using only
//!    async-signal-safe writes. This does **not** run for normal termination signals like `SIGTERM`
//!    or `SIGINT`.
//! 2. **Panics**: [`std::panic::set_hook`] in [`run`] appends a panic report to the same log file
//!    (normal Rust context; mutex-based logging is allowed).
//! 3. **Graceful shutdown**: Unix [`signal_hook::iterator::Signals`] (or `SIGTERM` fallback) and
//!    Windows [`ctrlc`] set a flag; [`super::renderide_app::RenderideApp`] polls it and exits the
//!    winit loop. Per-signal logging happens on a side thread (Unix) or a handler thread (Windows),
//!    not inside the raw async-signal handler.
//!
//! **Manual verification** (Linux/macOS): `kill -TERM <pid>`, `kill -INT <pid>`, `kill -HUP <pid>`;
//! Ctrl+C in a terminal; confirm an `info` line in `logs/renderer/*.log` and clean exit. **Windows:**
//! Ctrl+C in a console. **Crash path:** e.g. `kill -BUS <pid>` should still append a fatal line via
//! `fatal_crash_log` (not the graceful path).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use logger::{LogComponent, LogLevel};
use winit::event_loop::EventLoop;

use crate::config::{load_renderer_settings, log_config_resolve_trace, settings_handle_from};
use crate::connection::{get_connection_parameters, try_claim_renderer_singleton};
use crate::frontend::InitState;
use crate::ipc::get_headless_params;
use crate::run_error::RunError;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererInitData};

use super::headless::run_headless;
use super::renderide_app::RenderideApp;

/// Interval between log flushes when using file logging in the winit handler.
pub(super) const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Max time to wait for [`RendererInitData`] after IPC connect before exiting with an error.
const IPC_INIT_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Cooperative exit flag for OS-driven shutdown (Unix signals or Windows `Ctrl+C`).
///
/// When [`Self::log_when_checked`] is `true`, [`crate::app::renderide_app::RenderideApp::check_external_shutdown`] emits a log
/// line when the flag is observed (used only when Unix registration falls back to `SIGTERM` without
/// a [`signal_hook::iterator`] side thread).
pub(crate) struct ExternalShutdownCoordinator {
    /// Set by the graceful-shutdown path; polled by the winit loop.
    pub(crate) requested: Arc<AtomicBool>,
    /// When `true`, emit one [`logger::info!`] when the loop first observes [`Self::requested`].
    pub(crate) log_when_checked: bool,
}

#[cfg(unix)]
fn shutdown_signal_display_name(sig: i32) -> &'static str {
    match sig {
        s if s == libc::SIGTERM => "SIGTERM",
        s if s == libc::SIGINT => "SIGINT",
        s if s == libc::SIGHUP => "SIGHUP",
        _ => "unknown",
    }
}

/// Registers `SIGTERM`, `SIGINT`, and `SIGHUP` via [`signal_hook::iterator::Signals`] (side thread logs
/// each delivery). Falls back to [`signal_hook::flag`] for `SIGTERM` only if iterator registration fails.
///
/// Linux child processes may receive `SIGTERM` when the parent bootstrapper exits (e.g. `PR_SET_PDEATHSIG`);
/// that path does not run the panic hook or fatal-crash logger.
#[cfg(unix)]
fn install_external_shutdown_unix() -> ExternalShutdownCoordinator {
    use signal_hook::iterator::Signals;

    let flag = Arc::new(AtomicBool::new(false));
    match Signals::new([libc::SIGTERM, libc::SIGINT, libc::SIGHUP]) {
        Ok(mut signals) => {
            let f = Arc::clone(&flag);
            match thread::Builder::new()
                .name("shutdown-signals".to_owned())
                .spawn(move || {
                    for sig in signals.forever() {
                        logger::info!(
                            "Received shutdown signal ({}); cooperative exit",
                            shutdown_signal_display_name(sig)
                        );
                        f.store(true, Ordering::Relaxed);
                    }
                }) {
                Ok(_join) => ExternalShutdownCoordinator {
                    requested: flag,
                    log_when_checked: false,
                },
                Err(e) => {
                    logger::error!("Failed to spawn shutdown-signals thread: {e}");
                    if let Err(e2) = signal_hook::flag::register(libc::SIGTERM, Arc::clone(&flag)) {
                        logger::warn!("Failed to register SIGTERM fallback: {e2}");
                    }
                    ExternalShutdownCoordinator {
                        requested: flag,
                        log_when_checked: true,
                    }
                }
            }
        }
        Err(e) => {
            logger::warn!(
                "Failed to register graceful shutdown signals ({e}); falling back to SIGTERM only"
            );
            if let Err(e2) = signal_hook::flag::register(libc::SIGTERM, Arc::clone(&flag)) {
                logger::warn!("Failed to register SIGTERM fallback: {e2}");
            }
            ExternalShutdownCoordinator {
                requested: flag,
                log_when_checked: true,
            }
        }
    }
}

/// Registers `Ctrl+C` so the same cooperative flag and exit path as Unix graceful signals are used.
#[cfg(windows)]
fn install_external_shutdown_windows() -> ExternalShutdownCoordinator {
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    match ctrlc::set_handler(move || {
        logger::info!("Received Ctrl+C (console control); cooperative exit");
        f.store(true, Ordering::Relaxed);
    }) {
        Ok(()) => ExternalShutdownCoordinator {
            requested: flag,
            log_when_checked: false,
        },
        Err(e) => {
            logger::warn!("Failed to register Ctrl+C handler: {e}");
            ExternalShutdownCoordinator {
                requested: flag,
                log_when_checked: false,
            }
        }
    }
}

/// Installs [`ExternalShutdownCoordinator`] when the platform supports it; otherwise [`None`].
pub(crate) fn install_external_shutdown() -> Option<ExternalShutdownCoordinator> {
    #[cfg(unix)]
    {
        Some(install_external_shutdown_unix())
    }
    #[cfg(windows)]
    {
        Some(install_external_shutdown_windows())
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

/// Chooses the process max log level after [`logger::init_for`].
///
/// Precedence: **`-LogLevel`** (if present) always wins. If absent, [`crate::config::DebugSettings::log_verbose`]
/// selects [`LogLevel::Trace`] when true and [`LogLevel::Debug`] when false.
pub(super) fn effective_renderer_log_level(cli: Option<LogLevel>, log_verbose: bool) -> LogLevel {
    if let Some(level) = cli {
        level
    } else if log_verbose {
        LogLevel::Trace
    } else {
        LogLevel::Debug
    }
}

/// Runs the winit event loop until exit or window close.
pub fn run() -> Result<Option<i32>, RunError> {
    try_claim_renderer_singleton()?;

    let timestamp = logger::log_filename_timestamp();
    let log_level_cli = logger::parse_log_level_from_args();
    let initial_log_level = log_level_cli.unwrap_or(LogLevel::Debug);
    let log_path = logger::init_for(LogComponent::Renderer, &timestamp, initial_log_level, false)?;

    logger::info!("Logging to {}", log_path.display());

    // Vulkan validation (spirv-val, VK_LAYER_KHRONOS_validation) and other native code often use
    // stdout and/or stderr; forward both so messages reach the log file regardless of wgpu flags
    // or `VK_INSTANCE_LAYERS` (see `native_stdio`).
    crate::native_stdio::ensure_stdio_forwarded_to_logger();

    // Fatal faults (signals / SEH / Mach exceptions) do not run the panic hook; register after
    // stdio forwarding so a duplicate of the preserved terminal fd exists when tee is enabled.
    crate::fatal_crash_log::install(&log_path);

    let config_load = load_renderer_settings();
    logger::set_max_level(effective_renderer_log_level(
        log_level_cli,
        config_load.settings.debug.log_verbose,
    ));
    log_config_resolve_trace(&config_load.resolve);
    let settings_handle = settings_handle_from(&config_load);
    let initial_vsync = config_load.settings.rendering.vsync;
    let initial_gpu_validation = config_load.settings.debug.gpu_validation_layers;

    let log_path_hook = log_path.clone();
    std::panic::set_hook(Box::new(move |info| {
        let report = logger::panic_report(info);
        logger::append_panic_report_to_file(&log_path_hook, &report);
        crate::native_stdio::try_write_preserved_stderr(report.as_bytes());
    }));

    let params = get_connection_parameters();
    let mut runtime = RendererRuntime::new(
        params.clone(),
        settings_handle,
        config_load.save_path.clone(),
    );
    runtime.set_suppress_renderer_config_disk_writes(config_load.suppress_config_disk_writes);
    if let Err(e) = runtime.connect_ipc() {
        if params.is_some() {
            logger::error!("IPC connect failed: {e}");
            return Err(e.into());
        }
    }

    if params.is_some() && runtime.is_ipc_connected() {
        logger::info!("IPC connected (Primary/Background)");
        wait_for_renderer_init_data(&mut runtime)?;
    } else if params.is_some() {
        logger::warn!("IPC params present but connection state unexpected");
    } else {
        logger::info!("Standalone mode (no -QueueName/-QueueCapacity; desktop GPU, no host init)");
    }

    let external_shutdown = install_external_shutdown();

    if let Some(headless_params) = get_headless_params() {
        return run_headless(
            &mut runtime,
            headless_params,
            external_shutdown,
            initial_gpu_validation,
        );
    }

    let event_loop = EventLoop::new().map_err(|e| {
        logger::error!("EventLoop::new failed: {e}");
        e
    })?;

    let mut app = RenderideApp::new(
        runtime,
        initial_vsync,
        initial_gpu_validation,
        log_level_cli,
        external_shutdown,
    );

    let _ = event_loop.run_app(&mut app);
    Ok(app.exit_code)
}

/// Blocks until [`RendererInitData`] arrives or IPC fails (non-standalone only).
fn wait_for_renderer_init_data(runtime: &mut RendererRuntime) -> Result<(), RunError> {
    let deadline = Instant::now() + IPC_INIT_WAIT_TIMEOUT;
    while runtime.init_state() == InitState::Uninitialized {
        if Instant::now() > deadline {
            logger::error!("Timed out waiting for RendererInitData from host");
            return Err(RunError::RendererInitDataTimeout);
        }
        runtime.poll_ipc();
        if runtime.fatal_error() {
            logger::error!("Fatal IPC error while waiting for RendererInitData");
            return Err(RunError::RendererInitDataFatalIpc);
        }
        thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

/// Standalone runs have no host init; IPC runs should have [`RendererInitData`] before the window exists.
pub(super) fn effective_output_device_for_gpu(
    pending: Option<&RendererInitData>,
) -> HeadOutputDevice {
    pending
        .map(|i| i.output_device)
        .unwrap_or(HeadOutputDevice::Screen)
}

pub(super) fn apply_window_title_from_init(
    window: &Arc<winit::window::Window>,
    init: &RendererInitData,
) {
    if let Some(ref title) = init.window_title {
        window.set_title(title);
    }
}

#[cfg(test)]
mod effective_log_level_tests {
    use super::effective_renderer_log_level;
    use logger::LogLevel;

    #[test]
    fn cli_always_overrides_log_verbose() {
        assert_eq!(
            effective_renderer_log_level(Some(LogLevel::Warn), true),
            LogLevel::Warn
        );
    }

    #[test]
    fn no_cli_uses_trace_when_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, true), LogLevel::Trace);
    }

    #[test]
    fn no_cli_uses_debug_when_not_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, false), LogLevel::Debug);
    }
}
