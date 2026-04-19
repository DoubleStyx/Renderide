//! Full bootstrap sequence: IPC, Host spawn, watchdogs, queue loop, Wine cleanup.
//!
//! Shared-memory queue files use [`crate::ipc::interprocess_backing_dir`] unless overridden; see
//! [`crate::ipc::RENDERIDE_INTERPROCESS_DIR_ENV`].

use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::cleanup;
use crate::config::ResoBootConfig;
use crate::constants::{
    host_exit_watcher_poll_interval, initial_heartbeat_timeout, watchdog_poll_interval,
};
use crate::host;
use crate::ipc::{bootstrap_queue_base_names, BootstrapQueues};
use crate::protocol;
use crate::BootstrapError;

/// Paths and argv for a single bootstrap run (owned so a panic boundary can move it).
pub struct RunContext {
    /// Extra Host CLI args (before `-Invisible` / `-shmprefix` are appended).
    pub host_args: Vec<String>,
    /// Shared basename (no `.log`) for paths like `logs/host/{timestamp}.log` under [`logger::logs_root`].
    pub log_timestamp: String,
}

/// Logs Resonite / interprocess paths and queue names at bootstrap start.
fn log_run_intro(config: &ResoBootConfig) {
    if let Some(ref level) = config.renderide_log_level {
        logger::info!("Renderide log level: {}", level.as_arg());
    }

    logger::info!("Bootstrapper start");
    logger::info!("Shared memory prefix: {}", config.shared_memory_prefix);
    let backing = crate::ipc::interprocess_backing_dir();
    logger::info!(
        "Interprocess queue backing directory: {:?} (set {} to override; Host must match)",
        backing,
        crate::ipc::RENDERIDE_INTERPROCESS_DIR_ENV
    );
}

/// Appends `-shmprefix` and the generated prefix to Host argv.
fn assemble_host_args(mut host_args: Vec<String>, shared_memory_prefix: &str) -> Vec<String> {
    host_args.push("-shmprefix".to_string());
    host_args.push(shared_memory_prefix.to_string());
    host_args
}

/// Spawns the Host, raises priority, and starts stdout/stderr drainers into the host log file.
fn start_host_with_drainers(
    config: &ResoBootConfig,
    args: &[String],
    lifetime: &ChildLifetimeGroup,
    log_timestamp: &str,
) -> Result<Child, std::io::Error> {
    let mut child = host::spawn_host(config, args, lifetime)?;
    logger::info!("Process started. Id: {}", child.id());

    host::set_host_above_normal_priority(&child);

    logger::ensure_log_dir(logger::LogComponent::Host)?;
    let host_log_path = logger::log_file_path(logger::LogComponent::Host, log_timestamp);

    if let Some(stdout) = child.stdout.take() {
        host::spawn_output_drainer(host_log_path.clone(), stdout, "[Host stdout]");
    }
    if let Some(stderr) = child.stderr.take() {
        host::spawn_output_drainer(host_log_path, stderr, "[Host stderr]");
    }

    Ok(child)
}

/// Installs Ctrl+C handler on macOS to set `cancel`.
#[cfg(target_os = "macos")]
fn install_macos_signal_handler(cancel: &Arc<AtomicBool>) {
    let c = Arc::clone(cancel);
    if let Err(e) = ctrlc::set_handler(move || {
        c.store(true, Ordering::SeqCst);
    }) {
        logger::warn!("macOS: could not install ctrlc (SIGINT/SIGTERM) handler: {e}");
    }
}

/// Spawns the heartbeat watchdog; optionally spawns Host exit watcher when not under Wine.
fn spawn_watchdogs(
    config: &ResoBootConfig,
    cancel: Arc<AtomicBool>,
    heartbeat_deadline: Arc<Mutex<Instant>>,
    child: Child,
    log_timestamp: String,
) -> (JoinHandle<()>, Option<JoinHandle<()>>) {
    let heartbeat = spawn_heartbeat_watchdog(Arc::clone(&cancel), Arc::clone(&heartbeat_deadline));

    let host_exit = if !config.is_wine {
        logger::info!("Process watcher: cancel when Host process exits");
        Some(spawn_host_exit_watcher(
            child,
            Arc::clone(&cancel),
            log_timestamp,
        ))
    } else {
        logger::info!("Wine mode: Host exit watcher disabled (child is shell wrapper)");
        None
    };

    (heartbeat, host_exit)
}

/// macOS child teardown, Wine queue cleanup, and final log line.
fn finalize(config: &ResoBootConfig, lifetime: &ChildLifetimeGroup) {
    #[cfg(target_os = "macos")]
    lifetime.shutdown_tracked_children();
    #[cfg(not(target_os = "macos"))]
    let _ = lifetime;

    if config.is_wine {
        cleanup::remove_wine_queue_backing_files(&config.shared_memory_prefix);
    }

    logger::info!("Bootstrapper end");
}

/// Runs the bootstrapper main loop after logging is initialized.
pub fn run(config: &ResoBootConfig, ctx: RunContext) -> Result<(), BootstrapError> {
    log_run_intro(config);

    let lifetime = ChildLifetimeGroup::new().map_err(BootstrapError::Io)?;
    let mut queues = BootstrapQueues::open(&config.shared_memory_prefix)?;

    let (incoming_name, outgoing_name) = bootstrap_queue_base_names(&config.shared_memory_prefix);
    logger::info!(
        "Queues: incoming={incoming_name} outgoing={outgoing_name} (capacity {})",
        crate::ipc::BOOTSTRAP_QUEUE_CAPACITY
    );

    let RunContext {
        host_args,
        log_timestamp,
    } = ctx;

    let args = assemble_host_args(host_args, &config.shared_memory_prefix);
    logger::info!("Host args: {:?}", args);

    let child = start_host_with_drainers(config, &args, &lifetime, &log_timestamp)
        .map_err(BootstrapError::Io)?;

    let cancel = Arc::new(AtomicBool::new(false));

    #[cfg(target_os = "macos")]
    install_macos_signal_handler(&cancel);

    let heartbeat_deadline = Arc::new(Mutex::new(Instant::now() + initial_heartbeat_timeout()));
    let (_heartbeat_watchdog, _host_exit_watcher) = spawn_watchdogs(
        config,
        Arc::clone(&cancel),
        Arc::clone(&heartbeat_deadline),
        child,
        log_timestamp,
    );

    protocol::queue_loop(
        &mut queues.incoming,
        &mut queues.outgoing,
        config,
        &cancel,
        &lifetime,
        &heartbeat_deadline,
    );

    finalize(config, &lifetime);
    Ok(())
}

/// Thread: sets `cancel` when the IPC heartbeat deadline passes without refresh.
fn spawn_heartbeat_watchdog(
    cancel: Arc<AtomicBool>,
    heartbeat_deadline: Arc<Mutex<Instant>>,
) -> JoinHandle<()> {
    let cancel_wd = Arc::clone(&cancel);
    let deadline_wd = Arc::clone(&heartbeat_deadline);
    std::thread::spawn(move || {
        while !cancel_wd.load(Ordering::Relaxed) {
            std::thread::sleep(watchdog_poll_interval());
            let Ok(deadline) = deadline_wd.lock() else {
                continue;
            };
            if Instant::now() > *deadline {
                cancel_wd.store(true, Ordering::SeqCst);
                logger::info!("Bootstrapper messaging timeout!");
                break;
            }
        }
    })
}

/// Thread: sets `cancel` when the Host child exits (not used under Wine).
fn spawn_host_exit_watcher(
    mut child: Child,
    cancel: Arc<AtomicBool>,
    log_timestamp: String,
) -> JoinHandle<()> {
    let cancel_host = Arc::clone(&cancel);
    let host_out_name = format!("{log_timestamp}.log");
    std::thread::spawn(move || {
        let exit_status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {}
                Err(e) => {
                    logger::error!("Process watcher try_wait error: {}", e);
                    break None;
                }
            }
            std::thread::sleep(host_exit_watcher_poll_interval());
        };
        let exit_info = exit_status
            .as_ref()
            .map(|s| format!(" (exit code: {s})"))
            .unwrap_or_default();
        let msg = format!(
            "Host process exited{exit_info}. Check logs/host/{host_out_name} for stdout/stderr."
        );
        logger::info!("{msg}");
        cancel_host.store(true, Ordering::SeqCst);
    })
}
