//! Full bootstrap sequence: IPC, Host spawn, watchdogs, queue loop, Wine cleanup.
//!
//! Shared-memory queue files use [`crate::ipc::interprocess_backing_dir`] unless overridden; see
//! [`crate::ipc::RENDERIDE_INTERPROCESS_DIR_ENV`].

use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::BootstrapError;
use crate::child_lifetime::ChildLifetimeGroup;
use crate::cleanup;
use crate::config::ResoBootConfig;
use crate::constants::{
    host_exit_watcher_poll_interval, initial_heartbeat_timeout,
    renderer_exit_watcher_poll_interval, watchdog_poll_interval,
};
use crate::host;
use crate::ipc::{BootstrapQueues, bootstrap_queue_base_names};
use crate::protocol;

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
pub fn assemble_host_args(mut host_args: Vec<String>, shared_memory_prefix: &str) -> Vec<String> {
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

/// Spawns the heartbeat watchdog, optional Host exit watcher (non-Wine), and a renderer exit watcher
/// once [`crate::protocol_handlers::handle_start_renderer`] registers a [`Child`].
///
/// When the **renderer** exits first (e.g. user closes the window), the renderer watcher sets
/// `cancel`, terminates the **Host** [`Child`], and the queue loop ends so the bootstrapper process
/// exits--analogous to the engine-side watchdog that stops the session when the GPU process dies.
fn spawn_watchdogs(
    config: &ResoBootConfig,
    cancel: Arc<AtomicBool>,
    heartbeat_deadline: Arc<Mutex<Instant>>,
    host_child: Arc<Mutex<Option<Child>>>,
    renderer_child: Arc<Mutex<Option<Child>>>,
    log_timestamp: String,
) -> (JoinHandle<()>, Option<JoinHandle<()>>, JoinHandle<()>) {
    let heartbeat = spawn_heartbeat_watchdog(Arc::clone(&cancel), Arc::clone(&heartbeat_deadline));

    let host_exit = if config.is_wine {
        logger::info!("Wine mode: Host exit watcher disabled (child is shell wrapper)");
        None
    } else {
        logger::info!("Process watcher: cancel when Host process exits");
        Some(spawn_host_exit_watcher(
            Arc::clone(&host_child),
            Arc::clone(&cancel),
            log_timestamp,
        ))
    };

    let renderer_exit =
        spawn_renderer_exit_watcher(renderer_child, host_child, Arc::clone(&cancel));

    (heartbeat, host_exit, renderer_exit)
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

    let host_process = start_host_with_drainers(config, &args, &lifetime, &log_timestamp)
        .map_err(BootstrapError::Io)?;

    let host_child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(Some(host_process)));
    let renderer_child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));

    let cancel = Arc::new(AtomicBool::new(false));

    #[cfg(target_os = "macos")]
    install_macos_signal_handler(&cancel);

    let heartbeat_deadline = Arc::new(Mutex::new(Instant::now() + initial_heartbeat_timeout()));
    let (_heartbeat_watchdog, _host_exit_watcher, _renderer_exit_watcher) = spawn_watchdogs(
        config,
        Arc::clone(&cancel),
        Arc::clone(&heartbeat_deadline),
        Arc::clone(&host_child),
        Arc::clone(&renderer_child),
        log_timestamp,
    );

    protocol::queue_loop(
        &mut queues.incoming,
        &mut queues.outgoing,
        config,
        &cancel,
        &lifetime,
        &heartbeat_deadline,
        &renderer_child,
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
                logger::error!(
                    "heartbeat watchdog: deadline mutex poisoned, terminating watchdog and signalling cancel"
                );
                cancel_wd.store(true, Ordering::SeqCst);
                break;
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
///
/// [`Child`] is stored in `host_child` so [`spawn_renderer_exit_watcher`] can [`Child::kill`] the
/// Host when the renderer exits first.
fn spawn_host_exit_watcher(
    host_child: Arc<Mutex<Option<Child>>>,
    cancel: Arc<AtomicBool>,
    log_timestamp: String,
) -> JoinHandle<()> {
    let cancel_host = Arc::clone(&cancel);
    let host_out_name = format!("{log_timestamp}.log");
    std::thread::spawn(move || {
        loop {
            if cancel_host.load(Ordering::Relaxed) {
                break;
            }
            let outcome = {
                let Ok(mut guard) = host_child.lock() else {
                    logger::error!(
                        "host exit watcher: host_child mutex poisoned, terminating watchdog and signalling cancel"
                    );
                    cancel_host.store(true, Ordering::SeqCst);
                    break;
                };
                match guard.as_mut() {
                    None => break,
                    Some(child) => match child.try_wait() {
                        Ok(Some(status)) => Ok(Some(status)),
                        Ok(None) => Ok(None),
                        Err(e) => Err(e),
                    },
                }
            };
            match outcome {
                Ok(Some(status)) => {
                    let msg = format!(
                        "Host process exited (exit code: {status}). Check logs/host/{host_out_name} for stdout/stderr."
                    );
                    logger::info!("{msg}");
                    cancel_host.store(true, Ordering::SeqCst);
                    break;
                }
                Ok(None) => std::thread::sleep(host_exit_watcher_poll_interval()),
                Err(e) => {
                    logger::error!("Host process watcher try_wait error: {e}");
                    cancel_host.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
    })
}

/// Thread: when a registered renderer [`Child`] exits, terminates the Host and sets `cancel`.
fn spawn_renderer_exit_watcher(
    renderer_child: Arc<Mutex<Option<Child>>>,
    host_child: Arc<Mutex<Option<Child>>>,
    cancel: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let mut exited: Option<std::process::ExitStatus> = None;
            {
                let Ok(mut guard) = renderer_child.lock() else {
                    logger::error!(
                        "renderer exit watcher: renderer_child mutex poisoned, terminating watchdog and signalling cancel"
                    );
                    cancel.store(true, Ordering::SeqCst);
                    break;
                };
                if let Some(r) = guard.as_mut() {
                    match r.try_wait() {
                        Ok(Some(st)) => exited = Some(st),
                        Ok(None) => {}
                        Err(e) => {
                            logger::error!("Renderer exit watcher try_wait error: {e}");
                            cancel.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            }
            if let Some(status) = exited {
                logger::info!(
                    "Renderer process exited ({status}); terminating Host and stopping bootstrapper"
                );
                cancel.store(true, Ordering::SeqCst);
                match host_child.lock() {
                    Ok(mut h) => {
                        if let Some(mut hc) = h.take() {
                            logger::info!("Terminating Host PID {} after renderer exit", hc.id());
                            let _ = hc.kill();
                            let _ = hc.wait();
                        }
                    }
                    Err(_) => {
                        logger::error!(
                            "renderer exit watcher: host_child mutex poisoned during teardown; \
                         could not kill Host (relying on lifetime group cleanup)"
                        );
                    }
                }
                break;
            }
            std::thread::sleep(renderer_exit_watcher_poll_interval());
        }
    })
}

#[cfg(test)]
mod assemble_host_args_tests {
    use super::assemble_host_args;

    #[test]
    fn empty_argv_appends_shmprefix_and_prefix() {
        let out = assemble_host_args(vec![], "Ab12");
        assert_eq!(out, vec!["-shmprefix".to_string(), "Ab12".to_string()]);
    }

    #[test]
    fn preserves_order_and_appends_suffix() {
        let out = assemble_host_args(
            vec![
                "-Invisible".to_string(),
                "-Data".to_string(),
                "path".to_string(),
            ],
            "Z9",
        );
        assert_eq!(
            out,
            vec![
                "-Invisible".to_string(),
                "-Data".to_string(),
                "path".to_string(),
                "-shmprefix".to_string(),
                "Z9".to_string(),
            ]
        );
    }

    #[test]
    fn ends_with_shmprefix_then_prefix() {
        let prefix = "prefX";
        let out = assemble_host_args(vec!["a".into(), "b".into()], prefix);
        assert!(out.len() >= 2);
        assert_eq!(out[out.len() - 2], "-shmprefix");
        assert_eq!(out[out.len() - 1], prefix);
    }
}
