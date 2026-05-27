//! Process and heartbeat watchdog threads for the bootstrap sequence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::config::ResoBootConfig;
use crate::constants::{
    host_exit_watcher_poll_interval, renderer_exit_watcher_poll_interval, watchdog_poll_interval,
};
use crate::process_state::{ChildPoll, SharedChildSlot};

/// Watchdog thread handles kept alive for the duration of a bootstrap run.
pub(crate) struct WatchdogHandles {
    /// IPC heartbeat timeout watcher.
    _heartbeat: JoinHandle<()>,
    /// Host process exit watcher, disabled in Wine mode.
    _host_exit: Option<JoinHandle<()>>,
    /// Renderer process exit watcher.
    _renderer_exit: JoinHandle<()>,
}

/// Spawns heartbeat, optional Host exit, and renderer exit watchdogs.
pub(crate) fn spawn_watchdogs(
    config: &ResoBootConfig,
    cancel: Arc<AtomicBool>,
    heartbeat_deadline: Arc<Mutex<Instant>>,
    host_child: SharedChildSlot,
    renderer_child: SharedChildSlot,
    log_timestamp: String,
) -> WatchdogHandles {
    let heartbeat = spawn_heartbeat_watchdog(Arc::clone(&cancel), Arc::clone(&heartbeat_deadline));

    let host_exit = if config.is_wine {
        logger::info!("Wine mode: Host exit watcher disabled (child is shell wrapper)");
        None
    } else {
        logger::info!("Process watcher: cancel when Host process exits");
        Some(spawn_host_exit_watcher(
            host_child.clone(),
            Arc::clone(&cancel),
            log_timestamp,
        ))
    };

    let renderer_exit =
        spawn_renderer_exit_watcher(renderer_child, host_child, Arc::clone(&cancel));

    WatchdogHandles {
        _heartbeat: heartbeat,
        _host_exit: host_exit,
        _renderer_exit: renderer_exit,
    }
}

/// Thread: sets `cancel` when the IPC heartbeat deadline passes without refresh.
fn spawn_heartbeat_watchdog(
    cancel: Arc<AtomicBool>,
    heartbeat_deadline: Arc<Mutex<Instant>>,
) -> JoinHandle<()> {
    let cancel_wd = Arc::clone(&cancel);
    let deadline_wd = Arc::clone(&heartbeat_deadline);
    std::thread::spawn(move || {
        let start = Instant::now();
        while !cancel_wd.load(Ordering::Relaxed) {
            std::thread::sleep(watchdog_poll_interval());
            let Ok(deadline) = deadline_wd.lock() else {
                logger::error!(
                    "heartbeat watchdog: deadline mutex poisoned, terminating watchdog and signalling cancel"
                );
                cancel_wd.store(true, Ordering::SeqCst);
                break;
            };
            let now = Instant::now();
            if now > *deadline {
                cancel_wd.store(true, Ordering::SeqCst);
                logger::warn!(
                    "Bootstrapper messaging timeout: elapsed_s={:.3} overdue_ms={}",
                    start.elapsed().as_secs_f64(),
                    now.duration_since(*deadline).as_millis()
                );
                break;
            }
        }
    })
}

/// Thread: sets `cancel` when the Host child exits.
fn spawn_host_exit_watcher(
    host_child: SharedChildSlot,
    cancel: Arc<AtomicBool>,
    log_timestamp: String,
) -> JoinHandle<()> {
    let cancel_host = Arc::clone(&cancel);
    let host_out_name = format!("{log_timestamp}.log");
    std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            if cancel_host.load(Ordering::Relaxed) {
                break;
            }
            match host_child.poll_exit() {
                Ok(ChildPoll::Missing) => break,
                Ok(ChildPoll::Running) => std::thread::sleep(host_exit_watcher_poll_interval()),
                Ok(ChildPoll::Exited(status)) => {
                    let msg = format!(
                        "Host process exited after {:.3}s (exit code: {status}). Check logs/host/{host_out_name} for stdout/stderr.",
                        start.elapsed().as_secs_f64()
                    );
                    logger::info!("{msg}");
                    cancel_host.store(true, Ordering::SeqCst);
                    break;
                }
                Ok(ChildPoll::WaitError(e)) => {
                    logger::error!("Host process watcher try_wait error: {e}");
                    cancel_host.store(true, Ordering::SeqCst);
                    break;
                }
                Err(_) => {
                    logger::error!(
                        "host exit watcher: host_child mutex poisoned, terminating watchdog and signalling cancel"
                    );
                    cancel_host.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
    })
}

/// Thread: when a registered renderer child exits, terminates the Host and sets `cancel`.
fn spawn_renderer_exit_watcher(
    renderer_child: SharedChildSlot,
    host_child: SharedChildSlot,
    cancel: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            match renderer_child.poll_exit() {
                Ok(ChildPoll::Missing | ChildPoll::Running) => {}
                Ok(ChildPoll::Exited(status)) => {
                    logger::info!(
                        "Renderer process exited after {:.3}s ({status}); terminating Host and stopping bootstrapper",
                        start.elapsed().as_secs_f64()
                    );
                    cancel.store(true, Ordering::SeqCst);
                    terminate_host_after_renderer_exit(&host_child);
                    break;
                }
                Ok(ChildPoll::WaitError(e)) => {
                    logger::error!("Renderer exit watcher try_wait error: {e}");
                    cancel.store(true, Ordering::SeqCst);
                    break;
                }
                Err(_) => {
                    logger::error!(
                        "renderer exit watcher: renderer_child mutex poisoned, terminating watchdog and signalling cancel"
                    );
                    cancel.store(true, Ordering::SeqCst);
                    break;
                }
            }
            std::thread::sleep(renderer_exit_watcher_poll_interval());
        }
    })
}

/// Terminates the Host when the renderer exits first.
fn terminate_host_after_renderer_exit(host_child: &SharedChildSlot) {
    match host_child.take() {
        Ok(Some(mut host)) => {
            logger::info!("Terminating Host PID {} after renderer exit", host.id());
            let _ = host.kill();
            let _ = host.wait();
        }
        Ok(None) => {}
        Err(_) => {
            logger::error!(
                "renderer exit watcher: host_child mutex poisoned during teardown; \
                 could not kill Host (relying on lifetime group cleanup)"
            );
        }
    }
}
