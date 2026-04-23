//! Per-variant handling for [`crate::protocol::HostCommand`] messages from the Host.

use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use interprocess::Publisher;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::constants::heartbeat_refresh_timeout;
use crate::protocol::{HostCommand, LoopAction};
use crate::renderer_stub;

/// Extends the IPC watchdog deadline and logs receipt.
pub(crate) fn handle_heartbeat(heartbeat_deadline: &Arc<Mutex<Instant>>) -> LoopAction {
    if let Ok(mut d) = heartbeat_deadline.lock() {
        *d = Instant::now() + heartbeat_refresh_timeout();
    }
    logger::info!("Got heartbeat.");
    LoopAction::Continue
}

/// Acknowledges shutdown; the queue loop sets `cancel` when this returns [`LoopAction::Break`].
pub(crate) fn handle_shutdown() -> LoopAction {
    logger::info!("Got shutdown command");
    LoopAction::Break
}

/// Reads the system clipboard and enqueues UTF-8 bytes (empty string on failure).
pub(crate) fn handle_get_text(outgoing: &mut Publisher) -> LoopAction {
    logger::info!("Getting clipboard text");
    let text = arboard::Clipboard::new()
        .and_then(|mut c| c.get_text())
        .unwrap_or_default();
    let _ = outgoing.try_enqueue(text.as_bytes());
    LoopAction::Continue
}

/// Writes UTF-8 text to the system clipboard (best-effort).
pub(crate) fn handle_set_text(text: &str) -> LoopAction {
    logger::info!("Setting clipboard text");
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        let _ = clipboard.set_text(text);
    }
    LoopAction::Continue
}

/// Spawns the renderer with optional `-LogLevel`, registers it for lifetime management, stores the
/// [`Child`] in `renderer_child` for [`crate::orchestration::spawn_renderer_exit_watcher`], and
/// enqueues `RENDERITE_STARTED:{pid}`.
///
/// If a renderer was already registered (restart), the previous process is killed and reaped first.
pub(crate) fn handle_start_renderer(
    renderer_args: &[String],
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
    renderer_child: &Arc<Mutex<Option<Child>>>,
) -> LoopAction {
    let mut args: Vec<String> = renderer_args.to_vec();
    if let Some(ref level) = config.renderide_log_level {
        args.push("-LogLevel".to_string());
        args.push(level.as_arg().to_string());
    }
    let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();

    renderer_stub::ensure_link(config);

    logger::info!(
        "Spawning renderer: {:?} with args: {:?}",
        config.renderite_executable,
        args
    );
    let mut renderer_cmd = Command::new(&config.renderite_executable);
    renderer_cmd
        .args(&args_refs)
        .current_dir(&config.renderite_directory);
    lifetime.prepare_command(&mut renderer_cmd);
    match renderer_cmd.spawn() {
        Ok(mut process) => {
            if let Err(e) = lifetime.register_spawned(&process) {
                logger::error!("Renderer started but could not join lifetime group: {}", e);
                let _ = process.kill();
                let _ = process.wait();
            } else {
                let pid = process.id();
                if let Ok(mut slot) = renderer_child.lock() {
                    if let Some(mut old) = slot.take() {
                        logger::info!(
                            "Replacing previous renderer PID {} with new process",
                            old.id()
                        );
                        let _ = old.kill();
                        let _ = old.wait();
                    }
                    *slot = Some(process);
                }
                logger::info!("Renderer started PID {} with args: {}", pid, args.join(" "));
                let response = format!("RENDERITE_STARTED:{pid}");
                let _ = outgoing.try_enqueue(response.as_bytes());
            }
        }
        Err(e) => {
            logger::error!("Failed to start renderer: {}", e);
        }
    }
    LoopAction::Continue
}

/// Dispatches one parsed [`HostCommand`].
pub(crate) fn dispatch_command(
    cmd: HostCommand,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
    heartbeat_deadline: &Arc<Mutex<Instant>>,
    renderer_child: &Arc<Mutex<Option<Child>>>,
) -> LoopAction {
    match cmd {
        HostCommand::Heartbeat => handle_heartbeat(heartbeat_deadline),
        HostCommand::Shutdown => handle_shutdown(),
        HostCommand::GetText => handle_get_text(outgoing),
        HostCommand::SetText(text) => handle_set_text(&text),
        HostCommand::StartRenderer(args) => {
            handle_start_renderer(&args, outgoing, config, lifetime, renderer_child)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Child;
    use std::time::Duration;

    use interprocess::{Publisher, QueueOptions, Subscriber};
    use logger::LogLevel;

    use super::*;
    use crate::constants::heartbeat_refresh_timeout;

    fn renderer_slot() -> Arc<Mutex<Option<Child>>> {
        Arc::new(Mutex::new(None))
    }

    fn sample_config(exe: PathBuf, dir: PathBuf) -> ResoBootConfig {
        ResoBootConfig {
            current_directory: dir.clone(),
            runtime_config: dir.join("Renderite.Host.runtimeconfig.json"),
            renderite_directory: dir.clone(),
            renderite_executable: exe,
            shared_memory_prefix: "test".into(),
            is_wine: false,
            renderide_log_level: None,
        }
    }

    fn make_publisher_subscriber(dir: &std::path::Path) -> (Publisher, Subscriber) {
        let name = format!("ph_{}", std::process::id());
        let opts = QueueOptions::with_path_and_destroy(&name, dir, 4096, true).expect("opts");
        let publisher = Publisher::new(opts.clone()).expect("publisher");
        let subscriber = Subscriber::new(opts).expect("subscriber");
        (publisher, subscriber)
    }

    /// `true` is at `/usr/bin/true` on macOS; Linux typically has `/bin/true`. Fall back to `PATH`.
    #[cfg(unix)]
    fn unix_noop_executable() -> PathBuf {
        use std::path::Path;
        for candidate in ["/usr/bin/true", "/bin/true"] {
            if Path::new(candidate).exists() {
                return PathBuf::from(candidate);
            }
        }
        PathBuf::from("true")
    }

    #[test]
    fn heartbeat_advances_deadline() {
        let deadline = Arc::new(Mutex::new(Instant::now()));
        let before = *deadline.lock().expect("lock");
        std::thread::sleep(Duration::from_millis(20));
        handle_heartbeat(&deadline);
        let after = *deadline.lock().expect("lock");
        assert!(after > before);
        let cap = Instant::now() + heartbeat_refresh_timeout() + Duration::from_millis(500);
        assert!(after <= cap);
    }

    #[test]
    fn shutdown_returns_break() {
        assert_eq!(handle_shutdown(), LoopAction::Break);
    }

    #[test]
    fn get_text_does_not_panic() {
        let dir = std::env::temp_dir().join(format!("bootstrapper_ph_gt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (mut publisher, _) = make_publisher_subscriber(&dir);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle_get_text(&mut publisher);
        }));
        assert!(r.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_text_does_not_panic() {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle_set_text("hello");
        }));
        assert!(r.is_ok());
    }

    #[test]
    fn start_renderer_missing_executable_continues() {
        let dir = std::env::temp_dir().join(format!("bootstrapper_ph_se_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("resonite");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = sample_config(tmp.join("definitely_missing_exe_12345"), tmp);
        let lifetime = ChildLifetimeGroup::new().expect("lifetime");
        let (mut publisher, _) = make_publisher_subscriber(&dir);
        let slot = renderer_slot();
        assert_eq!(
            handle_start_renderer(&[], &mut publisher, &cfg, &lifetime, &slot),
            LoopAction::Continue
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn start_renderer_bin_true_enqueues_started() {
        let dir = std::env::temp_dir().join(format!("bootstrapper_ph_true_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("game");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = sample_config(unix_noop_executable(), tmp);
        let lifetime = ChildLifetimeGroup::new().expect("lifetime");
        let (mut publisher, mut subscriber) = make_publisher_subscriber(&dir);
        let slot = renderer_slot();
        assert_eq!(
            handle_start_renderer(&[], &mut publisher, &cfg, &lifetime, &slot),
            LoopAction::Continue
        );
        for _ in 0..50 {
            if let Some(body) = subscriber.try_dequeue() {
                let s = String::from_utf8(body).expect("utf8");
                assert!(
                    s.starts_with("RENDERITE_STARTED:"),
                    "unexpected message: {s}"
                );
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("expected RENDERITE_STARTED on queue");
    }

    #[test]
    fn dispatch_forwards_to_handlers() {
        let dir = std::env::temp_dir().join(format!("bootstrapper_ph_disp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("g");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = sample_config(tmp.join("missing"), tmp);
        let lifetime = ChildLifetimeGroup::new().expect("lifetime");
        let (mut publisher, _) = make_publisher_subscriber(&dir);
        let deadline = Arc::new(Mutex::new(Instant::now()));
        let slot = renderer_slot();
        assert_eq!(
            dispatch_command(
                HostCommand::Heartbeat,
                &mut publisher,
                &cfg,
                &lifetime,
                &deadline,
                &slot
            ),
            LoopAction::Continue
        );
        assert_eq!(
            dispatch_command(
                HostCommand::Shutdown,
                &mut publisher,
                &cfg,
                &lifetime,
                &deadline,
                &slot
            ),
            LoopAction::Break
        );
        assert_eq!(
            dispatch_command(
                HostCommand::SetText("clip".into()),
                &mut publisher,
                &cfg,
                &lifetime,
                &deadline,
                &slot
            ),
            LoopAction::Continue
        );
        assert_eq!(
            dispatch_command(
                HostCommand::GetText,
                &mut publisher,
                &cfg,
                &lifetime,
                &deadline,
                &slot
            ),
            LoopAction::Continue
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn start_renderer_appends_log_level() {
        let dir = std::env::temp_dir().join(format!("bootstrapper_ph_ll_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("g2");
        std::fs::create_dir_all(&tmp).unwrap();
        let mut cfg = sample_config(tmp.join("missing2"), tmp);
        cfg.renderide_log_level = Some(LogLevel::Warn);
        let lifetime = ChildLifetimeGroup::new().expect("lifetime");
        let (mut publisher, _) = make_publisher_subscriber(&dir);
        let slot = renderer_slot();
        assert_eq!(
            handle_start_renderer(&[], &mut publisher, &cfg, &lifetime, &slot),
            LoopAction::Continue
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
