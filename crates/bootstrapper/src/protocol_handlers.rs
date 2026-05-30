//! Per-variant handling for [`crate::protocol::HostCommand`] messages from the Host.

mod clipboard;
mod renderer;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use interprocess::Publisher;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::constants::heartbeat_refresh_timeout;
use crate::process_state::SharedChildSlot;
use crate::protocol::{HostCommand, LoopAction};

/// Extends the IPC watchdog deadline and logs receipt.
pub(super) fn handle_heartbeat(heartbeat_deadline: &Arc<Mutex<Instant>>) -> LoopAction {
    if let Ok(mut d) = heartbeat_deadline.lock() {
        *d = Instant::now() + heartbeat_refresh_timeout();
    }
    logger::debug!("Got heartbeat.");
    LoopAction::Continue
}

/// Acknowledges shutdown; the queue loop sets `cancel` when this returns [`LoopAction::Break`].
pub(super) fn handle_shutdown() -> LoopAction {
    logger::info!("Got shutdown command");
    LoopAction::Break
}

/// Dispatches one parsed [`HostCommand`].
pub(crate) fn dispatch_command(
    cmd: HostCommand,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
    heartbeat_deadline: &Arc<Mutex<Instant>>,
    renderer_child: &SharedChildSlot,
) -> LoopAction {
    match cmd {
        HostCommand::Heartbeat => handle_heartbeat(heartbeat_deadline),
        HostCommand::Shutdown => handle_shutdown(),
        HostCommand::GetText => clipboard::handle_get_text(outgoing),
        HostCommand::SetText(text) => clipboard::handle_set_text(&text),
        HostCommand::StartRenderer(args) => {
            renderer::handle_start_renderer(&args, outgoing, config, lifetime, renderer_child)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use interprocess::{Publisher, QueueOptions, Subscriber};
    use logger::LogLevel;

    use super::*;
    use crate::constants::heartbeat_refresh_timeout;

    fn renderer_slot() -> SharedChildSlot {
        SharedChildSlot::empty()
    }

    fn sample_config(exe: PathBuf, dir: PathBuf) -> ResoBootConfig {
        ResoBootConfig {
            current_directory: dir.clone(),
            runtime_config: dir.join("Renderite.Host.runtimeconfig.json"),
            renderite_directory: dir.clone(),
            renderite_executable: exe,
            resonite_dir: None,
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
            renderer::handle_start_renderer(&[], &mut publisher, &cfg, &lifetime, &slot),
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
            renderer::handle_start_renderer(&[], &mut publisher, &cfg, &lifetime, &slot),
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
            renderer::handle_start_renderer(&[], &mut publisher, &cfg, &lifetime, &slot),
            LoopAction::Continue
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_text_returns_continue() {
        assert_eq!(clipboard::handle_set_text("hello"), LoopAction::Continue);
        assert_eq!(clipboard::handle_set_text(""), LoopAction::Continue);
    }

    #[test]
    fn get_text_enqueues_response_and_continues() {
        let dir = std::env::temp_dir().join(format!("bootstrapper_ph_gt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (mut publisher, mut subscriber) = make_publisher_subscriber(&dir);
        assert_eq!(
            clipboard::handle_get_text(&mut publisher),
            LoopAction::Continue
        );
        let mut received = false;
        for _ in 0..50 {
            if subscriber.try_dequeue().is_some() {
                received = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(received, "expected GETTEXT response on outgoing queue");
    }

    #[test]
    fn dispatch_forwards_gettext_settext_and_start_renderer() {
        let dir =
            std::env::temp_dir().join(format!("bootstrapper_ph_disp2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("g");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = sample_config(tmp.join("definitely_missing_exe"), tmp);
        let lifetime = ChildLifetimeGroup::new().expect("lifetime");
        let (mut publisher, _subscriber) = make_publisher_subscriber(&dir);
        let deadline = Arc::new(Mutex::new(Instant::now()));
        let slot = renderer_slot();
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
        assert_eq!(
            dispatch_command(
                HostCommand::SetText("payload".into()),
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
                HostCommand::StartRenderer(Vec::new()),
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
}
