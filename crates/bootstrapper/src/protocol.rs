//! Host-to-bootstrapper queue messages: heartbeat, clipboard, renderer spawn.

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use interprocess::{Publisher, Subscriber};

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::fs;

/// Returns `true` when `lhs` and `rhs` refer to the same inode (e.g. a hard link to the renderer binary).
#[cfg(target_os = "macos")]
fn same_filesystem_inode(lhs: &std::path::Path, rhs: &std::path::Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (fs::metadata(lhs), fs::metadata(rhs)) {
        (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
        _ => false,
    }
}

/// Command sent from the Host over `bootstrapper_in`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostCommand {
    /// Extends the IPC watchdog deadline.
    Heartbeat,
    /// Clean shutdown request.
    Shutdown,
    /// Clipboard read request.
    GetText,
    /// Clipboard write (payload after `SETTEXT` prefix).
    SetText(String),
    /// Spawn renderer with argv-style tokens from the message (whitespace-separated).
    StartRenderer(Vec<String>),
}

/// Action for the queue loop after handling one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    /// Continue dequeuing.
    Continue,
    /// Exit the loop (e.g. `SHUTDOWN`).
    Break,
}

/// Parses a UTF-8 message from the Host into a [`HostCommand`].
pub fn parse_host_command(s: &str) -> HostCommand {
    match s {
        "HEARTBEAT" => HostCommand::Heartbeat,
        "SHUTDOWN" => HostCommand::Shutdown,
        "GETTEXT" => HostCommand::GetText,
        _ if s.starts_with("SETTEXT") => HostCommand::SetText(
            s.strip_prefix("SETTEXT")
                .map(str::to_string)
                .unwrap_or_default(),
        ),
        _ => HostCommand::StartRenderer(s.split_whitespace().map(String::from).collect()),
    }
}

/// Handles one command; updates `heartbeat_deadline` when [`HostCommand::Heartbeat`] is received.
pub fn handle_command(
    cmd: HostCommand,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
    heartbeat_deadline: &Arc<Mutex<Instant>>,
) -> LoopAction {
    match cmd {
        HostCommand::Heartbeat => {
            if let Ok(mut d) = heartbeat_deadline.lock() {
                *d = Instant::now() + Duration::from_secs(15);
            }
            logger::info!("Got heartbeat.");
            LoopAction::Continue
        }
        HostCommand::Shutdown => {
            logger::info!("Got shutdown command");
            LoopAction::Break
        }
        HostCommand::GetText => {
            logger::info!("Getting clipboard text");
            let text = arboard::Clipboard::new()
                .and_then(|mut c| c.get_text())
                .unwrap_or_default();
            let _ = outgoing.try_enqueue(text.as_bytes());
            LoopAction::Continue
        }
        HostCommand::SetText(text) => {
            logger::info!("Setting clipboard text");
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(&text);
            }
            LoopAction::Continue
        }
        HostCommand::StartRenderer(ref renderer_args) => {
            let mut args: Vec<String> = renderer_args.clone();
            if let Some(ref level) = config.renderide_log_level {
                args.push("-LogLevel".to_string());
                args.push(level.as_arg().to_string());
            }
            let args_refs: Vec<&str> = args.iter().map(String::as_str).collect();

            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                let symlink = &config.renderite_executable;
                let target = config.renderite_directory.join("renderide");
                let needs_renderer_stub = target.exists() && {
                    #[cfg(target_os = "linux")]
                    {
                        !symlink.exists() || fs::read_link(symlink).is_err()
                    }
                    #[cfg(target_os = "macos")]
                    {
                        !symlink.exists() || !same_filesystem_inode(symlink, &target)
                    }
                };
                if needs_renderer_stub {
                    let _ = fs::remove_file(symlink);
                    #[cfg(target_os = "linux")]
                    if let Err(e) = std::os::unix::fs::symlink(&target, symlink) {
                        logger::warn!("Failed to create Renderite.Renderer symlink: {}", e);
                    }
                    // Hard link so the Host path behaves like the real `renderide` binary; symlinks do
                    // not affect argv0 / process image naming the same way on macOS.
                    #[cfg(target_os = "macos")]
                    if let Err(e) = fs::hard_link(&target, symlink) {
                        logger::warn!("Failed to create Renderite.Renderer link: {}", e);
                    }
                }
            }

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
                        logger::info!(
                            "Renderer started PID {} with args: {}",
                            process.id(),
                            args.join(" ")
                        );
                        let response = format!("RENDERITE_STARTED:{}", process.id());
                        let _ = outgoing.try_enqueue(response.as_bytes());
                    }
                }
                Err(e) => {
                    logger::error!("Failed to start renderer: {}", e);
                }
            }
            LoopAction::Continue
        }
    }
}

/// Blocks on `incoming` until `cancel`, handling messages. Initial watchdog is 2 minutes, extended
/// to 15 seconds on each [`HostCommand::Heartbeat`] via `heartbeat_deadline`.
pub fn queue_loop(
    incoming: &mut Subscriber,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    cancel: &AtomicBool,
    lifetime: &ChildLifetimeGroup,
    heartbeat_deadline: &Arc<Mutex<Instant>>,
) {
    let start = Instant::now();
    let mut last_wait_log = Instant::now();
    let mut last_flush = Instant::now();
    let mut loop_iter: u64 = 0;

    logger::info!("Starting queue loop (2 min initial idle timeout; 15 s after each HEARTBEAT)");

    while !cancel.load(Ordering::Relaxed) {
        if last_flush.elapsed() >= Duration::from_secs(1) {
            logger::flush();
            last_flush = Instant::now();
        }
        loop_iter += 1;
        if loop_iter <= 3_u64 || loop_iter.is_multiple_of(1000) {
            logger::trace!(
                "queue_loop iter {} elapsed={:.1}s cancel={}",
                loop_iter,
                start.elapsed().as_secs_f64(),
                cancel.load(Ordering::Relaxed)
            );
        }

        let msg = incoming.dequeue(cancel);
        if msg.is_empty() {
            if cancel.load(Ordering::Relaxed) {
                logger::info!("Queue loop stopping (cancel set: host exit, SHUTDOWN, or timeout)");
                break;
            }
            if last_wait_log.elapsed() >= Duration::from_secs(5) {
                logger::info!(
                    "Still waiting for message from Host (elapsed {:.0}s). Check -shmprefix and BootstrapperManager.",
                    start.elapsed().as_secs_f64()
                );
                last_wait_log = Instant::now();
            }
            continue;
        }

        let arguments = match String::from_utf8(msg) {
            Ok(s) => s,
            Err(_) => continue,
        };

        logger::info!("Received message: {}", arguments);

        let cmd = parse_host_command(&arguments);
        if matches!(
            handle_command(cmd, outgoing, config, lifetime, heartbeat_deadline),
            LoopAction::Break
        ) {
            cancel.store(true, Ordering::SeqCst);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_host_command_fixed_tokens() {
        assert_eq!(parse_host_command("HEARTBEAT"), HostCommand::Heartbeat);
        assert_eq!(parse_host_command("SHUTDOWN"), HostCommand::Shutdown);
        assert_eq!(parse_host_command("GETTEXT"), HostCommand::GetText);
    }

    #[test]
    fn parse_host_command_settext() {
        assert!(matches!(
            parse_host_command("SETTEXThello"),
            HostCommand::SetText(ref s) if s == "hello"
        ));
    }

    #[test]
    fn parse_host_command_renderer_args() {
        let cmd = parse_host_command("-QueueName q -QueueCapacity 4096");
        assert!(matches!(
            cmd,
            HostCommand::StartRenderer(ref args)
                if args
                    == &vec!["-QueueName", "q", "-QueueCapacity", "4096"]
                        .into_iter()
                        .map(String::from)
                        .collect::<Vec<_>>()
        ));
    }

    #[test]
    fn parse_host_command_empty_message_is_start_renderer_empty() {
        assert!(matches!(
            parse_host_command(""),
            HostCommand::StartRenderer(ref args) if args.is_empty()
        ));
    }

    #[test]
    fn parse_host_command_settext_only() {
        assert!(matches!(
            parse_host_command("SETTEXT"),
            HostCommand::SetText(ref s) if s.is_empty()
        ));
    }

    #[test]
    fn parse_host_command_settext_preserves_utf8_payload() {
        let cmd = parse_host_command("SETTEXTこんにちは");
        assert!(matches!(
            cmd,
            HostCommand::SetText(ref s) if s == "こんにちは"
        ));
    }

    #[test]
    fn parse_host_command_whitespace_only_yields_empty_start_renderer() {
        assert!(matches!(
            parse_host_command("   \t  "),
            HostCommand::StartRenderer(ref args) if args.is_empty()
        ));
    }

    #[test]
    fn parse_host_command_unknown_token_becomes_start_renderer_argv() {
        let cmd = parse_host_command("CUSTOM opaque tail");
        assert!(matches!(
            cmd,
            HostCommand::StartRenderer(ref args)
                if args == &vec!["CUSTOM".to_string(), "opaque".to_string(), "tail".to_string()]
        ));
    }
}
