//! IPC queue command handling for Host-to-bootstrapper protocol.

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use interprocess::{Publisher, Subscriber};

use crate::config::ResoBootConfig;
use crate::process_lifetime::ChildLifetimeGroup;

#[cfg(target_os = "linux")]
use std::fs;

/// Parsed host command from the IPC queue.
#[derive(Debug)]
pub enum HostCommand {
    Heartbeat,
    Shutdown,
    GetText,
    SetText(String),
    StartRenderer(Vec<String>),
}

/// Result of handling a command: continue the loop or break.
#[derive(Debug)]
pub enum LoopAction {
    Continue,
    Break,
}

/// Parses a message string into a HostCommand.
pub fn parse_host_command(s: &str) -> HostCommand {
    match s {
        "HEARTBEAT" => HostCommand::Heartbeat,
        "SHUTDOWN" => HostCommand::Shutdown,
        "GETTEXT" => HostCommand::GetText,
        _ if s.starts_with("SETTEXT") => {
            HostCommand::SetText(s.strip_prefix("SETTEXT").unwrap_or("").to_string())
        }
        _ => HostCommand::StartRenderer(s.split_whitespace().map(String::from).collect()),
    }
}

/// Handles a host command and returns whether to continue or break the loop.
///
/// `lifetime` is applied when starting the renderer so it exits with the bootstrapper.
pub fn handle_command(
    cmd: HostCommand,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    lifetime: &ChildLifetimeGroup,
) -> LoopAction {
    match cmd {
        HostCommand::Heartbeat => {
            logger::info!("Got heartbeat.");
            LoopAction::Continue
        }
        HostCommand::Shutdown => {
            eprintln!("Host closed itself (SHUTDOWN command received)");
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

            #[cfg(target_os = "linux")]
            {
                let symlink = &config.renderite_executable;
                let target = config.renderite_directory.join("renderide");
                if target.exists() && (!symlink.exists() || fs::read_link(symlink).is_err()) {
                    let _ = fs::remove_file(symlink);
                    if let Err(e) = std::os::unix::fs::symlink("renderide", symlink) {
                        logger::warn!("Failed to create Renderite.Renderer symlink: {}", e);
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

/// Main queue loop: dequeue messages, parse, handle, and break on shutdown or cancel.
///
/// `lifetime` is forwarded when spawning the renderer from [`HostCommand::StartRenderer`].
pub fn queue_loop(
    incoming: &mut Subscriber,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    cancel: &AtomicBool,
    lifetime: &ChildLifetimeGroup,
) {
    let start = std::time::Instant::now();
    let mut last_wait_log = std::time::Instant::now();
    let mut last_flush = std::time::Instant::now();
    let mut loop_iter: u64 = 0;

    logger::info!("Starting queue loop");
    logger::info!(
        "Expected: Host sends first msg (renderer args), then HEARTBEAT every 5s, SHUTDOWN on exit",
    );
    logger::info!("dequeue() blocks until message or cancel; empty msg = cancel was set");

    while !cancel.load(Ordering::Relaxed) {
        if last_flush.elapsed() >= Duration::from_secs(1) {
            logger::flush();
            last_flush = std::time::Instant::now();
        }
        loop_iter += 1;
        if loop_iter <= 3 || loop_iter.is_multiple_of(1000) {
            logger::info!(
                "queue_loop iter {} elapsed={:.1}s cancel={}",
                loop_iter,
                start.elapsed().as_secs_f64(),
                cancel.load(Ordering::Relaxed)
            );
        }

        let msg = incoming.dequeue(cancel);
        if msg.is_empty() {
            if cancel.load(Ordering::Relaxed) {
                logger::info!("Host process exited (cancel set), stopping queue loop");
                break;
            }
            if last_wait_log.elapsed() >= Duration::from_secs(5) {
                logger::info!(
                    "Still waiting for message from Host (elapsed {:.0}s). Check: Host started with -shmprefix? Host reached BootstrapperManager?",
                    start.elapsed().as_secs_f64()
                );
                last_wait_log = std::time::Instant::now();
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
            handle_command(cmd, outgoing, config, lifetime),
            LoopAction::Break
        ) {
            break;
        }
    }
}
