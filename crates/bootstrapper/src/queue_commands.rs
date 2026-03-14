//! IPC queue command handling for Host-to-bootstrapper protocol.

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use interprocess::{Publisher, Subscriber};

use crate::config::ResoBootConfig;
use crate::logger::Logger;
use crate::orphan;

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
        _ if s.starts_with("SETTEXT") => HostCommand::SetText(s.strip_prefix("SETTEXT").unwrap_or("").to_string()),
        _ => HostCommand::StartRenderer(s.split_whitespace().map(String::from).collect()),
    }
}

/// Handles a host command and returns whether to continue or break the loop.
pub fn handle_command(
    cmd: HostCommand,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    logger: &mut Logger,
) -> LoopAction {
    match cmd {
        HostCommand::Heartbeat => {
            logger.log("Got heartbeat.");
            LoopAction::Continue
        }
        HostCommand::Shutdown => {
            logger.log("Got shutdown command");
            LoopAction::Break
        }
        HostCommand::GetText => {
            logger.log("Getting clipboard text");
            let text = arboard::Clipboard::new()
                .and_then(|mut c| c.get_text())
                .unwrap_or_default();
            let _ = outgoing.try_enqueue(text.as_bytes());
            LoopAction::Continue
        }
        HostCommand::SetText(text) => {
            logger.log("Setting clipboard text");
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(&text);
            }
            LoopAction::Continue
        }
        HostCommand::StartRenderer(ref renderer_args) => {
            let args: Vec<&str> = renderer_args.iter().map(String::as_str).collect();

            #[cfg(target_os = "linux")]
            {
                let symlink = &config.renderite_executable;
                let target = config.renderite_directory.join("renderide");
                if target.exists() && (!symlink.exists() || fs::read_link(symlink).is_err()) {
                    let _ = fs::remove_file(symlink);
                    if let Err(e) = std::os::unix::fs::symlink("renderide", symlink) {
                        logger.log(&format!("Failed to create Renderite.Renderer symlink: {}", e));
                    }
                }
            }

            logger.log(&format!(
                "Spawning renderer: {:?} with args: {:?}",
                config.renderite_executable, args
            ));
            match Command::new(&config.renderite_executable)
                .args(&args)
                .current_dir(&config.renderite_directory)
                .spawn()
            {
                Ok(process) => {
                    logger.log(&format!(
                        "Renderer started PID {} with args: {}",
                        process.id(),
                        renderer_args.join(" ")
                    ));
                    orphan::write_pid_file(process.id(), "renderer", logger);
                    let response = format!("RENDERITE_STARTED:{}", process.id());
                    let _ = outgoing.try_enqueue(response.as_bytes());
                }
                Err(e) => {
                    logger.log(&format!("Failed to start renderer: {}", e));
                }
            }
            LoopAction::Continue
        }
    }
}

/// Main queue loop: dequeue messages, parse, handle, and break on shutdown or cancel.
pub fn queue_loop(
    incoming: &mut Subscriber,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    cancel: &AtomicBool,
    logger: &mut Logger,
) {
    let start = std::time::Instant::now();
    let mut last_wait_log = std::time::Instant::now();
    let mut loop_iter: u64 = 0;

    logger.log("Starting queue loop");
    logger.log("Expected: Host sends first msg (renderer args), then HEARTBEAT every 5s, SHUTDOWN on exit");
    logger.log("dequeue() blocks until message or cancel; empty msg = cancel was set");

    while !cancel.load(Ordering::Relaxed) {
        loop_iter += 1;
        if loop_iter <= 3 || loop_iter % 1000 == 0 {
            logger.log(&format!(
                "queue_loop iter {} elapsed={:.1}s cancel={}",
                loop_iter,
                start.elapsed().as_secs_f64(),
                cancel.load(Ordering::Relaxed)
            ));
        }

        let msg = incoming.dequeue(cancel);
        if msg.is_empty() {
            if cancel.load(Ordering::Relaxed) {
                logger.log("Host process exited (cancel set), stopping queue loop");
                break;
            }
            if last_wait_log.elapsed() >= Duration::from_secs(5) {
                logger.log(&format!(
                    "Still waiting for message from Host (elapsed {:.0}s). Check: Host started with -shmprefix? Host reached BootstrapperManager?",
                    start.elapsed().as_secs_f64()
                ));
                last_wait_log = std::time::Instant::now();
            }
            continue;
        }

        let arguments = match String::from_utf8(msg) {
            Ok(s) => s,
            Err(_) => continue,
        };

        logger.log(&format!("Received message: {}", arguments));

        let cmd = parse_host_command(&arguments);
        if matches!(handle_command(cmd, outgoing, config, logger), LoopAction::Break) {
            break;
        }
    }
}
