//! Host-to-bootstrapper queue messages: heartbeat, clipboard, renderer spawn.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use interprocess::{Publisher, Subscriber};

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::constants::{
    queue_loop_flush_interval, queue_wait_log_interval, HEARTBEAT_REFRESH_TIMEOUT_SECS,
    INITIAL_HEARTBEAT_TIMEOUT_SECS,
};
use crate::protocol_handlers;

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

/// Returns `true` when queue-loop trace logging should run for this iteration counter.
pub fn should_trace_iter(loop_iter: u64) -> bool {
    loop_iter <= 3 || loop_iter.is_multiple_of(1000)
}

/// Blocks on `incoming` until `cancel`, handling messages. Initial watchdog uses
/// [`INITIAL_HEARTBEAT_TIMEOUT_SECS`], extended to [`HEARTBEAT_REFRESH_TIMEOUT_SECS`] on each
/// [`HostCommand::Heartbeat`] via `heartbeat_deadline`.
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

    logger::info!(
        "Starting queue loop ({} s initial idle timeout; {} s after each HEARTBEAT)",
        INITIAL_HEARTBEAT_TIMEOUT_SECS,
        HEARTBEAT_REFRESH_TIMEOUT_SECS
    );

    while !cancel.load(Ordering::Relaxed) {
        if last_flush.elapsed() >= queue_loop_flush_interval() {
            logger::flush();
            last_flush = Instant::now();
        }
        loop_iter += 1;
        if should_trace_iter(loop_iter) {
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
            if last_wait_log.elapsed() >= queue_wait_log_interval() {
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
            protocol_handlers::dispatch_command(
                cmd,
                outgoing,
                config,
                lifetime,
                heartbeat_deadline
            ),
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

    #[test]
    fn should_trace_iter_first_three_and_multiples_of_1000() {
        assert!(should_trace_iter(1));
        assert!(should_trace_iter(2));
        assert!(should_trace_iter(3));
        assert!(!should_trace_iter(4));
        assert!(!should_trace_iter(999));
        assert!(should_trace_iter(1000));
        assert!(!should_trace_iter(1001));
    }
}
