//! Bootstrap queue loop and per-loop accounting.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use interprocess::{Publisher, Subscriber};

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::constants::{
    HEARTBEAT_REFRESH_TIMEOUT_SECS, INITIAL_HEARTBEAT_TIMEOUT_SECS, queue_loop_flush_interval,
    queue_wait_log_interval,
};
use crate::process_state::SharedChildSlot;
use crate::protocol::HostCommand;
use crate::protocol_handlers;

/// Action for the queue loop after handling one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    /// Continue dequeuing.
    Continue,
    /// Exit the loop (e.g. `SHUTDOWN`).
    Break,
}

/// Returns `true` when queue-loop trace logging should run for this iteration counter.
pub const fn should_trace_iter(loop_iter: u64) -> bool {
    loop_iter <= 3 || loop_iter.is_multiple_of(1000)
}

#[derive(Default)]
struct QueueLoopStats {
    /// Number of valid UTF-8 command messages handled.
    messages: u64,
    /// Number of non-UTF-8 queue messages ignored.
    invalid_utf8: u64,
    /// Number of heartbeat commands handled.
    heartbeats: u64,
    /// Number of shutdown commands handled.
    shutdowns: u64,
    /// Number of clipboard read commands handled.
    get_text: u64,
    /// Number of clipboard write commands handled.
    set_text: u64,
    /// Number of renderer launch commands handled.
    start_renderer: u64,
}

impl QueueLoopStats {
    /// Records one parsed command in the loop summary counters.
    fn record_command(&mut self, cmd: &HostCommand) {
        self.messages = self.messages.saturating_add(1);
        match cmd {
            HostCommand::Heartbeat => self.heartbeats = self.heartbeats.saturating_add(1),
            HostCommand::Shutdown => self.shutdowns = self.shutdowns.saturating_add(1),
            HostCommand::GetText => self.get_text = self.get_text.saturating_add(1),
            HostCommand::SetText(_) => self.set_text = self.set_text.saturating_add(1),
            HostCommand::StartRenderer(_) => {
                self.start_renderer = self.start_renderer.saturating_add(1);
            }
        }
    }
}

/// Blocks on `incoming` until `cancel`, handling messages.
///
/// Initial watchdog uses [`INITIAL_HEARTBEAT_TIMEOUT_SECS`], extended to
/// [`HEARTBEAT_REFRESH_TIMEOUT_SECS`] on each [`HostCommand::Heartbeat`] via
/// `heartbeat_deadline`.
pub fn queue_loop(
    incoming: &mut Subscriber,
    outgoing: &mut Publisher,
    config: &ResoBootConfig,
    cancel: &AtomicBool,
    lifetime: &ChildLifetimeGroup,
    heartbeat_deadline: &Arc<Mutex<Instant>>,
    renderer_child: &SharedChildSlot,
) {
    let start = Instant::now();
    let mut last_wait_log = Instant::now();
    let mut last_flush = Instant::now();
    let mut loop_iter: u64 = 0;
    let mut stats = QueueLoopStats::default();
    let mut stop_reason = "cancel";

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
                logger::info!(
                    "Queue loop stopping (cancel set: host exit, renderer exit, SHUTDOWN, or timeout)"
                );
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

        let Ok(arguments) = String::from_utf8(msg) else {
            stats.invalid_utf8 = stats.invalid_utf8.saturating_add(1);
            continue;
        };

        logger::info!("Received message: {}", arguments);

        let cmd = crate::protocol::parse_host_command(&arguments);
        stats.record_command(&cmd);
        if matches!(
            protocol_handlers::dispatch_command(
                cmd,
                outgoing,
                config,
                lifetime,
                heartbeat_deadline,
                renderer_child,
            ),
            LoopAction::Break
        ) {
            cancel.store(true, Ordering::SeqCst);
            stop_reason = "command-break";
            break;
        }
    }
    logger::info!(
        "Queue loop summary: reason={} elapsed_s={:.3} iterations={} messages={} invalid_utf8={} heartbeats={} shutdowns={} get_text={} set_text={} start_renderer={}",
        stop_reason,
        start.elapsed().as_secs_f64(),
        loop_iter,
        stats.messages,
        stats.invalid_utf8,
        stats.heartbeats,
        stats.shutdowns,
        stats.get_text,
        stats.set_text,
        stats.start_renderer,
    );
}
