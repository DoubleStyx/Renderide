//! Dual-queue IPC: Primary and Background subscriber/publisher pairs for [`RendererCommand`].
//!
//! Naming matches the host client when the renderer is **non-authority**: subscribe on `...A`,
//! publish on `...S`.

use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use interprocess::{Publisher, QueueFactory, Subscriber};

use super::connection::{ConnectionParams, InitError, publisher_queue_name, subscriber_queue_name};
use super::dual_queue_reliable_outbox::ReliableBackgroundOutbox;
pub use super::dual_queue_shared::IpcDrainStats;
use super::dual_queue_shared::{
    decode_renderer_command_payload, encode_command, open_publisher, open_subscriber,
};
use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::shared::RendererCommand;

const SEND_BUFFER_CAP: usize = 65536;
const INBOUND_PUMP_IDLE_WAIT: Duration = Duration::from_millis(10);
const INBOUND_PUMP_BUFFERED_MESSAGES: usize = 1024;

/// Log prefix used when [`encode_command`] overflows the send buffer on the renderer side.
const ENCODE_OVERFLOW_LOG_PREFIX: &str = "IPC outgoing send: encode overflow";

/// Log prefix used when inbound polling decodes an invalid command on the renderer side.
const INVALID_MESSAGE_LOG_PREFIX: &str = "IPC";

/// After this many consecutive `try_enqueue` failures on one channel, log at [`logger::error!`].
const IPC_CONSECUTIVE_DROP_ERROR_AFTER: u32 = 16;

/// Renderer command annotated with the instant it was removed from the inbound IPC queue.
#[derive(Clone, Debug)]
pub struct TimedRendererCommand {
    /// Decoded renderer command.
    pub command: RendererCommand,
    /// Wall-clock instant when the raw command bytes were removed from the inbound queue.
    pub received_at: Instant,
}

impl TimedRendererCommand {
    /// Builds a timed command from an already-decoded command and explicit receive instant.
    pub fn new(command: RendererCommand, received_at: Instant) -> Self {
        Self {
            command,
            received_at,
        }
    }

    /// Builds a timed command stamped with [`Instant::now`].
    pub fn received_now(command: RendererCommand) -> Self {
        Self::new(command, Instant::now())
    }
}

struct RawTimedIpcMessage {
    payload: Vec<u8>,
    received_at: Instant,
}

struct InboundPump {
    primary_rx: Option<Receiver<RawTimedIpcMessage>>,
    background_rx: Option<Receiver<RawTimedIpcMessage>>,
    primary_cancel: Arc<AtomicBool>,
    background_cancel: Arc<AtomicBool>,
    primary_thread: Option<JoinHandle<()>>,
    background_thread: Option<JoinHandle<()>>,
}

impl InboundPump {
    fn new(primary: Subscriber, background: Subscriber) -> Result<Self, InitError> {
        let (primary_tx, primary_rx) = mpsc::sync_channel(INBOUND_PUMP_BUFFERED_MESSAGES);
        let (background_tx, background_rx) = mpsc::sync_channel(INBOUND_PUMP_BUFFERED_MESSAGES);
        let primary_cancel = Arc::new(AtomicBool::new(false));
        let background_cancel = Arc::new(AtomicBool::new(false));
        let primary_thread = spawn_inbound_pump(
            "renderide-ipc-primary-in",
            primary,
            primary_tx,
            Arc::clone(&primary_cancel),
        )?;
        let background_thread = spawn_inbound_pump(
            "renderide-ipc-background-in",
            background,
            background_tx,
            Arc::clone(&background_cancel),
        )?;
        Ok(Self {
            primary_rx: Some(primary_rx),
            background_rx: Some(background_rx),
            primary_cancel,
            background_cancel,
            primary_thread: Some(primary_thread),
            background_thread: Some(background_thread),
        })
    }
}

impl Drop for InboundPump {
    fn drop(&mut self) {
        self.primary_cancel.store(true, Ordering::Relaxed);
        self.background_cancel.store(true, Ordering::Relaxed);
        self.primary_rx.take();
        self.background_rx.take();
        join_inbound_pump_thread("primary", self.primary_thread.take());
        join_inbound_pump_thread("background", self.background_thread.take());
    }
}

/// Diagnostic counters collected while draining both renderer inbound queues.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DualQueueDrainStats {
    /// Primary subscriber drain counters.
    pub primary: IpcDrainStats,
    /// Background subscriber drain counters.
    pub background: IpcDrainStats,
}

impl DualQueueDrainStats {
    /// Returns an aggregate sample across both queues.
    pub fn total(&self) -> IpcDrainStats {
        let mut total = self.primary;
        total.add(self.background);
        total
    }
}

/// Diagnostic counters collected by a primary-wait poll.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DualQueuePollStats {
    /// Time spent waiting on the primary inbound semaphore.
    pub waited: Duration,
    /// Counters from the immediate pre-wait drain.
    pub initial_drain: DualQueueDrainStats,
    /// Counters from the post-wait drain after the primary queue became ready.
    pub post_wait_drain: DualQueueDrainStats,
    /// Whether the primary wait consumed the full timeout without a ready message.
    pub timed_out: bool,
}

impl DualQueuePollStats {
    /// Returns aggregate queue-drain counters from the full poll operation.
    pub fn total_drain(&self) -> IpcDrainStats {
        let mut total = self.initial_drain.total();
        total.add(self.post_wait_drain.total());
        total
    }
}

/// Host <-> renderer IPC over two Cloudtoid queue pairs (Primary and Background).
pub struct DualQueueIpc {
    inbound: InboundPump,
    primary_publisher: Publisher,
    background_publisher: Publisher,
    /// Reused across [`Self::poll_into`] calls so optional heap types during decode do not allocate a fresh pool each message.
    entity_pool: DefaultEntityPool,
    send_buffer: Vec<u8>,
    /// Count of dropped primary sends since last successful send (consecutive backpressure).
    primary_drops_since_log: u32,
    background_drops_since_log: u32,
    reliable_background_outbox: ReliableBackgroundOutbox,
    /// Set when a primary outbound send failed this winit tick (cleared in [`Self::reset_outbound_drop_tick_flags`]).
    had_primary_outbound_drop_this_tick: bool,
    had_background_outbound_drop_this_tick: bool,
}

impl DualQueueIpc {
    /// Opens all four queue endpoints. `params.queue_name` is the base prefix; `"Primary"` /
    /// `"Background"` are appended before the `A`/`S` suffixes.
    pub fn connect(params: &ConnectionParams) -> Result<Self, InitError> {
        Self::connect_inner(params, None)
    }

    /// Same as [`Self::connect`] but pins the backing directory explicitly.
    pub fn connect_with_dir(params: &ConnectionParams, dir: &Path) -> Result<Self, InitError> {
        Self::connect_inner(params, Some(dir))
    }

    fn connect_inner(
        params: &ConnectionParams,
        dir_override: Option<&Path>,
    ) -> Result<Self, InitError> {
        let factory = QueueFactory::new();
        let cap = params.queue_capacity;

        let primary_sub_name = subscriber_queue_name(&params.queue_name, "Primary");
        let background_sub_name = subscriber_queue_name(&params.queue_name, "Background");
        let primary_pub_name = publisher_queue_name(&params.queue_name, "Primary");
        let background_pub_name = publisher_queue_name(&params.queue_name, "Background");

        logger::info!(
            "IPC connect: base={} capacity={} primary_sub={} primary_pub={} background_sub={} background_pub={}",
            params.queue_name,
            cap,
            primary_sub_name,
            primary_pub_name,
            background_sub_name,
            background_pub_name,
        );

        let primary_sub = open_subscriber(factory, &primary_sub_name, cap, dir_override, false)?;
        let background_sub =
            open_subscriber(factory, &background_sub_name, cap, dir_override, false)?;
        let primary_pub = open_publisher(factory, &primary_pub_name, cap, dir_override, false)?;
        let background_pub =
            open_publisher(factory, &background_pub_name, cap, dir_override, false)?;
        let inbound = InboundPump::new(primary_sub, background_sub)?;

        Ok(Self {
            inbound,
            primary_publisher: primary_pub,
            background_publisher: background_pub,
            entity_pool: DefaultEntityPool,
            send_buffer: vec![0u8; SEND_BUFFER_CAP],
            primary_drops_since_log: 0,
            background_drops_since_log: 0,
            reliable_background_outbox: ReliableBackgroundOutbox::default(),
            had_primary_outbound_drop_this_tick: false,
            had_background_outbound_drop_this_tick: false,
        })
    }

    /// Clears per-tick outbound drop flags; call once at the start of each winit frame tick.
    pub const fn reset_outbound_drop_tick_flags(&mut self) {
        self.had_primary_outbound_drop_this_tick = false;
        self.had_background_outbound_drop_this_tick = false;
    }

    /// Whether any **primary** outbound send failed since the last [`Self::reset_outbound_drop_tick_flags`].
    pub const fn had_outbound_primary_drop_this_tick(&self) -> bool {
        self.had_primary_outbound_drop_this_tick
    }

    /// Whether any **background** outbound send failed since the last [`Self::reset_outbound_drop_tick_flags`].
    pub const fn had_outbound_background_drop_this_tick(&self) -> bool {
        self.had_background_outbound_drop_this_tick
    }

    /// Current consecutive primary-queue drop streak (resets on next successful enqueue).
    pub const fn consecutive_primary_drop_streak(&self) -> u32 {
        self.primary_drops_since_log
    }

    /// Current consecutive background-queue drop streak (resets on next successful enqueue).
    pub const fn consecutive_background_drop_streak(&self) -> u32 {
        self.background_drops_since_log
    }

    /// Number of reliable background messages waiting for queue capacity.
    pub fn reliable_background_pending_count(&self) -> usize {
        self.reliable_background_outbox.len()
    }

    /// Encoded byte count of reliable background messages waiting for queue capacity.
    pub fn reliable_background_pending_bytes(&self) -> usize {
        self.reliable_background_outbox.pending_bytes()
    }

    /// Drains both subscribers into `out` (Primary first, then Background; each channel fully drained in order).
    ///
    /// Clears `out` then drains both subscribers so each tick starts from an empty batch.
    pub fn poll_into(&mut self, out: &mut Vec<RendererCommand>) {
        let mut timed = Vec::new();
        let _stats = self.poll_timed_into_profiled(&mut timed);
        out.clear();
        out.extend(timed.into_iter().map(|timed| timed.command));
    }

    /// Drains both subscribers into `out` and returns diagnostic counters for the drain.
    pub fn poll_into_profiled(&mut self, out: &mut Vec<RendererCommand>) -> DualQueueDrainStats {
        let mut timed = Vec::new();
        let stats = self.poll_timed_into_profiled(&mut timed);
        out.clear();
        out.extend(timed.into_iter().map(|timed| timed.command));
        stats
    }

    /// Drains both subscribers into `out` with per-message receive timestamps.
    pub fn poll_timed_into(&mut self, out: &mut Vec<TimedRendererCommand>) {
        let _stats = self.poll_timed_into_profiled(out);
    }

    /// Drains both subscribers into `out` with per-message receive timestamps and counters.
    pub fn poll_timed_into_profiled(
        &mut self,
        out: &mut Vec<TimedRendererCommand>,
    ) -> DualQueueDrainStats {
        out.clear();
        self.flush_reliable_outbound();
        let primary = {
            profiling::scope!("ipc::primary_drain");
            drain_timed_receiver(
                self.inbound.primary_rx.as_ref(),
                &mut self.entity_pool,
                out,
                INVALID_MESSAGE_LOG_PREFIX,
            )
        };
        let background = {
            profiling::scope!("ipc::background_drain");
            drain_timed_receiver(
                self.inbound.background_rx.as_ref(),
                &mut self.entity_pool,
                out,
                INVALID_MESSAGE_LOG_PREFIX,
            )
        };
        DualQueueDrainStats {
            primary,
            background,
        }
    }

    /// Waits up to `timeout` for the primary inbound queue, then drains both inbound queues.
    ///
    /// The immediate drain first handles messages that were already queued before the caller
    /// decided to wait. The timed wait is primary-only because lock-step frame submissions and
    /// lifecycle commands travel on the primary channel.
    pub fn poll_into_after_primary_wait(
        &mut self,
        out: &mut Vec<RendererCommand>,
        timeout: Duration,
    ) -> Duration {
        let mut timed = Vec::new();
        let stats = self.poll_timed_into_after_primary_wait_profiled(&mut timed, timeout);
        out.clear();
        out.extend(timed.into_iter().map(|timed| timed.command));
        stats.waited
    }

    /// Waits up to `timeout` for primary inbound work and returns diagnostic counters.
    pub fn poll_into_after_primary_wait_profiled(
        &mut self,
        out: &mut Vec<RendererCommand>,
        timeout: Duration,
    ) -> DualQueuePollStats {
        let mut timed = Vec::new();
        let stats = self.poll_timed_into_after_primary_wait_profiled(&mut timed, timeout);
        out.clear();
        out.extend(timed.into_iter().map(|timed| timed.command));
        stats
    }

    /// Waits up to `timeout` for primary inbound work and returns timed commands.
    pub fn poll_timed_into_after_primary_wait(
        &mut self,
        out: &mut Vec<TimedRendererCommand>,
        timeout: Duration,
    ) -> Duration {
        self.poll_timed_into_after_primary_wait_profiled(out, timeout)
            .waited
    }

    /// Waits up to `timeout` for primary inbound work and returns timed commands with counters.
    pub fn poll_timed_into_after_primary_wait_profiled(
        &mut self,
        out: &mut Vec<TimedRendererCommand>,
        timeout: Duration,
    ) -> DualQueuePollStats {
        let initial_drain = {
            profiling::scope!("ipc::immediate_drain");
            self.poll_timed_into_profiled(out)
        };
        if !out.is_empty() || timeout.is_zero() {
            return DualQueuePollStats {
                initial_drain,
                ..DualQueuePollStats::default()
            };
        }
        let wait_started = Instant::now();
        let Some(primary_rx) = self.inbound.primary_rx.as_ref() else {
            logger::warn!("IPC primary inbound receiver disconnected before wait");
            return DualQueuePollStats {
                initial_drain,
                ..DualQueuePollStats::default()
            };
        };
        let ready = {
            profiling::scope!("ipc::host_wait::primary_queue");
            profiling::scope!("ipc::primary_wait");
            primary_rx.recv_timeout(timeout)
        };
        let waited = wait_started.elapsed();
        let first_primary = match ready {
            Ok(msg) => msg,
            Err(RecvTimeoutError::Timeout) => {
                return DualQueuePollStats {
                    waited,
                    initial_drain,
                    timed_out: true,
                    ..DualQueuePollStats::default()
                };
            }
            Err(RecvTimeoutError::Disconnected) => {
                logger::warn!("IPC primary inbound pump disconnected during wait");
                return DualQueuePollStats {
                    waited,
                    initial_drain,
                    ..DualQueuePollStats::default()
                };
            }
        };
        let post_wait_drain = {
            profiling::scope!("ipc::post_wait_drain");
            self.flush_reliable_outbound();
            let mut primary = IpcDrainStats::default();
            drain_timed_message(
                first_primary,
                &mut self.entity_pool,
                out,
                INVALID_MESSAGE_LOG_PREFIX,
                &mut primary,
            );
            primary.add(drain_timed_receiver(
                self.inbound.primary_rx.as_ref(),
                &mut self.entity_pool,
                out,
                INVALID_MESSAGE_LOG_PREFIX,
            ));
            let background = drain_timed_receiver(
                self.inbound.background_rx.as_ref(),
                &mut self.entity_pool,
                out,
                INVALID_MESSAGE_LOG_PREFIX,
            );
            DualQueueDrainStats {
                primary,
                background,
            }
        };
        DualQueuePollStats {
            waited,
            initial_drain,
            post_wait_drain,
            timed_out: false,
        }
    }

    /// Encodes and sends a command on the **Primary** publisher (frame handshake, init, etc.).
    ///
    /// Returns `true` if the message was queued, `false` if encoding produced no bytes or the queue was full.
    pub fn send_primary(&mut self, mut cmd: RendererCommand) -> bool {
        let written = encode_command(&mut cmd, &mut self.send_buffer, ENCODE_OVERFLOW_LOG_PREFIX);
        if written == 0 {
            return false;
        }
        let ok = send_on_publisher(
            &mut self.primary_publisher,
            &self.send_buffer[..written],
            &mut self.primary_drops_since_log,
            "primary",
        );
        if !ok {
            self.had_primary_outbound_drop_this_tick = true;
        }
        ok
    }

    /// Encodes and sends a command on the **Background** publisher (asset results, etc.).
    ///
    /// Returns `true` if the message was queued, `false` if encoding produced no bytes or the queue was full.
    pub fn send_background(&mut self, mut cmd: RendererCommand) -> bool {
        if !self.reliable_background_outbox.is_empty() {
            self.flush_reliable_outbound();
            if !self.reliable_background_outbox.is_empty() {
                self.had_background_outbound_drop_this_tick = true;
                return false;
            }
        }
        let written = encode_command(&mut cmd, &mut self.send_buffer, ENCODE_OVERFLOW_LOG_PREFIX);
        if written == 0 {
            return false;
        }
        let ok = send_on_publisher(
            &mut self.background_publisher,
            &self.send_buffer[..written],
            &mut self.background_drops_since_log,
            "background",
        );
        if !ok {
            self.had_background_outbound_drop_this_tick = true;
        }
        ok
    }

    /// Encodes a command for the **Background** publisher and retains it until it is sent.
    ///
    /// Returns `true` when the command was encoded and accepted into the reliable outbox. A `true`
    /// return does not guarantee the host has already received the command; call
    /// [`Self::flush_reliable_outbound`] to retry pending reliable messages.
    pub fn send_background_reliable(&mut self, cmd: RendererCommand) -> bool {
        if !self.enqueue_background_reliable(cmd) {
            return false;
        }
        self.flush_reliable_outbound();
        true
    }

    /// Encodes a reliable background command without flushing the reliable outbox immediately.
    ///
    /// Call [`Self::flush_reliable_outbound`] once after a batch of acknowledgements has been
    /// enqueued to amortize publisher calls.
    pub fn enqueue_background_reliable(&mut self, mut cmd: RendererCommand) -> bool {
        let written = encode_command(&mut cmd, &mut self.send_buffer, ENCODE_OVERFLOW_LOG_PREFIX);
        if written == 0 {
            return false;
        }
        if !self
            .reliable_background_outbox
            .enqueue(self.send_buffer[..written].to_vec())
        {
            logger::warn!(
                "IPC reliable background outbox full: pending_messages={} pending_bytes={}",
                self.reliable_background_outbox.len(),
                self.reliable_background_outbox.pending_bytes()
            );
            self.had_background_outbound_drop_this_tick = true;
            return false;
        }
        let pending_count = self.reliable_background_outbox.len();
        let pending_bytes = self.reliable_background_outbox.pending_bytes();
        if pending_count == 64 || (pending_count > 64 && pending_count.is_multiple_of(64)) {
            logger::warn!(
                "IPC reliable background outbox pressure: pending_messages={} pending_bytes={}",
                pending_count,
                pending_bytes
            );
        }
        true
    }

    /// Retries retained reliable background messages in FIFO order until the queue is full.
    pub fn flush_reliable_outbound(&mut self) {
        while let Some(payload) = self.reliable_background_outbox.front() {
            let ok = send_on_publisher(
                &mut self.background_publisher,
                payload,
                &mut self.background_drops_since_log,
                "background reliable",
            );
            if !ok {
                self.had_background_outbound_drop_this_tick = true;
                break;
            }
            self.reliable_background_outbox.mark_front_sent();
        }
    }
}

fn spawn_inbound_pump(
    name: &'static str,
    mut subscriber: Subscriber,
    tx: SyncSender<RawTimedIpcMessage>,
    cancel: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, InitError> {
    thread::Builder::new()
        .name(String::from(name))
        .spawn(move || {
            while !cancel.load(Ordering::Relaxed) {
                let ready = {
                    profiling::scope!("ipc::inbound_pump_wait");
                    subscriber.wait_for_message_timeout(INBOUND_PUMP_IDLE_WAIT)
                };
                if !ready {
                    continue;
                }
                while let Some(payload) = {
                    profiling::scope!("ipc::inbound_pump_dequeue");
                    subscriber.try_dequeue()
                } {
                    let received_at = Instant::now();
                    if tx
                        .send(RawTimedIpcMessage {
                            payload,
                            received_at,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
        })
        .map_err(|e| InitError::IpcConnect(format!("failed to spawn {name}: {e}")))
}

fn join_inbound_pump_thread(name: &'static str, thread: Option<JoinHandle<()>>) {
    let Some(thread) = thread else {
        return;
    };
    if thread.join().is_err() {
        logger::warn!("IPC {name} inbound pump thread panicked during shutdown");
    }
}

fn drain_timed_receiver(
    rx: Option<&Receiver<RawTimedIpcMessage>>,
    pool: &mut DefaultEntityPool,
    out: &mut Vec<TimedRendererCommand>,
    invalid_log_prefix: &'static str,
) -> IpcDrainStats {
    let mut stats = IpcDrainStats::default();
    let Some(rx) = rx else {
        return stats;
    };
    while let Ok(msg) = rx.try_recv() {
        drain_timed_message(msg, pool, out, invalid_log_prefix, &mut stats);
    }
    stats
}

fn drain_timed_message(
    msg: RawTimedIpcMessage,
    pool: &mut DefaultEntityPool,
    out: &mut Vec<TimedRendererCommand>,
    invalid_log_prefix: &'static str,
    stats: &mut IpcDrainStats,
) {
    stats.bytes += msg.payload.len();
    let decoded = decode_renderer_command_payload(&msg.payload, pool, invalid_log_prefix);
    stats.decode_duration += decoded.duration;
    match decoded.command {
        Some(command) => {
            stats.messages += 1;
            out.push(TimedRendererCommand::new(command, msg.received_at));
        }
        None => {
            stats.invalid_messages += 1;
        }
    }
}

fn send_on_publisher(
    publisher: &mut Publisher,
    payload: &[u8],
    drops_since_log: &mut u32,
    channel: &'static str,
) -> bool {
    if publisher.try_enqueue(payload) {
        *drops_since_log = 0;
        return true;
    }
    *drops_since_log += 1;
    if *drops_since_log == 1 {
        logger::warn!(
            "IPC {channel} queue full, dropped outgoing command ({} bytes)",
            payload.len()
        );
    } else if *drops_since_log >= IPC_CONSECUTIVE_DROP_ERROR_AFTER
        && ((*drops_since_log == IPC_CONSECUTIVE_DROP_ERROR_AFTER)
            || (*drops_since_log - IPC_CONSECUTIVE_DROP_ERROR_AFTER)
                .is_multiple_of(IPC_CONSECUTIVE_DROP_ERROR_AFTER))
    {
        logger::error!(
            "IPC {channel} queue full: {} consecutive dropped outgoing sends (backpressure)",
            *drops_since_log
        );
    } else if drops_since_log.is_multiple_of(128) {
        logger::warn!(
            "IPC {channel} queue full: {} additional drops since last summary",
            128
        );
    }
    false
}

#[cfg(test)]
mod timed_ipc_tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::DualQueueIpc;
    use crate::ipc::connection::ConnectionParams;
    use crate::ipc::host_dual_queue::HostDualQueueIpc;
    use crate::shared::{FrameSubmitData, KeepAlive, QualityConfig, RendererCommand};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_params() -> (ConnectionParams, PathBuf) {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "renderide_timed_ipc_{}_{}",
            std::process::id(),
            seq
        ));
        std::fs::create_dir_all(&dir).expect("create IPC test dir");
        (
            ConnectionParams {
                queue_name: format!("renderide_timed_ipc_{seq}"),
                queue_capacity: 4096,
            },
            dir,
        )
    }

    #[test]
    fn timed_poll_preserves_primary_before_background_when_both_are_ready() {
        let (params, dir) = unique_params();
        let mut host = HostDualQueueIpc::connect_with_dir(&params, &dir).expect("host connect");
        let mut renderer = DualQueueIpc::connect_with_dir(&params, &dir).expect("renderer connect");
        let sent_at = Instant::now();

        assert!(host.send_background(RendererCommand::QualityConfig(QualityConfig::default(),)));
        assert!(host.send_primary(RendererCommand::KeepAlive(KeepAlive::default())));
        std::thread::sleep(Duration::from_millis(50));

        let mut out = Vec::new();
        renderer.poll_timed_into(&mut out);

        assert_eq!(out.len(), 2);
        assert!(matches!(out[0].command, RendererCommand::KeepAlive(_)));
        assert!(matches!(out[1].command, RendererCommand::QualityConfig(_)));
        assert!(out.iter().all(|cmd| cmd.received_at >= sent_at));
        drop(renderer);
        drop(host);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn primary_wait_returns_receive_timestamped_command() {
        let (params, dir) = unique_params();
        let mut host = HostDualQueueIpc::connect_with_dir(&params, &dir).expect("host connect");
        let mut renderer = DualQueueIpc::connect_with_dir(&params, &dir).expect("renderer connect");
        let sent_at = Instant::now();

        assert!(host.send_primary(RendererCommand::FrameSubmitData(FrameSubmitData::default(),)));

        let mut out = Vec::new();
        let waited =
            renderer.poll_timed_into_after_primary_wait(&mut out, Duration::from_millis(250));

        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0].command,
            RendererCommand::FrameSubmitData(_)
        ));
        assert!(out[0].received_at >= sent_at);
        assert!(waited <= Duration::from_millis(250));
        drop(renderer);
        drop(host);
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod renderer_command_roundtrip_tests {
    use super::ENCODE_OVERFLOW_LOG_PREFIX;
    use crate::ipc::dual_queue_shared::encode_command;
    use crate::packing::default_entity_pool::DefaultEntityPool;
    use crate::packing::memory_unpacker::MemoryUnpacker;
    use crate::shared::{
        FrameSubmitData, FreeSharedMemoryView, KeepAlive, RendererCommand, RendererShutdown,
        decode_renderer_command,
    };

    fn assert_roundtrip(mut cmd: RendererCommand) {
        let expect = format!("{cmd:?}");
        let mut buf = vec![0u8; 65536];
        let n = encode_command(&mut cmd, &mut buf, ENCODE_OVERFLOW_LOG_PREFIX);
        let mut pool = DefaultEntityPool;
        let mut unpacker = MemoryUnpacker::new(&buf[..n], &mut pool);
        let decoded = decode_renderer_command(&mut unpacker).expect("decode");
        assert_eq!(
            expect,
            format!("{decoded:?}"),
            "RendererCommand wire roundtrip"
        );
        assert_eq!(unpacker.remaining_data(), 0, "no trailing bytes");
    }

    #[test]
    fn encode_command_reports_zero_when_output_buffer_is_too_small() {
        let mut cmd = RendererCommand::FrameSubmitData(FrameSubmitData::default());
        let mut tiny = [0u8; 1];

        let written = encode_command(&mut cmd, &mut tiny, ENCODE_OVERFLOW_LOG_PREFIX);

        assert_eq!(written, 0);
    }

    #[test]
    fn roundtrip_keep_alive() {
        assert_roundtrip(RendererCommand::KeepAlive(KeepAlive {}));
    }

    #[test]
    fn roundtrip_renderer_shutdown() {
        assert_roundtrip(RendererCommand::RendererShutdown(RendererShutdown {}));
    }

    #[test]
    fn roundtrip_frame_submit_default() {
        assert_roundtrip(RendererCommand::FrameSubmitData(FrameSubmitData::default()));
    }

    #[test]
    fn roundtrip_free_shared_memory_view() {
        assert_roundtrip(RendererCommand::FreeSharedMemoryView(
            FreeSharedMemoryView { buffer_id: 42 },
        ));
    }
}
