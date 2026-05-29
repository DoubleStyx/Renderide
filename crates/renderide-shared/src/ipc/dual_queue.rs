//! Dual-queue IPC: Primary and Background subscriber/publisher pairs for [`RendererCommand`].
//!
//! Naming matches the host client when the renderer is **non-authority**: subscribe on `...A`,
//! publish on `...S`.

use std::time::Duration;

use interprocess::{Publisher, QueueFactory, Subscriber};

use super::connection::{ConnectionParams, InitError, publisher_queue_name, subscriber_queue_name};
use super::dual_queue_reliable_outbox::ReliableBackgroundOutbox;
pub use super::dual_queue_shared::IpcDrainStats;
use super::dual_queue_shared::{drain_subscriber, encode_command, open_publisher, open_subscriber};
use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::shared::RendererCommand;

const SEND_BUFFER_CAP: usize = 65536;

/// Log prefix used when [`encode_command`] overflows the send buffer on the renderer side.
const ENCODE_OVERFLOW_LOG_PREFIX: &str = "IPC outgoing send: encode overflow";

/// Log prefix used when [`drain_subscriber`] decodes an invalid command on the renderer side.
const INVALID_MESSAGE_LOG_PREFIX: &str = "IPC";

/// After this many consecutive `try_enqueue` failures on one channel, log at [`logger::error!`].
const IPC_CONSECUTIVE_DROP_ERROR_AFTER: u32 = 16;

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
    primary_subscriber: Subscriber,
    background_subscriber: Subscriber,
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

        let primary_sub = open_subscriber(factory, &primary_sub_name, cap, None, false)?;
        let background_sub = open_subscriber(factory, &background_sub_name, cap, None, false)?;
        let primary_pub = open_publisher(factory, &primary_pub_name, cap, None, false)?;
        let background_pub = open_publisher(factory, &background_pub_name, cap, None, false)?;

        Ok(Self {
            primary_subscriber: primary_sub,
            background_subscriber: background_sub,
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
        let _stats = self.poll_into_profiled(out);
    }

    /// Drains both subscribers into `out` and returns diagnostic counters for the drain.
    pub fn poll_into_profiled(&mut self, out: &mut Vec<RendererCommand>) -> DualQueueDrainStats {
        out.clear();
        self.flush_reliable_outbound();
        let primary = {
            profiling::scope!("ipc::primary_drain");
            drain_subscriber(
                &mut self.primary_subscriber,
                &mut self.entity_pool,
                out,
                INVALID_MESSAGE_LOG_PREFIX,
            )
        };
        let background = {
            profiling::scope!("ipc::background_drain");
            drain_subscriber(
                &mut self.background_subscriber,
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
        self.poll_into_after_primary_wait_profiled(out, timeout)
            .waited
    }

    /// Waits up to `timeout` for primary inbound work and returns diagnostic counters.
    pub fn poll_into_after_primary_wait_profiled(
        &mut self,
        out: &mut Vec<RendererCommand>,
        timeout: Duration,
    ) -> DualQueuePollStats {
        let initial_drain = {
            profiling::scope!("ipc::immediate_drain");
            self.poll_into_profiled(out)
        };
        if !out.is_empty() || timeout.is_zero() {
            return DualQueuePollStats {
                initial_drain,
                ..DualQueuePollStats::default()
            };
        }
        let wait_started = std::time::Instant::now();
        let ready = {
            profiling::scope!("ipc::host_wait::primary_queue");
            profiling::scope!("ipc::primary_wait");
            self.primary_subscriber.wait_for_message_timeout(timeout)
        };
        let waited = wait_started.elapsed();
        if !ready {
            return DualQueuePollStats {
                waited,
                initial_drain,
                timed_out: true,
                ..DualQueuePollStats::default()
            };
        }
        let post_wait_drain = {
            profiling::scope!("ipc::post_wait_drain");
            self.poll_into_profiled(out)
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
    pub fn send_background_reliable(&mut self, mut cmd: RendererCommand) -> bool {
        let written = encode_command(&mut cmd, &mut self.send_buffer, ENCODE_OVERFLOW_LOG_PREFIX);
        if written == 0 {
            return false;
        }
        self.reliable_background_outbox
            .enqueue(self.send_buffer[..written].to_vec());
        let pending_count = self.reliable_background_outbox.len();
        let pending_bytes = self.reliable_background_outbox.pending_bytes();
        if pending_count == 64 || (pending_count > 64 && pending_count.is_multiple_of(64)) {
            logger::warn!(
                "IPC reliable background outbox pressure: pending_messages={} pending_bytes={}",
                pending_count,
                pending_bytes
            );
        }
        self.flush_reliable_outbound();
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
