//! Dual-queue IPC: Primary and Background subscriber/publisher pairs for [`RendererCommand`].
//!
//! Naming matches the managed client when the renderer is **non-authority**: subscribe on `…A`,
//! publish on `…S` (see `Renderite.Shared.MessagingManager.Connect`).

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use crate::connection::{publisher_queue_name, subscriber_queue_name, ConnectionParams, InitError};
use crate::shared::{
    decode_renderer_command, default_entity_pool::DefaultEntityPool, memory_packer::MemoryPacker,
    memory_unpacker::MemoryUnpacker, polymorphic_memory_packable_entity::PolymorphicEncode,
    RendererCommand, WireDecodeError,
};

const SEND_BUFFER_CAP: usize = 65536;

/// After this many consecutive `try_enqueue` failures on one channel, log at [`logger::error!`].
const IPC_CONSECUTIVE_DROP_ERROR_AFTER: u32 = 16;

/// Host ↔ renderer IPC over two Cloudtoid queue pairs (Primary and Background).
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

        let primary_sub = open_subscriber(&factory, params, "Primary", cap)?;
        let background_sub = open_subscriber(&factory, params, "Background", cap)?;
        let primary_pub = open_publisher(&factory, params, "Primary", cap)?;
        let background_pub = open_publisher(&factory, params, "Background", cap)?;

        Ok(Self {
            primary_subscriber: primary_sub,
            background_subscriber: background_sub,
            primary_publisher: primary_pub,
            background_publisher: background_pub,
            entity_pool: DefaultEntityPool,
            send_buffer: vec![0u8; SEND_BUFFER_CAP],
            primary_drops_since_log: 0,
            background_drops_since_log: 0,
            had_primary_outbound_drop_this_tick: false,
            had_background_outbound_drop_this_tick: false,
        })
    }

    /// Clears per-tick outbound drop flags; call once at the start of each winit frame tick.
    pub fn reset_outbound_drop_tick_flags(&mut self) {
        self.had_primary_outbound_drop_this_tick = false;
        self.had_background_outbound_drop_this_tick = false;
    }

    /// Whether any **primary** outbound send failed since the last [`Self::reset_outbound_drop_tick_flags`].
    pub fn had_outbound_primary_drop_this_tick(&self) -> bool {
        self.had_primary_outbound_drop_this_tick
    }

    /// Whether any **background** outbound send failed since the last [`Self::reset_outbound_drop_tick_flags`].
    pub fn had_outbound_background_drop_this_tick(&self) -> bool {
        self.had_background_outbound_drop_this_tick
    }

    /// Current consecutive primary-queue drop streak (resets on next successful enqueue).
    pub fn consecutive_primary_drop_streak(&self) -> u32 {
        self.primary_drops_since_log
    }

    /// Current consecutive background-queue drop streak (resets on next successful enqueue).
    pub fn consecutive_background_drop_streak(&self) -> u32 {
        self.background_drops_since_log
    }

    /// Drains both subscribers into `out` (Primary first, then Background; each channel fully drained in order).
    ///
    /// Clears `out` then drains both subscribers so each tick starts from an empty batch.
    pub fn poll_into(&mut self, out: &mut Vec<RendererCommand>) {
        out.clear();
        drain_subscriber(&mut self.primary_subscriber, &mut self.entity_pool, out);
        drain_subscriber(&mut self.background_subscriber, &mut self.entity_pool, out);
    }

    /// Encodes and sends a command on the **Primary** publisher (frame handshake, init, etc.).
    ///
    /// Returns `true` if the message was queued, `false` if encoding produced no bytes or the queue was full.
    pub fn send_primary(&mut self, mut cmd: RendererCommand) -> bool {
        let written = encode_command(&mut cmd, &mut self.send_buffer);
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
        let written = encode_command(&mut cmd, &mut self.send_buffer);
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

fn encode_command(cmd: &mut RendererCommand, buf: &mut [u8]) -> usize {
    let total_len = buf.len();
    let mut packer = MemoryPacker::new(buf);
    cmd.encode(&mut packer);
    total_len - packer.remaining_len()
}

fn drain_subscriber(
    sub: &mut Subscriber,
    pool: &mut DefaultEntityPool,
    out: &mut Vec<RendererCommand>,
) {
    while let Some(msg) = sub.try_dequeue() {
        let mut unpacker = MemoryUnpacker::new(&msg, pool);
        match decode_renderer_command(&mut unpacker) {
            Ok(cmd) => out.push(cmd),
            Err(e) => log_invalid_renderer_command(e),
        }
    }
}

fn log_invalid_renderer_command(err: WireDecodeError) {
    logger::warn!("IPC: dropped message ({err})");
}

fn open_subscriber(
    factory: &QueueFactory,
    params: &ConnectionParams,
    channel: &str,
    capacity: i64,
) -> Result<Subscriber, InitError> {
    let name = subscriber_queue_name(&params.queue_name, channel);
    let options = QueueOptions::new(&name, capacity).map_err(InitError::IpcConnect)?;
    factory
        .create_subscriber(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

fn open_publisher(
    factory: &QueueFactory,
    params: &ConnectionParams,
    channel: &str,
    capacity: i64,
) -> Result<Publisher, InitError> {
    let name = publisher_queue_name(&params.queue_name, channel);
    let options = QueueOptions::new(&name, capacity).map_err(InitError::IpcConnect)?;
    factory
        .create_publisher(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

#[cfg(test)]
mod renderer_command_roundtrip_tests {
    use super::encode_command;
    use crate::shared::{
        decode_renderer_command, default_entity_pool::DefaultEntityPool,
        memory_unpacker::MemoryUnpacker, FrameSubmitData, FreeSharedMemoryView, KeepAlive,
        RendererCommand, RendererShutdown,
    };

    fn assert_roundtrip(mut cmd: RendererCommand) {
        let expect = format!("{cmd:?}");
        let mut buf = vec![0u8; 65536];
        let n = encode_command(&mut cmd, &mut buf);
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
