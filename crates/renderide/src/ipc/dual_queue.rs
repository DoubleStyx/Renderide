//! Dual-queue IPC: Primary and Background subscriber/publisher pairs for [`RendererCommand`].
//!
//! Naming matches the managed client when the renderer is **non-authority**: subscribe on `…A`,
//! publish on `…S` (see `Renderite.Shared.MessagingManager.Connect`).

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use crate::connection::{publisher_queue_name, subscriber_queue_name, ConnectionParams, InitError};
use crate::shared::{
    decode_renderer_command, default_entity_pool::DefaultEntityPool, memory_packer::MemoryPacker,
    memory_unpacker::MemoryUnpacker, polymorphic_memory_packable_entity::PolymorphicEncode,
    RendererCommand,
};

const SEND_BUFFER_CAP: usize = 65536;

/// Host ↔ renderer IPC over two Cloudtoid queue pairs (Primary and Background).
pub struct DualQueueIpc {
    primary_subscriber: Subscriber,
    background_subscriber: Subscriber,
    primary_publisher: Publisher,
    background_publisher: Publisher,
    send_buffer: Vec<u8>,
    /// Count of dropped primary sends since last log (for burst summary).
    primary_drops_since_log: u32,
    background_drops_since_log: u32,
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
            send_buffer: vec![0u8; SEND_BUFFER_CAP],
            primary_drops_since_log: 0,
            background_drops_since_log: 0,
        })
    }

    /// Drains both subscribers and returns decoded commands (Primary first, then Background; each
    /// channel fully drained in order).
    pub fn poll(&mut self) -> Vec<RendererCommand> {
        let mut commands = Vec::new();
        drain_subscriber(&mut self.primary_subscriber, &mut commands);
        drain_subscriber(&mut self.background_subscriber, &mut commands);
        commands
    }

    /// Encodes and sends a command on the **Primary** publisher (frame handshake, init, etc.).
    pub fn send_primary(&mut self, mut cmd: RendererCommand) {
        let written = encode_command(&mut cmd, &mut self.send_buffer);
        if written == 0 {
            return;
        }
        if self
            .primary_publisher
            .try_enqueue(&self.send_buffer[..written])
        {
            self.primary_drops_since_log = 0;
            return;
        }
        self.primary_drops_since_log += 1;
        if self.primary_drops_since_log == 1 {
            logger::warn!(
                "IPC primary queue full, dropped outgoing command ({} bytes)",
                written
            );
        } else if self.primary_drops_since_log.is_multiple_of(128) {
            logger::warn!(
                "IPC primary queue full: {} additional drops since last summary",
                128
            );
        }
    }

    /// Encodes and sends a command on the **Background** publisher (asset results, etc.).
    pub fn send_background(&mut self, mut cmd: RendererCommand) {
        let written = encode_command(&mut cmd, &mut self.send_buffer);
        if written == 0 {
            return;
        }
        if self
            .background_publisher
            .try_enqueue(&self.send_buffer[..written])
        {
            self.background_drops_since_log = 0;
            return;
        }
        self.background_drops_since_log += 1;
        if self.background_drops_since_log == 1 {
            logger::warn!(
                "IPC background queue full, dropped outgoing command ({} bytes)",
                written
            );
        } else if self.background_drops_since_log.is_multiple_of(128) {
            logger::warn!(
                "IPC background queue full: {} additional drops since last summary",
                128
            );
        }
    }
}

fn encode_command(cmd: &mut RendererCommand, buf: &mut [u8]) -> usize {
    let total_len = buf.len();
    let mut packer = MemoryPacker::new(buf);
    cmd.encode(&mut packer);
    total_len - packer.remaining_len()
}

fn drain_subscriber(sub: &mut Subscriber, out: &mut Vec<RendererCommand>) {
    while let Some(msg) = sub.try_dequeue() {
        let mut pool = DefaultEntityPool;
        let mut unpacker = MemoryUnpacker::new(&msg, &mut pool);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decode_renderer_command(&mut unpacker)
        })) {
            Ok(cmd) => out.push(cmd),
            Err(e) => logger::log_panic_payload(e, "IPC decode panic"),
        }
    }
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
