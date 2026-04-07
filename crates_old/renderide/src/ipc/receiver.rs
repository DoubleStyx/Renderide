//! IPC command receiver: polls subscribers and encodes outgoing commands.
//!
//! Extension point for IPC message handling.

use std::time::Duration;

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use crate::diagnostics::{DropLogEvent, ThrottledDropLog};
use crate::session::init::{ConnectionParams, InitError, get_connection_parameters};
use crate::shared::RendererCommand;
use crate::shared::decode_renderer_command;
use crate::shared::default_entity_pool::DefaultEntityPool;
use crate::shared::memory_packer::MemoryPacker;
use crate::shared::memory_unpacker::MemoryUnpacker;
use crate::shared::polymorphic_memory_packable_entity::PolymorphicEncode;

/// Minimum interval between aggregated "queue full" summaries per channel.
const IPC_QUEUE_DROP_LOG_INTERVAL: Duration = Duration::from_secs(2);

/// Polls IPC queues and decodes incoming commands.
pub struct CommandReceiver {
    primary_subscriber: Option<Subscriber>,
    background_subscriber: Option<Subscriber>,
    primary_publisher: Option<Publisher>,
    background_publisher: Option<Publisher>,
    send_buffer: Vec<u8>,
    primary_drop_throttle: ThrottledDropLog,
    background_drop_throttle: ThrottledDropLog,
}

impl CommandReceiver {
    /// Creates a new receiver (not yet connected).
    pub fn new() -> Self {
        Self {
            primary_subscriber: None,
            background_subscriber: None,
            primary_publisher: None,
            background_publisher: None,
            send_buffer: vec![0u8; 65536],
            primary_drop_throttle: ThrottledDropLog::new(IPC_QUEUE_DROP_LOG_INTERVAL),
            background_drop_throttle: ThrottledDropLog::new(IPC_QUEUE_DROP_LOG_INTERVAL),
        }
    }

    /// Connects to IPC queues. Returns Ok(()) on success.
    /// When no connection params, leaves subscribers/publisher as None (standalone mode).
    pub fn connect(&mut self) -> Result<(), InitError> {
        let params = match get_connection_parameters() {
            Some(p) => p,
            None => return Ok(()),
        };

        let primary_sub = create_subscriber(&params, "Primary")?;
        let background_sub = create_subscriber(&params, "Background")?;
        let primary_pub = create_publisher(&params, "Primary")?;
        let background_pub = create_publisher(&params, "Background")?;

        self.primary_subscriber = Some(primary_sub);
        self.background_subscriber = Some(background_sub);
        self.primary_publisher = Some(primary_pub);
        self.background_publisher = Some(background_pub);
        Ok(())
    }

    /// Whether the receiver is connected to IPC.
    pub fn is_connected(&self) -> bool {
        self.primary_subscriber.is_some()
    }

    /// Polls both subscribers and returns all decoded commands.
    pub fn poll(&mut self) -> Vec<RendererCommand> {
        let mut commands = Vec::new();
        let mut pool = DefaultEntityPool;

        if let Some(ref mut s) = self.primary_subscriber {
            while let Some(msg) = s.try_dequeue() {
                let mut unpacker = MemoryUnpacker::new(&msg, &mut pool);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    decode_renderer_command(&mut unpacker)
                })) {
                    Ok(cmd) => commands.push(cmd),
                    Err(e) => logger::log_panic_payload(e, "IPC decode panic (primary)"),
                }
            }
        }
        if let Some(ref mut s) = self.background_subscriber {
            while let Some(msg) = s.try_dequeue() {
                let mut unpacker = MemoryUnpacker::new(&msg, &mut pool);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    decode_renderer_command(&mut unpacker)
                })) {
                    Ok(cmd) => commands.push(cmd),
                    Err(e) => logger::log_panic_payload(e, "IPC decode panic (background)"),
                }
            }
        }
        commands
    }

    /// Sends a command to the primary queue (frame data, init result, etc.).
    pub fn send(&mut self, mut cmd: RendererCommand) {
        let total_len = self.send_buffer.len();
        let written = {
            let mut packer = MemoryPacker::new(&mut self.send_buffer[..]);
            cmd.encode(&mut packer);
            total_len - packer.remaining_len()
        };
        if written > 0
            && let Some(ref mut pub_) = self.primary_publisher
            && !pub_.try_enqueue(&self.send_buffer[..written])
            && let Some(ev) = self.primary_drop_throttle.record_drop(written)
        {
            match ev {
                DropLogEvent::First { bytes } => {
                    logger::warn!(
                        "IPC primary queue full, dropped outgoing command ({} bytes)",
                        bytes
                    );
                }
                DropLogEvent::Burst { count, bytes } => {
                    logger::warn!(
                        "IPC primary queue full, dropped {} additional outgoing command(s) ({} bytes) since last log",
                        count,
                        bytes
                    );
                }
            }
        }
    }

    /// Sends an asset result command to the background queue (MeshUploadResult, etc.).
    /// Must match the channel the host uses for asset updates.
    pub fn send_background(&mut self, mut cmd: RendererCommand) {
        let total_len = self.send_buffer.len();
        let written = {
            let mut packer = MemoryPacker::new(&mut self.send_buffer[..]);
            cmd.encode(&mut packer);
            total_len - packer.remaining_len()
        };
        if written > 0
            && let Some(ref mut pub_) = self.background_publisher
            && !pub_.try_enqueue(&self.send_buffer[..written])
            && let Some(ev) = self.background_drop_throttle.record_drop(written)
        {
            match ev {
                DropLogEvent::First { bytes } => {
                    logger::warn!(
                        "IPC background queue full, dropped outgoing command ({} bytes)",
                        bytes
                    );
                }
                DropLogEvent::Burst { count, bytes } => {
                    logger::warn!(
                        "IPC background queue full, dropped {} additional outgoing command(s) ({} bytes) since last log",
                        count,
                        bytes
                    );
                }
            }
        }
    }
}

impl Default for CommandReceiver {
    fn default() -> Self {
        Self::new()
    }
}

fn create_subscriber(params: &ConnectionParams, suffix: &str) -> Result<Subscriber, InitError> {
    let queue_name = format!("{}{}A", params.queue_name, suffix);
    let options = QueueOptions::new(&queue_name, params.queue_capacity);
    let factory = QueueFactory::new();
    factory
        .create_subscriber(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

fn create_publisher(params: &ConnectionParams, suffix: &str) -> Result<Publisher, InitError> {
    let queue_name = format!("{}{}S", params.queue_name, suffix);
    let options = QueueOptions::new(&queue_name, params.queue_capacity);
    let factory = QueueFactory::new();
    factory
        .create_publisher(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}
