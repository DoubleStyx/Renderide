//! IPC command receiver: polls subscribers and encodes outgoing commands.

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use crate::init::{get_connection_parameters, ConnectionParams, InitError};
use crate::shared::decode_renderer_command;
use crate::shared::default_entity_pool::DefaultEntityPool;
use crate::shared::memory_packer::MemoryPacker;
use crate::shared::memory_unpacker::MemoryUnpacker;
use crate::shared::polymorphic_memory_packable_entity::PolymorphicEncode;
use crate::shared::RendererCommand;

/// Polls IPC queues and decodes incoming commands.
pub struct CommandReceiver {
    primary_subscriber: Option<Subscriber>,
    background_subscriber: Option<Subscriber>,
    primary_publisher: Option<Publisher>,
    background_publisher: Option<Publisher>,
    send_buffer: Vec<u8>,
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
        }
    }

    /// Connects to IPC queues. Returns Ok(()) on success.
    /// When no connection params, leaves subscribers/publisher as None (standalone mode).
    pub fn connect(&mut self) -> Result<(), InitError> {
        let params = match get_connection_parameters() {
            Some(p) => p,
            None => return Ok(()),
        };

        crate::log::log_write(&format!(
            "[Renderide] Connecting to queue {} (capacity={})",
            params.queue_name, params.queue_capacity
        ));

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
                if let Ok(cmd) =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        decode_renderer_command(&mut unpacker)
                    }))
                {
                    commands.push(cmd);
                }
            }
        }
        if let Some(ref mut s) = self.background_subscriber {
            while let Some(msg) = s.try_dequeue() {
                let mut unpacker = MemoryUnpacker::new(&msg, &mut pool);
                if let Ok(cmd) =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        decode_renderer_command(&mut unpacker)
                    }))
                {
                    commands.push(cmd);
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
        if written > 0 {
            if let Some(ref mut pub_) = self.primary_publisher {
                let _ = pub_.try_enqueue(&self.send_buffer[..written]);
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
        if written > 0 {
            if let Some(ref mut pub_) = self.background_publisher {
                let _ = pub_.try_enqueue(&self.send_buffer[..written]);
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
    Ok(factory.create_subscriber(options))
}

fn create_publisher(params: &ConnectionParams, suffix: &str) -> Result<Publisher, InitError> {
    let queue_name = format!("{}{}S", params.queue_name, suffix);
    let options = QueueOptions::new(&queue_name, params.queue_capacity);
    let factory = QueueFactory::new();
    Ok(factory.create_publisher(options))
}
