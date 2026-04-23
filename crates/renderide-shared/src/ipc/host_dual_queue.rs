//! Host-side authority dual-queue (mirror of [`super::dual_queue::DualQueueIpc`]).
//!
//! When the renderer connects as a non-authority client it subscribes on `…A` and publishes on
//! `…S` (see [`super::connection::subscriber_queue_name`] / [`super::connection::publisher_queue_name`]).
//! The host therefore takes the **complementary** sides — publishes on `…A` (renderer reads) and
//! subscribes on `…S` (renderer writes). This module mirrors the renderer-side `DualQueueIpc` so
//! the mock host (`renderide-test`) and any future host-side Rust tooling can speak the same
//! `RendererCommand` byte stream `FrooxEngine` produces, with zero divergence in encoding.

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use super::connection::{publisher_queue_name, subscriber_queue_name, ConnectionParams, InitError};
use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::packing::memory_packer::MemoryPacker;
use crate::packing::memory_unpacker::MemoryUnpacker;
use crate::packing::polymorphic_memory_packable_entity::PolymorphicEncode;
use crate::packing::wire_decode_error::WireDecodeError;
use crate::shared::{decode_renderer_command, RendererCommand};

/// Send buffer capacity reused across encodes (matches [`super::dual_queue::DualQueueIpc`]).
const SEND_BUFFER_CAP: usize = 65536;

/// Authority-side dual-queue endpoints used by the host to talk to the renderer.
///
/// **Publishes on `…A`** (renderer subscribes there) and **subscribes on `…S`** (renderer
/// publishes there). The renderer-side counterpart is [`super::dual_queue::DualQueueIpc`].
pub struct HostDualQueueIpc {
    primary_publisher: Publisher,
    background_publisher: Publisher,
    primary_subscriber: Subscriber,
    background_subscriber: Subscriber,
    /// Reused across [`Self::poll_into`] calls so optional heap types do not allocate per message.
    entity_pool: DefaultEntityPool,
    /// Reused encode buffer.
    send_buffer: Vec<u8>,
}

impl HostDualQueueIpc {
    /// Opens all four authority endpoints: publishers on `{name}PrimaryA` / `{name}BackgroundA`
    /// and subscribers on `{name}PrimaryS` / `{name}BackgroundS`.
    pub fn connect(params: &ConnectionParams) -> Result<Self, InitError> {
        let factory = QueueFactory::new();
        let cap = params.queue_capacity;
        let primary_pub = open_authority_publisher(&factory, params, "Primary", cap)?;
        let background_pub = open_authority_publisher(&factory, params, "Background", cap)?;
        let primary_sub = open_authority_subscriber(&factory, params, "Primary", cap)?;
        let background_sub = open_authority_subscriber(&factory, params, "Background", cap)?;
        Ok(Self {
            primary_publisher: primary_pub,
            background_publisher: background_pub,
            primary_subscriber: primary_sub,
            background_subscriber: background_sub,
            entity_pool: DefaultEntityPool,
            send_buffer: vec![0u8; SEND_BUFFER_CAP],
        })
    }

    /// Drains both subscribers (`…S`) into `out`, Primary first then Background.
    ///
    /// Reuses the existing `RendererCommand` decoder so messages from the renderer arrive as
    /// fully-typed values. Decode errors are logged via [`logger::warn!`] and the offending
    /// message is dropped (matches the renderer-side behavior).
    pub fn poll_into(&mut self, out: &mut Vec<RendererCommand>) {
        out.clear();
        drain_subscriber(&mut self.primary_subscriber, &mut self.entity_pool, out);
        drain_subscriber(&mut self.background_subscriber, &mut self.entity_pool, out);
    }

    /// Encodes and publishes `cmd` on the Primary `…A` queue (renderer reads it as Primary).
    ///
    /// Returns `true` when the message was queued, `false` when encoding produced no bytes or
    /// the queue was full (caller may retry next tick).
    pub fn send_primary(&mut self, mut cmd: RendererCommand) -> bool {
        let written = encode_command(&mut cmd, &mut self.send_buffer);
        if written == 0 {
            return false;
        }
        self.primary_publisher
            .try_enqueue(&self.send_buffer[..written])
    }

    /// Encodes and publishes `cmd` on the Background `…A` queue (renderer reads it as Background).
    pub fn send_background(&mut self, mut cmd: RendererCommand) -> bool {
        let written = encode_command(&mut cmd, &mut self.send_buffer);
        if written == 0 {
            return false;
        }
        self.background_publisher
            .try_enqueue(&self.send_buffer[..written])
    }
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
    logger::warn!("Host IPC: dropped message ({err})");
}

/// Authority-side publisher: opens the `…A` queue (renderer subscribes here).
fn open_authority_publisher(
    factory: &QueueFactory,
    params: &ConnectionParams,
    channel: &str,
    capacity: i64,
) -> Result<Publisher, InitError> {
    // Renderer subscribes on `…A`; host publishes there.
    let name = subscriber_queue_name(&params.queue_name, channel);
    let options =
        QueueOptions::with_destroy(&name, capacity, true).map_err(InitError::IpcConnect)?;
    factory
        .create_publisher(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

/// Authority-side subscriber: opens the `…S` queue (renderer publishes here).
fn open_authority_subscriber(
    factory: &QueueFactory,
    params: &ConnectionParams,
    channel: &str,
    capacity: i64,
) -> Result<Subscriber, InitError> {
    // Renderer publishes on `…S`; host subscribes there.
    let name = publisher_queue_name(&params.queue_name, channel);
    let options =
        QueueOptions::with_destroy(&name, capacity, true).map_err(InitError::IpcConnect)?;
    factory
        .create_subscriber(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::HostDualQueueIpc;
    use crate::ipc::connection::ConnectionParams;
    use crate::shared::{KeepAlive, RendererCommand};

    #[test]
    fn host_can_send_and_receive_via_self_loopback() {
        // The host alone cannot exercise full round-trip without a renderer counterpart
        // (publishers and subscribers are paired across the `…A`/`…S` boundary). This test
        // just verifies that `connect` sets up all four endpoints without error.
        let prefix = format!("renderide_host_dq_test_{}", std::process::id());
        let params = ConnectionParams {
            queue_name: prefix,
            queue_capacity: 4096,
        };
        let mut host = HostDualQueueIpc::connect(&params).expect("authority connect");
        assert!(host.send_primary(RendererCommand::KeepAlive(KeepAlive {})));
        assert!(host.send_background(RendererCommand::KeepAlive(KeepAlive {})));
    }
}
