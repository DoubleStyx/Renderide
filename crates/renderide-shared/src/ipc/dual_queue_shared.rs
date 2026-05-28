//! Helpers shared by [`super::dual_queue::DualQueueIpc`] (renderer side) and
//! [`super::host_dual_queue::HostDualQueueIpc`] (host side).
//!
//! Both wrappers encode `RendererCommand` payloads, drain subscribers, and open Cloudtoid queues
//! in the same way. The renderer-side wrapper additionally tracks backpressure and a reliable
//! background outbox (see [`super::dual_queue_reliable_outbox`]), while the host-side wrapper
//! inverts the `...A` / `...S` suffix convention so renderer and host meet on the same queues.
//! Per-side specifics live in the wrappers; the encode/drain/open primitives below are common.

use std::path::Path;
use std::time::{Duration, Instant};

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use super::connection::InitError;
use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::packing::memory_packer::MemoryPacker;
use crate::packing::memory_unpacker::MemoryUnpacker;
use crate::packing::polymorphic_memory_packable_entity::PolymorphicEncode;
use crate::packing::wire_decode_error::WireDecodeError;
use crate::shared::{RendererCommand, decode_renderer_command};

/// Diagnostic counters collected while draining one IPC subscriber.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IpcDrainStats {
    /// Successfully decoded command count.
    pub messages: usize,
    /// Total encoded payload bytes consumed from the queue, including invalid messages.
    pub bytes: usize,
    /// Messages dropped because their payload did not decode as a renderer command.
    pub invalid_messages: usize,
    /// Wall-clock time spent in renderer-command decode.
    pub decode_duration: Duration,
}

impl IpcDrainStats {
    /// Adds another drain sample into this one.
    pub fn add(&mut self, other: Self) {
        self.messages += other.messages;
        self.bytes += other.bytes;
        self.invalid_messages += other.invalid_messages;
        self.decode_duration += other.decode_duration;
    }
}

/// Encodes `cmd` into `buf`, returning the number of bytes written.
///
/// Returns `0` (and logs at [`logger::error!`] under `overflow_log_prefix`) when the encode buffer
/// was too small for the command. Callers treat `0` as "nothing to enqueue" -- sending a truncated
/// frame would surface as a confusing decoder underrun on the other side.
pub(super) fn encode_command(
    cmd: &mut RendererCommand,
    buf: &mut [u8],
    overflow_log_prefix: &'static str,
) -> usize {
    let total_len = buf.len();
    let mut packer = MemoryPacker::new(buf);
    cmd.encode(&mut packer);
    if let Some(err) = packer.overflow_error() {
        logger::error!(
            "{overflow_log_prefix} ({err}); dropping {} byte buffer",
            total_len
        );
        return 0;
    }
    total_len - packer.remaining_len()
}

/// Drains `sub` into `out`, decoding each message as a [`RendererCommand`].
///
/// Decode failures are logged at [`logger::warn!`] under `invalid_log_prefix` and the offending
/// message is dropped.
pub(super) fn drain_subscriber(
    sub: &mut Subscriber,
    pool: &mut DefaultEntityPool,
    out: &mut Vec<RendererCommand>,
    invalid_log_prefix: &'static str,
) -> IpcDrainStats {
    let mut stats = IpcDrainStats::default();
    while let Some(msg) = sub.try_dequeue() {
        stats.bytes += msg.len();
        let mut unpacker = MemoryUnpacker::new(&msg, pool);
        let decode_started = Instant::now();
        let decoded = {
            profiling::scope!("ipc::decode");
            decode_renderer_command(&mut unpacker)
        };
        stats.decode_duration += decode_started.elapsed();
        match decoded {
            Ok(cmd) => {
                stats.messages += 1;
                out.push(cmd);
            }
            Err(e) => {
                stats.invalid_messages += 1;
                log_invalid_renderer_command(invalid_log_prefix, e);
            }
        }
    }
    stats
}

/// Logs an invalid-renderer-command decode failure at [`logger::warn!`].
fn log_invalid_renderer_command(prefix: &'static str, err: WireDecodeError) {
    logger::warn!("{prefix}: dropped message ({err})");
}

/// Builds a [`QueueOptions`] for `name` with the given `capacity`, optional backing-directory
/// override, and destroy-on-drop flag.
///
/// `dir_override = None` resolves the backing directory via [`interprocess::default_memory_dir`].
/// `destroy_on_drop = true` is used by the queue owner (the host); clients (the renderer) leave
/// the queue alone so the owner controls its lifetime.
pub(super) fn queue_options(
    name: &str,
    capacity: i64,
    dir_override: Option<&Path>,
    destroy_on_drop: bool,
) -> Result<QueueOptions, InitError> {
    let result = match (dir_override, destroy_on_drop) {
        (Some(dir), true) => QueueOptions::with_path_and_destroy(name, dir, capacity, true),
        (Some(dir), false) => QueueOptions::with_path(name, dir, capacity),
        (None, true) => QueueOptions::with_destroy(name, capacity, true),
        (None, false) => QueueOptions::new(name, capacity),
    };
    result.map_err(InitError::IpcConnect)
}

/// Opens a [`Subscriber`] on the queue named `name`.
pub(super) fn open_subscriber(
    factory: QueueFactory,
    name: &str,
    capacity: i64,
    dir_override: Option<&Path>,
    destroy_on_drop: bool,
) -> Result<Subscriber, InitError> {
    let options = queue_options(name, capacity, dir_override, destroy_on_drop)?;
    factory
        .create_subscriber(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

/// Opens a [`Publisher`] on the queue named `name`.
pub(super) fn open_publisher(
    factory: QueueFactory,
    name: &str,
    capacity: i64,
    dir_override: Option<&Path>,
    destroy_on_drop: bool,
) -> Result<Publisher, InitError> {
    let options = queue_options(name, capacity, dir_override, destroy_on_drop)?;
    factory
        .create_publisher(options)
        .map_err(|e| InitError::IpcConnect(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{KeepAlive, RendererCommand, decode_renderer_command};

    const TEST_OVERFLOW_PREFIX: &str = "test::overflow";

    #[test]
    fn encode_command_returns_zero_when_buffer_too_small() {
        let mut cmd = RendererCommand::KeepAlive(KeepAlive {});
        let mut buf = [0u8; 1];
        let n = encode_command(&mut cmd, &mut buf, TEST_OVERFLOW_PREFIX);
        assert_eq!(n, 0, "buffer too small must signal nothing-to-enqueue");
    }

    #[test]
    fn encode_command_returns_byte_count_on_success_and_decodes_back() {
        let mut cmd = RendererCommand::KeepAlive(KeepAlive {});
        let mut buf = vec![0u8; 64];
        let n = encode_command(&mut cmd, &mut buf, TEST_OVERFLOW_PREFIX);
        assert!(n > 0, "should write at least one byte");
        assert!(n <= buf.len(), "must not exceed buffer length");

        let mut pool = DefaultEntityPool;
        let mut unpacker = MemoryUnpacker::new(&buf[..n], &mut pool);
        let decoded = decode_renderer_command(&mut unpacker).expect("decode");
        assert!(matches!(decoded, RendererCommand::KeepAlive(_)));
        assert_eq!(unpacker.remaining_data(), 0, "no trailing bytes");
    }
}
