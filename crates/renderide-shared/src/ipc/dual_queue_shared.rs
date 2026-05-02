//! Helpers shared by [`super::dual_queue::DualQueueIpc`] (renderer side) and
//! [`super::host_dual_queue::HostDualQueueIpc`] (host side).
//!
//! The two wrappers diverge in queue-suffix conventions and renderer-side backpressure tracking,
//! but they both encode `RendererCommand` payloads and drain subscribers in identical ways. The
//! only difference between their respective implementations was the log-label prefix used when
//! reporting overflow or invalid messages, so those prefixes are passed in as parameters here.

use interprocess::Subscriber;

use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::packing::memory_packer::MemoryPacker;
use crate::packing::memory_unpacker::MemoryUnpacker;
use crate::packing::polymorphic_memory_packable_entity::PolymorphicEncode;
use crate::packing::wire_decode_error::WireDecodeError;
use crate::shared::{RendererCommand, decode_renderer_command};

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
) {
    while let Some(msg) = sub.try_dequeue() {
        let mut unpacker = MemoryUnpacker::new(&msg, pool);
        match decode_renderer_command(&mut unpacker) {
            Ok(cmd) => out.push(cmd),
            Err(e) => log_invalid_renderer_command(invalid_log_prefix, e),
        }
    }
}

/// Logs an invalid-renderer-command decode failure at [`logger::warn!`].
fn log_invalid_renderer_command(prefix: &'static str, err: WireDecodeError) {
    logger::warn!("{prefix}: dropped message ({err})");
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
