//! Binary layout shared with the managed interprocess queue implementation.

use std::mem::size_of;
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};

/// Offset of the ring buffer from the start of the mapping (after [`QueueHeader`]).
pub const BUFFER_BYTE_OFFSET: usize = size_of::<QueueHeader>();

/// Fixed header at the start of the mapped queue file; ring bytes follow at the byte offset given by the crate-root `BUFFER_BYTE_OFFSET` constant.
///
/// `read_offset` / `write_offset` use logical positions that may wrap using modulo `capacity`.
/// Fields use [`AtomicI64`] so lock-free access matches the same sizes and alignment as `i64` in the managed layout.
#[repr(C)]
pub struct QueueHeader {
    /// Logical read position in the ring (may exceed `capacity`; use modulo for physical index).
    pub read_offset: AtomicI64,
    /// Logical write position in the ring.
    pub write_offset: AtomicI64,
    /// Lock timestamp in 100 ns ticks (same epoch as `System.DateTime.UtcNow.Ticks` in .NET).
    pub read_lock_timestamp: AtomicI64,
    /// Reserved for alignment / future use.
    pub reserved: AtomicI64,
}

impl Default for QueueHeader {
    fn default() -> Self {
        Self {
            read_offset: AtomicI64::new(0),
            write_offset: AtomicI64::new(0),
            read_lock_timestamp: AtomicI64::new(0),
            reserved: AtomicI64::new(0),
        }
    }
}

impl QueueHeader {
    /// Returns `true` when no message is queued.
    pub fn is_empty(&self) -> bool {
        self.read_offset.load(Ordering::SeqCst) == self.write_offset.load(Ordering::SeqCst)
    }
}

/// Per-message prefix (8 bytes) at the start of each slot in the ring; body follows, then padding to eight bytes.
///
/// `state` uses [`AtomicI32`] for compare-exchange with the subscriber; `body_length` is written by the publisher before `state` becomes [`STATE_READY`].
#[repr(C)]
pub struct MessageHeader {
    /// See [`STATE_WRITING`], [`STATE_LOCKED`], [`STATE_READY`].
    pub state: AtomicI32,
    /// Payload length in bytes.
    pub body_length: i32,
}

/// Publisher is writing the message body.
pub const STATE_WRITING: i32 = 0;
/// Subscriber holds the message for consumption.
pub const STATE_LOCKED: i32 = 1;
/// Message is ready for the subscriber.
pub const STATE_READY: i32 = 2;

/// Little-endian wire bytes for the 8-byte [`MessageHeader`] prefix in the ring (matches `i32`/`i32` layout).
pub(crate) fn message_header_wire_bytes(state: i32, body_length: i32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0..4].copy_from_slice(&state.to_le_bytes());
    b[4..8].copy_from_slice(&body_length.to_le_bytes());
    b
}

/// Returns the wire size of a message (header + body + padding) for a given body length.
pub fn padded_message_length(body_len: i64) -> i64 {
    let total = size_of::<MessageHeader>() as i64 + body_len;
    ((total + 7) / 8) * 8
}

/// Byte offset from the start of a message slot to its body.
pub const MESSAGE_BODY_OFFSET: i64 = size_of::<MessageHeader>() as i64;

/// Ticks per second in the same 100 ns unit as .NET `DateTime.Ticks`.
pub const TICKS_PER_SECOND: i64 = 10_000_000;

/// Ten seconds expressed in 100 ns ticks (matches dequeue lock / spin timeouts).
pub const TICKS_FOR_TEN_SECONDS: i64 = 10 * TICKS_PER_SECOND;

#[cfg(test)]
mod tests {
    use std::mem::align_of;
    use std::mem::offset_of;
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn header_sizes_match_contract() {
        assert_eq!(size_of::<QueueHeader>(), 32);
        assert_eq!(size_of::<MessageHeader>(), 8);
        assert_eq!(align_of::<QueueHeader>(), 8);
        assert_eq!(align_of::<MessageHeader>(), 4);
    }

    #[test]
    fn queue_header_field_offsets_match_i64_layout() {
        assert_eq!(offset_of!(QueueHeader, read_offset), 0);
        assert_eq!(offset_of!(QueueHeader, write_offset), 8);
        assert_eq!(offset_of!(QueueHeader, read_lock_timestamp), 16);
        assert_eq!(offset_of!(QueueHeader, reserved), 24);
    }

    #[test]
    fn message_header_field_offsets_match_i32_layout() {
        assert_eq!(offset_of!(MessageHeader, state), 0);
        assert_eq!(offset_of!(MessageHeader, body_length), 4);
    }

    #[test]
    fn padded_length_aligns_to_eight() {
        for body in [0i64, 1, 7, 8, 9, 100] {
            let p = padded_message_length(body);
            assert_eq!(p % 8, 0);
            assert!(p >= size_of::<MessageHeader>() as i64 + body);
        }
    }

    #[test]
    fn padded_message_length_exact_for_small_bodies() {
        assert_eq!(padded_message_length(0), 8);
        assert_eq!(padded_message_length(1), 16);
        assert_eq!(padded_message_length(8), 16);
        assert_eq!(padded_message_length(9), 24);
    }

    #[test]
    fn queue_header_is_empty_when_offsets_equal() {
        let h = QueueHeader::default();
        assert!(h.is_empty());
        h.read_offset.store(0, Ordering::SeqCst);
        h.write_offset.store(8, Ordering::SeqCst);
        assert!(!h.is_empty());
    }
}
