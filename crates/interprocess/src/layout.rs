//! Binary layout shared with the managed interprocess queue implementation.

use std::mem::size_of;

/// Offset of the ring buffer from the start of the mapping (after [`QueueHeader`]).
pub const BUFFER_BYTE_OFFSET: usize = size_of::<QueueHeader>();

/// Queue header size on the wire (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QueueHeader {
    /// Logical read position in the ring (may exceed `capacity`; use modulo for physical index).
    pub read_offset: i64,
    /// Logical write position in the ring.
    pub write_offset: i64,
    /// Lock timestamp in 100 ns ticks (same epoch as `System.DateTime.UtcNow.Ticks` in .NET).
    pub read_lock_timestamp: i64,
    /// Reserved for alignment / future use.
    pub reserved: i64,
}

impl QueueHeader {
    /// Returns `true` when no message is queued.
    pub fn is_empty(&self) -> bool {
        self.read_offset == self.write_offset
    }
}

/// Per-message header (8 bytes), immediately followed by the body then padding to eight bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MessageHeader {
    /// See [`STATE_WRITING`], [`STATE_LOCKED`], [`STATE_READY`].
    pub state: i32,
    /// Payload length in bytes.
    pub body_length: i32,
}

/// Publisher is writing the message body.
pub const STATE_WRITING: i32 = 0;
/// Subscriber holds the message for consumption.
pub const STATE_LOCKED: i32 = 1;
/// Message is ready for the subscriber.
pub const STATE_READY: i32 = 2;

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

    use super::*;

    #[test]
    fn header_sizes_match_contract() {
        assert_eq!(size_of::<QueueHeader>(), 32);
        assert_eq!(size_of::<MessageHeader>(), 8);
        assert_eq!(align_of::<QueueHeader>(), 8);
        assert_eq!(align_of::<MessageHeader>(), 4);
    }

    #[test]
    fn padded_length_aligns_to_eight() {
        for body in [0i64, 1, 7, 8, 9, 100] {
            let p = padded_message_length(body);
            assert_eq!(p % 8, 0);
            assert!(p >= size_of::<MessageHeader>() as i64 + body);
        }
    }
}
