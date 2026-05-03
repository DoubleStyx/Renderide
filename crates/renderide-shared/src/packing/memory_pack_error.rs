//! Error returned when a [`super::memory_packer::MemoryPacker`] runs out of buffer space.

use thiserror::Error;

/// Failure encountered while packing into a fixed-size IPC buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryPackError {
    /// One or more writes were skipped because the destination buffer ran out of space.
    ///
    /// The first `needed`/`remaining` pair is captured at the point of overflow; subsequent
    /// writes silently no-op so the encoder cursor remains coherent.
    #[error(
        "packer buffer too small: needed {needed} byte(s) for {ty}, {remaining} byte(s) remaining"
    )]
    BufferTooSmall {
        /// Short type name of the value whose write first ran out of room.
        ty: &'static str,
        /// Bytes the offending write required.
        needed: usize,
        /// Bytes still free at the moment of the offending write.
        remaining: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_too_small_display_is_stable() {
        let err = MemoryPackError::BufferTooSmall {
            ty: "i32",
            needed: 4,
            remaining: 1,
        };
        assert_eq!(
            err.to_string(),
            "packer buffer too small: needed 4 byte(s) for i32, 1 byte(s) remaining"
        );
    }

    #[test]
    fn buffer_too_small_display_handles_zero_remaining() {
        let err = MemoryPackError::BufferTooSmall {
            ty: "u8",
            needed: 1,
            remaining: 0,
        };
        assert_eq!(
            err.to_string(),
            "packer buffer too small: needed 1 byte(s) for u8, 0 byte(s) remaining"
        );
    }

    #[test]
    fn equality_and_copy_round_trip() {
        let a = MemoryPackError::BufferTooSmall {
            ty: "T",
            needed: 8,
            remaining: 4,
        };
        let b = a;
        assert_eq!(a, b);

        let different = MemoryPackError::BufferTooSmall {
            ty: "T",
            needed: 8,
            remaining: 5,
        };
        assert_ne!(a, different);
    }
}
