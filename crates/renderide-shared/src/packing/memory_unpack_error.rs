//! Errors when an IPC byte buffer ends before a full value has been read.

use thiserror::Error;

/// Failure while advancing a [`super::memory_unpacker::MemoryUnpacker`] cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryUnpackError {
    /// Not enough bytes remained for the requested read.
    #[error("buffer underrun: need {needed} bytes for {ty}, {remaining} byte(s) remaining")]
    Underrun {
        /// Short type name for logs (e.g. `i32`).
        ty: &'static str,
        /// Bytes required for this read.
        needed: usize,
        /// Bytes left in the buffer.
        remaining: usize,
    },
    /// `count * size_of::<T>()` overflowed `usize`.
    #[error("length overflow for POD access")]
    LengthOverflow,
    /// A string field declared more UTF-16 code units than [`super::memory_unpacker::MAX_STRING_LEN`].
    #[error("string length {requested} exceeds MAX_STRING_LEN ({max})")]
    StringTooLong {
        /// UTF-16 code units the wire field requested.
        requested: usize,
        /// Cap enforced by the unpacker.
        max: usize,
    },
}

impl MemoryUnpackError {
    /// Underrun for a single POD `T` (uses `std::any::type_name` for diagnostics).
    pub fn pod_underrun<T>(needed: usize, remaining: usize) -> Self {
        Self::Underrun {
            ty: short_type_name::<T>(),
            needed,
            remaining,
        }
    }
}

fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn underrun_display_is_stable() {
        let err = MemoryUnpackError::Underrun {
            ty: "u32",
            needed: 4,
            remaining: 2,
        };
        assert_eq!(
            err.to_string(),
            "buffer underrun: need 4 bytes for u32, 2 byte(s) remaining"
        );
    }

    #[test]
    fn length_overflow_display_is_stable() {
        assert_eq!(
            MemoryUnpackError::LengthOverflow.to_string(),
            "length overflow for POD access"
        );
    }

    #[test]
    fn string_too_long_display_is_stable() {
        let err = MemoryUnpackError::StringTooLong {
            requested: 99_999,
            max: 1_024,
        };
        assert_eq!(
            err.to_string(),
            "string length 99999 exceeds MAX_STRING_LEN (1024)"
        );
    }

    #[test]
    fn pod_underrun_records_short_type_name_for_module_path() {
        let err = MemoryUnpackError::pod_underrun::<i32>(8, 3);
        assert_eq!(
            err,
            MemoryUnpackError::Underrun {
                ty: "i32",
                needed: 8,
                remaining: 3,
            }
        );
    }

    #[test]
    fn pod_underrun_handles_single_segment_type_name() {
        let err = MemoryUnpackError::pod_underrun::<u8>(1, 0);
        match err {
            MemoryUnpackError::Underrun {
                ty,
                needed,
                remaining,
            } => {
                assert_eq!(ty, "u8");
                assert_eq!(needed, 1);
                assert_eq!(remaining, 0);
            }
            _ => panic!("expected Underrun variant"),
        }
    }

    #[test]
    fn pod_underrun_strips_module_path_for_qualified_type() {
        struct LocalDummy;
        let err = MemoryUnpackError::pod_underrun::<LocalDummy>(2, 0);
        match err {
            MemoryUnpackError::Underrun { ty, .. } => {
                assert_eq!(ty, "LocalDummy");
            }
            _ => panic!("expected Underrun variant"),
        }
    }

    #[test]
    fn equality_and_copy_round_trip() {
        let a = MemoryUnpackError::Underrun {
            ty: "i64",
            needed: 8,
            remaining: 0,
        };
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, MemoryUnpackError::LengthOverflow);
    }
}
