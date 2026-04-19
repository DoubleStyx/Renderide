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
