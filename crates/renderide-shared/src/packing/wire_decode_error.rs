//! Unified decode failure for generated wire payloads (tag mismatch or truncated buffer).

use thiserror::Error;

use super::memory_unpack_error::MemoryUnpackError;
use super::polymorphic_decode_error::PolymorphicDecodeError;

/// Error returned when decoding a [`crate::shared::RendererCommand`] or nested polymorphic payload fails.
#[derive(Debug, Error)]
pub enum WireDecodeError {
    /// Discriminator did not match any known variant for the tagged union.
    #[error(transparent)]
    Polymorphic(#[from] PolymorphicDecodeError),
    /// The buffer ended before a typed field could be read.
    #[error(transparent)]
    Unpack(#[from] MemoryUnpackError),
}
