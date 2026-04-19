//! Errors when a polymorphic `i32` discriminator does not match any known variant.

use thiserror::Error;

/// Discriminator read from the wire did not match any known variant for the tagged union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid polymorphic tag {discriminator} for {union}")]
pub struct PolymorphicDecodeError {
    /// Raw `i32` tag from the buffer.
    pub discriminator: i32,
    /// Union name for diagnostics (for example `RendererCommand`).
    pub union: &'static str,
}
