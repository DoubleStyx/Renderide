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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_polymorphic_decode_error_routes_to_polymorphic_variant() {
        let inner = PolymorphicDecodeError {
            discriminator: 7,
            union: "RendererCommand",
        };
        let err: WireDecodeError = inner.into();
        match err {
            WireDecodeError::Polymorphic(e) => assert_eq!(e, inner),
            WireDecodeError::Unpack(_) => panic!("expected Polymorphic variant"),
        }
    }

    #[test]
    fn from_memory_unpack_error_routes_to_unpack_variant() {
        let inner = MemoryUnpackError::LengthOverflow;
        let err: WireDecodeError = inner.into();
        match err {
            WireDecodeError::Unpack(e) => assert_eq!(e, inner),
            WireDecodeError::Polymorphic(_) => panic!("expected Unpack variant"),
        }
    }

    #[test]
    fn display_is_transparent_for_polymorphic() {
        let inner = PolymorphicDecodeError {
            discriminator: -3,
            union: "Foo",
        };
        let wire: WireDecodeError = inner.into();
        assert_eq!(wire.to_string(), inner.to_string());
    }

    #[test]
    fn display_is_transparent_for_unpack() {
        let inner = MemoryUnpackError::Underrun {
            ty: "i32",
            needed: 4,
            remaining: 1,
        };
        let wire: WireDecodeError = inner.into();
        assert_eq!(wire.to_string(), inner.to_string());
    }
}
