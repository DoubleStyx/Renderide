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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_stable_for_positive_discriminator() {
        let err = PolymorphicDecodeError {
            discriminator: 42,
            union: "RendererCommand",
        };
        assert_eq!(
            err.to_string(),
            "invalid polymorphic tag 42 for RendererCommand"
        );
    }

    #[test]
    fn display_is_stable_for_negative_discriminator() {
        let err = PolymorphicDecodeError {
            discriminator: -1,
            union: "AssetUpdate",
        };
        assert_eq!(
            err.to_string(),
            "invalid polymorphic tag -1 for AssetUpdate"
        );
    }

    #[test]
    fn equality_and_copy_round_trip() {
        let a = PolymorphicDecodeError {
            discriminator: 7,
            union: "Foo",
        };
        let b = a;
        assert_eq!(a, b);

        let different = PolymorphicDecodeError {
            discriminator: 8,
            union: "Foo",
        };
        assert_ne!(a, different);
    }
}
