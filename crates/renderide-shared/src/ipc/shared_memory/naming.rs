//! Memory view naming and Unix `.qu` backing paths.

#[cfg(unix)]
use std::path::PathBuf;

/// Environment variable overriding the Unix directory for `.qu` MMF files (must match host /
/// bootstrapper). Same value as `bootstrapper::ipc::RENDERIDE_INTERPROCESS_DIR_ENV`.
///
/// Only read by `unix_mmf_backing_dir` on Unix; Windows builds keep this symbol for API parity
/// with the bootstrapper constant name.
pub const RENDERIDE_INTERPROCESS_DIR_ENV: &str = "RENDERIDE_INTERPROCESS_DIR";

/// Maximum accepted shared-memory session prefix length.
pub const MAX_SHARED_MEMORY_PREFIX_LEN: usize = 64;

/// Reason a shared-memory session prefix cannot be used as a backing-name component.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(super) enum SharedMemoryPrefixValidationError {
    /// Prefix exceeds [`MAX_SHARED_MEMORY_PREFIX_LEN`].
    #[error("length {byte_len} exceeds max {max_byte_len}")]
    TooLong {
        /// Prefix length in bytes.
        byte_len: usize,
        /// Maximum accepted prefix length in bytes.
        max_byte_len: usize,
    },
    /// Prefix contains a byte outside the Renderite token-safe filename/object alphabet.
    #[error("invalid byte 0x{byte:02X} at offset {offset}")]
    InvalidByte {
        /// Byte offset of the rejected byte.
        offset: usize,
        /// Rejected byte.
        byte: u8,
    },
}

/// Validates that `prefix` is safe to use as one component of a shared-memory backing name.
pub(super) fn validate_shared_memory_prefix(
    prefix: &str,
) -> Result<(), SharedMemoryPrefixValidationError> {
    if prefix.len() > MAX_SHARED_MEMORY_PREFIX_LEN {
        return Err(SharedMemoryPrefixValidationError::TooLong {
            byte_len: prefix.len(),
            max_byte_len: MAX_SHARED_MEMORY_PREFIX_LEN,
        });
    }
    for (offset, byte) in prefix.bytes().enumerate() {
        if !is_valid_shared_memory_prefix_byte(byte) {
            return Err(SharedMemoryPrefixValidationError::InvalidByte { offset, byte });
        }
    }
    Ok(())
}

/// Returns whether `prefix` is safe to use as one component of a shared-memory backing name.
pub fn is_valid_shared_memory_prefix(prefix: &str) -> bool {
    validate_shared_memory_prefix(prefix).is_ok()
}

fn is_valid_shared_memory_prefix_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'+' | b'=')
}

/// Composes the memory view name per Renderite `Helper.ComposeMemoryViewName` (prefix + hex id).
pub fn compose_memory_view_name(prefix: &str, buffer_id: i32) -> String {
    debug_assert!(is_valid_shared_memory_prefix(prefix));
    format!("{prefix}_{buffer_id:X}")
}

/// Unix-only: resolved directory containing `{composed}.qu` backing files.
#[cfg(unix)]
pub(super) fn unix_mmf_backing_dir() -> PathBuf {
    std::env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map_or_else(interprocess::default_memory_dir, PathBuf::from)
}

/// Full path to the `.qu` file for a buffer on Unix.
#[cfg(unix)]
pub(super) fn unix_backing_file_path(prefix: &str, buffer_id: i32) -> PathBuf {
    unix_mmf_backing_dir().join(format!(
        "{}.qu",
        compose_memory_view_name(prefix, buffer_id)
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_SHARED_MEMORY_PREFIX_LEN, SharedMemoryPrefixValidationError, compose_memory_view_name,
        is_valid_shared_memory_prefix, validate_shared_memory_prefix,
    };

    #[test]
    fn compose_memory_view_name_matches_renderite_helper() {
        assert_eq!(compose_memory_view_name("", 0), "_0");
        assert_eq!(compose_memory_view_name("sess", 255), "sess_FF");
        assert_eq!(compose_memory_view_name("p", 0), "p_0");
    }

    #[test]
    fn shared_memory_prefix_validation_accepts_identifier_like_prefixes() {
        assert!(is_valid_shared_memory_prefix(""));
        assert!(is_valid_shared_memory_prefix("Renderide_123-abc"));
        assert!(is_valid_shared_memory_prefix("Renderide_123+abc="));
        assert!(is_valid_shared_memory_prefix("a"));
        assert!(is_valid_shared_memory_prefix(&"a".repeat(64)));
    }

    #[test]
    fn shared_memory_prefix_validation_accepts_renderite_crypto_token_prefix() {
        let token = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi+jklmnop=";
        assert_eq!(token.len(), 44);
        let prefix = format!("abcdefghijklmnop_{token}");

        assert_eq!(prefix.len(), 61);
        assert!(is_valid_shared_memory_prefix(&prefix));
        assert_eq!(
            compose_memory_view_name(&prefix, 255),
            format!("{prefix}_FF")
        );
    }

    #[test]
    fn shared_memory_prefix_validation_rejects_paths_and_invalid_values() {
        assert!(!is_valid_shared_memory_prefix("../session"));
        assert!(!is_valid_shared_memory_prefix("/tmp/session"));
        assert!(!is_valid_shared_memory_prefix(r"a\b"));
        assert!(!is_valid_shared_memory_prefix("session.name"));
        assert!(!is_valid_shared_memory_prefix("session\0name"));
        assert!(!is_valid_shared_memory_prefix("session name"));
        assert!(!is_valid_shared_memory_prefix("session\nname"));
        assert!(!is_valid_shared_memory_prefix(&"a".repeat(65)));
    }

    #[test]
    fn shared_memory_prefix_validation_reports_reason_without_full_prefix() {
        let too_long = validate_shared_memory_prefix(&"a".repeat(65))
            .expect_err("overlong prefix should be rejected");
        assert_eq!(
            too_long,
            SharedMemoryPrefixValidationError::TooLong {
                byte_len: 65,
                max_byte_len: MAX_SHARED_MEMORY_PREFIX_LEN,
            }
        );

        let invalid = validate_shared_memory_prefix("ab/c").expect_err("slash should be rejected");
        assert_eq!(
            invalid,
            SharedMemoryPrefixValidationError::InvalidByte {
                offset: 2,
                byte: b'/',
            }
        );
    }
}
