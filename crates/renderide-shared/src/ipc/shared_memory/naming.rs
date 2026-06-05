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

/// Returns whether `prefix` is safe to use as one component of a shared-memory backing name.
pub fn is_valid_shared_memory_prefix(prefix: &str) -> bool {
    !prefix.is_empty()
        && prefix.len() <= MAX_SHARED_MEMORY_PREFIX_LEN
        && prefix
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
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
    use super::{compose_memory_view_name, is_valid_shared_memory_prefix};

    #[test]
    fn compose_memory_view_name_matches_renderite_helper() {
        assert_eq!(compose_memory_view_name("sess", 255), "sess_FF");
        assert_eq!(compose_memory_view_name("p", 0), "p_0");
    }

    #[test]
    fn shared_memory_prefix_validation_accepts_identifier_like_prefixes() {
        assert!(is_valid_shared_memory_prefix("Renderide_123-abc"));
        assert!(is_valid_shared_memory_prefix("a"));
        assert!(is_valid_shared_memory_prefix(&"a".repeat(64)));
    }

    #[test]
    fn shared_memory_prefix_validation_rejects_paths_and_empty_values() {
        assert!(!is_valid_shared_memory_prefix(""));
        assert!(!is_valid_shared_memory_prefix("../session"));
        assert!(!is_valid_shared_memory_prefix("/tmp/session"));
        assert!(!is_valid_shared_memory_prefix(r"a\b"));
        assert!(!is_valid_shared_memory_prefix("session.name"));
        assert!(!is_valid_shared_memory_prefix("session\0name"));
        assert!(!is_valid_shared_memory_prefix(&"a".repeat(65)));
    }
}
