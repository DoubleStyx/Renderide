//! Memory view naming and Unix `.qu` backing paths.

#[cfg(unix)]
use std::path::PathBuf;

/// Environment variable overriding the Unix directory for `.qu` MMF files (must match host /
/// bootstrapper). Same value as `bootstrapper::ipc::RENDERIDE_INTERPROCESS_DIR_ENV`.
///
/// Only read by [`unix_mmf_backing_dir`] on Unix; Windows builds keep this symbol for API parity
/// with the bootstrapper constant name.
pub const RENDERIDE_INTERPROCESS_DIR_ENV: &str = "RENDERIDE_INTERPROCESS_DIR";

/// Composes the memory view name per Renderite `Helper.ComposeMemoryViewName` (prefix + hex id).
pub fn compose_memory_view_name(prefix: &str, buffer_id: i32) -> String {
    format!("{}_{:X}", prefix, buffer_id)
}

/// Unix-only: resolved directory containing `{composed}.qu` backing files.
#[cfg(unix)]
pub(super) fn unix_mmf_backing_dir() -> PathBuf {
    std::env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(interprocess::default_memory_dir)
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
    use super::compose_memory_view_name;

    #[test]
    fn compose_memory_view_name_matches_renderite_helper() {
        assert_eq!(compose_memory_view_name("sess", 255), "sess_FF");
        assert_eq!(compose_memory_view_name("p", 0), "p_0");
    }
}
