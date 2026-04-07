//! Configuration for opening a shared-memory queue.

use std::path::{Path, PathBuf};

/// Linux tmpfs directory used for file-backed queues and for interop with stacks that expect `/dev/shm`.
pub const LINUX_SHM_MEMORY_DIR: &str = "/dev/shm/.cloudtoid/interprocess/mmf";

/// Linux-only default directory (same as [`LINUX_SHM_MEMORY_DIR`]).
#[deprecated(
    note = "use LINUX_SHM_MEMORY_DIR for Linux-specific paths, or default_memory_dir() for portable defaults"
)]
pub const DEFAULT_MEMORY_DIR: &str = LINUX_SHM_MEMORY_DIR;

/// Legacy alias for [`LINUX_SHM_MEMORY_DIR`].
#[deprecated(note = "use LINUX_SHM_MEMORY_DIR or default_memory_dir()")]
pub const MEMORY_FILE_PATH: &str = LINUX_SHM_MEMORY_DIR;

/// Returns the default directory for `.qu` backing files used by [`QueueOptions::new`] and [`QueueOptions::with_destroy`].
///
/// - **Linux**: [`LINUX_SHM_MEMORY_DIR`] under `/dev/shm` (tmpfs, matches typical managed layouts).
/// - **Other Unix** (macOS, BSD, etc.): `std::env::temp_dir()/.cloudtoid/interprocess/mmf`.
/// - **Windows**: same temp-dir layout (the named mapping does not use this path, but [`QueueOptions::path`] is populated for consistency).
pub fn default_memory_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        PathBuf::from(LINUX_SHM_MEMORY_DIR)
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        std::env::temp_dir().join(".cloudtoid/interprocess/mmf")
    }
    #[cfg(windows)]
    {
        std::env::temp_dir().join(".cloudtoid/interprocess/mmf")
    }
}

/// Options for creating a [`crate::Publisher`] or [`crate::Subscriber`].
#[derive(Clone)]
pub struct QueueOptions {
    /// Logical queue name (maps to `{dir}/{name}.qu` on Unix and `CT_IP_{name}` on Windows).
    pub memory_view_name: String,
    /// Directory containing `.qu` files on Unix; ignored for the default Windows named-mapping backend.
    pub path: PathBuf,
    /// Ring buffer capacity in bytes (user data only; excludes [`crate::layout::QueueHeader`]).
    pub capacity: i64,
    /// When `true`, remove the backing file (Unix) when the handle is dropped.
    pub destroy_on_dispose: bool,
}

impl QueueOptions {
    const MIN_CAPACITY: i64 = 17;

    /// Validates `capacity` and returns an error message if invalid.
    fn validate_capacity(capacity: i64) -> Result<(), String> {
        if capacity <= Self::MIN_CAPACITY {
            return Err(format!(
                "capacity must be greater than {} (got {capacity})",
                Self::MIN_CAPACITY
            ));
        }
        if capacity % 8 != 0 {
            return Err(format!(
                "capacity must be a multiple of 8 bytes (got {capacity})"
            ));
        }
        Ok(())
    }

    /// Builds options with [`default_memory_dir()`] and `destroy_on_dispose = false`.
    pub fn new(queue_name: &str, capacity: i64) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: default_memory_dir(),
            capacity,
            destroy_on_dispose: false,
        })
    }

    /// Same as [`Self::new`] but controls whether the backing file is removed on drop (Unix).
    pub fn with_destroy(
        queue_name: &str,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: default_memory_dir(),
            capacity,
            destroy_on_dispose,
        })
    }

    /// Full control over the backing directory.
    pub fn with_path(
        queue_name: &str,
        path: impl AsRef<Path>,
        capacity: i64,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: path.as_ref().to_path_buf(),
            capacity,
            destroy_on_dispose: false,
        })
    }

    /// Full control over directory and `destroy_on_dispose`.
    pub fn with_path_and_destroy(
        queue_name: &str,
        path: impl AsRef<Path>,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path: path.as_ref().to_path_buf(),
            capacity,
            destroy_on_dispose,
        })
    }

    /// Total file / mapping size: header + ring capacity.
    pub fn actual_storage_size(&self) -> i64 {
        crate::layout::BUFFER_BYTE_OFFSET as i64 + self.capacity
    }

    /// Path to the `.qu` backing file on Unix.
    pub fn file_path(&self) -> PathBuf {
        self.path.join(format!("{}.qu", self.memory_view_name))
    }

    /// POSIX semaphore name (`/ct.ip.{memory_view_name}`).
    pub fn posix_semaphore_name(&self) -> String {
        format!("/ct.ip.{}", self.memory_view_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MM_SUBDIR: &str = ".cloudtoid/interprocess/mmf";

    #[test]
    fn default_memory_dir_linux_matches_shm_path() {
        if !cfg!(target_os = "linux") {
            return;
        }
        assert_eq!(default_memory_dir(), PathBuf::from(LINUX_SHM_MEMORY_DIR));
    }

    #[test]
    fn default_memory_dir_non_linux_unix_uses_temp_subdir() {
        if !cfg!(unix) || cfg!(target_os = "linux") {
            return;
        }
        let d = default_memory_dir();
        let tmp = std::env::temp_dir();
        assert!(
            d.starts_with(&tmp) && d.as_os_str().to_string_lossy().contains(MM_SUBDIR),
            "expected path under temp containing {MM_SUBDIR}, got {d:?}"
        );
    }

    #[test]
    fn default_memory_dir_windows_uses_temp_subdir() {
        if !cfg!(windows) {
            return;
        }
        let d = default_memory_dir();
        let tmp = std::env::temp_dir();
        assert!(
            d.starts_with(&tmp) && d.as_os_str().to_string_lossy().contains(MM_SUBDIR),
            "expected path under temp containing {MM_SUBDIR}, got {d:?}"
        );
    }

    #[test]
    fn queue_options_new_paths_default_memory_dir() {
        let o = QueueOptions::new("q", 4096).expect("valid");
        assert_eq!(o.path, default_memory_dir());
    }
}
