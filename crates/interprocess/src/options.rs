//! Configuration for opening a shared-memory queue.

use std::path::{Path, PathBuf};

/// Linux tmpfs directory used for file-backed queues and for interop with stacks that expect `/dev/shm`.
pub const LINUX_SHM_MEMORY_DIR: &str = "/dev/shm/.cloudtoid/interprocess/mmf";

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

    /// Ensures `capacity` is above [`Self::MIN_CAPACITY`] and 8-byte aligned (layout requirement).
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

    fn build(
        queue_name: &str,
        path: PathBuf,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::validate_capacity(capacity)?;
        Ok(Self {
            memory_view_name: queue_name.to_string(),
            path,
            capacity,
            destroy_on_dispose,
        })
    }

    /// Builds options with [`default_memory_dir()`] and `destroy_on_dispose = false`.
    pub fn new(queue_name: &str, capacity: i64) -> Result<Self, String> {
        Self::build(queue_name, default_memory_dir(), capacity, false)
    }

    /// Same as [`Self::new`] but controls whether the backing file is removed on drop (Unix).
    pub fn with_destroy(
        queue_name: &str,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::build(
            queue_name,
            default_memory_dir(),
            capacity,
            destroy_on_dispose,
        )
    }

    /// Full control over the backing directory.
    pub fn with_path(
        queue_name: &str,
        path: impl AsRef<Path>,
        capacity: i64,
    ) -> Result<Self, String> {
        Self::build(queue_name, path.as_ref().to_path_buf(), capacity, false)
    }

    /// Full control over directory and `destroy_on_dispose`.
    pub fn with_path_and_destroy(
        queue_name: &str,
        path: impl AsRef<Path>,
        capacity: i64,
        destroy_on_dispose: bool,
    ) -> Result<Self, String> {
        Self::build(
            queue_name,
            path.as_ref().to_path_buf(),
            capacity,
            destroy_on_dispose,
        )
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

    #[test]
    fn queue_options_rejects_capacity_at_or_below_min() {
        assert!(QueueOptions::new("q", 17).is_err());
        assert!(QueueOptions::new("q", 16).is_err());
    }

    #[test]
    fn queue_options_rejects_non_multiple_of_eight() {
        assert!(QueueOptions::new("q", 4097).is_err());
        assert!(QueueOptions::new("q", 18).is_err());
    }

    #[test]
    fn queue_options_accepts_minimum_valid_capacity() {
        let o = QueueOptions::new("q", 24).expect("24 > 17 and aligned");
        assert_eq!(o.capacity, 24);
    }

    #[test]
    fn queue_options_actual_storage_size_includes_header() {
        let o = QueueOptions::new("q", 4096).expect("valid");
        assert_eq!(
            o.actual_storage_size(),
            crate::layout::BUFFER_BYTE_OFFSET as i64 + 4096
        );
    }

    #[test]
    fn queue_options_file_path_and_posix_semaphore_name() {
        let base = std::env::temp_dir().join("interprocess_opts_path_test");
        let o = QueueOptions::with_path("my_queue", &base, 4096).expect("valid");
        assert_eq!(o.file_path(), base.join("my_queue.qu"));
        assert_eq!(o.posix_semaphore_name(), "/ct.ip.my_queue");
    }

    #[test]
    fn queue_options_with_destroy_sets_flag() {
        let o = QueueOptions::with_destroy("q", 4096, true).expect("valid");
        assert!(o.destroy_on_dispose);
        let o2 = QueueOptions::with_destroy("q", 4096, false).expect("valid");
        assert!(!o2.destroy_on_dispose);
    }

    #[test]
    fn queue_options_with_path_and_destroy() {
        let base = std::env::temp_dir().join("interprocess_opts_destroy");
        let o = QueueOptions::with_path_and_destroy("q", &base, 4096, true).expect("valid");
        assert_eq!(o.path, base);
        assert!(o.destroy_on_dispose);
    }
}
