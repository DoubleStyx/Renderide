//! File-backed `mmap` on Unix (including macOS).

use std::fs::{self, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// File-backed queue: keeps the `.qu` file open alongside a writable [`memmap2::MmapMut`].
pub(super) struct UnixMapping {
    /// Open file handle; must outlive `mmap`.
    _file: fs::File,
    /// Writable mapping of the entire file.
    mmap: memmap2::MmapMut,
    /// Path passed to [`crate::QueueOptions::file_path`].
    file_path: PathBuf,
    /// Byte length of the mapping (header plus ring).
    len: usize,
}

impl UnixMapping {
    /// Returns the start of the mapped file.
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Length of the mapping in bytes.
    pub(super) const fn len(&self) -> usize {
        self.len
    }

    /// Path to the backing `.qu` file (always [`Some`] on Unix).
    pub(super) const fn backing_file_path(&self) -> Option<&PathBuf> {
        Some(&self.file_path)
    }
}

/// Opens or creates the `.qu` file, sets its length, maps it read/write, and opens the POSIX semaphore.
pub(super) fn open_queue(options: &QueueOptions) -> Result<(UnixMapping, Semaphore), OpenError> {
    let path = options.file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(OpenError)?;
    }

    let storage_size_u64 =
        u64::try_from(options.actual_storage_size()).map_err(|e: std::num::TryFromIntError| {
            OpenError(io::Error::other(format!(
                "queue storage size does not fit u64 (capacity {}): {e}",
                options.capacity
            )))
        })?;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // Do not truncate: additional participants must retain existing queue contents.
        .truncate(false)
        // Reject symlinks at the queue path. `/dev/shm` is a sticky world-writable tmpfs;
        // without `O_NOFOLLOW` a co-resident user could plant a symlink targeting an
        // arbitrary user-writable file, and the subsequent `set_len` + mmap write would
        // truncate and overwrite that target with queue header bytes.
        .custom_flags(libc::O_NOFOLLOW)
        // Owner-only access: queue contents may include in-flight IPC bytes that should not
        // be readable by other local users sharing the tmpfs.
        .mode(0o600)
        .open(&path)
        .map_err(OpenError)?;

    let existing_len = file.metadata().map_err(OpenError)?.len();
    if existing_len < storage_size_u64 {
        file.set_len(storage_size_u64).map_err(OpenError)?;
    }

    let map_len = storage_size_u64 as usize;
    // SAFETY: `memmap2::MmapMut` is unsafe because the file's contents may be mutated by other
    // processes; this is intentional -- the cross-process ring protocol provides all synchronisation
    // via atomics and single-writer / single-reader slot discipline. The mapping length is no
    // greater than the just-set file length.
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .len(map_len)
            .map_mut(&file)
            .map_err(|e| OpenError(io::Error::other(format!("mmap failed: {e}"))))?
    };

    let sem = Semaphore::open(options.memory_view_name.as_str()).map_err(OpenError)?;

    Ok((
        UnixMapping {
            _file: file,
            mmap,
            file_path: path,
            len: map_len,
        },
        sem,
    ))
}

#[cfg(test)]
mod tests {
    use crate::memory::SharedMapping;
    use crate::options::QueueOptions;

    #[test]
    fn queue_backing_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let opts = QueueOptions::with_path("mm_mode", dir.path(), 4096).expect("valid");
        let path = opts.file_path();
        let (_m, _s) = SharedMapping::open_queue(&opts).expect("open");
        let mode = std::fs::metadata(&path).expect("meta").permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "queue backing file must be owner-only (got {mode:o})",
        );
    }

    #[test]
    fn queue_backing_file_open_rejects_symlink() {
        let dir = tempfile::tempdir().expect("tempdir");
        let opts = QueueOptions::with_path("mm_symlink", dir.path(), 4096).expect("valid");
        let path = opts.file_path();
        let target = dir.path().join("symlink_target_should_not_be_truncated");
        std::fs::write(&target, b"sentinel-bytes-must-survive").expect("write target");
        std::os::unix::fs::symlink(&target, &path).expect("create symlink");

        let result = SharedMapping::open_queue(&opts);
        assert!(
            result.is_err(),
            "open_queue must refuse to follow a symlink at the queue path",
        );
        let preserved = std::fs::read(&target).expect("read target");
        assert_eq!(
            preserved, b"sentinel-bytes-must-survive",
            "symlink target must not be truncated by the failed open",
        );
    }

    #[test]
    fn open_twice_same_path_same_file_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let opts = QueueOptions::with_path("mm_reopen", dir.path(), 4096).expect("valid");
        let path = opts.file_path();
        let (m1, _s1) = SharedMapping::open_queue(&opts).expect("open1");
        let len1 = std::fs::metadata(&path).expect("meta").len();
        assert_eq!(len1, opts.actual_storage_size() as u64);
        drop(m1);
        let (_m2, _s2) = SharedMapping::open_queue(&opts).expect("open2");
        let len2 = std::fs::metadata(&path).expect("meta").len();
        assert_eq!(len1, len2);
    }

    #[test]
    fn larger_existing_file_is_not_truncated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let opts = QueueOptions::with_path("mm_large", dir.path(), 4096).expect("valid");
        let path = opts.file_path();
        let big = opts.actual_storage_size() as u64 + 4096;
        std::fs::write(&path, vec![0u8; big as usize]).expect("seed");
        let (_m, _s) = SharedMapping::open_queue(&opts).expect("open");
        let len = std::fs::metadata(&path).expect("meta").len();
        assert_eq!(len, big);
    }
}
