//! File-backed `mmap` on Unix (including macOS).

use std::fs::{self, OpenOptions};
use std::io;
use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// File-backed queue: keeps the `.qu` file open alongside a writable [`memmap2::MmapMut`].
pub(super) struct UnixMapping {
    /// Open file handle; must outlive `mmap`.
    _file: std::fs::File,
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
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Path to the backing `.qu` file (always [`Some`] on Unix).
    pub(super) fn backing_file_path(&self) -> Option<&PathBuf> {
        Some(&self.file_path)
    }
}

/// Opens or creates the `.qu` file, sets its length, maps it read/write, and opens the POSIX semaphore.
pub(super) fn open_queue(options: &QueueOptions) -> Result<(UnixMapping, Semaphore), OpenError> {
    let path = options.file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(OpenError)?;
    }

    let storage_size_u64 = u64::try_from(options.actual_storage_size()).map_err(|_| {
        OpenError(io::Error::other(format!(
            "queue storage size does not fit u64 (capacity {})",
            options.capacity
        )))
    })?;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // Do not truncate: additional participants must retain existing queue contents.
        .truncate(false)
        .open(&path)
        .map_err(OpenError)?;

    let existing_len = file.metadata().map_err(OpenError)?.len();
    if existing_len < storage_size_u64 {
        file.set_len(storage_size_u64).map_err(OpenError)?;
    }

    let map_len = storage_size_u64 as usize;
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
