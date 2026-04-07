//! File-backed `mmap` on Unix (including macOS).

use std::fs::{self, OpenOptions};
use std::io;
use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// Unix mapping: keeps the file open alongside a writable [`memmap2::MmapMut`].
pub(super) struct UnixMapping {
    _file: std::fs::File,
    mmap: memmap2::MmapMut,
    file_path: PathBuf,
    len: usize,
}

impl UnixMapping {
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    pub(super) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn backing_file_path(&self) -> Option<&PathBuf> {
        Some(&self.file_path)
    }
}

/// Opens or creates the `.qu` file, ensures length, maps RW, and opens the semaphore.
pub(super) fn open_queue(options: &QueueOptions) -> Result<(UnixMapping, Semaphore), OpenError> {
    let path = options.file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(OpenError)?;
    }

    let storage_size = options.actual_storage_size() as u64;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // Do not truncate: additional participants must retain existing queue contents.
        .truncate(false)
        .open(&path)
        .map_err(OpenError)?;

    file.set_len(storage_size).map_err(OpenError)?;

    let mmap = unsafe {
        memmap2::MmapMut::map_mut(&file)
            .map_err(|e| OpenError(io::Error::other(format!("mmap failed: {e}"))))?
    };

    let sem = Semaphore::open(options.memory_view_name.as_str()).map_err(OpenError)?;

    let len = storage_size as usize;
    Ok((
        UnixMapping {
            _file: file,
            mmap,
            file_path: path,
            len,
        },
        sem,
    ))
}
