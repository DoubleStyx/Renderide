//! Unix implementation: read/write mmap of `{composed}.qu` under [`super::naming::unix_mmf_backing_dir`].

use std::fs::OpenOptions;
use std::io;

use memmap2::MmapMut;

use super::bounds::byte_subrange;
use super::naming::unix_backing_file_path;

/// Single mapped host buffer backing file (`.qu`).
pub(super) struct SharedMemoryView {
    mmap: MmapMut,
}

impl SharedMemoryView {
    /// Opens the backing file and maps it read/write.
    pub(super) fn new(prefix: &str, buffer_id: i32, _capacity: i32) -> io::Result<Self> {
        let path = unix_backing_file_path(prefix, buffer_id);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| {
                io::Error::new(io::ErrorKind::NotFound, format!("{}: {e}", path.display()))
            })?;
        // SAFETY: the bulk-data wire protocol (see `ipc/shared_memory`) is the caller's
        // synchronisation between host and renderer -- ownership of each region alternates per
        // message; `memmap2::MmapMut::map_mut` is sound under those external guarantees.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap })
    }

    /// Returns `descriptor` region as an immutable slice.
    pub(super) fn slice(&self, offset: i32, length: i32) -> Option<&[u8]> {
        let (start, end) = byte_subrange(self.mmap.len(), offset, length)?;
        Some(&self.mmap[start..end])
    }

    /// Returns `descriptor` region as a mutable slice.
    pub(super) fn slice_mut(&mut self, offset: i32, length: i32) -> Option<&mut [u8]> {
        let (start, end) = byte_subrange(self.mmap.len(), offset, length)?;
        Some(&mut self.mmap[start..end])
    }

    /// Flushes `offset..offset+length` so other processes observe writes (best-effort).
    pub(super) fn flush_range(&self, offset: i32, length: i32) {
        if let Some((start, end)) = byte_subrange(self.mmap.len(), offset, length) {
            let len = end - start;
            if len > 0 {
                let _ = self.mmap.flush_range(start, len);
            }
        }
    }

    /// Length of the mapped file in bytes.
    pub(super) fn len(&self) -> usize {
        self.mmap.len()
    }
}
