//! Unix host-side shared-memory writer backend.

use std::env;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

use memmap2::MmapMut;

use super::{SharedMemoryWriterConfig, SharedMemoryWriterError};
use crate::ipc::shared_memory::naming::{RENDERIDE_INTERPROCESS_DIR_ENV, compose_memory_view_name};

/// Platform-specific writer for one Unix `.qu` backing file.
#[derive(Debug)]
pub(super) struct PlatformWriter {
    file_path: PathBuf,
    _file: File,
    mmap: MmapMut,
    destroy_on_drop: bool,
}

impl PlatformWriter {
    /// Creates or opens the backing file for `buffer_id` and maps it read/write.
    pub(super) fn new(
        cfg: &SharedMemoryWriterConfig,
        buffer_id: i32,
        capacity_bytes: i32,
    ) -> Result<Self, SharedMemoryWriterError> {
        let dir = cfg.dir_override.clone().unwrap_or_else(|| {
            env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
                .filter(|s| !s.is_empty())
                .map_or_else(interprocess::default_memory_dir, PathBuf::from)
        });
        std::fs::create_dir_all(&dir).map_err(SharedMemoryWriterError::Io)?;
        let file_path = dir.join(format!(
            "{}.qu",
            compose_memory_view_name(&cfg.prefix, buffer_id)
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)
            .map_err(|e| SharedMemoryWriterError::Map(format!("{}: {e}", file_path.display())))?;
        file.set_len(capacity_bytes as u64)
            .map_err(SharedMemoryWriterError::Io)?;
        // SAFETY: the writer holds exclusive ownership of the backing `.qu` file; cross-process
        // readers synchronise via the IPC wire protocol.
        let mmap = unsafe { MmapMut::map_mut(&file) }
            .map_err(|e| SharedMemoryWriterError::Map(e.to_string()))?;
        Ok(Self {
            file_path,
            _file: file,
            mmap,
            destroy_on_drop: cfg.destroy_on_drop,
        })
    }

    /// Writes bytes into the already bounds-checked mapped range.
    pub(super) fn write_at(
        &mut self,
        offset: usize,
        data: &[u8],
    ) -> Result<(), SharedMemoryWriterError> {
        self.mmap[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Flushes a mapped byte range so the renderer can observe it.
    pub(super) fn flush_range(&self, offset: usize, len: usize) {
        let _ = self.mmap.flush_range(offset, len);
    }

    /// Returns the mapped file length in bytes.
    pub(super) fn len(&self) -> usize {
        self.mmap.len()
    }
}

impl Drop for PlatformWriter {
    fn drop(&mut self) {
        if self.destroy_on_drop {
            let _ = std::fs::remove_file(&self.file_path);
        }
    }
}
