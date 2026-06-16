//! Host-side shared memory writer (mirror of [`SharedMemoryAccessor`](super::SharedMemoryAccessor)).
//!
//! The renderer reads host-supplied shared-memory regions (mesh vertex/index buffers, texture pixel
//! data, material-batch side buffers) via `SharedMemoryAccessor`. This writer is the inverse: a
//! mock host (currently `renderide-test`) creates and writes the same Cloudtoid backing files so
//! the renderer's reads pick up the written bytes through `SharedMemoryBufferDescriptor` lookups.
//!
//! ## Naming
//!
//! Matches `Helper.ComposeMemoryViewName` exactly (see [`super::compose_memory_view_name`]) so
//! renderer-side reads find the mapping.
//!
//! - Unix: `{prefix}_{bufferId:X}.qu` under `{RENDERIDE_INTERPROCESS_DIR or default_memory_dir()}`.
//! - Windows: named file mapping `CT_IP_{prefix}_{bufferId:X}` (anonymous backing -- no on-disk file).
//!
//! ## Lifetime
//!
//! On Unix, the file is truncated to `capacity` bytes on first open and removed on drop when the
//! writer is configured with `destroy_on_drop` (the host typically uses true so the test crate
//! does not litter `/dev/shm/.cloudtoid/...`). On Windows the named mapping handle is closed on
//! drop; the kernel reaps the section automatically.

use std::io;
use std::path::PathBuf;

use super::naming::is_valid_shared_memory_prefix;
use crate::buffer::SharedMemoryBufferDescriptor;

#[cfg(unix)]
use super::RENDERIDE_INTERPROCESS_DIR_ENV;

#[cfg(unix)]
use std::env;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
use unix::PlatformWriter;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows::PlatformWriter;

/// Unix backing-file directory matching the renderer's resolution.
///
/// Returns `RENDERIDE_INTERPROCESS_DIR` when set (and non-empty), else the platform-specific
/// `interprocess::default_memory_dir()` (Linux: `/dev/shm/.cloudtoid/interprocess/mmf`, others:
/// `temp_dir()/.cloudtoid/interprocess/mmf`). Both writer and renderer must agree on the directory.
#[cfg(unix)]
pub fn host_writer_backing_dir() -> PathBuf {
    env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map_or_else(interprocess::default_memory_dir, PathBuf::from)
}

/// Windows fallback: returns the same name the renderer-side `SharedMemoryAccessor` derives so
/// callers can log it for diagnostics. The host-side writer holds the named-section handle, not a
/// file.
#[cfg(windows)]
pub fn host_writer_backing_dir() -> PathBuf {
    PathBuf::from("CT_IP_*")
}

/// Identifier for a shared-memory writer instance.
///
/// Holds the configuration needed to derive `SharedMemoryBufferDescriptor`s without re-reading
/// the prefix and capacity at every call site.
#[derive(Clone, Debug, Default)]
pub struct SharedMemoryWriterConfig {
    /// Session prefix, matching `RendererInitData.shared_memory_prefix`.
    pub prefix: String,
    /// Whether to remove backing files on drop (Unix only; ignored on Windows).
    pub destroy_on_drop: bool,
    /// Optional explicit backing directory. When `Some`, takes precedence over the
    /// `RENDERIDE_INTERPROCESS_DIR` environment variable; lets concurrent in-process callers
    /// (notably the integration harness) avoid mutating the process-global env. Unix only;
    /// ignored on Windows where the named-mapping backend is path-independent.
    pub dir_override: Option<PathBuf>,
}

/// Failure to create or write a host-side shared-memory buffer.
#[derive(Debug, thiserror::Error)]
pub enum SharedMemoryWriterError {
    /// Filesystem (mkdir, file create, truncate) failed on Unix.
    #[error("io: {0}")]
    Io(#[source] io::Error),
    /// Capacity must fit in `i32` (matches [`SharedMemoryBufferDescriptor::buffer_capacity`]).
    #[error("capacity {0} does not fit in i32")]
    CapacityOverflow(usize),
    /// Capacity must be > 0.
    #[error("capacity must be > 0")]
    CapacityZero,
    /// Write would extend past the buffer capacity.
    #[error("write of {len} bytes at offset {offset} exceeds capacity {capacity}")]
    OutOfBounds {
        /// Byte offset of the requested write.
        offset: i32,
        /// Length of the requested write.
        len: i32,
        /// Total buffer capacity.
        capacity: i32,
    },
    /// Platform mapping failed (file open / `CreateFileMappingW`).
    #[error("platform mapping: {0}")]
    Map(String),
    /// Shared-memory prefix is not safe as a backing-name component.
    #[error("invalid shared-memory prefix")]
    InvalidPrefix,
}

impl From<io::Error> for SharedMemoryWriterError {
    fn from(e: io::Error) -> Self {
        SharedMemoryWriterError::Io(e)
    }
}

/// Single host-side shared-memory buffer (one Cloudtoid `.qu` file or named mapping).
///
/// Writes from the host are visible to the renderer's [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor)
/// as soon as [`SharedMemoryWriter::flush`] is called. Use [`SharedMemoryWriter::descriptor_for`]
/// to embed the byte range in `RendererCommand` payloads (e.g. `MeshUploadData.buffer`).
#[derive(Debug)]
pub struct SharedMemoryWriter {
    cfg: SharedMemoryWriterConfig,
    buffer_id: i32,
    capacity_bytes: i32,
    inner: PlatformWriter,
}

impl SharedMemoryWriter {
    /// Creates (or opens, on Windows) the host-side mapping for `buffer_id` with `capacity` bytes.
    ///
    /// On Unix, the backing file is truncated to `capacity` and `mmap`'d for read+write. On
    /// Windows, the named mapping is created (or opened if it already exists). Errors when the
    /// directory cannot be created, the file cannot be opened/sized, or the mapping fails.
    pub fn open(
        cfg: SharedMemoryWriterConfig,
        buffer_id: i32,
        capacity_bytes: usize,
    ) -> Result<Self, SharedMemoryWriterError> {
        if capacity_bytes == 0 {
            return Err(SharedMemoryWriterError::CapacityZero);
        }
        if !is_valid_shared_memory_prefix(&cfg.prefix) {
            return Err(SharedMemoryWriterError::InvalidPrefix);
        }
        #[expect(
            clippy::map_err_ignore,
            reason = "CapacityOverflow carries the value; `TryFromIntError` adds no detail"
        )]
        let capacity_i32: i32 = capacity_bytes
            .try_into()
            .map_err(|_| SharedMemoryWriterError::CapacityOverflow(capacity_bytes))?;
        let inner = PlatformWriter::new(&cfg, buffer_id, capacity_i32)?;
        Ok(Self {
            cfg,
            buffer_id,
            capacity_bytes: capacity_i32,
            inner,
        })
    }

    /// Writes `data` at `offset` (bytes from the start of the mapping).
    ///
    /// Returns [`SharedMemoryWriterError::OutOfBounds`] if `offset + data.len()` exceeds the
    /// mapping size. Use [`Self::flush`] before publishing the descriptor on the IPC queue so the
    /// renderer reads see the writes (mmap is best-effort coherent across processes on Unix).
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<(), SharedMemoryWriterError> {
        let total = offset
            .checked_add(data.len())
            .ok_or(SharedMemoryWriterError::OutOfBounds {
                offset: offset as i32,
                len: data.len() as i32,
                capacity: self.capacity_bytes,
            })?;
        if total > self.inner.len() {
            return Err(SharedMemoryWriterError::OutOfBounds {
                offset: offset as i32,
                len: data.len() as i32,
                capacity: self.capacity_bytes,
            });
        }
        self.inner.write_at(offset, data)?;
        Ok(())
    }

    /// Flushes the entire mapping so the renderer process observes the writes (best-effort).
    pub fn flush(&self) {
        self.inner.flush_range(0, self.inner.len());
    }

    /// Flushes a specific byte range.
    pub fn flush_range(&self, offset: usize, len: usize) {
        self.inner.flush_range(offset, len);
    }

    /// Builds a [`SharedMemoryBufferDescriptor`] referencing the byte range `[offset, offset+length)`.
    ///
    /// The descriptor is embedded in `RendererCommand` payloads (e.g. `MeshUploadData.buffer`,
    /// `SetTexture2DData.data`, `MaterialsUpdateBatch.material_updates[i]`). The renderer's
    /// `SharedMemoryAccessor` opens the mapping for `self.buffer_id` and reads from
    /// `[offset, offset+length)`.
    pub const fn descriptor_for(&self, offset: i32, length: i32) -> SharedMemoryBufferDescriptor {
        SharedMemoryBufferDescriptor {
            buffer_id: self.buffer_id,
            buffer_capacity: self.capacity_bytes,
            offset,
            length,
        }
    }

    /// Buffer id (matches `SharedMemoryBufferDescriptor::buffer_id`).
    pub const fn buffer_id(&self) -> i32 {
        self.buffer_id
    }

    /// Capacity in bytes.
    pub const fn capacity_bytes(&self) -> i32 {
        self.capacity_bytes
    }

    /// Configured prefix and `destroy_on_drop` flag.
    pub const fn config(&self) -> &SharedMemoryWriterConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::{SharedMemoryWriter, SharedMemoryWriterConfig, SharedMemoryWriterError};

    #[test]
    fn capacity_zero_rejected() {
        let cfg = SharedMemoryWriterConfig {
            prefix: "test_capacity_zero".into(),
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let err = SharedMemoryWriter::open(cfg, 0, 0).expect_err("capacity zero");
        assert!(matches!(err, SharedMemoryWriterError::CapacityZero));
    }

    #[test]
    fn invalid_prefix_rejected() {
        let cfg = SharedMemoryWriterConfig {
            prefix: "../bad".into(),
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let err = SharedMemoryWriter::open(cfg, 0, 1024).expect_err("invalid prefix");
        assert!(matches!(err, SharedMemoryWriterError::InvalidPrefix));
    }

    #[test]
    fn descriptor_for_round_trip() {
        // Use a unique prefix so concurrent test runs don't collide on the backing file/section.
        let prefix = format!("renderide_test_writer_{}", std::process::id());
        let cfg = SharedMemoryWriterConfig {
            prefix,
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let writer = SharedMemoryWriter::open(cfg, 1, 1024).expect("open writer");
        let d = writer.descriptor_for(16, 64);
        assert_eq!(d.buffer_id, 1);
        assert_eq!(d.buffer_capacity, 1024);
        assert_eq!(d.offset, 16);
        assert_eq!(d.length, 64);
    }

    #[test]
    fn write_then_flush_succeeds_within_capacity() {
        let prefix = format!("renderide_test_writer_wf_{}", std::process::id());
        let cfg = SharedMemoryWriterConfig {
            prefix,
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let mut writer = SharedMemoryWriter::open(cfg, 7, 256).expect("open writer");
        writer.write_at(0, b"hello world").expect("write");
        writer.flush();
    }

    #[test]
    fn write_out_of_bounds_rejected() {
        let prefix = format!("renderide_test_writer_oob_{}", std::process::id());
        let cfg = SharedMemoryWriterConfig {
            prefix,
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let mut writer = SharedMemoryWriter::open(cfg, 9, 16).expect("open writer");
        let err = writer.write_at(10, b"too long").expect_err("oob");
        assert!(matches!(err, SharedMemoryWriterError::OutOfBounds { .. }));
    }
}
