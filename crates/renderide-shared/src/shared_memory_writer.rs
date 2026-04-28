//! Host-side shared memory writer (mirror of [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor)).
//!
//! The renderer reads host-supplied shared-memory regions (mesh vertex/index buffers, texture pixel
//! data, material-batch side buffers) via `SharedMemoryAccessor`. This writer is the inverse: a
//! mock host (currently `renderide-test`) creates and writes the same Cloudtoid backing files so
//! the renderer's reads pick up the written bytes through `SharedMemoryBufferDescriptor` lookups.
//!
//! ## Naming
//!
//! Matches `Helper.ComposeMemoryViewName` exactly (see
//! [`crate::ipc::compose_memory_view_name`]) so renderer-side reads find the mapping.
//!
//! - Unix: `{prefix}_{bufferId:X}.qu` under `{RENDERIDE_INTERPROCESS_DIR or default_memory_dir()}`.
//! - Windows: named file mapping `CT_IP_{prefix}_{bufferId:X}` (anonymous backing — no on-disk file).
//!
//! ## Lifetime
//!
//! On Unix, the file is truncated to `capacity` bytes on first open and removed on drop when the
//! writer is configured with `destroy_on_drop` (the host typically uses true so the test crate
//! does not litter `/dev/shm/.cloudtoid/...`). On Windows the named mapping handle is closed on
//! drop; the kernel reaps the section automatically.

use std::io;
use std::path::PathBuf;

#[cfg(unix)]
use std::fs::{File, OpenOptions};

#[cfg(unix)]
use memmap2::MmapMut;

use crate::buffer::SharedMemoryBufferDescriptor;
use crate::ipc::shared_memory::compose_memory_view_name;

#[cfg(unix)]
use std::env;

#[cfg(unix)]
const RENDERIDE_INTERPROCESS_DIR_ENV: &str = "RENDERIDE_INTERPROCESS_DIR";

/// Unix backing-file directory matching the renderer's resolution.
///
/// Returns `RENDERIDE_INTERPROCESS_DIR` when set (and non-empty), else the platform-specific
/// `interprocess::default_memory_dir()` (Linux: `/dev/shm/.cloudtoid/interprocess/mmf`, others:
/// `temp_dir()/.cloudtoid/interprocess/mmf`). Both writer and renderer must agree on the directory.
#[cfg(unix)]
pub fn host_writer_backing_dir() -> PathBuf {
    env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(interprocess::default_memory_dir)
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
#[derive(Clone, Debug)]
pub struct SharedMemoryWriterConfig {
    /// Session prefix, matching `RendererInitData.shared_memory_prefix`.
    pub prefix: String,
    /// Whether to remove backing files on drop (Unix only; ignored on Windows).
    pub destroy_on_drop: bool,
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
}

impl From<io::Error> for SharedMemoryWriterError {
    fn from(e: io::Error) -> Self {
        SharedMemoryWriterError::Io(e)
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use crate::ipc::shared_memory::RENDERIDE_INTERPROCESS_DIR_ENV;

    #[derive(Debug)]
    pub(super) struct PlatformWriter {
        file_path: PathBuf,
        _file: File,
        mmap: MmapMut,
        destroy_on_drop: bool,
    }

    impl PlatformWriter {
        pub(super) fn new(
            cfg: &SharedMemoryWriterConfig,
            buffer_id: i32,
            capacity_bytes: i32,
        ) -> Result<Self, SharedMemoryWriterError> {
            let dir = env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(interprocess::default_memory_dir);
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
                .map_err(|e| {
                    SharedMemoryWriterError::Map(format!("{}: {e}", file_path.display()))
                })?;
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

        pub(super) fn write_at(
            &mut self,
            offset: usize,
            data: &[u8],
        ) -> Result<(), SharedMemoryWriterError> {
            self.mmap[offset..offset + data.len()].copy_from_slice(data);
            Ok(())
        }

        pub(super) fn flush_range(&self, offset: usize, len: usize) {
            let _ = self.mmap.flush_range(offset, len);
        }

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
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FlushViewOfFile, MapViewOfFile, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
        MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
    };

    const MAP_NAME_PREFIX: &str = "CT_IP_";

    pub(super) struct PlatformWriter {
        handle: HANDLE,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
        len: usize,
    }

    /// [`MEMORY_MAPPED_VIEW_ADDRESS`] does not implement [`std::fmt::Debug`]; we print the mapped base pointer.
    impl std::fmt::Debug for PlatformWriter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PlatformWriter")
                .field("handle", &self.handle)
                .field("view", &self.view.Value)
                .field("len", &self.len)
                .finish()
        }
    }

    impl PlatformWriter {
        pub(super) fn new(
            cfg: &SharedMemoryWriterConfig,
            buffer_id: i32,
            capacity_bytes: i32,
        ) -> Result<Self, SharedMemoryWriterError> {
            let _ = cfg.destroy_on_drop;
            let name = format!(
                "{}{}",
                MAP_NAME_PREFIX,
                compose_memory_view_name(&cfg.prefix, buffer_id)
            );
            let name_wide: Vec<u16> = OsStr::new(&name)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let size = capacity_bytes as usize;
            // SAFETY: `name_wide` is a NUL-terminated wide string; `INVALID_HANDLE_VALUE` requests
            // an anonymous pagefile-backed mapping.
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    null(),
                    PAGE_READWRITE,
                    (size >> 32) as u32,
                    (size & 0xFFFF_FFFF) as u32,
                    name_wide.as_ptr(),
                )
            };
            if handle.is_null() || handle == INVALID_HANDLE_VALUE {
                return Err(SharedMemoryWriterError::Map(format!(
                    "CreateFileMappingW failed for {name}"
                )));
            }
            // SAFETY: `handle` was just returned valid.
            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size) };
            if view.Value.is_null() {
                // SAFETY: `handle` is live; closed once on this error path.
                unsafe {
                    CloseHandle(handle);
                }
                return Err(SharedMemoryWriterError::Map(format!(
                    "MapViewOfFile failed for {name}"
                )));
            }
            Ok(Self {
                handle,
                view,
                len: size,
            })
        }

        pub(super) fn write_at(
            &mut self,
            offset: usize,
            data: &[u8],
        ) -> Result<(), SharedMemoryWriterError> {
            // SAFETY: caller-facing bounds are checked by `write` in the outer struct; `self.view`
            // is the mapping base; `&mut self` ensures no other writer is active here.
            unsafe {
                let dst = self.view.Value.add(offset).cast::<u8>();
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
            Ok(())
        }

        pub(super) fn flush_range(&self, offset: usize, len: usize) {
            if len == 0 {
                return;
            }
            // SAFETY: `offset + len <= self.len` is the caller's contract (the outer `flush_range`
            // bounds-checks); `self.view.Value` is the non-null live mapping base.
            unsafe {
                let base = self.view.Value.add(offset).cast_const();
                let _ = FlushViewOfFile(base, len);
            }
        }

        pub(super) fn len(&self) -> usize {
            self.len
        }
    }

    impl Drop for PlatformWriter {
        fn drop(&mut self) {
            if !self.view.Value.is_null() {
                // SAFETY: `self.view` was mapped in `new`; unmapped exactly once on drop.
                unsafe {
                    UnmapViewOfFile(self.view);
                }
            }
            if !self.handle.is_null() && self.handle != INVALID_HANDLE_VALUE {
                // SAFETY: `self.handle` was opened in `new`; closed exactly once on drop.
                unsafe {
                    CloseHandle(self.handle);
                }
            }
        }
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
    inner: platform::PlatformWriter,
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
        #[expect(
            clippy::map_err_ignore,
            reason = "CapacityOverflow carries the value; `TryFromIntError` adds no detail"
        )]
        let capacity_i32: i32 = capacity_bytes
            .try_into()
            .map_err(|_| SharedMemoryWriterError::CapacityOverflow(capacity_bytes))?;
        let inner = platform::PlatformWriter::new(&cfg, buffer_id, capacity_i32)?;
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
    pub fn descriptor_for(&self, offset: i32, length: i32) -> SharedMemoryBufferDescriptor {
        SharedMemoryBufferDescriptor {
            buffer_id: self.buffer_id,
            buffer_capacity: self.capacity_bytes,
            offset,
            length,
        }
    }

    /// Buffer id (matches `SharedMemoryBufferDescriptor::buffer_id`).
    pub fn buffer_id(&self) -> i32 {
        self.buffer_id
    }

    /// Capacity in bytes.
    pub fn capacity_bytes(&self) -> i32 {
        self.capacity_bytes
    }

    /// Configured prefix and `destroy_on_drop` flag.
    pub fn config(&self) -> &SharedMemoryWriterConfig {
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
        };
        let err = SharedMemoryWriter::open(cfg, 0, 0).expect_err("capacity zero");
        assert!(matches!(err, SharedMemoryWriterError::CapacityZero));
    }

    #[test]
    fn descriptor_for_round_trip() {
        // Use a unique prefix so concurrent test runs don't collide on the backing file/section.
        let prefix = format!("renderide_test_writer_{}", std::process::id());
        let cfg = SharedMemoryWriterConfig {
            prefix,
            destroy_on_drop: true,
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
        };
        let mut writer = SharedMemoryWriter::open(cfg, 9, 16).expect("open writer");
        let err = writer.write_at(10, b"too long").expect_err("oob");
        assert!(matches!(err, SharedMemoryWriterError::OutOfBounds { .. }));
    }
}
