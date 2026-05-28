//! Windows host-side shared-memory writer backend.

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, FlushViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
};

use super::{SharedMemoryWriterConfig, SharedMemoryWriterError};
use crate::ipc::shared_memory::naming::compose_memory_view_name;

/// Prefix used by Cloudtoid named file mappings.
const MAP_NAME_PREFIX: &str = "CT_IP_";

/// Platform-specific writer for one Windows named mapping.
pub(super) struct PlatformWriter {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

/// [`MEMORY_MAPPED_VIEW_ADDRESS`] does not implement [`std::fmt::Debug`]; print the mapped base pointer.
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
    /// Creates or opens the named pagefile-backed mapping for `buffer_id`.
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

    /// Writes bytes into the already bounds-checked mapped range.
    pub(super) fn write_at(
        &mut self,
        offset: usize,
        data: &[u8],
    ) -> Result<(), SharedMemoryWriterError> {
        // SAFETY: caller-facing bounds are checked by `write` in the outer struct; `self.view` is
        // the mapping base; `&mut self` ensures no other writer is active here.
        unsafe {
            let dst = self.view.Value.add(offset).cast::<u8>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    /// Flushes a mapped byte range so the renderer can observe it.
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

    /// Returns the mapped section length in bytes.
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
