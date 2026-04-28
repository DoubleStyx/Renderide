//! Windows implementation: named file mapping `CT_IP_{prefix}_{bufferId:X}`.

use std::ffi::OsStr;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FlushViewOfFile, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile,
    FILE_MAP_ALL_ACCESS, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};

use super::bounds::byte_subrange;
use super::naming::compose_memory_view_name;

const MAP_NAME_PREFIX: &str = "CT_IP_";

/// Single mapped host buffer (named file mapping).
pub struct SharedMemoryView {
    map_handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl SharedMemoryView {
    /// Opens or creates the mapping and maps `capacity` bytes (host already sized the section).
    pub fn new(prefix: &str, buffer_id: i32, capacity: i32) -> io::Result<Self> {
        let name = format!(
            "{}{}",
            MAP_NAME_PREFIX,
            compose_memory_view_name(prefix, buffer_id)
        );
        let size = capacity as usize;

        let name_wide: Vec<u16> = OsStr::new(&name)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let map_handle = create_or_open_file_mapping(&name_wide, size)?;

        // SAFETY: `map_handle` was just returned valid by `create_or_open_file_mapping`.
        let view =
            unsafe { MapViewOfFile(map_handle, FILE_MAP_ALL_ACCESS | FILE_MAP_WRITE, 0, 0, size) };

        if view.Value.is_null() {
            // SAFETY: `map_handle` is live; closed once on this error path.
            unsafe {
                CloseHandle(map_handle);
            }
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("MapViewOfFile failed for {name}"),
            ));
        }

        Ok(Self {
            map_handle,
            view,
            len: size,
        })
    }

    /// Returns the byte subregion or `None` if out of bounds.
    pub fn slice(&self, offset: i32, length: i32) -> Option<&[u8]> {
        let (start, end) = byte_subrange(self.len, offset, length)?;
        if self.view.Value.is_null() {
            return None;
        }
        // SAFETY: `(start, end)` was validated against `self.len`, which is the mapping size, and
        // `self.view.Value` is the non-null mapping base. Borrow lives no longer than `&self`.
        Some(unsafe {
            std::slice::from_raw_parts(
                self.view.Value.add(start).cast::<u8>().cast_const(),
                end - start,
            )
        })
    }

    /// Returns the mutable byte subregion or `None` if out of bounds.
    pub fn slice_mut(&mut self, offset: i32, length: i32) -> Option<&mut [u8]> {
        let (start, end) = byte_subrange(self.len, offset, length)?;
        if self.view.Value.is_null() {
            return None;
        }
        // SAFETY: same bounds-checked region as `slice`, but `&mut self` guarantees unique access
        // on the Rust side; cross-process exclusivity is enforced by the IPC protocol.
        Some(unsafe {
            std::slice::from_raw_parts_mut(self.view.Value.add(start).cast::<u8>(), end - start)
        })
    }

    /// Flushes the given view range (best-effort).
    pub fn flush_range(&self, offset: i32, length: i32) {
        let Some((start, end)) = byte_subrange(self.len, offset, length) else {
            return;
        };
        let range_len = end - start;
        if range_len == 0 || self.view.Value.is_null() {
            return;
        }
        // SAFETY: `start + range_len <= self.len`; `self.view.Value` is the live mapping base.
        let base = unsafe { self.view.Value.add(start).cast_const() };
        // SAFETY: `base` and `range_len` describe a subrange of the mapping.
        unsafe {
            let _ = FlushViewOfFile(base, range_len);
        }
    }

    /// Mapped span length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for SharedMemoryView {
    fn drop(&mut self) {
        if !self.view.Value.is_null() {
            // SAFETY: `self.view` was mapped in `new`; unmapped exactly once on drop.
            unsafe {
                UnmapViewOfFile(self.view);
            }
        }
        if is_valid_handle(self.map_handle) {
            // SAFETY: `self.map_handle` was opened in `new`; closed exactly once on drop.
            unsafe {
                CloseHandle(self.map_handle);
            }
        }
    }
}

fn is_valid_handle(h: HANDLE) -> bool {
    !h.is_null() && h != INVALID_HANDLE_VALUE
}

fn create_or_open_file_mapping(name: &[u16], size: usize) -> io::Result<HANDLE> {
    // SAFETY: `name` is a NUL-terminated wide string; `INVALID_HANDLE_VALUE` requests an anonymous
    // pagefile-backed mapping.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            null(),
            PAGE_READWRITE,
            (size >> 32) as u32,
            (size & 0xFFFF_FFFF) as u32,
            name.as_ptr(),
        )
    };

    if is_valid_handle(handle) {
        return Ok(handle);
    }

    // SAFETY: `name` is a NUL-terminated wide string.
    let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, name.as_ptr()) };

    if is_valid_handle(handle) {
        return Ok(handle);
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Failed to create or open file mapping for shared memory buffer",
    ))
}
