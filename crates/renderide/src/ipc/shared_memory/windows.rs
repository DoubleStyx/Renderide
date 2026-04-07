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

use super::{byte_subrange, compose_memory_view_name};

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

        let view =
            unsafe { MapViewOfFile(map_handle, FILE_MAP_ALL_ACCESS | FILE_MAP_WRITE, 0, 0, size) };

        if view.Value.is_null() {
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
        Some(unsafe {
            std::slice::from_raw_parts(self.view.Value.add(start) as *const u8, end - start)
        })
    }

    /// Returns the mutable byte subregion or `None` if out of bounds.
    pub fn slice_mut(&mut self, offset: i32, length: i32) -> Option<&mut [u8]> {
        let (start, end) = byte_subrange(self.len, offset, length)?;
        if self.view.Value.is_null() {
            return None;
        }
        Some(unsafe {
            std::slice::from_raw_parts_mut(self.view.Value.add(start) as *mut u8, end - start)
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
        let base = unsafe { self.view.Value.add(start) as *const std::ffi::c_void };
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
            unsafe {
                UnmapViewOfFile(self.view);
            }
        }
        if is_valid_handle(self.map_handle) {
            unsafe {
                CloseHandle(self.map_handle);
            }
        }
    }
}

fn is_valid_handle(h: HANDLE) -> bool {
    h != 0 && h != INVALID_HANDLE_VALUE
}

fn create_or_open_file_mapping(name: &[u16], size: usize) -> io::Result<HANDLE> {
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

    let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, name.as_ptr()) };

    if is_valid_handle(handle) {
        return Ok(handle);
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Failed to create or open file mapping for shared memory buffer",
    ))
}
