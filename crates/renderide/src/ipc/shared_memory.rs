//! Shared memory accessor for reading and writing host-rendered data.
//!
//! - Unix: file-based mmap at /dev/shm/.cloudtoid/interprocess/mmf/{prefix}_{buffer_id:X}.qu
//! - Windows: named memory-mapped file CT_IP_{prefix}_{buffer_id:X} (matches Cloudtoid MemoryFileWindows)

use std::collections::HashMap;
use std::io;

use bytemuck::{Pod, Zeroable};

use crate::shared::buffer::SharedMemoryBufferDescriptor;

/// Composes the memory view name per Renderite.Shared.Helper.ComposeMemoryViewName.
fn compose_memory_view_name(prefix: &str, buffer_id: i32) -> String {
    format!("{}_{:X}", prefix, buffer_id)
}

#[cfg(unix)]
mod imp {
    use std::fs::OpenOptions;
    use std::path::PathBuf;

    use memmap2::MmapMut;

    use super::*;

    const MEMORY_FILE_PATH: &str = "/dev/shm/.cloudtoid/interprocess/mmf";

    pub struct SharedMemoryView {
        pub mmap: MmapMut,
    }

    impl SharedMemoryView {
        pub fn new(prefix: &str, buffer_id: i32, _capacity: i32) -> io::Result<Self> {
            let name = compose_memory_view_name(prefix, buffer_id);
            let path = PathBuf::from(MEMORY_FILE_PATH).join(format!("{}.qu", name));
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("{}: {}", path.display(), e),
                    )
                })?;
            let mmap = unsafe { MmapMut::map_mut(&file)? };
            Ok(Self { mmap })
        }

        pub fn slice(&self, offset: i32, length: i32) -> Option<&[u8]> {
            let offset = offset as usize;
            let length = length as usize;
            if offset + length <= self.mmap.len() {
                Some(&self.mmap[offset..offset + length])
            } else {
                None
            }
        }

        pub fn slice_mut(&mut self, offset: i32, length: i32) -> Option<&mut [u8]> {
            let offset = offset as usize;
            let length = length as usize;
            if offset + length <= self.mmap.len() {
                Some(&mut self.mmap[offset..offset + length])
            } else {
                None
            }
        }

        pub fn flush_range(&self, offset: i32, length: i32) {
            let offset = offset as usize;
            let length = length as usize;
            if offset + length <= self.mmap.len() && length > 0 {
                let _ = self.mmap.flush_range(offset, length);
            }
        }

        pub fn len(&self) -> usize {
            self.mmap.len()
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Memory::{
        MapViewOfFile, UnmapViewOfFile, FILE_MAP_ALL_ACCESS, FILE_MAP_WRITE,
    };

    use super::*;

    const MAP_NAME_PREFIX: &str = "CT_IP_";

    pub struct SharedMemoryView {
        map_handle: HANDLE,
        view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
        len: usize,
    }

    impl SharedMemoryView {
        pub fn new(prefix: &str, buffer_id: i32, capacity: i32) -> io::Result<Self> {
            let name = format!("{}{}", MAP_NAME_PREFIX, compose_memory_view_name(prefix, buffer_id));
            let size = capacity as usize;

            let name_wide: Vec<u16> = OsStr::new(&name)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();

            let map_handle = create_or_open_file_mapping(&name_wide, size);

            let view = unsafe {
                MapViewOfFile(
                    map_handle,
                    FILE_MAP_ALL_ACCESS | FILE_MAP_WRITE,
                    0,
                    0,
                    size,
                )
            };

            if view.Value.is_null() {
                unsafe { CloseHandle(map_handle) };
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("MapViewOfFile failed for {}", name),
                ));
            }

            Ok(Self {
                map_handle,
                view,
                len: size,
            })
        }

        pub fn slice(&self, offset: i32, length: i32) -> Option<&[u8]> {
            let offset = offset as usize;
            let length = length as usize;
            if offset + length <= self.len && !self.view.Value.is_null() {
                Some(unsafe {
                    std::slice::from_raw_parts(
                        self.view.Value.add(offset) as *const u8,
                        length,
                    )
                })
            } else {
                None
            }
        }

        pub fn slice_mut(&mut self, offset: i32, length: i32) -> Option<&mut [u8]> {
            let offset = offset as usize;
            let length = length as usize;
            if offset + length <= self.len && !self.view.Value.is_null() {
                Some(unsafe {
                    std::slice::from_raw_parts_mut(
                        self.view.Value.add(offset) as *mut u8,
                        length,
                    )
                })
            } else {
                None
            }
        }

        pub fn flush_range(&self, offset: i32, length: i32) {
            let offset = offset as usize;
            let length = length as usize;
            if offset + length <= self.len && length > 0 && !self.view.Value.is_null() {
                let base = unsafe { self.view.Value.add(offset) as *const std::ffi::c_void };
                let _ = unsafe {
                    windows_sys::Win32::System::Memory::FlushViewOfFile(base, length)
                };
            }
        }

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
            if self.map_handle != 0 && self.map_handle != -1 {
                unsafe {
                    CloseHandle(self.map_handle);
                }
            }
        }
    }

    fn create_or_open_file_mapping(name: &[u16], size: usize) -> HANDLE {
        use std::ptr::null;
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, OpenFileMappingW, FILE_MAP_ALL_ACCESS, PAGE_READWRITE,
        };

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

        if handle != 0 && handle != -1 {
            return handle;
        }

        let handle = unsafe {
            OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, name.as_ptr())
        };

        if handle != 0 && handle != -1 {
            return handle;
        }

        panic!(
            "Failed to create or open file mapping for shared memory buffer"
        );
    }
}

use imp::SharedMemoryView;

/// Accessor for shared memory buffers written by the host.
/// Creates views lazily and caches them by buffer_id.
pub struct SharedMemoryAccessor {
    prefix: String,
    views: HashMap<i32, SharedMemoryView>,
}

impl SharedMemoryAccessor {
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            views: HashMap::new(),
        }
    }

    fn compose_memory_view_name(&self, buffer_id: i32) -> String {
        compose_memory_view_name(&self.prefix, buffer_id)
    }

    /// Returns true if the accessor has a valid prefix (can attempt access).
    pub fn is_available(&self) -> bool {
        !self.prefix.is_empty()
    }

    /// Returns the path/name we use for the given buffer_id (for diagnostic logging).
    /// Unix: /dev/shm/.cloudtoid/interprocess/mmf/{prefix}_{buffer_id:X}.qu
    /// Windows: CT_IP_{prefix}_{buffer_id:X}
    pub fn shm_path_for_buffer(&self, buffer_id: i32) -> String {
        let name = self.compose_memory_view_name(buffer_id);
        #[cfg(unix)]
        {
            std::path::PathBuf::from("/dev/shm/.cloudtoid/interprocess/mmf")
                .join(format!("{}.qu", name))
                .display()
                .to_string()
        }
        #[cfg(windows)]
        {
            format!("CT_IP_{}", name)
        }
    }

    /// Copy data from shared memory into a Vec. Returns None if descriptor is empty,
    /// prefix is missing, or the file cannot be opened.
    ///
    /// For safety across frames, we copy rather than return references, since the
    /// host may reuse or free the buffer.
    pub fn access_copy<T: Pod + Zeroable>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
    ) -> Option<Vec<T>> {
        self.access_copy_diagnostic(descriptor).ok()
    }

    /// Max bytes we will allocate for a single access_copy (guards against OOM from corrupt host data).
    const MAX_ACCESS_COPY_BYTES: i32 = 64 * 1024 * 1024; // 64 MiB

    /// Like access_copy but returns Err with a diagnostic string on failure.
    pub fn access_copy_diagnostic<T: Pod + Zeroable>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
    ) -> Result<Vec<T>, String> {
        if descriptor.length <= 0 {
            return Err("length<=0".into());
        }
        if descriptor.length > Self::MAX_ACCESS_COPY_BYTES {
            return Err(format!(
                "length {} exceeds max {} (buffer_id={})",
                descriptor.length,
                Self::MAX_ACCESS_COPY_BYTES,
                descriptor.buffer_id
            ));
        }
        let buffer_id = descriptor.buffer_id;
        let capacity = descriptor
            .buffer_capacity
            .max(descriptor.offset + descriptor.length);
        if capacity <= 0 {
            return Err(format!(
                "capacity<=0 (buffer_id={} offset={} length={})",
                buffer_id, descriptor.offset, descriptor.length
            ));
        }
        let view = match self.get_view(descriptor) {
            Some(v) => v,
            None => {
                return Err(format!(
                    "get_view failed buffer_id={} name={}",
                    buffer_id,
                    self.compose_memory_view_name(buffer_id)
                ));
            }
        };
        let bytes = view
            .slice(descriptor.offset, descriptor.length)
            .ok_or_else(|| {
                format!(
                    "slice failed buffer_id={} offset={} length={} view_len={}",
                    buffer_id,
                    descriptor.offset,
                    descriptor.length,
                    view.len()
                )
            })?;
        let count = descriptor.length as usize / std::mem::size_of::<T>();
        if count == 0 {
            return Ok(Vec::new());
        }
        // Copy to aligned buffer: mmap slices at arbitrary offsets may be unaligned for T
        // (e.g. i32 needs 4-byte alignment), causing bytemuck::try_cast_slice to fail.
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        let slice =
            bytemuck::try_cast_slice::<u8, T>(&aligned).map_err(|_| "try_cast_slice failed")?;
        if slice.len() < count {
            return Err(format!("slice.len()<count {}<{}", slice.len(), count));
        }
        Ok(slice[..count].to_vec())
    }

    /// Mutably access shared memory for writing (e.g. ReflectionProbeSH2Task results).
    /// Returns false if descriptor is empty or access fails.
    /// Flushes the modified region so the host process sees our writes.
    pub fn access_mut<T: Pod + Zeroable, F>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        f: F,
    ) -> bool
    where
        F: FnOnce(&mut [T]),
    {
        if descriptor.length <= 0 {
            return false;
        }
        let view = match self.get_view(descriptor) {
            Some(v) => v,
            None => return false,
        };
        let bytes = match view.slice_mut(descriptor.offset, descriptor.length) {
            Some(b) => b,
            None => return false,
        };
        let count = descriptor.length as usize / std::mem::size_of::<T>();
        if count == 0 {
            return false;
        }
        // Copy to aligned buffer for same reason as access_copy (unaligned mmap offsets).
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        let slice = match bytemuck::try_cast_slice_mut::<u8, T>(&mut aligned) {
            Ok(s) => s,
            Err(_) => return false,
        };
        if slice.len() < count {
            return false;
        }
        f(&mut slice[..count]);
        // Copy back and flush so the host process sees our writes.
        bytes.copy_from_slice(bytemuck::cast_slice(slice));
        view.flush_range(descriptor.offset, descriptor.length);
        true
    }

    /// Mutably access shared memory as raw bytes for types that don't implement Pod.
    /// Use for manually patching fields (e.g. ReflectionProbeSH2Task.result).
    /// Flushes the modified region so the host process sees our writes.
    pub fn access_mut_bytes<F>(&mut self, descriptor: &SharedMemoryBufferDescriptor, f: F) -> bool
    where
        F: FnOnce(&mut [u8]),
    {
        if descriptor.length <= 0 {
            return false;
        }
        let view = match self.get_view(descriptor) {
            Some(v) => v,
            None => return false,
        };
        let bytes = match view.slice_mut(descriptor.offset, descriptor.length) {
            Some(b) => b,
            None => return false,
        };
        f(bytes);
        view.flush_range(descriptor.offset, descriptor.length);
        true
    }

    fn get_view(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
    ) -> Option<&mut SharedMemoryView> {
        if descriptor.length <= 0 {
            return None;
        }
        let buffer_id = descriptor.buffer_id;
        let capacity = descriptor
            .buffer_capacity
            .max(descriptor.offset + descriptor.length);
        if capacity <= 0 {
            return None;
        }
        if !self.views.contains_key(&buffer_id) {
            let view = SharedMemoryView::new(&self.prefix, buffer_id, capacity).ok()?;
            self.views.insert(buffer_id, view);
        }
        self.views.get_mut(&buffer_id)
    }

    /// Release a view (e.g. when host sends FreeSharedMemoryView).
    pub fn release_view(&mut self, buffer_id: i32) {
        self.views.remove(&buffer_id);
    }
}
