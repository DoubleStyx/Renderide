//! Shared memory accessor for reading and writing host-rendered data.
//!
//! Mirrors Renderite.Unity.SharedMemoryAccessor - opens mmap files at
//! /dev/shm/.cloudtoid/interprocess/mmf/{prefix}_{buffer_id:X}.qu

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io;
use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};
use memmap2::MmapMut;

use super::buffer::SharedMemoryBufferDescriptor;

const MEMORY_FILE_PATH: &str = "/dev/shm/.cloudtoid/interprocess/mmf";

/// Composes the memory view name per Renderite.Shared.Helper.ComposeMemoryViewName.
fn compose_memory_view_name(prefix: &str, buffer_id: i32) -> String {
    format!("{}_{:X}", prefix, buffer_id)
}

/// Cached mmap view for a buffer (mutable for writing back results).
struct SharedMemoryView {
    mmap: MmapMut,
}

impl SharedMemoryView {
    fn new(prefix: &str, buffer_id: i32, _capacity: i32) -> io::Result<Self> {
        let name = compose_memory_view_name(prefix, buffer_id);
        let path = PathBuf::from(MEMORY_FILE_PATH).join(format!("{}.qu", name));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("{}: {}", path.display(), e)))?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap })
    }

    fn slice(&self, offset: i32, length: i32) -> Option<&[u8]> {
        let offset = offset as usize;
        let length = length as usize;
        if offset + length <= self.mmap.len() {
            Some(&self.mmap[offset..offset + length])
        } else {
            None
        }
    }

    fn slice_mut(&mut self, offset: i32, length: i32) -> Option<&mut [u8]> {
        let offset = offset as usize;
        let length = length as usize;
        if offset + length <= self.mmap.len() {
            Some(&mut self.mmap[offset..offset + length])
        } else {
            None
        }
    }

    /// Flush modified region to backing store so other processes (e.g. host) see writes.
    fn flush_range(&self, offset: i32, length: i32) {
        let offset = offset as usize;
        let length = length as usize;
        if offset + length <= self.mmap.len() && length > 0 {
            let _ = self.mmap.flush_range(offset, length);
        }
    }
}

/// Accessor for shared memory buffers written by the host.
/// Creates views lazily and caches them by buffer_id.
pub struct SharedMemoryAccessor {
    prefix: String,
    views: HashMap<i32, SharedMemoryView>,
}

impl SharedMemoryView {
    fn mmap_len(&self) -> usize {
        self.mmap.len()
    }
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

    /// Returns the filesystem path we use for the given buffer_id (for diagnostic logging).
    /// Matches Cloudtoid: MEMORY_FILE_PATH / "{prefix}_{buffer_id:X}.qu"
    pub fn shm_path_for_buffer(&self, buffer_id: i32) -> String {
        let name = self.compose_memory_view_name(buffer_id);
        PathBuf::from(MEMORY_FILE_PATH)
            .join(format!("{}.qu", name))
            .display()
            .to_string()
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
        let capacity = descriptor.buffer_capacity.max(descriptor.offset + descriptor.length);
        if capacity <= 0 {
            return Err(format!("capacity<=0 (buffer_id={} offset={} length={})", buffer_id, descriptor.offset, descriptor.length));
        }
        let view = match self.get_view(descriptor) {
            Some(v) => v,
            None => {
                let name = self.compose_memory_view_name(buffer_id);
                let path = PathBuf::from(MEMORY_FILE_PATH).join(format!("{}.qu", name));
                return Err(format!("get_view failed buffer_id={} path={}", buffer_id, path.display()));
            }
        };
        let bytes = view.slice(descriptor.offset, descriptor.length).ok_or_else(|| {
            format!("slice failed buffer_id={} offset={} length={} mmap_len={}", buffer_id, descriptor.offset, descriptor.length, view.mmap_len())
        })?;
        let count = descriptor.length as usize / std::mem::size_of::<T>();
        if count == 0 {
            return Err("count==0".into());
        }
        // Copy to aligned buffer: mmap slices at arbitrary offsets may be unaligned for T
        // (e.g. i32 needs 4-byte alignment), causing bytemuck::try_cast_slice to fail.
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        let slice = bytemuck::try_cast_slice::<u8, T>(&aligned).map_err(|_| "try_cast_slice failed")?;
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

    fn get_view(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<&mut SharedMemoryView> {
        if descriptor.length <= 0 {
            return None;
        }
        let buffer_id = descriptor.buffer_id;
        let capacity = descriptor.buffer_capacity.max(descriptor.offset + descriptor.length);
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
