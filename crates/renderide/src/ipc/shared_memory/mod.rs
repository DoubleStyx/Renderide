//! Memory-mapped access to host-owned shared buffers (mesh payloads, textures, probe results, etc.).
//!
//! Naming follows Renderite `Helper.ComposeMemoryViewName` (see [`compose_memory_view_name`]).
//!
//! # Layout (Cloudtoid / Renderite interop)
//!
//! - **Windows**: named section `CT_IP_{prefix}_{bufferId:X}` via
//!   [`windows_sys`] file mapping (same prefix as [`interprocess`] on Windows).
//! - **Unix**: file `{composed}.qu` in the MMF directory (see below).
//!
//! # Backing directory on Unix
//!
//! Uses the same directory resolution as the bootstrapper
//! ([`RENDERIDE_INTERPROCESS_DIR_ENV`]): if set to a non-empty path, that directory holds
//! `{name}.qu` files; otherwise [`interprocess::default_memory_dir`] — Linux:
//! `/dev/shm/.cloudtoid/interprocess/mmf`, **macOS and other Unix**: `std::env::temp_dir()` +
//! `.cloudtoid/interprocess/mmf`. This matches the workspace `interprocess` crate and avoids
//! assuming `/dev/shm` exists on non-Linux Unix.
//!
//! Managed Cloudtoid historically used `/dev/shm` for any `PlatformID.Unix`; portable Rust stacks
//! should set [`RENDERIDE_INTERPROCESS_DIR_ENV`] consistently on host and renderer when defaults
//! differ from the host implementation.

use std::collections::HashMap;
use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};

use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::default_entity_pool::DefaultEntityPool;
use crate::shared::memory_packable::MemoryPackable;
use crate::shared::memory_unpacker::MemoryUnpacker;

/// Environment variable overriding the Unix directory for `.qu` MMF files (must match host /
/// bootstrapper). Same value as `bootstrapper::ipc::RENDERIDE_INTERPROCESS_DIR_ENV`.
pub const RENDERIDE_INTERPROCESS_DIR_ENV: &str = "RENDERIDE_INTERPROCESS_DIR";

/// Composes the memory view name per Renderite `Helper.ComposeMemoryViewName` (prefix + hex id).
pub fn compose_memory_view_name(prefix: &str, buffer_id: i32) -> String {
    format!("{}_{:X}", prefix, buffer_id)
}

/// Unix-only: resolved directory containing `{composed}.qu` backing files.
#[cfg(unix)]
pub(super) fn unix_mmf_backing_dir() -> PathBuf {
    std::env::var_os(RENDERIDE_INTERPROCESS_DIR_ENV)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(interprocess::default_memory_dir)
}

/// Full path to the `.qu` file for a buffer on Unix.
#[cfg(unix)]
pub(super) fn unix_backing_file_path(prefix: &str, buffer_id: i32) -> PathBuf {
    unix_mmf_backing_dir().join(format!(
        "{}.qu",
        compose_memory_view_name(prefix, buffer_id)
    ))
}

/// Converts `offset`/`length` into a valid byte subrange of `total_len`, or `None`.
pub(super) fn byte_subrange(total_len: usize, offset: i32, length: i32) -> Option<(usize, usize)> {
    let offset = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    let end = offset.checked_add(length)?;
    if end <= total_len {
        Some((offset, end))
    } else {
        None
    }
}

fn required_view_capacity(d: &SharedMemoryBufferDescriptor) -> Option<i32> {
    if d.length <= 0 {
        return None;
    }
    let cap = d.buffer_capacity.max(d.offset.saturating_add(d.length));
    if cap > 0 {
        Some(cap)
    } else {
        None
    }
}

#[cfg(unix)]
mod unix;

#[cfg(windows)]
mod windows;

#[cfg(unix)]
use unix::SharedMemoryView;

#[cfg(windows)]
use windows::SharedMemoryView;

/// Lazy mapping cache keyed by `buffer_id` for host shared buffers.
pub struct SharedMemoryAccessor {
    prefix: String,
    views: HashMap<i32, SharedMemoryView>,
}

impl SharedMemoryAccessor {
    /// Builds an accessor with the session prefix from [`RendererInitData::shared_memory_prefix`](crate::shared::RendererInitData::shared_memory_prefix).
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            views: HashMap::new(),
        }
    }

    /// Returns `true` if host buffers can be opened (`prefix` non-empty).
    pub fn is_available(&self) -> bool {
        !self.prefix.is_empty()
    }

    /// Diagnostic path (Unix) or mapping name (Windows) for `buffer_id`.
    pub fn shm_path_for_buffer(&self, buffer_id: i32) -> String {
        #[cfg(unix)]
        {
            unix_backing_file_path(&self.prefix, buffer_id)
                .display()
                .to_string()
        }
        #[cfg(windows)]
        {
            format!(
                "CT_IP_{}",
                compose_memory_view_name(&self.prefix, buffer_id)
            )
        }
    }

    /// Maps `descriptor` to a byte slice and runs `f` without copying the payload.
    ///
    /// The closure must not retain references beyond its return: the host may reuse the mapping.
    pub fn with_read_bytes<R>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        f: impl FnOnce(&[u8]) -> Option<R>,
    ) -> Option<R> {
        if descriptor.length <= 0 {
            return None;
        }
        let view = self.get_view(descriptor)?;
        let bytes = view.slice(descriptor.offset, descriptor.length)?;
        f(bytes)
    }

    /// Releases a cached view (e.g. after [`RendererCommand::free_shared_memory_view`](crate::shared::shared::RendererCommand::free_shared_memory_view)).
    pub fn release_view(&mut self, buffer_id: i32) {
        self.views.remove(&buffer_id);
    }

    fn get_view(&mut self, d: &SharedMemoryBufferDescriptor) -> Option<&mut SharedMemoryView> {
        let capacity = required_view_capacity(d)?;
        let buffer_id = d.buffer_id;
        if !self.views.contains_key(&buffer_id) {
            let view = SharedMemoryView::new(&self.prefix, buffer_id, capacity).ok()?;
            self.views.insert(buffer_id, view);
        }
        self.views.get_mut(&buffer_id)
    }

    /// Maximum bytes allocated for a single [`Self::access_copy`] (guards corrupt `length`).
    const MAX_ACCESS_COPY_BYTES: i32 = 64 * 1024 * 1024;

    /// Copy helper for small typed reads (tests / diagnostics). Prefer [`Self::with_read_bytes`] for large meshes.
    pub fn access_copy<T: Pod + Zeroable>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
    ) -> Option<Vec<T>> {
        self.access_copy_diagnostic(descriptor).ok()
    }

    /// Like [`Self::access_copy`] but returns a diagnostic error string.
    pub fn access_copy_diagnostic<T: Pod + Zeroable>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
    ) -> Result<Vec<T>, String> {
        self.access_copy_diagnostic_with_context(descriptor, None)
    }

    /// Like [`Self::access_copy_diagnostic`] with optional caller context for errors.
    pub fn access_copy_diagnostic_with_context<T: Pod + Zeroable>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        context: Option<&str>,
    ) -> Result<Vec<T>, String> {
        let prefix_err = |msg: &str| {
            if let Some(ctx) = context {
                format!("{ctx}: {msg}")
            } else {
                msg.to_string()
            }
        };
        if descriptor.length <= 0 {
            return Err(prefix_err(&format!(
                "length<=0 (buffer_id={} offset={} length={})",
                descriptor.buffer_id, descriptor.offset, descriptor.length
            )));
        }
        if descriptor.length > Self::MAX_ACCESS_COPY_BYTES {
            return Err(prefix_err(&format!(
                "length {} exceeds max {} (buffer_id={})",
                descriptor.length,
                Self::MAX_ACCESS_COPY_BYTES,
                descriptor.buffer_id
            )));
        }
        let buffer_id = descriptor.buffer_id;
        let view = match self.get_view(descriptor) {
            Some(v) => v,
            None => {
                return Err(prefix_err(&format!(
                    "get_view failed buffer_id={} path/name={}",
                    buffer_id,
                    self.shm_path_for_buffer(buffer_id)
                )));
            }
        };
        let bytes = view
            .slice(descriptor.offset, descriptor.length)
            .ok_or_else(|| {
                prefix_err(&format!(
                    "slice failed buffer_id={} offset={} length={} view_len={}",
                    buffer_id,
                    descriptor.offset,
                    descriptor.length,
                    view.len()
                ))
            })?;
        let type_size = std::mem::size_of::<T>();
        let count = descriptor.length as usize / type_size;
        if count == 0 {
            return Ok(Vec::new());
        }
        let length = descriptor.length as usize;
        let remainder = length % type_size;
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        let slice = bytemuck::try_cast_slice::<u8, T>(&aligned).map_err(|_| {
            prefix_err(&format!(
                "try_cast_slice failed: length={length} bytes, type_size={type_size}, length%type_size={remainder}"
            ))
        })?;
        if slice.len() < count {
            return Err(prefix_err(&format!(
                "slice.len()<count {}<{count}",
                slice.len()
            )));
        }
        Ok(slice[..count].to_vec())
    }

    /// Copies shared memory into host-sized rows and decodes each with [`MemoryPackable::unpack`].
    ///
    /// Use when `T` is not [`Pod`] but the host still blits rows of the same sequential byte layout as
    /// [`MemoryPackable`] (e.g. SIMD-aligned composites). `element_stride` must match the host record size.
    pub fn access_copy_memory_packable_rows<T: MemoryPackable + Default>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        element_stride: usize,
        context: Option<&str>,
    ) -> Result<Vec<T>, String> {
        let prefix_err = |msg: &str| {
            if let Some(ctx) = context {
                format!("{ctx}: {msg}")
            } else {
                msg.to_string()
            }
        };
        if element_stride == 0 {
            return Err(prefix_err("element_stride must be nonzero"));
        }
        if descriptor.length <= 0 {
            return Err(prefix_err(&format!(
                "length<=0 (buffer_id={} offset={} length={})",
                descriptor.buffer_id, descriptor.offset, descriptor.length
            )));
        }
        if descriptor.length > Self::MAX_ACCESS_COPY_BYTES {
            return Err(prefix_err(&format!(
                "length {} exceeds max {} (buffer_id={})",
                descriptor.length,
                Self::MAX_ACCESS_COPY_BYTES,
                descriptor.buffer_id
            )));
        }
        let buffer_id = descriptor.buffer_id;
        let view = match self.get_view(descriptor) {
            Some(v) => v,
            None => {
                return Err(prefix_err(&format!(
                    "get_view failed buffer_id={} path/name={}",
                    buffer_id,
                    self.shm_path_for_buffer(buffer_id)
                )));
            }
        };
        let bytes = view
            .slice(descriptor.offset, descriptor.length)
            .ok_or_else(|| {
                prefix_err(&format!(
                    "slice failed buffer_id={} offset={} length={} view_len={}",
                    buffer_id,
                    descriptor.offset,
                    descriptor.length,
                    view.len()
                ))
            })?;
        let length = descriptor.length as usize;
        let remainder = length % element_stride;
        if remainder != 0 {
            return Err(prefix_err(&format!(
                "length {length} is not a multiple of element_stride {element_stride} (remainder {remainder})"
            )));
        }
        let count = length / element_stride;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        let mut out = Vec::with_capacity(count);
        for chunk in aligned.chunks_exact(element_stride) {
            let mut pool = DefaultEntityPool;
            let mut unpacker = MemoryUnpacker::new(chunk, &mut pool);
            let mut row = T::default();
            row.unpack(&mut unpacker);
            if unpacker.remaining_data() != 0 {
                return Err(prefix_err(&format!(
                    "unpack left {} bytes unconsumed (stride {element_stride})",
                    unpacker.remaining_data()
                )));
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Mutably accesses shared memory as `T` slices: read-modify-write with flush so the host sees updates.
    ///
    /// Uses a temporary aligned buffer because mmap offsets may be unaligned for `T`.
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
        let Some(view) = self.get_view(descriptor) else {
            return false;
        };
        let Some(bytes) = view.slice_mut(descriptor.offset, descriptor.length) else {
            return false;
        };
        let type_size = std::mem::size_of::<T>();
        let count = descriptor.length as usize / type_size;
        if count == 0 {
            return false;
        }
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        let Ok(slice) = bytemuck::try_cast_slice_mut::<u8, T>(&mut aligned) else {
            return false;
        };
        if slice.len() < count {
            return false;
        }
        f(&mut slice[..count]);
        bytes.copy_from_slice(bytemuck::cast_slice(slice));
        view.flush_range(descriptor.offset, descriptor.length);
        true
    }

    /// Mutably accesses raw bytes (no `Pod` requirement). Flushes after `f` returns.
    pub fn access_mut_bytes<F>(&mut self, descriptor: &SharedMemoryBufferDescriptor, f: F) -> bool
    where
        F: FnOnce(&mut [u8]),
    {
        if descriptor.length <= 0 {
            return false;
        }
        let Some(view) = self.get_view(descriptor) else {
            return false;
        };
        let Some(bytes) = view.slice_mut(descriptor.offset, descriptor.length) else {
            return false;
        };
        f(bytes);
        view.flush_range(descriptor.offset, descriptor.length);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_memory_view_name_matches_renderite_helper() {
        assert_eq!(compose_memory_view_name("sess", 255), "sess_FF");
        assert_eq!(compose_memory_view_name("p", 0), "p_0");
    }

    #[test]
    fn byte_subrange_ok_and_rejects_overflow() {
        assert_eq!(byte_subrange(100, 10, 5), Some((10, 15)));
        assert_eq!(byte_subrange(100, 0, 100), Some((0, 100)));
        assert_eq!(byte_subrange(100, 99, 2), None);
        assert_eq!(byte_subrange(100, -1, 5), None);
    }
}
