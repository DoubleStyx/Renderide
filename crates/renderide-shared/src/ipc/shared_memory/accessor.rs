//! [`SharedMemoryAccessor`]: lazy map cache for host shared buffers.

use hashbrown::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::buffer::SharedMemoryBufferDescriptor;
use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::packing::memory_packable::MemoryPackable;
use crate::packing::memory_unpacker::MemoryUnpacker;
use crate::packing::wire_decode_error::WireDecodeError;

#[cfg(windows)]
use super::naming::compose_memory_view_name;
#[cfg(unix)]
use super::naming::unix_backing_file_path;
#[cfg(unix)]
use super::unix::SharedMemoryView;
#[cfg(windows)]
use super::windows::SharedMemoryView;

use super::bounds::required_view_capacity;

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
        profiling::scope!("shared_memory::with_read_bytes");
        if descriptor.length <= 0 {
            return None;
        }
        let view = self.get_view(descriptor)?;
        let bytes = view.slice(descriptor.offset, descriptor.length)?;
        f(bytes)
    }

    /// Releases a cached view (e.g. after [`RendererCommand::FreeSharedMemoryView`](crate::shared::RendererCommand::FreeSharedMemoryView)).
    pub fn release_view(&mut self, buffer_id: i32) {
        profiling::scope!("shared_memory::release_view");
        self.views.remove(&buffer_id);
    }

    fn get_view(&mut self, d: &SharedMemoryBufferDescriptor) -> Option<&mut SharedMemoryView> {
        profiling::scope!("shared_memory::get_view");
        let capacity = required_view_capacity(d)?;
        let buffer_id = d.buffer_id;
        if !self.views.contains_key(&buffer_id) {
            profiling::scope!("shared_memory::map_view");
            let view = SharedMemoryView::new(&self.prefix, buffer_id, capacity).ok()?;
            self.views.insert(buffer_id, view);
        }
        self.views.get_mut(&buffer_id)
    }

    /// Maximum bytes allocated for a single [`Self::access_copy`] (guards corrupt `length`).
    pub const MAX_ACCESS_COPY_BYTES: i32 = 64 * 1024 * 1024;

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
        profiling::scope!("shared_memory::access_copy");
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
        let Some(view) = self.get_view(descriptor) else {
            return Err(prefix_err(&format!(
                "get_view failed buffer_id={} path/name={}",
                buffer_id,
                self.shm_path_for_buffer(buffer_id)
            )));
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
        let type_size = size_of::<T>();
        let length = descriptor.length as usize;
        let remainder = length % type_size;
        if remainder != 0 {
            return Err(prefix_err(&format!(
                "length {length} is not a multiple of type size {type_size} (remainder {remainder})"
            )));
        }
        let count = length / type_size;
        if count == 0 {
            return Ok(Vec::new());
        }

        let align = align_of::<T>();
        let base = bytes.as_ptr() as usize;
        if base.is_multiple_of(align) {
            if let Ok(slice) = bytemuck::try_cast_slice::<u8, T>(bytes) {
                if slice.len() >= count {
                    return Ok(slice[..count].to_vec());
                }
            }
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * type_size;
            let chunk = bytes
                .get(start..start + type_size)
                .ok_or_else(|| prefix_err("pod chunk subslice"))?;
            let value = bytemuck::try_pod_read_unaligned::<T>(chunk)
                .map_err(|e| prefix_err(&format!("pod_read_unaligned: {e:?}")))?;
            out.push(value);
        }
        Ok(out)
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
        profiling::scope!("shared_memory::access_packable_rows");
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
        let Some(view) = self.get_view(descriptor) else {
            return Err(prefix_err(&format!(
                "get_view failed buffer_id={} path/name={}",
                buffer_id,
                self.shm_path_for_buffer(buffer_id)
            )));
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
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(element_stride) {
            let mut pool = DefaultEntityPool;
            let mut unpacker = MemoryUnpacker::new(chunk, &mut pool);
            let mut row = T::default();
            row.unpack(&mut unpacker).map_err(|e: WireDecodeError| {
                prefix_err(&format!("MemoryPackable::unpack: {e}"))
            })?;
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
        profiling::scope!("shared_memory::access_mut");
        if descriptor.length <= 0 {
            return false;
        }
        let Some(view) = self.get_view(descriptor) else {
            return false;
        };
        let Some(bytes) = view.slice_mut(descriptor.offset, descriptor.length) else {
            return false;
        };
        let type_size = size_of::<T>();
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
        {
            profiling::scope!("shared_memory::flush_mut");
            bytes.copy_from_slice(bytemuck::cast_slice(slice));
            view.flush_range(descriptor.offset, descriptor.length);
        }
        true
    }

    /// Mutably accesses raw bytes (no `Pod` requirement). Flushes after `f` returns.
    pub fn access_mut_bytes<F>(&mut self, descriptor: &SharedMemoryBufferDescriptor, f: F) -> bool
    where
        F: FnOnce(&mut [u8]),
    {
        profiling::scope!("shared_memory::access_mut_bytes");
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
        {
            profiling::scope!("shared_memory::flush_mut_bytes");
            view.flush_range(descriptor.offset, descriptor.length);
        }
        true
    }
}

#[cfg(test)]
mod access_copy_diagnostic_tests {
    use crate::buffer::SharedMemoryBufferDescriptor;

    use super::SharedMemoryAccessor;

    #[test]
    fn access_copy_rejects_non_positive_length() {
        let mut acc = SharedMemoryAccessor::new("pfx".into());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 1,
            buffer_capacity: 4096,
            offset: 0,
            length: 0,
        };
        let err = acc.access_copy_diagnostic::<u32>(&d).expect_err("length 0");
        assert!(err.contains("length<=0"), "unexpected message: {err}");
    }

    #[test]
    fn access_copy_rejects_length_above_max() {
        let mut acc = SharedMemoryAccessor::new("pfx".into());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 2,
            buffer_capacity: 4096,
            offset: 0,
            length: SharedMemoryAccessor::MAX_ACCESS_COPY_BYTES + 1,
        };
        let err = acc
            .access_copy_diagnostic::<u32>(&d)
            .expect_err("too large");
        assert!(err.contains("exceeds max"), "unexpected message: {err}");
    }

    #[test]
    fn access_copy_diagnostic_prefixes_context_on_early_validation_error() {
        let mut acc = SharedMemoryAccessor::new("pfx".into());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 3,
            buffer_capacity: 100,
            offset: 0,
            length: -1,
        };
        let err = acc
            .access_copy_diagnostic_with_context::<u32>(&d, Some("mesh_upload"))
            .expect_err("negative length");
        assert!(err.starts_with("mesh_upload:"), "unexpected message: {err}");
        assert!(err.contains("length<=0"), "unexpected message: {err}");
    }
}
