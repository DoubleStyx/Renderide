//! [`SharedMemoryAccessor`]: lazy map cache for host shared buffers.

use hashbrown::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::buffer::SharedMemoryBufferDescriptor;
use crate::packing::memory_packable::MemoryPackable;

#[cfg(windows)]
use super::naming::compose_memory_view_name;
#[cfg(unix)]
use super::naming::unix_backing_file_path;
#[cfg(unix)]
use super::unix::SharedMemoryView;
#[cfg(windows)]
use super::windows::SharedMemoryView;

use super::bounds::required_view_capacity;
use super::diagnostics::{
    describe_descriptor_failure, describe_get_view_failure, describe_slice_failure,
    describe_slice_failure_with_descriptor, log_shared_memory_read_failure, make_context_prefixer,
};

mod copy;
mod rows;
mod validation;

use copy::copy_pod_slice;
use rows::{pack_memory_packable_row, unpack_memory_packable_row};
use validation::{validate_access_copy_descriptor, validate_memory_packable_row_descriptor};

/// Lazy mapping cache keyed by `buffer_id` for host shared buffers.
pub struct SharedMemoryAccessor {
    prefix: String,
    views: HashMap<i32, SharedMemoryView>,
}

impl SharedMemoryAccessor {
    /// Maximum bytes allocated for a single [`Self::access_copy`] (guards corrupt `length`).
    pub const MAX_ACCESS_COPY_BYTES: i32 = 64 * 1024 * 1024;

    /// Builds an accessor with the session prefix from [`RendererInitData::shared_memory_prefix`](crate::shared::RendererInitData::shared_memory_prefix).
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            views: HashMap::new(),
        }
    }

    /// Returns `true` if host buffers can be opened (`prefix` non-empty).
    pub const fn is_available(&self) -> bool {
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
            log_shared_memory_read_failure(&describe_descriptor_failure(
                descriptor,
                "non-positive length",
                None,
            ));
            return None;
        }
        let path_for_diag = self.shm_path_for_buffer(descriptor.buffer_id);
        let Some(view) = self.get_view(descriptor) else {
            log_shared_memory_read_failure(&describe_descriptor_failure(
                descriptor,
                "get_view failed",
                Some(path_for_diag.as_str()),
            ));
            return None;
        };
        let Some(bytes) = view.slice(descriptor.offset, descriptor.length) else {
            log_shared_memory_read_failure(&describe_slice_failure_with_descriptor(
                descriptor,
                view.len(),
            ));
            return None;
        };
        f(bytes)
    }

    /// Releases a cached view (e.g. after [`RendererCommand::FreeSharedMemoryView`](crate::shared::RendererCommand::FreeSharedMemoryView)).
    pub fn release_view(&mut self, buffer_id: i32) {
        profiling::scope!("shared_memory::release_view");
        self.views.remove(&buffer_id);
    }

    fn get_view(&mut self, d: &SharedMemoryBufferDescriptor) -> Option<&mut SharedMemoryView> {
        profiling::scope!("shared_memory::get_view");
        if self.prefix.is_empty() {
            return None;
        }
        let capacity = required_view_capacity(d)?;
        let buffer_id = d.buffer_id;
        if !self.views.contains_key(&buffer_id) {
            profiling::scope!("shared_memory::map_view");
            let view = SharedMemoryView::new(&self.prefix, buffer_id, capacity).ok()?;
            self.views.insert(buffer_id, view);
        }
        self.views.get_mut(&buffer_id)
    }

    /// Resolves `descriptor` to an immutable byte slice and runs `f` against it.
    ///
    /// Routes `get_view` / `slice` failures through `prefix_err` and the diagnostics formatters so
    /// every `access_copy_*` flavour produces the same error wording for the mapping/slicing
    /// steps. Callers handle their own descriptor- and payload-shape validation before invoking.
    fn with_validated_slice<R>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        prefix_err: &impl Fn(&str) -> String,
        f: impl FnOnce(&[u8]) -> Result<R, String>,
    ) -> Result<R, String> {
        let buffer_id = descriptor.buffer_id;
        let path_for_diag = self.shm_path_for_buffer(buffer_id);
        let Some(view) = self.get_view(descriptor) else {
            return Err(prefix_err(&describe_get_view_failure(
                buffer_id,
                &path_for_diag,
            )));
        };
        let view_len = view.len();
        let bytes = view
            .slice(descriptor.offset, descriptor.length)
            .ok_or_else(|| {
                prefix_err(&describe_slice_failure(
                    buffer_id,
                    descriptor.offset,
                    descriptor.length,
                    view_len,
                ))
            })?;
        f(bytes)
    }

    /// Resolves `descriptor` to a mutable byte slice, runs `f`, and flushes the range when `f`
    /// returns `true`. Returns `false` on any pre-`f` failure (or when `f` itself returns `false`).
    fn with_validated_slice_mut<F: FnOnce(&mut [u8]) -> bool>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        f: F,
    ) -> bool {
        if descriptor.length <= 0 {
            return false;
        }
        let Some(view) = self.get_view(descriptor) else {
            return false;
        };
        let Some(bytes) = view.slice_mut(descriptor.offset, descriptor.length) else {
            return false;
        };
        if !f(bytes) {
            return false;
        }
        view.flush_range(descriptor.offset, descriptor.length);
        true
    }

    /// Resolves `descriptor` to a mutable byte slice, runs `f`, and flushes the range after a
    /// successful mutation. Errors are routed through `prefix_err` for call-site diagnostics.
    fn with_validated_slice_mut_result<R>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        prefix_err: &impl Fn(&str) -> String,
        f: impl FnOnce(&mut [u8]) -> Result<R, String>,
    ) -> Result<R, String> {
        let buffer_id = descriptor.buffer_id;
        let path_for_diag = self.shm_path_for_buffer(buffer_id);
        let Some(view) = self.get_view(descriptor) else {
            return Err(prefix_err(&describe_get_view_failure(
                buffer_id,
                &path_for_diag,
            )));
        };
        let view_len = view.len();
        let result = {
            let bytes = view
                .slice_mut(descriptor.offset, descriptor.length)
                .ok_or_else(|| {
                    prefix_err(&describe_slice_failure(
                        buffer_id,
                        descriptor.offset,
                        descriptor.length,
                        view_len,
                    ))
                })?;
            f(bytes)?
        };
        view.flush_range(descriptor.offset, descriptor.length);
        Ok(result)
    }

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
        let prefix_err = make_context_prefixer(context);
        validate_access_copy_descriptor(descriptor, Self::MAX_ACCESS_COPY_BYTES, &prefix_err)?;
        self.with_validated_slice(descriptor, &prefix_err, |bytes| {
            copy_pod_slice::<T>(bytes, descriptor.length as usize, &prefix_err)
        })
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
        self.access_copy_memory_packable_rows_with_max(
            descriptor,
            element_stride,
            Self::MAX_ACCESS_COPY_BYTES,
            context,
        )
    }

    /// Copies shared memory into host-sized rows with a caller-specified byte ceiling.
    ///
    /// Use this for row buffers that are expected to exceed [`Self::MAX_ACCESS_COPY_BYTES`] after
    /// independent protocol validation has established a safe upper bound.
    pub fn access_copy_memory_packable_rows_with_max<T: MemoryPackable + Default>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        element_stride: usize,
        max_bytes: i32,
        context: Option<&str>,
    ) -> Result<Vec<T>, String> {
        profiling::scope!("shared_memory::access_packable_rows");
        let prefix_err = make_context_prefixer(context);
        validate_memory_packable_row_descriptor(
            descriptor,
            element_stride,
            max_bytes,
            &prefix_err,
        )?;
        self.with_validated_slice(descriptor, &prefix_err, |bytes| {
            let count = descriptor.length as usize / element_stride;
            if count == 0 {
                return Ok(Vec::new());
            }
            let mut out = Vec::with_capacity(count);
            for chunk in bytes.chunks_exact(element_stride) {
                out.push(unpack_memory_packable_row::<T>(
                    chunk,
                    element_stride,
                    &prefix_err,
                )?);
            }
            Ok(out)
        })
    }

    /// Copies host-sized [`MemoryPackable`] rows until `stop_after` returns `true` for a decoded row.
    ///
    /// The matching row is included in the returned vector. Use this for shared-memory buffers
    /// whose descriptor length may cover a larger reserved slab while an in-band sentinel marks
    /// the active prefix.
    pub fn access_copy_memory_packable_rows_until_with_max<T, F>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        element_stride: usize,
        max_bytes: i32,
        context: Option<&str>,
        mut stop_after: F,
    ) -> Result<Vec<T>, String>
    where
        T: MemoryPackable + Default,
        F: FnMut(&T) -> bool,
    {
        profiling::scope!("shared_memory::access_packable_rows_until");
        let prefix_err = make_context_prefixer(context);
        validate_memory_packable_row_descriptor(
            descriptor,
            element_stride,
            max_bytes,
            &prefix_err,
        )?;
        self.with_validated_slice(descriptor, &prefix_err, |bytes| {
            let count = descriptor.length as usize / element_stride;
            if count == 0 {
                return Ok(Vec::new());
            }
            let mut out = Vec::with_capacity(count.min(1024));
            for chunk in bytes.chunks_exact(element_stride) {
                let row = unpack_memory_packable_row::<T>(chunk, element_stride, &prefix_err)?;
                let stop = stop_after(&row);
                out.push(row);
                if stop {
                    break;
                }
            }
            Ok(out)
        })
    }

    /// Mutates host-sized [`MemoryPackable`] rows until `mutate` returns `true`.
    ///
    /// The matching row is decoded, offered to `mutate`, repacked, and included in the returned
    /// count. Use this for host writeback buffers that carry sentinel-terminated rows whose
    /// element type is not [`Pod`].
    pub fn access_mut_memory_packable_rows_until_with_max<T, F>(
        &mut self,
        descriptor: &SharedMemoryBufferDescriptor,
        element_stride: usize,
        max_bytes: i32,
        context: Option<&str>,
        mut mutate: F,
    ) -> Result<usize, String>
    where
        T: MemoryPackable + Default,
        F: FnMut(&mut T) -> bool,
    {
        profiling::scope!("shared_memory::access_mut_packable_rows_until");
        let prefix_err = make_context_prefixer(context);
        validate_memory_packable_row_descriptor(
            descriptor,
            element_stride,
            max_bytes,
            &prefix_err,
        )?;
        self.with_validated_slice_mut_result(descriptor, &prefix_err, |bytes| {
            let mut count = 0usize;
            for chunk in bytes.chunks_exact_mut(element_stride) {
                let mut row = unpack_memory_packable_row::<T>(chunk, element_stride, &prefix_err)?;
                let stop = mutate(&mut row);
                pack_memory_packable_row(&mut row, chunk, element_stride, &prefix_err)?;
                count = count.saturating_add(1);
                if stop {
                    break;
                }
            }
            Ok(count)
        })
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
        let type_size = size_of::<T>();
        let count = descriptor.length as usize / type_size;
        if count == 0 {
            return false;
        }
        self.with_validated_slice_mut(descriptor, |bytes| {
            let mut aligned = vec![0u8; bytes.len()];
            aligned.copy_from_slice(bytes);
            let Ok(slice) = bytemuck::try_cast_slice_mut::<u8, T>(&mut aligned) else {
                return false;
            };
            if slice.len() < count {
                return false;
            }
            f(&mut slice[..count]);
            profiling::scope!("shared_memory::flush_mut");
            bytes.copy_from_slice(bytemuck::cast_slice(slice));
            true
        })
    }

    /// Mutably accesses raw bytes (no `Pod` requirement). Flushes after `f` returns.
    pub fn access_mut_bytes<F>(&mut self, descriptor: &SharedMemoryBufferDescriptor, f: F) -> bool
    where
        F: FnOnce(&mut [u8]),
    {
        profiling::scope!("shared_memory::access_mut_bytes");
        self.with_validated_slice_mut(descriptor, |bytes| {
            f(bytes);
            profiling::scope!("shared_memory::flush_mut_bytes");
            true
        })
    }
}

#[cfg(test)]
mod access_copy_diagnostic_tests {
    use crate::buffer::SharedMemoryBufferDescriptor;
    use crate::ipc::shared_memory::diagnostics::{
        describe_descriptor_failure, describe_get_view_failure, describe_slice_failure,
        make_context_prefixer,
    };
    use crate::ipc::shared_memory::writer::{SharedMemoryWriter, SharedMemoryWriterConfig};
    use crate::packing::extras::SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES;
    use crate::packing::memory_packable::MemoryPackable;
    use crate::packing::memory_packer::MemoryPacker;
    use crate::shared::{
        RenderBoundingBox, RenderTransform, SkinnedMeshRealtimeBoundsUpdate,
        TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES, TransformPoseUpdate,
    };
    use crate::wire_writer::{TransformPoseRow, encode_transform_pose_updates};

    use super::{
        SharedMemoryAccessor, validate_access_copy_descriptor,
        validate_memory_packable_row_descriptor,
    };

    fn unique_prefix(label: &str) -> String {
        format!("renderide_test_accessor_{label}_{}", std::process::id())
    }

    #[test]
    fn with_read_bytes_returns_none_when_prefix_is_empty() {
        let mut acc = SharedMemoryAccessor::new(String::new());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 0,
            buffer_capacity: 16,
            offset: 0,
            length: 16,
        };

        let read = acc.with_read_bytes(&d, |_bytes| Some(()));

        assert!(!acc.is_available());
        assert!(read.is_none());
    }

    #[test]
    fn with_read_bytes_returns_none_when_mapping_is_missing() {
        let prefix = unique_prefix("missing_read");
        let mut acc = SharedMemoryAccessor::new(prefix);
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 41,
            buffer_capacity: 16,
            offset: 0,
            length: 16,
        };

        let read = acc.with_read_bytes(&d, |_bytes| Some(()));

        assert!(read.is_none());
    }

    fn encode_realtime_bounds_rows(rows: &mut [SkinnedMeshRealtimeBoundsUpdate]) -> Vec<u8> {
        let mut bytes = vec![0u8; rows.len() * SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES];
        for (row, chunk) in rows
            .iter_mut()
            .zip(bytes.chunks_exact_mut(SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES))
        {
            let mut packer = MemoryPacker::new(chunk);
            row.pack(&mut packer);
            assert_eq!(packer.remaining_len(), 0, "test row must fill host stride");
        }
        bytes
    }

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
    fn context_prefixer_adds_context_only_when_present() {
        let with_context = make_context_prefixer(Some("mesh"));
        let without_context = make_context_prefixer(None);

        assert_eq!(with_context("bad descriptor"), "mesh: bad descriptor");
        assert_eq!(without_context("bad descriptor"), "bad descriptor");
    }

    #[test]
    fn descriptor_validation_accepts_length_at_maximum() {
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 1,
            buffer_capacity: 8,
            offset: 0,
            length: 8,
        };
        let prefix = make_context_prefixer(Some("copy"));

        assert!(validate_access_copy_descriptor(&d, 8, &prefix).is_ok());
    }

    #[test]
    fn descriptor_failure_includes_descriptor_fields() {
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 17,
            buffer_capacity: 4096,
            offset: 12,
            length: 34,
        };
        let msg = describe_descriptor_failure(&d, "get_view failed", Some("path"));
        assert!(msg.contains("buffer_id=17"), "message={msg}");
        assert!(msg.contains("offset=12"), "message={msg}");
        assert!(msg.contains("length=34"), "message={msg}");
        assert!(msg.contains("capacity=4096"), "message={msg}");
        assert!(msg.contains("path/name=path"), "message={msg}");
    }

    #[test]
    fn get_view_and_slice_failure_messages_include_mapping_context() {
        let get = describe_get_view_failure(44, "/tmp/missing.qu");
        assert!(get.contains("buffer_id=44"), "message={get}");
        assert!(get.contains("/tmp/missing.qu"), "message={get}");

        let slice = describe_slice_failure(45, 4, 12, 8);
        assert!(slice.contains("buffer_id=45"), "message={slice}");
        assert!(slice.contains("offset=4"), "message={slice}");
        assert!(slice.contains("length=12"), "message={slice}");
        assert!(slice.contains("view_len=8"), "message={slice}");
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
    fn memory_packable_row_descriptor_rejects_zero_stride_before_mapping() {
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 7,
            buffer_capacity: 128,
            offset: 0,
            length: 32,
        };
        let prefix = make_context_prefixer(Some("rows"));

        let err =
            validate_memory_packable_row_descriptor(&d, 0, 128, &prefix).expect_err("zero stride");

        assert_eq!(err, "rows: element_stride must be nonzero");
    }

    #[test]
    fn memory_packable_row_descriptor_rejects_misaligned_length_before_mapping() {
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 8,
            buffer_capacity: 128,
            offset: 0,
            length: 10,
        };
        let prefix = make_context_prefixer(None);

        let err = validate_memory_packable_row_descriptor(&d, 4, 128, &prefix)
            .expect_err("misaligned length");

        assert!(
            err.contains("is not a multiple of element_stride 4"),
            "unexpected message: {err}"
        );
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

    #[test]
    fn access_copy_rejects_non_multiple_type_size_after_mapping() {
        let prefix = unique_prefix("non_multiple");
        let cfg = SharedMemoryWriterConfig {
            prefix: prefix.clone(),
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let mut writer = SharedMemoryWriter::open(cfg, 12, 3).expect("open writer");
        writer.write_at(0, &[1, 2, 3]).expect("write bytes");
        writer.flush();
        let descriptor = writer.descriptor_for(0, 3);

        let mut acc = SharedMemoryAccessor::new(prefix);
        let err = acc
            .access_copy_diagnostic::<u16>(&descriptor)
            .expect_err("length is not multiple of u16");

        assert!(
            err.contains("is not a multiple of type size 2"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn access_mut_bytes_writes_through_cached_mapping() {
        let prefix = unique_prefix("mut_bytes");
        let cfg = SharedMemoryWriterConfig {
            prefix: prefix.clone(),
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let mut writer = SharedMemoryWriter::open(cfg, 13, 4).expect("open writer");
        writer.write_at(0, &[1, 2, 3, 4]).expect("write bytes");
        writer.flush();
        let descriptor = writer.descriptor_for(0, 4);

        let mut acc = SharedMemoryAccessor::new(prefix);
        assert!(acc.access_mut_bytes(&descriptor, |bytes| {
            bytes.copy_from_slice(&[9, 8, 7, 6]);
        }));
        let readback = acc
            .access_copy_diagnostic::<u8>(&descriptor)
            .expect("read mutated bytes");

        assert_eq!(readback, vec![9, 8, 7, 6]);
    }

    #[test]
    fn access_copy_packable_rows_respects_caller_max() {
        let mut acc = SharedMemoryAccessor::new("pfx".into());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 4,
            buffer_capacity: 4096,
            offset: 0,
            length: 128,
        };
        let err = acc
            .access_copy_memory_packable_rows_with_max::<TransformPoseUpdate>(
                &d,
                TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
                64,
                Some("pose_rows"),
            )
            .expect_err("custom max should reject before mapping");
        assert!(err.starts_with("pose_rows:"), "unexpected message: {err}");
        assert!(err.contains("exceeds max 64"), "unexpected message: {err}");
    }

    #[test]
    fn access_copy_packable_rows_until_stops_after_matching_row() {
        let prefix = unique_prefix("until_stops");
        let bytes = encode_transform_pose_updates(&[
            TransformPoseRow {
                transform_id: 7,
                pose: RenderTransform::default(),
            },
            TransformPoseRow {
                transform_id: -1,
                pose: RenderTransform::default(),
            },
            TransformPoseRow {
                transform_id: 99,
                pose: RenderTransform::default(),
            },
        ]);
        let cfg = SharedMemoryWriterConfig {
            prefix: prefix.clone(),
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let mut writer = SharedMemoryWriter::open(cfg, 11, bytes.len()).expect("open writer");
        writer.write_at(0, &bytes).expect("write rows");
        writer.flush();
        let descriptor = writer.descriptor_for(0, bytes.len() as i32);

        let mut acc = SharedMemoryAccessor::new(prefix);
        let rows = acc
            .access_copy_memory_packable_rows_until_with_max::<TransformPoseUpdate, _>(
                &descriptor,
                TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
                descriptor.length,
                Some("pose_rows"),
                |row| row.transform_id < 0,
            )
            .expect("decode rows");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].transform_id, 7);
        assert_eq!(rows[1].transform_id, -1);
    }

    #[test]
    fn access_copy_packable_rows_until_rejects_misaligned_length() {
        let mut acc = SharedMemoryAccessor::new("pfx".into());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 5,
            buffer_capacity: 100,
            offset: 0,
            length: TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES as i32 + 1,
        };
        let err = acc
            .access_copy_memory_packable_rows_until_with_max::<TransformPoseUpdate, _>(
                &d,
                TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
                100,
                Some("pose_rows"),
                |row| row.transform_id < 0,
            )
            .expect_err("misaligned length");

        assert!(err.starts_with("pose_rows:"), "unexpected message: {err}");
        assert!(
            err.contains("is not a multiple of element_stride"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn access_copy_packable_rows_until_rejects_over_max_before_mapping() {
        let mut acc = SharedMemoryAccessor::new("pfx".into());
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 6,
            buffer_capacity: 4096,
            offset: 0,
            length: 128,
        };
        let err = acc
            .access_copy_memory_packable_rows_until_with_max::<TransformPoseUpdate, _>(
                &d,
                TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
                64,
                Some("pose_rows"),
                |row| row.transform_id < 0,
            )
            .expect_err("custom max should reject before mapping");

        assert!(err.starts_with("pose_rows:"), "unexpected message: {err}");
        assert!(err.contains("exceeds max 64"), "unexpected message: {err}");
    }

    #[test]
    fn access_mut_packable_rows_until_writes_rows_and_stops_at_sentinel() {
        let prefix = unique_prefix("mut_packable_until");
        let mut source_rows = [
            SkinnedMeshRealtimeBoundsUpdate {
                renderable_index: 0,
                computed_global_bounds: RenderBoundingBox {
                    center: glam::Vec3::new(1.0, 2.0, 3.0),
                    extents: glam::Vec3::ONE,
                },
            },
            SkinnedMeshRealtimeBoundsUpdate {
                renderable_index: -1,
                computed_global_bounds: RenderBoundingBox {
                    center: glam::Vec3::new(4.0, 5.0, 6.0),
                    extents: glam::Vec3::ONE,
                },
            },
            SkinnedMeshRealtimeBoundsUpdate {
                renderable_index: 99,
                computed_global_bounds: RenderBoundingBox {
                    center: glam::Vec3::new(7.0, 8.0, 9.0),
                    extents: glam::Vec3::ONE,
                },
            },
        ];
        let bytes = encode_realtime_bounds_rows(&mut source_rows);
        let cfg = SharedMemoryWriterConfig {
            prefix: prefix.clone(),
            destroy_on_drop: true,
            ..SharedMemoryWriterConfig::default()
        };
        let mut writer = SharedMemoryWriter::open(cfg, 14, bytes.len()).expect("open writer");
        writer.write_at(0, &bytes).expect("write rows");
        writer.flush();
        let descriptor = writer.descriptor_for(0, bytes.len() as i32);

        let mut acc = SharedMemoryAccessor::new(prefix);
        let visited = acc
            .access_mut_memory_packable_rows_until_with_max::<SkinnedMeshRealtimeBoundsUpdate, _>(
                &descriptor,
                SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES,
                descriptor.length,
                Some("realtime_bounds"),
                |row| {
                    if row.renderable_index < 0 {
                        return true;
                    }
                    row.computed_global_bounds.center.x += 10.0;
                    false
                },
            )
            .expect("mutate rows");
        let rows = acc
            .access_copy_memory_packable_rows::<SkinnedMeshRealtimeBoundsUpdate>(
                &descriptor,
                SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES,
                Some("realtime_bounds"),
            )
            .expect("read rows");

        assert_eq!(visited, 2);
        assert_eq!(rows[0].computed_global_bounds.center.x, 11.0);
        assert_eq!(rows[1].computed_global_bounds.center.x, 4.0);
        assert_eq!(rows[2].computed_global_bounds.center.x, 7.0);
    }
}
