//! Helpers for [`super::GpuMesh::upload`](GpuMesh::upload); keeps the `impl` readable.

mod accounting;
mod blendshape;
mod bone_skin;
mod extended_streams;

pub(super) use accounting::resident_bytes_for_mesh_upload;
pub(super) use blendshape::{padded_sparse_bytes, upload_blendshape_buffer};
pub(super) use bone_skin::upload_bone_and_skin_buffers;
pub(super) use extended_streams::{
    ExtendedVertexUploadSource, UvVertexUploadSource, upload_default_extended_vertex_streams,
    upload_default_raw_tangent_vertex_stream, upload_default_tangent_vertex_stream,
    upload_default_uv_vertex_stream, upload_default_wide_uv_vertex_stream,
    upload_extended_vertex_streams, upload_raw_tangent_vertex_stream, upload_tangent_vertex_stream,
    upload_uv_vertex_stream, upload_wide_uv_vertex_stream,
};
#[cfg(test)]
use extended_streams::{tangent_stream_usage, vertex_stream_usage};

use std::borrow::Cow;
use std::sync::Arc;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::shared::MeshUploadData;

use super::super::layout::{
    MeshBufferLayout, color_float4_stream_bytes, compute_index_count, compute_vertex_stride,
    extract_float3_position_normal_as_vec4_streams, uv0_float2_stream_bytes,
};
use super::hints::wgpu_index_format;

/// Pair of optional derived vertex buffers.
type OptionalBufferPair = (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>);

/// Interleaved VB, IB, and layout-derived scalars after validation.
pub(super) struct CoreBuffers {
    pub vb: wgpu::Buffer,
    pub ib: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub vertex_stride: u32,
    pub vertex_stride_us: usize,
    pub index_count_u32: u32,
}

/// GPU handles and mapped-buffer generation captured for one mesh upload.
#[derive(Clone, Copy)]
pub(crate) struct MeshGpuUploadContext<'a> {
    /// Logical device used to create mesh buffers.
    pub device: &'a wgpu::Device,
    /// Queue used to initialize mesh buffers without mapped-at-creation writes.
    pub queue: &'a wgpu::Queue,
    /// Effective device limits used for upload validation.
    pub gpu_limits: &'a GpuLimits,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub mapped_buffer_health: &'a GpuMappedBufferHealth,
    /// Invalidation generation captured before the upload began.
    pub mapped_buffer_generation: u64,
}

/// Position/normal streams, UV0, and vertex color.
pub(super) struct DerivedStreams {
    pub positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub normals_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv0_buffer: Option<Arc<wgpu::Buffer>>,
    pub color_buffer: Option<Arc<wgpu::Buffer>>,
    pub tangent_buffer: Option<Arc<wgpu::Buffer>>,
    pub raw_tangent_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv1_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv2_buffer: Option<Arc<wgpu::Buffer>>,
    pub uv3_buffer: Option<Arc<wgpu::Buffer>>,
    pub wide_uv_buffer: Option<Arc<wgpu::Buffer>>,
}

/// Validates raw length and device buffer-size limits, including per-derived-stream sizes
/// that would otherwise reach `device.create_buffer_init` and trigger a fatal panic in
/// `wgpu`'s `get_mapped_range` when the underlying buffer creation fails validation.
///
/// Returns `false` when the upload must abort.
pub(super) fn validate_mesh_upload_layout(
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    gpu_limits: &GpuLimits,
) -> bool {
    if raw.len() < layout.total_buffer_length {
        logger::error!(
            "mesh {}: raw too short (need {}, got {})",
            data.asset_id,
            layout.total_buffer_length,
            raw.len()
        );
        return false;
    }

    let max_buf = gpu_limits.max_buffer_size();
    let max_storage = gpu_limits.max_storage_buffer_binding_size();
    let vc = data.vertex_count.max(0) as u64;

    // (label, size in bytes, requires storage-binding limit)
    //
    // Derived per-vertex storage streams (positions, normals, tangents, bone_indices,
    // bone_weights_vec4) are all `vc * 16` and bound to STORAGE bindings, so a single
    // entry covers all of them. Likewise the largest VERTEX-only derived stream (color)
    // is `vc * 16`.
    let checks: [(&str, u64, bool); 7] = [
        ("interleaved vertices", layout.vertex_size as u64, false),
        ("indices", layout.index_buffer_length as u64, false),
        (
            "total mesh layout",
            layout.total_buffer_length as u64,
            false,
        ),
        ("derived per-vertex storage stream", vc * 16, true),
        ("derived per-vertex vertex stream", vc * 16, false),
        ("bone_counts", layout.bone_counts_length as u64, true),
        ("bind_poses", layout.bind_poses_length as u64, true),
    ];

    for (label, size, is_storage) in checks {
        if size > max_buf {
            logger::warn!(
                "mesh {}: {} buffer ({} B) exceeds max_buffer_size ({} B)",
                data.asset_id,
                label,
                size,
                max_buf
            );
            return false;
        }
        if is_storage && size > max_storage {
            logger::warn!(
                "mesh {}: {} buffer ({} B) exceeds max_storage_buffer_binding_size ({} B)",
                data.asset_id,
                label,
                size,
                max_storage
            );
            return false;
        }
    }

    true
}

/// Returns the GPU buffer size needed for queue-backed initialization of `contents_len` bytes.
pub(super) fn queue_init_buffer_size(contents_len: usize) -> wgpu::BufferAddress {
    let unpadded_size = contents_len as wgpu::BufferAddress;
    if unpadded_size == 0 {
        return 0;
    }

    let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    ((unpadded_size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT)
}

/// Returns whether `actual_size` matches the queue-backed allocation size for `contents_len`.
pub(super) fn queue_init_buffer_size_matches(actual_size: u64, contents_len: usize) -> bool {
    actual_size == queue_init_buffer_size(contents_len)
}

fn queue_write_bytes(contents: &[u8]) -> Cow<'_, [u8]> {
    let padded_size = queue_init_buffer_size(contents.len()) as usize;
    if contents.len() == padded_size {
        return Cow::Borrowed(contents);
    }

    let mut padded = Vec::with_capacity(padded_size);
    padded.extend_from_slice(contents);
    padded.resize(padded_size, 0);
    Cow::Owned(padded)
}

/// Writes mesh bytes through `queue.write_buffer`, padding the payload length when wgpu requires it.
pub(super) fn write_mesh_queue_buffer(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
    contents: &[u8],
) {
    if contents.is_empty() {
        return;
    }

    let bytes = queue_write_bytes(contents);
    queue.write_buffer(buffer, offset, bytes.as_ref());
}

fn mapped_buffer_generation_still_current(
    health: &GpuMappedBufferHealth,
    expected_generation: u64,
) -> bool {
    health.generation() == expected_generation
}

fn reject_if_mapped_buffer_generation_changed(
    health: &GpuMappedBufferHealth,
    expected_generation: u64,
    label: Option<&str>,
) -> bool {
    if mapped_buffer_generation_still_current(health, expected_generation) {
        return false;
    }
    let current_generation = health.generation();
    logger::debug!(
        "mesh upload: buffer {:?} rejected after mapped-buffer invalidation generation changed (expected={}, current={})",
        label,
        expected_generation,
        current_generation
    );
    true
}

/// Creates a buffer with initial contents.
///
/// This intentionally avoids [`wgpu::util::DeviceExt::create_buffer_init`]'s
/// mapped-at-creation path. Device-loss and surface-validation failures can leave
/// new buffers invalid, and asking wgpu for a mapped range on that invalid buffer
/// is fatal. Queue-backed initialization lets the caller's upload-level error
/// scope catch validation once while this helper rejects work when the shared
/// invalidation generation changes.
///
/// Returns [`None`] when the mapped-buffer invalidation generation changed.
fn try_create_buffer_init(
    ctx: MeshGpuUploadContext<'_>,
    desc: &wgpu::util::BufferInitDescriptor<'_>,
) -> Option<wgpu::Buffer> {
    if reject_if_mapped_buffer_generation_changed(
        ctx.mapped_buffer_health,
        ctx.mapped_buffer_generation,
        desc.label,
    ) {
        return None;
    }

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: desc.label,
        size: queue_init_buffer_size(desc.contents.len()),
        usage: desc.usage,
        mapped_at_creation: false,
    });

    if reject_if_mapped_buffer_generation_changed(
        ctx.mapped_buffer_health,
        ctx.mapped_buffer_generation,
        desc.label,
    ) {
        return None;
    }

    if !desc.contents.is_empty() {
        write_mesh_queue_buffer(ctx.queue, &buffer, 0, desc.contents);
    }

    if reject_if_mapped_buffer_generation_changed(
        ctx.mapped_buffer_health,
        ctx.mapped_buffer_generation,
        desc.label,
    ) {
        None
    } else {
        Some(buffer)
    }
}

/// Creates core vertex and index buffers from the layout-validated `raw` slice.
///
/// Returns [`None`] when either buffer fails wgpu validation; the caller must
/// abort the mesh upload in that case.
pub(super) fn create_core_vertex_index_buffers(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
) -> Option<CoreBuffers> {
    profiling::scope!("asset::mesh_create_core_buffers");
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
    let vertex_stride_us = vertex_stride as usize;
    let index_count = compute_index_count(&data.submeshes);
    let index_count_u32 = index_count.max(0) as u32;

    let vb = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} vertices", data.asset_id)),
            contents: &raw[..layout.vertex_size],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_core_vertices");

    let ib_slice =
        &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
    let ib = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} indices", data.asset_id)),
            contents: ib_slice,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_core_indices");

    let index_format = wgpu_index_format(data.index_buffer_format);

    Some(CoreBuffers {
        vb,
        ib,
        index_format,
        vertex_stride,
        vertex_stride_us,
        index_count_u32,
    })
}

fn upload_positions_normals(
    ctx: MeshGpuUploadContext<'_>,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> Option<OptionalBufferPair> {
    if let Some((pb, nb)) = extract_float3_position_normal_as_vec4_streams(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    ) {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pbuf = try_create_buffer_init(
            ctx,
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} positions_stream", data.asset_id)),
                contents: &pb,
                usage,
            },
        )?;
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_positions_stream");
        let nbuf = try_create_buffer_init(
            ctx,
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} normals_stream", data.asset_id)),
                contents: &nb,
                usage,
            },
        )?;
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_normals_stream");
        Some((Some(Arc::new(pbuf)), Some(Arc::new(nbuf))))
    } else {
        logger::warn!(
            "mesh {}: missing float3 position+normal attributes -- debug/deform path disabled",
            data.asset_id
        );
        Some((None, None))
    }
}

fn upload_uv0_color(
    ctx: MeshGpuUploadContext<'_>,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> Option<OptionalBufferPair> {
    let uv0_buffer = match uv0_float2_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    ) {
        Some(uv_bytes) => {
            let buffer = Arc::new(try_create_buffer_init(
                ctx,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh {} uv0_stream", data.asset_id)),
                    contents: &uv_bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            )?);
            crate::profiling::note_resource_churn!(Buffer, "assets::mesh_uv0_stream");
            Some(buffer)
        }
        None => None,
    };
    let color_buffer = match color_float4_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    ) {
        Some(color_bytes) => {
            let buffer = Arc::new(try_create_buffer_init(
                ctx,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("mesh {} color_stream", data.asset_id)),
                    contents: &color_bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            )?);
            crate::profiling::note_resource_churn!(Buffer, "assets::mesh_color_stream");
            Some(buffer)
        }
        None => None,
    };
    Some((uv0_buffer, color_buffer))
}

/// Builds optional position/normal streams plus UV0 and vertex color buffers.
pub(super) fn extract_derived_vertex_streams(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    core: &CoreBuffers,
) -> Option<DerivedStreams> {
    profiling::scope!("asset::mesh_extract_derived_streams");
    let vc_usize = data.vertex_count.max(0) as usize;
    let vertex_slice = &raw[..layout.vertex_size];
    let (positions_buffer, normals_buffer) =
        upload_positions_normals(ctx, data, vertex_slice, vc_usize, core.vertex_stride_us)?;
    let (uv0_buffer, color_buffer) =
        upload_uv0_color(ctx, data, vertex_slice, vc_usize, core.vertex_stride_us)?;
    //perf xlinka: tangent/UV1-3 are big; build them only if a shader actually asks for them.
    Some(DerivedStreams {
        positions_buffer,
        normals_buffer,
        uv0_buffer,
        color_buffer,
        tangent_buffer: None,
        raw_tangent_buffer: None,
        uv1_buffer: None,
        uv2_buffer: None,
        uv3_buffer: None,
        wide_uv_buffer: None,
    })
}

#[cfg(test)]
mod tests {
    use crate::gpu::GpuMappedBufferHealth;

    use super::{
        mapped_buffer_generation_still_current, queue_init_buffer_size,
        queue_init_buffer_size_matches, queue_write_bytes, tangent_stream_usage,
        vertex_stream_usage,
    };

    #[test]
    fn queue_init_buffer_size_matches_wgpu_copy_alignment() {
        assert_eq!(queue_init_buffer_size(0), 0);
        assert_eq!(queue_init_buffer_size(1), wgpu::COPY_BUFFER_ALIGNMENT);
        assert_eq!(queue_init_buffer_size(6), wgpu::COPY_BUFFER_ALIGNMENT * 2);
        assert_eq!(
            queue_init_buffer_size(wgpu::COPY_BUFFER_ALIGNMENT as usize),
            wgpu::COPY_BUFFER_ALIGNMENT
        );
        assert_eq!(
            queue_init_buffer_size(wgpu::COPY_BUFFER_ALIGNMENT as usize + 1),
            wgpu::COPY_BUFFER_ALIGNMENT * 2
        );
    }

    #[test]
    fn queue_init_buffer_size_match_accepts_padded_six_byte_index_buffer() {
        assert!(queue_init_buffer_size_matches(8, 6));
        assert!(!queue_init_buffer_size_matches(6, 6));
    }

    #[test]
    fn queue_write_bytes_pads_unaligned_payloads_with_zeroes() {
        let bytes = queue_write_bytes(&[1, 2, 3, 4, 5, 6]);

        assert_eq!(bytes.as_ref(), &[1, 2, 3, 4, 5, 6, 0, 0]);
    }

    #[test]
    fn queue_write_bytes_borrows_aligned_payloads() {
        let source = [1, 2, 3, 4];
        let bytes = queue_write_bytes(&source);

        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
        assert_eq!(bytes.as_ref(), &source);
    }

    #[test]
    fn queue_write_bytes_leaves_empty_payloads_empty() {
        let bytes = queue_write_bytes(&[]);

        assert!(bytes.is_empty());
    }

    #[test]
    fn mapped_buffer_generation_check_rejects_stale_uploads() {
        let health = GpuMappedBufferHealth::new();
        let generation = health.generation();

        assert!(mapped_buffer_generation_still_current(&health, generation));

        health.mark_invalid("test invalidation");

        assert!(!mapped_buffer_generation_still_current(&health, generation));
    }

    #[test]
    fn tangent_streams_can_feed_deform_compute_and_forward_draws() {
        let usage = tangent_stream_usage();

        assert!(usage.contains(wgpu::BufferUsages::VERTEX));
        assert!(usage.contains(wgpu::BufferUsages::STORAGE));
        assert!(usage.contains(wgpu::BufferUsages::COPY_DST));
        assert!(usage.contains(wgpu::BufferUsages::COPY_SRC));
    }

    #[test]
    fn ordinary_vertex_streams_remain_vertex_only_upload_targets() {
        let usage = vertex_stream_usage();

        assert!(usage.contains(wgpu::BufferUsages::VERTEX));
        assert!(usage.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!usage.contains(wgpu::BufferUsages::STORAGE));
        assert!(!usage.contains(wgpu::BufferUsages::COPY_SRC));
    }
}
