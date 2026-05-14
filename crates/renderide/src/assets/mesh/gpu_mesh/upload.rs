//! Helpers for [`super::GpuMesh::upload`](GpuMesh::upload); keeps the `impl` readable.

use std::borrow::Cow;
use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::mesh_deform::{
    BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES, blendshape_sparse_buffers_fit_device,
};
use crate::shared::{
    IndexBufferFormat, MeshUploadData, SubmeshBufferDescriptor, VertexAttributeDescriptor,
    VertexAttributeType,
};

use super::super::layout::{
    BlendshapeFrameRange, BlendshapeFrameSpan, MeshBufferLayout, WIDE_UV_VERTEX_STRIDE_BYTES,
    color_float4_stream_bytes, compute_index_count, compute_vertex_stride, extract_bind_poses,
    extract_blendshape_offsets, extract_float3_position_normal_as_vec4_streams,
    split_bone_weights_tail_for_gpu, uv0_float2_stream_bytes, vertex_float2_stream_bytes,
    wide_uv_stream_bytes,
};
use super::hints::wgpu_index_format;
use super::tangent_generation::{
    TangentStreamSource, raw_tangent_payload_stream_bytes, tangent_stream_bytes,
};

/// Tangent plus UV1-UV3 optional vertex buffers from extended stream upload.
type ExtendedVertexStreams = (
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
);

/// Pair of optional derived vertex buffers.
type OptionalBufferPair = (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>);

/// CPU-side mesh source required to build lazy tangent and UV1-UV3 vertex streams.
pub(super) struct ExtendedVertexUploadSource<'a> {
    /// Interleaved vertex bytes from the host mesh payload.
    pub vertex_slice: &'a [u8],
    /// Index bytes from the host mesh payload.
    pub index_slice: &'a [u8],
    /// Number of vertices in `vertex_slice`.
    pub vertex_count: usize,
    /// Byte stride of one interleaved vertex.
    pub vertex_stride: usize,
    /// Host vertex attribute descriptors, in interleaved order.
    pub vertex_attributes: &'a [VertexAttributeDescriptor],
    /// Host index-buffer format.
    pub index_format: IndexBufferFormat,
    /// Host submesh descriptors.
    pub submeshes: &'a [SubmeshBufferDescriptor],
}

impl<'a> ExtendedVertexUploadSource<'a> {
    fn tangent_source(&self) -> TangentStreamSource<'a> {
        TangentStreamSource {
            vertex_data: self.vertex_slice,
            index_data: self.index_slice,
            vertex_count: self.vertex_count,
            stride: self.vertex_stride,
            attrs: self.vertex_attributes,
            index_format: self.index_format,
            submeshes: self.submeshes,
        }
    }
}

/// CPU-side mesh source required to build one lazy UV vertex stream.
#[derive(Copy, Clone)]
pub(super) struct UvVertexUploadSource<'a> {
    /// Interleaved vertex bytes from the host mesh payload.
    pub vertex_slice: &'a [u8],
    /// Number of vertices in `vertex_slice`.
    pub vertex_count: usize,
    /// Byte stride of one interleaved vertex.
    pub vertex_stride: usize,
    /// Host vertex attribute descriptors, in interleaved order.
    pub vertex_attributes: &'a [VertexAttributeDescriptor],
    /// UV attribute to extract.
    pub target: VertexAttributeType,
    /// Debug label suffix for the generated GPU buffer.
    pub label: &'a str,
}

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

/// Aggregated bone/skin GPU state and skinning matrices.
pub(super) struct BoneSkinUpload {
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    pub skinning_bind_matrices: Vec<Mat4>,
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

/// Creates a buffer with initial contents while capturing validation errors.
///
/// This intentionally avoids [`wgpu::util::DeviceExt::create_buffer_init`]'s
/// mapped-at-creation path. Device-loss and surface-validation failures can leave
/// new buffers invalid, and asking wgpu for a mapped range on that invalid buffer
/// is fatal. Queue-backed initialization lets validation stay inside error scopes
/// and lets the caller reject work when the shared invalidation generation changes.
///
/// Returns [`None`] when buffer creation failed validation; the helper logs the
/// underlying wgpu error with label, content size, and usage flags.
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

    let error_scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: desc.label,
        size: queue_init_buffer_size(desc.contents.len()),
        usage: desc.usage,
        mapped_at_creation: false,
    });
    if let Some(err) = pollster::block_on(error_scope.pop()) {
        logger::error!(
            "mesh upload: buffer create failed for {:?} ({} B, usage {:?}): {}",
            desc.label,
            desc.contents.len(),
            desc.usage,
            err,
        );
        return None;
    }

    if reject_if_mapped_buffer_generation_changed(
        ctx.mapped_buffer_health,
        ctx.mapped_buffer_generation,
        desc.label,
    ) {
        return None;
    }

    if !desc.contents.is_empty() {
        let error_scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        write_mesh_queue_buffer(ctx.queue, &buffer, 0, desc.contents);
        if let Some(err) = pollster::block_on(error_scope.pop()) {
            logger::error!(
                "mesh upload: queue write failed for {:?} ({} B, usage {:?}): {}",
                desc.label,
                desc.contents.len(),
                desc.usage,
                err,
            );
            return None;
        }
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

fn float4_default_stream_bytes(vertex_count: usize, default: [f32; 4]) -> Vec<u8> {
    let mut out = vec![0u8; vertex_count * 16];
    for chunk in out.chunks_exact_mut(16) {
        for (component, value) in default.iter().enumerate() {
            let o = component * 4;
            chunk[o..o + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn float2_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * 8]
}

fn wide_uv_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * WIDE_UV_VERTEX_STRIDE_BYTES]
}

#[inline]
fn vertex_stream_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
}

#[inline]
fn tangent_stream_usage() -> wgpu::BufferUsages {
    vertex_stream_usage() | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
}

fn create_vertex_stream_buffer(
    device: &wgpu::Device,
    asset_id: i32,
    label: &str,
    bytes: &[u8],
) -> Arc<wgpu::Buffer> {
    let buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} {label}_stream")),
            contents: bytes,
            usage: vertex_stream_usage(),
        }),
    );
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_vertex_stream");
    buffer
}

fn create_tangent_stream_buffer(
    device: &wgpu::Device,
    asset_id: i32,
    bytes: &[u8],
) -> Arc<wgpu::Buffer> {
    let buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} tangent_stream")),
            contents: bytes,
            usage: tangent_stream_usage(),
        }),
    );
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_tangent_stream");
    buffer
}

pub(super) fn upload_extended_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
    generate_missing_tangents: bool,
) -> ExtendedVertexStreams {
    let vc_usize = source.vertex_count;
    if vc_usize == 0 {
        return (None, None, None, None);
    }

    let tangent_bytes = tangent_stream_bytes(source.tangent_source(), generate_missing_tangents)
        .unwrap_or_else(|| float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]));

    let make_uv = |target: VertexAttributeType, label: &str| {
        let bytes = vertex_float2_stream_bytes(
            source.vertex_slice,
            vc_usize,
            source.vertex_stride,
            source.vertex_attributes,
            target,
        )
        .unwrap_or_else(|| float2_zero_stream_bytes(vc_usize));
        create_vertex_stream_buffer(device, asset_id, label, &bytes)
    };

    (
        Some(create_tangent_stream_buffer(
            device,
            asset_id,
            &tangent_bytes,
        )),
        Some(make_uv(VertexAttributeType::UV1, "uv1")),
        Some(make_uv(VertexAttributeType::UV2, "uv2")),
        Some(make_uv(VertexAttributeType::UV3, "uv3")),
    )
}

pub(super) fn upload_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
    generate_missing_tangents: bool,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let tangent_bytes = tangent_stream_bytes(source.tangent_source(), generate_missing_tangents)
        .unwrap_or_else(|| float4_default_stream_bytes(source.vertex_count, [1.0, 0.0, 0.0, 1.0]));
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

pub(super) fn upload_raw_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let tangent_bytes = raw_tangent_payload_stream_bytes(source.tangent_source())
        .unwrap_or_else(|| float4_default_stream_bytes(source.vertex_count, [1.0; 4]));
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

pub(super) fn upload_default_raw_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0; 4]);
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

pub(super) fn upload_default_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]);
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

pub(super) fn upload_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = vertex_float2_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
        source.target,
    )
    .unwrap_or_else(|| float2_zero_stream_bytes(source.vertex_count));
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        source.label,
        &uv_bytes,
    ))
}

pub(super) fn upload_wide_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = wide_uv_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    )
    .unwrap_or_else(|| wide_uv_zero_stream_bytes(source.vertex_count));
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        source.label,
        &uv_bytes,
    ))
}

pub(super) fn upload_default_wide_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv_bytes = wide_uv_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device, asset_id, "wide_uv", &uv_bytes,
    ))
}

pub(super) fn upload_default_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
    label: &str,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv_bytes = float2_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device, asset_id, label, &uv_bytes,
    ))
}

pub(super) fn upload_default_extended_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> ExtendedVertexStreams {
    if vc_usize == 0 {
        return (None, None, None, None);
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]);
    let uv_bytes = float2_zero_stream_bytes(vc_usize);
    (
        Some(create_tangent_stream_buffer(
            device,
            asset_id,
            &tangent_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv1", &uv_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv2", &uv_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv3", &uv_bytes,
        )),
    )
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

fn upload_skeleton_bone_buffers(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    profiling::scope!("asset::mesh_upload_skeleton_buffers");
    let bp_raw = &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
    let bind_poses_arr = extract_bind_poses(bp_raw, data.bone_count as usize)?;
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    let skinning: Vec<Mat4> = bind_poses_arr
        .iter()
        .map(Mat4::from_cols_array_2d)
        .collect();

    let bc = &raw[layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
    let bw =
        &raw[layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
    let (bi_buf, bw_buf) = if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(bc, bw, vc_usize)
    {
        let bi = try_create_buffer_init(
            ctx,
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_indices", data.asset_id)),
                contents: &ib,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bone_indices");
        let bwt = try_create_buffer_init(
            ctx,
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights_vec4", data.asset_id)),
                contents: &wb,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bone_weights_vec4");
        (Some(Arc::new(bi)), Some(Arc::new(bwt)))
    } else {
        logger::warn!(
            "mesh {}: bone weight tail could not be repacked for GPU skinning",
            data.asset_id
        );
        (None, None)
    };

    let bc_buf = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} bone_counts", data.asset_id)),
            contents: bc,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bone_counts");
    let bp_buf = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} bind_poses", data.asset_id)),
            contents: &bp_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bind_poses");
    Some(BoneSkinUpload {
        bone_counts_buffer: Some(Arc::new(bc_buf)),
        bone_indices_buffer: bi_buf,
        bone_weights_vec4_buffer: bw_buf,
        bind_poses_buffer: Some(Arc::new(bp_buf)),
        skinning_bind_matrices: skinning,
    })
}

/// Bone indices/weights, bind poses, and skinning matrices for real skeleton paths.
///
/// Returns [`None`] when the real-skeleton bind-pose slice is invalid ([`extract_bind_poses`]).
pub(super) fn upload_bone_and_skin_buffers(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    profiling::scope!("asset::mesh_upload_bone_skin_buffers");
    if data.bone_count > 0 {
        upload_skeleton_bone_buffers(ctx, raw, data, layout, vc_usize)
    } else {
        Some(BoneSkinUpload {
            bone_counts_buffer: None,
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bind_poses_buffer: None,
            skinning_bind_matrices: Vec::new(),
        })
    }
}

/// Sparse GPU buffers and CPU scatter ranges produced by `layout::extract_blendshape_offsets`.
pub(super) struct BlendshapeBuffersUpload {
    /// Storage buffer of packed sparse channel deltas (padded when empty; see `BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES`).
    pub sparse_buffer: Option<Arc<wgpu::Buffer>>,
    /// Copy of frame rows for CPU-side scatter dispatch.
    pub frame_ranges: Vec<BlendshapeFrameRange>,
    /// Per-shape spans into [`Self::frame_ranges`].
    pub shape_frame_spans: Vec<BlendshapeFrameSpan>,
    /// Logical blendshape slot count for weight indexing.
    pub num_blendshapes: u32,
    /// Whether any sparse row carries a nonzero position delta.
    pub has_position_deltas: bool,
    /// Whether any sparse row carries a nonzero normal delta.
    pub has_normal_deltas: bool,
    /// Whether any sparse row carries a nonzero tangent delta.
    pub has_tangent_deltas: bool,
}

/// Pads sparse CPU bytes to at least [`BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES`] for `wgpu` buffers.
pub(super) fn padded_sparse_bytes(sparse_deltas: &[u8]) -> Vec<u8> {
    let mut v = sparse_deltas.to_vec();
    let min = BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES as usize;
    if v.len() < min {
        v.resize(min, 0);
    }
    v
}

/// Builds sparse / descriptor GPU buffers (or empty upload when blendshapes are disabled).
pub(super) fn upload_blendshape_buffer(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    use_blendshapes: bool,
    max_buf: u64,
) -> Option<BlendshapeBuffersUpload> {
    profiling::scope!("asset::mesh_upload_blendshape_buffers");
    if !use_blendshapes {
        return Some(BlendshapeBuffersUpload {
            sparse_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            has_position_deltas: false,
            has_normal_deltas: false,
            has_tangent_deltas: false,
        });
    }
    let Some(pack) =
        extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count)
    else {
        return Some(BlendshapeBuffersUpload {
            sparse_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            has_position_deltas: false,
            has_normal_deltas: false,
            has_tangent_deltas: false,
        });
    };

    let n_u32 = pack.num_blendshapes.max(0) as u32;
    if !blendshape_sparse_buffers_fit_device(
        &pack,
        max_buf,
        ctx.gpu_limits.wgpu.max_storage_buffer_binding_size,
    ) {
        logger::warn!(
            "mesh {}: blendshapes dropped (sparse bytes exceed buffer / binding limits)",
            data.asset_id
        );
        return Some(BlendshapeBuffersUpload {
            sparse_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            has_position_deltas: false,
            has_normal_deltas: false,
            has_tangent_deltas: false,
        });
    }
    if pack.clamped_packed_deltas {
        logger::warn!(
            "mesh {}: blendshape normal/tangent deltas exceeded packed range and were clamped",
            data.asset_id
        );
    }

    let sparse_bytes = padded_sparse_bytes(&pack.sparse_deltas);

    let sparse_label = format!("mesh {} blendshape_sparse", data.asset_id);
    let sparse_buf = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&sparse_label),
            contents: &sparse_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_blendshape_sparse");

    Some(BlendshapeBuffersUpload {
        sparse_buffer: Some(Arc::new(sparse_buf)),
        frame_ranges: pack.frame_ranges,
        shape_frame_spans: pack.shape_frame_spans,
        num_blendshapes: n_u32,
        has_position_deltas: pack.has_position_deltas,
        has_normal_deltas: pack.has_normal_deltas,
        has_tangent_deltas: pack.has_tangent_deltas,
    })
}

pub(super) fn sum_optional_buffer_bytes(buffers: &[Option<&Arc<wgpu::Buffer>>]) -> u64 {
    buffers
        .iter()
        .filter_map(|o| o.as_ref().map(|b| b.size()))
        .sum()
}

/// Sums VRAM for all optional mesh buffers plus fixed vertex/index sizes.
pub(super) fn resident_bytes_for_mesh_upload(
    core_vb: &wgpu::Buffer,
    core_ib: &wgpu::Buffer,
    derived: &DerivedStreams,
    bone_skin: &BoneSkinUpload,
    blend_sparse: Option<&Arc<wgpu::Buffer>>,
) -> u64 {
    let mut n = core_vb.size() + core_ib.size();
    n += sum_optional_buffer_bytes(&[
        bone_skin.bone_counts_buffer.as_ref(),
        bone_skin.bone_indices_buffer.as_ref(),
        bone_skin.bone_weights_vec4_buffer.as_ref(),
        bone_skin.bind_poses_buffer.as_ref(),
        derived.positions_buffer.as_ref(),
        derived.normals_buffer.as_ref(),
        derived.uv0_buffer.as_ref(),
        derived.color_buffer.as_ref(),
        derived.tangent_buffer.as_ref(),
        derived.raw_tangent_buffer.as_ref(),
        derived.uv1_buffer.as_ref(),
        derived.uv2_buffer.as_ref(),
        derived.uv3_buffer.as_ref(),
        derived.wide_uv_buffer.as_ref(),
    ]);
    if let Some(b) = blend_sparse {
        n += b.size();
    }
    n
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
