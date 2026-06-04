//! Helpers for [`super::GpuMesh::upload`](GpuMesh::upload); keeps the `impl` readable.

mod accounting;
mod blendshape;
mod bone_skin;
mod derived_streams;
mod extended_streams;
mod generated;

pub(super) use accounting::resident_bytes_for_mesh_upload;
pub(super) use blendshape::{padded_sparse_bytes, upload_blendshape_buffer};
pub(super) use bone_skin::upload_bone_and_skin_buffers;
pub(crate) use derived_streams::{PreparedDerivedStreams, prepare_derived_stream_bytes};
pub(super) use extended_streams::{
    ExtendedVertexUploadSource, UvVertexUploadSource, upload_color_vertex_stream,
    upload_default_extended_vertex_streams, upload_default_raw_tangent_vertex_stream,
    upload_default_tangent_vertex_stream, upload_default_uv_vertex_stream,
    upload_default_wide_high_uv_vertex_stream, upload_default_wide_low_uv_vertex_stream,
    upload_extended_vertex_streams, upload_position_normal_vertex_streams,
    upload_raw_tangent_vertex_stream, upload_tangent_vertex_stream, upload_uv_vertex_stream,
    upload_uv0_vertex_stream, upload_wide_high_uv_vertex_stream, upload_wide_low_uv_vertex_stream,
};
#[cfg(test)]
use extended_streams::{tangent_stream_usage, vertex_stream_usage};
pub(crate) use generated::try_upload_generated_mesh_from_parts;

#[cfg(test)]
use std::borrow::Cow;
use std::sync::Arc;

use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::shared::MeshUploadData;

use super::super::layout::{MeshBufferLayout, compute_index_count, compute_vertex_stride};
use super::demand::{MeshDerivedStreamDemand, MeshDerivedStreamMask};
use super::hints::wgpu_index_format;

/// Interleaved VB, IB, and layout-derived scalars after validation.
pub(super) struct CoreBuffers {
    pub vb: wgpu::Buffer,
    pub ib: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub vertex_stride: u32,
    pub index_count_u32: u32,
}

/// GPU handles and mapped-buffer generation captured for one mesh upload.
#[derive(Clone, Copy)]
pub(crate) struct MeshGpuUploadContext<'a> {
    /// Logical device used to create mesh buffers.
    pub device: &'a wgpu::Device,
    /// Sink used to initialize mesh buffers without mapped-at-creation writes.
    pub upload_sink: &'a dyn MeshBufferUploadSink,
    /// Precomputed derived-stream bytes for full uploads.
    pub prepared_derived_streams: Option<&'a PreparedDerivedStreams>,
    /// Effective device limits used for upload validation.
    pub gpu_limits: &'a GpuLimits,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub mapped_buffer_health: &'a GpuMappedBufferHealth,
    /// Invalidation generation captured before the upload began.
    pub mapped_buffer_generation: u64,
    /// Derived streams requested by current or pending material reflection.
    pub derived_stream_demand: MeshDerivedStreamDemand,
    /// Whether this upload should wrap per-mesh GPU writes in a wgpu validation scope.
    pub validation_scopes_enabled: bool,
}

/// Upload sink for mesh buffer writes.
pub(crate) trait MeshBufferUploadSink {
    /// Queues or immediately performs a buffer write.
    fn write_buffer(&self, buffer: &wgpu::Buffer, offset: wgpu::BufferAddress, contents: &[u8]);

    /// Queues a buffer write whose payload needs zero padding to satisfy wgpu copy alignment.
    fn write_buffer_padded(
        &self,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        contents: &[u8],
        padded_size: usize,
    ) {
        let mut padded = Vec::with_capacity(padded_size);
        padded.extend_from_slice(contents);
        padded.resize(padded_size, 0);
        self.write_buffer(buffer, offset, &padded);
    }
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
    pub wide_low_uv_buffer: Option<Arc<wgpu::Buffer>>,
    pub wide_high_uv_buffer: Option<Arc<wgpu::Buffer>>,
}

impl DerivedStreams {
    pub(super) fn available_mask(&self) -> MeshDerivedStreamMask {
        let mut mask = MeshDerivedStreamMask::EMPTY;
        if self.positions_buffer.is_some() {
            mask |= MeshDerivedStreamMask::POSITION;
        }
        if self.normals_buffer.is_some() {
            mask |= MeshDerivedStreamMask::NORMAL;
        }
        if self.uv0_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV0;
        }
        if self.color_buffer.is_some() {
            mask |= MeshDerivedStreamMask::COLOR;
        }
        if self.tangent_buffer.is_some() {
            mask |= MeshDerivedStreamMask::TANGENT;
        }
        if self.raw_tangent_buffer.is_some() {
            mask |= MeshDerivedStreamMask::RAW_TANGENT;
        }
        if self.uv1_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV1;
        }
        if self.uv2_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV2;
        }
        if self.uv3_buffer.is_some() {
            mask |= MeshDerivedStreamMask::UV3;
        }
        if self.wide_low_uv_buffer.is_some() {
            mask |= MeshDerivedStreamMask::WIDE_UV_LOW;
        }
        if self.wide_high_uv_buffer.is_some() {
            mask |= MeshDerivedStreamMask::WIDE_UV_HIGH;
        }
        mask
    }
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
    let checks: [(&str, u64, bool); 9] = [
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
        ("bone_influence_offsets", (vc + 1) * 4, true),
        (
            "bone_influences",
            layout.bone_weights_length.max(8) as u64,
            true,
        ),
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

/// Returns the grow-only allocation size used for generated particle mesh buffers.
pub(crate) fn generated_particle_buffer_capacity(
    required_len: usize,
    max_buffer_size: u64,
) -> Option<u64> {
    let required = queue_init_buffer_size(required_len);
    if required > max_buffer_size {
        return None;
    }
    if required == 0 {
        return Some(0);
    }
    let grown = required.checked_next_power_of_two().unwrap_or(required);
    Some(grown.min(max_buffer_size).max(required))
}

#[cfg(test)]
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

/// Writes mesh bytes through `sink`, padding the payload length when wgpu requires it.
pub(super) fn write_mesh_upload_buffer(
    sink: &dyn MeshBufferUploadSink,
    buffer: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
    contents: &[u8],
) {
    if contents.is_empty() {
        return;
    }

    let padded_size = queue_init_buffer_size(contents.len()) as usize;
    if contents.len() == padded_size {
        sink.write_buffer(buffer, offset, contents);
    } else {
        sink.write_buffer_padded(buffer, offset, contents, padded_size);
    }
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
        write_mesh_upload_buffer(ctx.upload_sink, &buffer, 0, desc.contents);
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
        index_count_u32,
    })
}

#[derive(Clone, Copy)]
enum DerivedBufferProfile {
    Positions,
    Normals,
    Uv0,
    Color,
    Tangent,
    RawTangent,
    Uv1,
    Uv2,
    Uv3,
    WideLowUv,
    WideHighUv,
}

impl DerivedBufferProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Positions => "positions",
            Self::Normals => "normals",
            Self::Uv0 => "uv0",
            Self::Color => "color",
            Self::Tangent => "tangent",
            Self::RawTangent => "raw_tangent",
            Self::Uv1 => "uv1",
            Self::Uv2 => "uv2",
            Self::Uv3 => "uv3",
            Self::WideLowUv => "wide_low_uv",
            Self::WideHighUv => "wide_high_uv",
        }
    }

    fn note_resource_churn(self) {
        match self {
            Self::Positions => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_positions_stream");
            }
            Self::Normals => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_normals_stream");
            }
            Self::Uv0 => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_uv0_stream");
            }
            Self::Color => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_color_stream");
            }
            Self::Tangent => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_tangent_stream");
            }
            Self::RawTangent => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_raw_tangent_stream");
            }
            Self::Uv1 => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_uv1_stream");
            }
            Self::Uv2 => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_uv2_stream");
            }
            Self::Uv3 => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_uv3_stream");
            }
            Self::WideLowUv => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_wide_low_uv_stream");
            }
            Self::WideHighUv => {
                crate::profiling::note_resource_churn!(Buffer, "assets::mesh_wide_high_uv_stream");
            }
        }
    }
}

enum DerivedBufferUpload {
    Skipped,
    Uploaded(Arc<wgpu::Buffer>),
}

impl DerivedBufferUpload {
    fn into_buffer(self) -> Option<Arc<wgpu::Buffer>> {
        match self {
            Self::Skipped => None,
            Self::Uploaded(buffer) => Some(buffer),
        }
    }
}

fn upload_derived_buffer(
    ctx: MeshGpuUploadContext<'_>,
    asset_id: i32,
    profile: DerivedBufferProfile,
    bytes: Option<&[u8]>,
    usage: wgpu::BufferUsages,
) -> Option<DerivedBufferUpload> {
    let Some(bytes) = bytes else {
        return Some(DerivedBufferUpload::Skipped);
    };
    let buffer = Arc::new(try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} {}_stream", profile.label())),
            contents: bytes,
            usage,
        },
    )?);
    profile.note_resource_churn();
    Some(DerivedBufferUpload::Uploaded(buffer))
}

#[derive(Clone, Copy)]
struct DerivedBufferUsages {
    primary: wgpu::BufferUsages,
    vertex: wgpu::BufferUsages,
    tangent: wgpu::BufferUsages,
}

impl DerivedBufferUsages {
    fn new() -> Self {
        let primary = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let vertex = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let tangent = vertex | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        Self {
            primary,
            vertex,
            tangent,
        }
    }
}

fn upload_prepared_derived_streams(
    ctx: MeshGpuUploadContext<'_>,
    asset_id: i32,
    prepared: &PreparedDerivedStreams,
    usages: DerivedBufferUsages,
) -> Option<DerivedStreams> {
    Some(DerivedStreams {
        positions_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Positions,
            prepared.positions.as_deref(),
            usages.primary,
        )?
        .into_buffer(),
        normals_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Normals,
            prepared.normals.as_deref(),
            usages.primary,
        )?
        .into_buffer(),
        uv0_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Uv0,
            prepared.uv0.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
        color_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Color,
            prepared.color.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
        tangent_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Tangent,
            prepared.tangent.as_deref(),
            usages.tangent,
        )?
        .into_buffer(),
        raw_tangent_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::RawTangent,
            prepared.raw_tangent.as_deref(),
            usages.tangent,
        )?
        .into_buffer(),
        uv1_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Uv1,
            prepared.uv1.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
        uv2_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Uv2,
            prepared.uv2.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
        uv3_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::Uv3,
            prepared.uv3.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
        wide_low_uv_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::WideLowUv,
            prepared.wide_low_uv.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
        wide_high_uv_buffer: upload_derived_buffer(
            ctx,
            asset_id,
            DerivedBufferProfile::WideHighUv,
            prepared.wide_high_uv.as_deref(),
            usages.vertex,
        )?
        .into_buffer(),
    })
}

/// Builds optional position/normal streams plus UV0 and vertex color buffers.
pub(super) fn extract_derived_vertex_streams(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    _core: &CoreBuffers,
) -> Option<DerivedStreams> {
    profiling::scope!("asset::mesh_extract_derived_streams");
    let prepared_owned;
    let prepared = if let Some(prepared) = ctx.prepared_derived_streams {
        prepared
    } else {
        prepared_owned = prepare_derived_stream_bytes(raw, data, layout, ctx.derived_stream_demand);
        &prepared_owned
    };
    upload_prepared_derived_streams(ctx, data.asset_id, prepared, DerivedBufferUsages::new())
}

#[cfg(test)]
mod tests {
    use super::super::super::layout::{
        color_float4_stream_bytes, compute_mesh_buffer_layout, compute_vertex_stride,
        extract_float3_position_normal_as_vec4_streams, uv0_float2_stream_bytes,
    };

    use crate::gpu::GpuMappedBufferHealth;
    use crate::shared::{
        IndexBufferFormat, MeshUploadData, VertexAttributeDescriptor, VertexAttributeFormat,
        VertexAttributeType,
    };

    use super::{
        MeshDerivedStreamDemand, MeshDerivedStreamMask, generated_particle_buffer_capacity,
        mapped_buffer_generation_still_current, prepare_derived_stream_bytes,
        queue_init_buffer_size, queue_init_buffer_size_matches, queue_write_bytes,
        tangent_stream_usage, vertex_stream_usage,
    };

    fn float_attr(attribute: VertexAttributeType, dimensions: i32) -> VertexAttributeDescriptor {
        VertexAttributeDescriptor {
            attribute,
            format: VertexAttributeFormat::Float32,
            dimensions,
        }
    }

    fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

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
    fn generated_particle_capacity_grows_geometrically_and_stays_within_limits() {
        assert_eq!(generated_particle_buffer_capacity(0, 1024), Some(0));
        assert_eq!(generated_particle_buffer_capacity(6, 1024), Some(8));
        assert_eq!(generated_particle_buffer_capacity(9, 1024), Some(16));
        assert_eq!(generated_particle_buffer_capacity(513, 768), Some(768));
        assert_eq!(generated_particle_buffer_capacity(1025, 1024), None);
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

    #[test]
    fn prepared_derived_streams_match_layout_extractors() {
        let attrs = vec![
            float_attr(VertexAttributeType::Position, 3),
            float_attr(VertexAttributeType::Normal, 3),
            float_attr(VertexAttributeType::UV0, 2),
            float_attr(VertexAttributeType::Color, 4),
        ];
        let stride = compute_vertex_stride(&attrs);
        let vertex_count = 2;
        let layout =
            compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, None).expect("layout");
        let data = MeshUploadData {
            vertex_count,
            index_buffer_format: IndexBufferFormat::UInt16,
            vertex_attributes: attrs,
            ..Default::default()
        };
        let mut raw = Vec::with_capacity(layout.total_buffer_length);
        push_f32s(&mut raw, &[1.0, 2.0, 3.0]);
        push_f32s(&mut raw, &[0.0, 1.0, 0.0]);
        push_f32s(&mut raw, &[0.25, 0.75]);
        push_f32s(&mut raw, &[0.1, 0.2, 0.3, 0.4]);
        push_f32s(&mut raw, &[4.0, 5.0, 6.0]);
        push_f32s(&mut raw, &[0.0, 0.0, 1.0]);
        push_f32s(&mut raw, &[0.5, 0.125]);
        push_f32s(&mut raw, &[0.9, 0.8, 0.7, 0.6]);
        assert_eq!(raw.len(), layout.vertex_size);
        raw.resize(layout.total_buffer_length, 0);

        let demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::POSITION
                | MeshDerivedStreamMask::NORMAL
                | MeshDerivedStreamMask::UV0
                | MeshDerivedStreamMask::COLOR,
            ..MeshDerivedStreamDemand::EMPTY
        };
        let prepared = prepare_derived_stream_bytes(&raw, &data, &layout, demand);
        let vertex_slice = &raw[..layout.vertex_size];
        let stride = compute_vertex_stride(&data.vertex_attributes) as usize;
        let vertex_count = data.vertex_count as usize;
        let (positions, normals) = extract_float3_position_normal_as_vec4_streams(
            vertex_slice,
            vertex_count,
            stride,
            &data.vertex_attributes,
        )
        .expect("position and normal streams");
        let uv0 =
            uv0_float2_stream_bytes(vertex_slice, vertex_count, stride, &data.vertex_attributes)
                .expect("uv0 stream");
        let color =
            color_float4_stream_bytes(vertex_slice, vertex_count, stride, &data.vertex_attributes)
                .expect("color stream");

        assert_eq!(prepared.positions.as_deref(), Some(positions.as_slice()));
        assert_eq!(prepared.normals.as_deref(), Some(normals.as_slice()));
        assert_eq!(prepared.uv0.as_deref(), Some(uv0.as_slice()));
        assert_eq!(prepared.color.as_deref(), Some(color.as_slice()));
        assert!(prepared.tangent.is_none());
        assert!(prepared.raw_tangent.is_none());
        assert!(prepared.uv1.is_none());
        assert!(prepared.uv2.is_none());
        assert!(prepared.uv3.is_none());
        assert!(prepared.wide_low_uv.is_none());
        assert!(prepared.wide_high_uv.is_none());
    }
}
