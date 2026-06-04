//! Upload path for renderer-generated particle mesh buffers.

use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::materials::EmbeddedTangentFallbackMode;
use crate::shared::MeshUploadData;

use super::super::super::layout::{MeshBufferLayout, compute_index_count, compute_vertex_stride};
use super::super::demand::MeshDerivedStreamState;
use super::super::hints::{
    validated_submesh_ranges, validated_submesh_topologies, wgpu_index_format,
};
use super::super::{EMPTY_MESH_PLACEHOLDER_BYTES, GpuMesh};
use super::{
    DerivedBufferProfile, DerivedBufferUsages, DerivedStreams, MeshGpuUploadContext,
    PreparedDerivedStreams, accounting, generated_particle_buffer_capacity, queue_init_buffer_size,
    reject_if_mapped_buffer_generation_changed, write_mesh_upload_buffer,
};

fn try_create_generated_buffer(
    ctx: MeshGpuUploadContext<'_>,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> Option<Arc<wgpu::Buffer>> {
    if reject_if_mapped_buffer_generation_changed(
        ctx.mapped_buffer_health,
        ctx.mapped_buffer_generation,
        Some(label),
    ) {
        return None;
    }
    let buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }));
    if reject_if_mapped_buffer_generation_changed(
        ctx.mapped_buffer_health,
        ctx.mapped_buffer_generation,
        Some(label),
    ) {
        None
    } else {
        Some(buffer)
    }
}

#[derive(Clone, Copy)]
enum GeneratedCoreBufferProfile {
    Vertices,
    Indices,
}

impl GeneratedCoreBufferProfile {
    fn note_resource_churn(self) {
        match self {
            Self::Vertices => {
                crate::profiling::note_resource_churn!(
                    Buffer,
                    "assets::generated_particle_core_vertices"
                );
            }
            Self::Indices => {
                crate::profiling::note_resource_churn!(
                    Buffer,
                    "assets::generated_particle_core_indices"
                );
            }
        }
    }
}

fn upload_generated_core_buffer(
    ctx: MeshGpuUploadContext<'_>,
    existing: Option<&Arc<wgpu::Buffer>>,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
    profile: GeneratedCoreBufferProfile,
) -> Option<Arc<wgpu::Buffer>> {
    let required = queue_init_buffer_size(contents.len());
    let minimum_size = required.max(EMPTY_MESH_PLACEHOLDER_BYTES);
    let buffer = if let Some(existing) = existing
        && existing.size() >= minimum_size
    {
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::generated_mesh_buffer_reuses", 1.0);
        Arc::clone(existing)
    } else {
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::generated_mesh_buffer_grows", 1.0);
        let size = if contents.is_empty() {
            EMPTY_MESH_PLACEHOLDER_BYTES
        } else {
            generated_particle_buffer_capacity(contents.len(), ctx.gpu_limits.max_buffer_size())?
        };
        let buffer = try_create_generated_buffer(ctx, label, size, usage)?;
        profile.note_resource_churn();
        buffer
    };
    write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, contents);
    Some(buffer)
}

enum GeneratedDerivedBufferUpload {
    Missing,
    Ready(Arc<wgpu::Buffer>),
}

impl GeneratedDerivedBufferUpload {
    fn into_buffer(self) -> Option<Arc<wgpu::Buffer>> {
        match self {
            Self::Missing => None,
            Self::Ready(buffer) => Some(buffer),
        }
    }
}

fn upload_generated_derived_buffer(
    ctx: MeshGpuUploadContext<'_>,
    existing: Option<&Arc<wgpu::Buffer>>,
    asset_id: i32,
    profile: DerivedBufferProfile,
    contents: Option<&[u8]>,
    usage: wgpu::BufferUsages,
    max_size: u64,
) -> Option<GeneratedDerivedBufferUpload> {
    let Some(contents) = contents else {
        return Some(GeneratedDerivedBufferUpload::Missing);
    };
    if contents.is_empty() {
        return Some(
            existing
                .cloned()
                .map(GeneratedDerivedBufferUpload::Ready)
                .unwrap_or(GeneratedDerivedBufferUpload::Missing),
        );
    }
    let required = queue_init_buffer_size(contents.len());
    let buffer = if let Some(existing) = existing
        && existing.size() >= required
    {
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::generated_mesh_buffer_reuses", 1.0);
        Arc::clone(existing)
    } else {
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::generated_mesh_buffer_grows", 1.0);
        let label = format!("mesh {asset_id} {}_stream", profile.label());
        let size = generated_particle_buffer_capacity(contents.len(), max_size)?;
        let buffer = try_create_generated_buffer(ctx, &label, size, usage)?;
        profile.note_resource_churn();
        buffer
    };
    write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, contents);
    Some(GeneratedDerivedBufferUpload::Ready(buffer))
}

fn generated_particle_mesh_input_is_valid(
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vertices: &[u8],
    indices: &[u8],
    prepared: &PreparedDerivedStreams,
    gpu_limits: &GpuLimits,
) -> bool {
    if vertices.len() != layout.vertex_size || indices.len() != layout.index_buffer_length {
        logger::warn!(
            "mesh {}: generated particle payload does not match layout (vertices {} != {}, indices {} != {})",
            data.asset_id,
            vertices.len(),
            layout.vertex_size,
            indices.len(),
            layout.index_buffer_length
        );
        return false;
    }
    let max_buf = gpu_limits.max_buffer_size();
    let max_storage = gpu_limits.max_storage_buffer_binding_size();
    for (label, bytes) in [
        ("interleaved vertices", vertices),
        ("indices", indices),
        ("uv0", prepared.uv0.as_deref().unwrap_or(&[])),
        ("color", prepared.color.as_deref().unwrap_or(&[])),
        ("uv1", prepared.uv1.as_deref().unwrap_or(&[])),
        ("uv2", prepared.uv2.as_deref().unwrap_or(&[])),
        ("uv3", prepared.uv3.as_deref().unwrap_or(&[])),
        (
            "wide_low_uv",
            prepared.wide_low_uv.as_deref().unwrap_or(&[]),
        ),
        (
            "wide_high_uv",
            prepared.wide_high_uv.as_deref().unwrap_or(&[]),
        ),
    ] {
        let size = queue_init_buffer_size(bytes.len());
        if size > max_buf {
            logger::warn!(
                "mesh {}: generated particle {label} buffer ({} B) exceeds max_buffer_size ({} B)",
                data.asset_id,
                bytes.len(),
                max_buf
            );
            return false;
        }
    }
    for (label, bytes) in [
        ("positions", prepared.positions.as_deref().unwrap_or(&[])),
        ("normals", prepared.normals.as_deref().unwrap_or(&[])),
        ("tangent", prepared.tangent.as_deref().unwrap_or(&[])),
        (
            "raw_tangent",
            prepared.raw_tangent.as_deref().unwrap_or(&[]),
        ),
    ] {
        let size = queue_init_buffer_size(bytes.len());
        if size > max_buf || size > max_storage {
            logger::warn!(
                "mesh {}: generated particle {label} storage buffer ({} B) exceeds device limits max_buffer={} max_storage={}",
                data.asset_id,
                bytes.len(),
                max_buf,
                max_storage
            );
            return false;
        }
    }
    true
}

fn upload_generated_core_buffers(
    ctx: MeshGpuUploadContext<'_>,
    data: &MeshUploadData,
    existing: Option<&GpuMesh>,
    vertices: &[u8],
    indices: &[u8],
) -> Option<(Arc<wgpu::Buffer>, Arc<wgpu::Buffer>)> {
    let vertex_buffer = upload_generated_core_buffer(
        ctx,
        existing.map(|mesh| &mesh.vertex_buffer),
        &format!("mesh {} generated vertices", data.asset_id),
        vertices,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        GeneratedCoreBufferProfile::Vertices,
    )?;
    let index_buffer = upload_generated_core_buffer(
        ctx,
        existing.map(|mesh| &mesh.index_buffer),
        &format!("mesh {} generated indices", data.asset_id),
        indices,
        wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        GeneratedCoreBufferProfile::Indices,
    )?;
    Some((vertex_buffer, index_buffer))
}

#[derive(Clone, Copy)]
struct GeneratedDerivedUploadLimits {
    usages: DerivedBufferUsages,
    storage_size_limit: u64,
    max_buffer_size: u64,
}

impl GeneratedDerivedUploadLimits {
    fn from_context(ctx: MeshGpuUploadContext<'_>) -> Self {
        let usages = DerivedBufferUsages::new();
        let max_buffer_size = ctx.gpu_limits.max_buffer_size();
        let storage_size_limit =
            max_buffer_size.min(ctx.gpu_limits.max_storage_buffer_binding_size());
        Self {
            usages,
            storage_size_limit,
            max_buffer_size,
        }
    }
}

struct GeneratedPrimaryDerivedBuffers {
    positions_buffer: Option<Arc<wgpu::Buffer>>,
    normals_buffer: Option<Arc<wgpu::Buffer>>,
    uv0_buffer: Option<Arc<wgpu::Buffer>>,
    color_buffer: Option<Arc<wgpu::Buffer>>,
}

struct GeneratedExtendedDerivedBuffers {
    tangent_buffer: Option<Arc<wgpu::Buffer>>,
    raw_tangent_buffer: Option<Arc<wgpu::Buffer>>,
    uv1_buffer: Option<Arc<wgpu::Buffer>>,
    uv2_buffer: Option<Arc<wgpu::Buffer>>,
    uv3_buffer: Option<Arc<wgpu::Buffer>>,
    wide_low_uv_buffer: Option<Arc<wgpu::Buffer>>,
    wide_high_uv_buffer: Option<Arc<wgpu::Buffer>>,
}

fn upload_generated_primary_derived_streams(
    ctx: MeshGpuUploadContext<'_>,
    asset_id: i32,
    existing: Option<&GpuMesh>,
    prepared: &PreparedDerivedStreams,
    limits: GeneratedDerivedUploadLimits,
) -> Option<GeneratedPrimaryDerivedBuffers> {
    Some(GeneratedPrimaryDerivedBuffers {
        positions_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.positions_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Positions,
            prepared.positions.as_deref(),
            limits.usages.primary,
            limits.storage_size_limit,
        )?
        .into_buffer(),
        normals_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.normals_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Normals,
            prepared.normals.as_deref(),
            limits.usages.primary,
            limits.storage_size_limit,
        )?
        .into_buffer(),
        uv0_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.uv0_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Uv0,
            prepared.uv0.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
        color_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.color_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Color,
            prepared.color.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
    })
}

fn upload_generated_extended_derived_streams(
    ctx: MeshGpuUploadContext<'_>,
    asset_id: i32,
    existing: Option<&GpuMesh>,
    prepared: &PreparedDerivedStreams,
    limits: GeneratedDerivedUploadLimits,
) -> Option<GeneratedExtendedDerivedBuffers> {
    Some(GeneratedExtendedDerivedBuffers {
        tangent_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.tangent_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Tangent,
            prepared.tangent.as_deref(),
            limits.usages.tangent,
            limits.storage_size_limit,
        )?
        .into_buffer(),
        raw_tangent_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.raw_tangent_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::RawTangent,
            prepared.raw_tangent.as_deref(),
            limits.usages.tangent,
            limits.storage_size_limit,
        )?
        .into_buffer(),
        uv1_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.uv1_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Uv1,
            prepared.uv1.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
        uv2_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.uv2_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Uv2,
            prepared.uv2.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
        uv3_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.uv3_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::Uv3,
            prepared.uv3.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
        wide_low_uv_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.wide_low_uv_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::WideLowUv,
            prepared.wide_low_uv.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
        wide_high_uv_buffer: upload_generated_derived_buffer(
            ctx,
            existing.and_then(|mesh| mesh.wide_high_uv_buffer.as_ref()),
            asset_id,
            DerivedBufferProfile::WideHighUv,
            prepared.wide_high_uv.as_deref(),
            limits.usages.vertex,
            limits.max_buffer_size,
        )?
        .into_buffer(),
    })
}

fn upload_generated_derived_streams(
    ctx: MeshGpuUploadContext<'_>,
    data: &MeshUploadData,
    existing: Option<&GpuMesh>,
    prepared: &PreparedDerivedStreams,
) -> Option<DerivedStreams> {
    let limits = GeneratedDerivedUploadLimits::from_context(ctx);
    let primary =
        upload_generated_primary_derived_streams(ctx, data.asset_id, existing, prepared, limits)?;
    let extended =
        upload_generated_extended_derived_streams(ctx, data.asset_id, existing, prepared, limits)?;
    Some(DerivedStreams {
        positions_buffer: primary.positions_buffer,
        normals_buffer: primary.normals_buffer,
        uv0_buffer: primary.uv0_buffer,
        color_buffer: primary.color_buffer,
        tangent_buffer: extended.tangent_buffer,
        raw_tangent_buffer: extended.raw_tangent_buffer,
        uv1_buffer: extended.uv1_buffer,
        uv2_buffer: extended.uv2_buffer,
        uv3_buffer: extended.uv3_buffer,
        wide_low_uv_buffer: extended.wide_low_uv_buffer,
        wide_high_uv_buffer: extended.wide_high_uv_buffer,
    })
}

fn generated_mesh_resident_bytes(
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    derived: &DerivedStreams,
) -> u64 {
    vertex_buffer.size()
        + index_buffer.size()
        + accounting::sum_optional_buffer_bytes(&[
            derived.positions_buffer.as_ref(),
            derived.normals_buffer.as_ref(),
            derived.uv0_buffer.as_ref(),
            derived.color_buffer.as_ref(),
            derived.tangent_buffer.as_ref(),
            derived.raw_tangent_buffer.as_ref(),
            derived.uv1_buffer.as_ref(),
            derived.uv2_buffer.as_ref(),
            derived.uv3_buffer.as_ref(),
            derived.wide_low_uv_buffer.as_ref(),
            derived.wide_high_uv_buffer.as_ref(),
        ])
}

/// Uploads renderer-generated particle geometry with grow-only GPU buffers.
pub(crate) fn try_upload_generated_mesh_from_parts(
    ctx: MeshGpuUploadContext<'_>,
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vertices: &[u8],
    indices: &[u8],
    prepared: &PreparedDerivedStreams,
    existing: Option<GpuMesh>,
) -> Option<GpuMesh> {
    profiling::scope!("asset::generated_particle_mesh_upload");
    if !generated_particle_mesh_input_is_valid(
        data,
        layout,
        vertices,
        indices,
        prepared,
        ctx.gpu_limits,
    ) {
        return None;
    }
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
    let index_count = compute_index_count(&data.submeshes);
    let index_count_u32 = index_count.max(0) as u32;
    let existing = existing.as_ref();
    let (vertex_buffer, index_buffer) =
        upload_generated_core_buffers(ctx, data, existing, vertices, indices)?;
    let derived = upload_generated_derived_streams(ctx, data, existing, prepared)?;
    let derived_available_mask = derived.available_mask();
    let derived_stream_state = MeshDerivedStreamState::after_full_upload(
        ctx.derived_stream_demand,
        derived_available_mask,
        derived_available_mask,
    );
    let resident_bytes = generated_mesh_resident_bytes(&vertex_buffer, &index_buffer, &derived);

    Some(GpuMesh {
        asset_id: data.asset_id,
        vertex_buffer,
        index_buffer,
        index_format: wgpu_index_format(data.index_buffer_format),
        index_count: index_count_u32,
        submeshes: validated_submesh_ranges(&data.submeshes, index_count_u32),
        submesh_topologies: validated_submesh_topologies(&data.submeshes, index_count_u32),
        vertex_count: data.vertex_count.max(0) as u32,
        vertex_stride,
        bounds: data.bounds,
        bone_counts_buffer: None,
        bone_indices_buffer: None,
        bone_weights_vec4_buffer: None,
        bone_influence_offsets_buffer: None,
        bone_influences_buffer: None,
        bind_poses_buffer: None,
        blendshape_sparse_buffer: None,
        blendshape_frame_ranges: Vec::new(),
        blendshape_shape_frame_spans: Vec::new(),
        num_blendshapes: 0,
        blendshape_has_position_deltas: false,
        blendshape_has_normal_deltas: false,
        blendshape_has_tangent_deltas: false,
        positions_buffer: derived.positions_buffer,
        normals_buffer: derived.normals_buffer,
        uv0_buffer: derived.uv0_buffer,
        color_buffer: derived.color_buffer,
        tangent_buffer: derived.tangent_buffer,
        raw_tangent_buffer: derived.raw_tangent_buffer,
        tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
        uv1_buffer: derived.uv1_buffer,
        uv2_buffer: derived.uv2_buffer,
        uv3_buffer: derived.uv3_buffer,
        wide_low_uv_buffer: derived.wide_low_uv_buffer,
        wide_high_uv_buffer: derived.wide_high_uv_buffer,
        derived_stream_state,
        extended_vertex_stream_source: None,
        has_skeleton: false,
        skinning_bind_matrices: Vec::new(),
        resident_bytes,
    })
}
