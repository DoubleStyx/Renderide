use glam::{Vec2, Vec3, Vec4};

use crate::assets::mesh::{
    GpuMesh, MeshGpuUploadContext, PreparedDerivedStreams, compute_and_validate_mesh_layout,
    try_upload_generated_mesh_from_parts,
};
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{
    IndexBufferFormat, MeshUploadData, RenderBoundingBox, SubmeshBufferDescriptor, SubmeshTopology,
    VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

use super::types::ParticleRenderBufferError;

/// Optional generated streams that are not present in the interleaved upload bytes.
#[derive(Debug, Default)]
pub(super) struct GeneratedExtraStreams {
    /// Raw tangent payload stream at shader location 4.
    pub(super) raw_tangent: Option<Vec<u8>>,
    /// Geometric tangent stream at shader location 4.
    pub(super) tangent: Option<Vec<u8>>,
    /// UV1 stream at shader location 5.
    pub(super) uv1: Option<Vec<u8>>,
}

pub(super) fn generated_vertex_stride() -> usize {
    12 + 12 + 8 + 16
}

pub(super) fn push_generated_vertex(
    out: &mut Vec<u8>,
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    color: Vec4,
) {
    push_f32s(out, &position.to_array());
    push_f32s(out, &normal.to_array());
    push_f32s(out, &uv.to_array());
    push_f32s(out, &color.to_array());
}

fn push_f32s<const N: usize>(out: &mut Vec<u8>, values: &[f32; N]) {
    out.extend_from_slice(bytemuck::cast_slice(values));
}

/// Writes one generated vertex into an already sized byte slice.
pub(super) fn write_generated_vertex(
    out: &mut [u8],
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    color: Vec4,
) {
    let mut cursor = 0usize;
    cursor += write_f32s(&mut out[cursor..], &position.to_array());
    cursor += write_f32s(&mut out[cursor..], &normal.to_array());
    cursor += write_f32s(&mut out[cursor..], &uv.to_array());
    let _ = write_f32s(&mut out[cursor..], &color.to_array());
}

/// Writes tightly packed `f32` values into `out` and returns the byte count written.
fn write_f32s<const N: usize>(out: &mut [u8], values: &[f32; N]) -> usize {
    let bytes = bytemuck::cast_slice(values);
    out[..bytes.len()].copy_from_slice(bytes);
    bytes.len()
}

/// Writes tightly packed `u32` values into `out`.
pub(super) fn write_u32s(out: &mut [u8], values: &[u32]) {
    out.copy_from_slice(bytemuck::cast_slice(values));
}

/// Builds derived stream bytes for generated particle vertices on the worker thread.
pub(super) fn prepared_generated_derived_streams(
    vertices: &[u8],
    vertex_count: usize,
    extra: GeneratedExtraStreams,
) -> PreparedDerivedStreams {
    profiling::scope!("particle::prepare_generated_derived_streams");
    let stride = generated_vertex_stride();
    let mut positions = vec![0u8; vertex_count * 16];
    let mut normals = vec![0u8; vertex_count * 16];
    let mut uv0 = vec![0u8; vertex_count * 8];
    let mut color = vec![0u8; vertex_count * 16];
    let one = 1.0f32.to_le_bytes();
    for (index, vertex) in vertices.chunks_exact(stride).take(vertex_count).enumerate() {
        let position_start = index * 16;
        positions[position_start..position_start + 12].copy_from_slice(&vertex[0..12]);
        positions[position_start + 12..position_start + 16].copy_from_slice(&one);
        normals[position_start..position_start + 12].copy_from_slice(&vertex[12..24]);
        let uv_start = index * 8;
        uv0[uv_start..uv_start + 8].copy_from_slice(&vertex[24..32]);
        color[position_start..position_start + 16].copy_from_slice(&vertex[32..48]);
    }
    PreparedDerivedStreams {
        positions: Some(positions),
        normals: Some(normals),
        uv0: Some(uv0),
        color: Some(color),
        tangent: extra.tangent,
        raw_tangent: extra.raw_tangent,
        uv1: extra.uv1,
        ..PreparedDerivedStreams::default()
    }
}

/// Inputs needed to publish one renderer-generated mesh into the mesh pool.
#[derive(Debug)]
pub(crate) struct GeneratedMeshUploadInput {
    /// Human-readable source kind used in diagnostics.
    pub(crate) kind: &'static str,
    /// Host asset id that produced the generated mesh.
    pub(crate) source_asset_id: i32,
    /// Renderer-generated mesh asset id.
    pub(crate) mesh_asset_id: i32,
    /// Packed interleaved vertex bytes.
    pub(crate) vertices: Vec<u8>,
    /// Packed `u32` index bytes.
    pub(crate) indices: Vec<u8>,
    /// Worker-prepared derived stream bytes for generated vertices.
    pub(crate) prepared_derived_streams: PreparedDerivedStreams,
    /// Number of vertices in `vertices`.
    pub(crate) vertex_count: usize,
    /// Number of indices in `indices`.
    pub(crate) index_count: usize,
    /// Local-space bounds for the generated mesh.
    pub(crate) bounds: RenderBoundingBox,
}

pub(crate) fn upload_generated_mesh(
    gpu: MeshGpuUploadContext<'_>,
    input: GeneratedMeshUploadInput,
    existing: Option<GpuMesh>,
) -> Result<GpuMesh, ParticleRenderBufferError> {
    let data = generated_mesh_upload_data(
        input.mesh_asset_id,
        input.vertex_count,
        input.index_count,
        input.bounds,
    )?;
    let layout = compute_and_validate_mesh_layout(&data).ok_or(
        ParticleRenderBufferError::InvalidMeshLayout {
            kind: input.kind,
            asset_id: input.source_asset_id,
        },
    )?;
    let gpu = MeshGpuUploadContext {
        prepared_derived_streams: Some(&input.prepared_derived_streams),
        ..gpu
    };
    let mesh = if gpu.validation_scopes_enabled {
        profiling::scope!("particle::generated_mesh_validation_scope");
        let validation_scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mesh = try_upload_generated_mesh_from_parts(
            gpu,
            &data,
            &layout,
            &input.vertices,
            &input.indices,
            &input.prepared_derived_streams,
            existing,
        );
        let validation_error = pollster::block_on(validation_scope.pop());
        if let Some(err) = validation_error {
            logger::error!(
                "{} render buffer {}: generated mesh GPU validation failed: {}",
                input.kind,
                input.source_asset_id,
                err
            );
            return Err(ParticleRenderBufferError::GpuUploadFailed {
                kind: input.kind,
                asset_id: input.source_asset_id,
            });
        }
        mesh
    } else {
        try_upload_generated_mesh_from_parts(
            gpu,
            &data,
            &layout,
            &input.vertices,
            &input.indices,
            &input.prepared_derived_streams,
            existing,
        )
    };
    mesh.ok_or(ParticleRenderBufferError::GpuUploadFailed {
        kind: input.kind,
        asset_id: input.source_asset_id,
    })
}

/// Builds the synthetic mesh-upload metadata describing one generated particle mesh.
pub(super) fn generated_mesh_upload_data(
    mesh_asset_id: i32,
    vertex_count: usize,
    index_count: usize,
    bounds: RenderBoundingBox,
) -> Result<MeshUploadData, ParticleRenderBufferError> {
    let vertex_count_i32 = i32::try_from(vertex_count).map_err(|_conversion_error| {
        ParticleRenderBufferError::MeshTooLarge {
            kind: "generated",
            asset_id: mesh_asset_id,
        }
    })?;
    let index_count_i32 = i32::try_from(index_count).map_err(|_conversion_error| {
        ParticleRenderBufferError::MeshTooLarge {
            kind: "generated",
            asset_id: mesh_asset_id,
        }
    })?;
    Ok(MeshUploadData {
        buffer: SharedMemoryBufferDescriptor {
            length: 1,
            ..Default::default()
        },
        vertex_count: vertex_count_i32,
        index_buffer_format: IndexBufferFormat::UInt32,
        vertex_attributes: vec![
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::Position,
                format: VertexAttributeFormat::Float32,
                dimensions: 3,
            },
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::Normal,
                format: VertexAttributeFormat::Float32,
                dimensions: 3,
            },
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::UV0,
                format: VertexAttributeFormat::Float32,
                dimensions: 2,
            },
            VertexAttributeDescriptor {
                attribute: VertexAttributeType::Color,
                format: VertexAttributeFormat::Float32,
                dimensions: 4,
            },
        ],
        submeshes: if index_count_i32 > 0 {
            vec![SubmeshBufferDescriptor {
                topology: SubmeshTopology::Triangles,
                index_start: 0,
                index_count: index_count_i32,
                bounds,
            }]
        } else {
            Vec::new()
        },
        bounds,
        asset_id: mesh_asset_id,
        ..Default::default()
    })
}
