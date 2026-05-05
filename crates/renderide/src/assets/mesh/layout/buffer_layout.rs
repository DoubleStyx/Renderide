//! Host mesh buffer layout and basic packed-buffer region extraction.

use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, SubmeshBufferDescriptor,
    VertexAttributeDescriptor, VertexAttributeFormat,
};

pub(super) fn vertex_format_size(format: VertexAttributeFormat) -> i32 {
    match format {
        VertexAttributeFormat::Float32 => 4,
        VertexAttributeFormat::Half16 => 2,
        VertexAttributeFormat::UNorm8 => 1,
        VertexAttributeFormat::UNorm16 => 2,
        VertexAttributeFormat::SInt8 => 1,
        VertexAttributeFormat::SInt16 => 2,
        VertexAttributeFormat::SInt32 => 4,
        VertexAttributeFormat::UInt8 => 1,
        VertexAttributeFormat::UInt16 => 2,
        VertexAttributeFormat::UInt32 => 4,
    }
}

/// Interleaved vertex stride from [`VertexAttributeDescriptor`] list (host order).
pub fn compute_vertex_stride(attrs: &[VertexAttributeDescriptor]) -> i32 {
    attrs
        .iter()
        .map(|a| vertex_format_size(a.format) * a.dimensions)
        .sum()
}

/// Total index count from submeshes (`max(index_start + index_count)`).
pub fn compute_index_count(submeshes: &[SubmeshBufferDescriptor]) -> i32 {
    submeshes
        .iter()
        .map(|s| s.index_start + s.index_count)
        .max()
        .unwrap_or(0)
}

/// Bytes per index for [`IndexBufferFormat`].
pub fn index_bytes_per_element(format: IndexBufferFormat) -> i32 {
    match format {
        IndexBufferFormat::UInt16 => 2,
        IndexBufferFormat::UInt32 => 4,
    }
}

/// Maximum allowed mesh buffer size in bytes (`MeshBuffer.MAX_BUFFER_SIZE`).
pub const MAX_BUFFER_SIZE: usize = 2_147_483_648;

/// Byte offsets for each region of the host mesh payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshBufferLayout {
    /// Byte length of the interleaved vertex region at the start of the host buffer.
    pub vertex_size: usize,
    /// Byte offset where the index buffer begins.
    pub index_buffer_start: usize,
    /// Byte length of the index buffer region.
    pub index_buffer_length: usize,
    /// Byte offset where optional per-vertex bone count bytes begin.
    pub bone_counts_start: usize,
    /// Byte length of the bone counts region (or zero).
    pub bone_counts_length: usize,
    /// Byte offset where packed bone weight tail data begins.
    pub bone_weights_start: usize,
    /// Byte length of the bone weights region (or zero).
    pub bone_weights_length: usize,
    /// Byte offset where inverse bind-pose matrices begin.
    pub bind_poses_start: usize,
    /// Byte length of bind pose data (or zero).
    pub bind_poses_length: usize,
    /// Byte offset where packed blendshape delta payload begins.
    pub blendshape_data_start: usize,
    /// Byte length of blendshape payload (or zero).
    pub blendshape_data_length: usize,
    /// Total bytes required to cover all regions (validation vs mapped SHM).
    pub total_buffer_length: usize,
}

fn compute_blendshape_data_length(
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: i32,
) -> usize {
    let vertex_count = vertex_count.max(0) as usize;
    let bytes_per_channel = 12 * vertex_count;
    blendshape_buffers
        .iter()
        .map(|d| {
            let mut len = 0;
            if d.data_flags.positions() {
                len += bytes_per_channel;
            }
            if d.data_flags.normals() {
                len += bytes_per_channel;
            }
            if d.data_flags.tangets() {
                len += bytes_per_channel;
            }
            len
        })
        .sum()
}

/// Computes layout per `MeshBuffer.ComputeBufferLayout`.
pub fn compute_mesh_buffer_layout(
    vertex_stride: i32,
    vertex_count: i32,
    index_count: i32,
    index_bytes: i32,
    bone_count: i32,
    bone_weight_count: i32,
    blendshape_buffers: Option<&[BlendshapeBufferDescriptor]>,
) -> Result<MeshBufferLayout, &'static str> {
    let vertex_stride = vertex_stride.max(0) as usize;
    let vertex_count = vertex_count.max(0) as usize;
    let index_count = index_count.max(0) as usize;
    let index_bytes = index_bytes.max(0) as usize;
    let bone_count = bone_count.max(0) as usize;
    let bone_weight_count = bone_weight_count.max(0) as usize;

    let vertex_size = vertex_stride
        .checked_mul(vertex_count)
        .ok_or("Mesh buffer size overflow")?;
    let index_buffer_length = index_count
        .checked_mul(index_bytes)
        .ok_or("Mesh buffer size overflow")?;
    let index_buffer_start = vertex_size;
    let bone_counts_start = index_buffer_start + index_buffer_length;
    let bone_counts_length = vertex_count;
    let bone_weights_start = bone_counts_start + bone_counts_length;
    let bone_weights_length = bone_weight_count
        .checked_mul(8)
        .ok_or("Mesh buffer size overflow")?;
    let bind_poses_start = bone_weights_start + bone_weights_length;
    let bind_poses_length = bone_count
        .checked_mul(64)
        .ok_or("Mesh buffer size overflow")?;
    let blendshape_data_start = bind_poses_start + bind_poses_length;
    let blendshape_data_length = blendshape_buffers.map_or(0, |b| {
        compute_blendshape_data_length(b, vertex_count as i32)
    });
    let total_buffer_length = blendshape_data_start + blendshape_data_length;

    if total_buffer_length > MAX_BUFFER_SIZE {
        return Err("Mesh buffer size exceeds maximum allowed size of 2 GB.");
    }

    Ok(MeshBufferLayout {
        vertex_size,
        index_buffer_start,
        index_buffer_length,
        bone_counts_start,
        bone_counts_length,
        bone_weights_start,
        bone_weights_length,
        bind_poses_start,
        bind_poses_length,
        blendshape_data_start,
        blendshape_data_length,
        total_buffer_length,
    })
}

/// Extracts bind pose matrices from raw bytes (64 bytes per matrix).
pub fn extract_bind_poses(raw: &[u8], bone_count: usize) -> Option<Vec<[[f32; 4]; 4]>> {
    const MATRIX_BYTES: usize = 64;
    let need = bone_count.checked_mul(MATRIX_BYTES)?;
    if raw.len() < need {
        return None;
    }
    let mut poses = Vec::with_capacity(bone_count);
    for i in 0..bone_count {
        let start = i * MATRIX_BYTES;
        let slice = &raw[start..start + MATRIX_BYTES];
        poses.push(bytemuck::pod_read_unaligned(slice));
    }
    Some(poses)
}
