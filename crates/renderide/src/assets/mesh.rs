//! Mesh asset type and vertex layout helpers.

use super::Asset;
use super::AssetId;
use crate::shared::{
    IndexBufferFormat, RenderBoundingBox, SubmeshBufferDescriptor, VertexAttributeDescriptor,
    VertexAttributeFormat, VertexAttributeType,
};

fn vertex_format_size(format: VertexAttributeFormat) -> i32 {
    match format {
        VertexAttributeFormat::float32 => 4,
        VertexAttributeFormat::half16 => 2,
        VertexAttributeFormat::u_norm8 => 1,
        VertexAttributeFormat::u_norm16 => 2,
        VertexAttributeFormat::s_int8 => 1,
        VertexAttributeFormat::s_int16 => 2,
        VertexAttributeFormat::s_int32 => 4,
        VertexAttributeFormat::u_int8 => 1,
        VertexAttributeFormat::u_int16 => 2,
        VertexAttributeFormat::u_int32 => 4,
    }
}

/// Computes the interleaved vertex stride from vertex attributes.
pub fn compute_vertex_stride(attrs: &[VertexAttributeDescriptor]) -> i32 {
    attrs.iter()
        .map(|a| vertex_format_size(a.format) * a.dimensions)
        .sum()
}

/// Returns (offset_bytes, size_bytes) for the first attribute of the given type, or None.
pub fn attribute_offset_and_size(
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<(usize, usize)> {
    attribute_offset_size_format(attrs, target).map(|(o, s, _)| (o, s))
}

/// Returns (offset_bytes, size_bytes, format) for the first attribute of the given type, or None.
pub fn attribute_offset_size_format(
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<(usize, usize, VertexAttributeFormat)> {
    let mut offset: i32 = 0;
    for a in attrs {
        let size = (vertex_format_size(a.format) * a.dimensions) as usize;
        if (a.attribute as i16) == (target as i16) {
            return Some((offset as usize, size, a.format));
        }
        offset += size as i32;
    }
    None
}

fn compute_index_count(submeshes: &[SubmeshBufferDescriptor]) -> i32 {
    submeshes
        .iter()
        .map(|s| s.index_start + s.index_count)
        .max()
        .unwrap_or(0)
}

fn index_bytes_per_element(format: IndexBufferFormat) -> i32 {
    match format {
        IndexBufferFormat::u_int16 => 2,
        IndexBufferFormat::u_int32 => 4,
    }
}

/// Stored mesh geometry for GPU upload.
pub struct MeshAsset {
    /// Unique identifier for this mesh.
    pub id: AssetId,
    /// Raw vertex buffer data.
    pub vertex_data: Vec<u8>,
    /// Raw index buffer data.
    pub index_data: Vec<u8>,
    /// Vertex count.
    pub vertex_count: i32,
    /// Index count.
    pub index_count: i32,
    /// Index format (u16 or u32).
    pub index_format: IndexBufferFormat,
    /// Per-submesh (index_start, index_count).
    pub submeshes: Vec<SubmeshBufferDescriptor>,
    /// Vertex layout for parsing position, UVs, etc.
    pub vertex_attributes: Vec<VertexAttributeDescriptor>,
    /// Bounding box (center + extents).
    pub bounds: RenderBoundingBox,
    /// Bind poses (inverse bind matrices), one per bone. Only present for skinned meshes.
    pub bind_poses: Option<Vec<[[f32; 4]; 4]>>,
    /// Per-vertex bone count (1 byte each). Only present for skinned meshes.
    pub bone_counts: Option<Vec<u8>>,
    /// Flat bone weights (weight f32, bone_index i32 per entry). Only present for skinned meshes.
    pub bone_weights: Option<Vec<u8>>,
}

impl MeshAsset {
    /// Number of bones in the skeleton. Zero for non-skinned meshes.
    pub fn bone_count(&self) -> i32 {
        self.bind_poses
            .as_ref()
            .map(|v| v.len() as i32)
            .unwrap_or(0)
    }

    /// Number of bone weights across all vertices (BoneWeight entries, each 8 bytes).
    pub fn bone_weight_count(&self) -> i32 {
        self.bone_weights
            .as_ref()
            .map(|v| (v.len() / 8) as i32)
            .unwrap_or(0)
    }
}

impl Asset for MeshAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}

/// Layout offsets computed per MeshBuffer.ComputeBufferLayout.
#[derive(Clone, Copy)]
pub struct MeshBufferLayout {
    pub vertex_size: usize,
    pub index_buffer_start: usize,
    pub index_buffer_length: usize,
    pub bone_counts_start: usize,
    pub bone_counts_length: usize,
    pub bone_weights_start: usize,
    pub bone_weights_length: usize,
    pub bind_poses_start: usize,
    pub bind_poses_length: usize,
}

/// Computes buffer layout matching MeshBuffer.ComputeBufferLayout.
pub fn compute_mesh_buffer_layout(
    vertex_stride: i32,
    vertex_count: i32,
    index_count: i32,
    index_bytes: i32,
    bone_count: i32,
    bone_weight_count: i32,
) -> MeshBufferLayout {
    let vertex_size = (vertex_stride * vertex_count) as usize;
    let index_buffer_length = (index_count * index_bytes) as usize;
    let index_buffer_start = vertex_size;
    let bone_counts_start = index_buffer_start + index_buffer_length;
    let bone_counts_length = vertex_count as usize;
    let bone_weights_start = bone_counts_start + bone_counts_length;
    let bone_weights_length = (bone_weight_count * 8) as usize; // BoneWeight = 8 bytes
    let bind_poses_start = bone_weights_start + bone_weights_length;
    let bind_poses_length = (bone_count * 64) as usize; // Matrix4x4 = 64 bytes
    MeshBufferLayout {
        vertex_size,
        index_buffer_start,
        index_buffer_length,
        bone_counts_start,
        bone_counts_length,
        bone_weights_start,
        bone_weights_length,
        bind_poses_start,
        bind_poses_length,
    }
}

/// Extracts bind poses (4x4 matrices) from raw buffer. Expects 64 bytes per matrix.
pub fn extract_bind_poses(raw: &[u8], bone_count: usize) -> Option<Vec<[[f32; 4]; 4]>> {
    const MATRIX_BYTES: usize = 64;
    if raw.len() < bone_count * MATRIX_BYTES {
        return None;
    }
    let mut poses = Vec::with_capacity(bone_count);
    for i in 0..bone_count {
        let start = i * MATRIX_BYTES;
        let slice = &raw[start..start + MATRIX_BYTES];
        let mat: [[f32; 4]; 4] = bytemuck::pod_read_unaligned(slice);
        poses.push(mat);
    }
    Some(poses)
}
