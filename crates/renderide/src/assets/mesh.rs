//! Mesh asset type and vertex layout helpers.

use super::Asset;
use super::AssetId;
use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, RenderBoundingBox, SubmeshBufferDescriptor,
    VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};
use bytemuck::{Pod, Zeroable};

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
    attrs
        .iter()
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

/// Computes total index count from submeshes (max of index_start + index_count across submeshes).
pub fn compute_index_count(submeshes: &[SubmeshBufferDescriptor]) -> i32 {
    submeshes
        .iter()
        .map(|s| s.index_start + s.index_count)
        .max()
        .unwrap_or(0)
}

/// Returns bytes per index element for the given format.
pub fn index_bytes_per_element(format: IndexBufferFormat) -> i32 {
    match format {
        IndexBufferFormat::u_int16 => 2,
        IndexBufferFormat::u_int32 => 4,
    }
}

/// Stride in bytes for one BlendshapeOffset in GPU storage buffer (WGSL vec3 alignment).
pub const BLENDSHAPE_OFFSET_GPU_STRIDE: usize = 48;

/// Per-vertex blendshape offset in GPU-friendly format.
///
/// Extraction outputs 48 bytes per element (WGSL vec3 16-byte alignment): position at 0,
/// normal at 16, tangent at 32.
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct BlendshapeOffset {
    pub position_offset: [f32; 3],
    pub normal_offset: [f32; 3],
    pub tangent_offset: [f32; 3],
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
    /// Blendshape offsets in GPU-friendly format: for each blendshape index, for each vertex,
    /// position (0-12), normal (16-28), tangent (32-44) per [`BLENDSHAPE_OFFSET_GPU_STRIDE`].
    pub blendshape_offsets: Option<Vec<u8>>,
    /// Number of blendshape slots (max(blendshape_index + 1) over descriptors). Zero when no blendshapes.
    pub num_blendshapes: i32,
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

/// Maximum allowed mesh buffer size in bytes (matches Renderite.Shared.MeshBuffer.MAX_BUFFER_SIZE).
pub const MAX_BUFFER_SIZE: usize = 2_147_483_648;

/// Layout offsets computed per Renderite.Shared.MeshBuffer.ComputeBufferLayout.
///
/// Regions are laid out in order: vertices, indices, bone_counts, bone_weights, bind_poses,
/// blendshape_data. The blendshape region starts immediately after bind poses.
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
    /// Byte offset where blendshape data begins (immediately after bind poses).
    pub blendshape_data_start: usize,
    /// Total byte length of the blendshape region.
    pub blendshape_data_length: usize,
    /// Total buffer length in bytes. Must not exceed [`MAX_BUFFER_SIZE`].
    pub total_buffer_length: usize,
}

/// Computes blendshape region size by iterating descriptors and adding 12 × vertex_count bytes
/// for each of positions, normals, tangents when the corresponding [`BlendshapeDataFlags`] bit is set.
/// Note: The tangents flag is named `tangets` in the IPC contract to match the host.
fn compute_blendshape_data_length(
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: i32,
) -> usize {
    let vertex_count = vertex_count.max(0) as usize;
    let bytes_per_channel = 12 * vertex_count; // 3 floats × 4 bytes
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

/// Computes buffer layout matching Renderite.Shared.MeshBuffer.ComputeBufferLayout.
///
/// Regions: vertices, indices, bone_counts, bone_weights, bind_poses, blendshape_data.
/// The blendshape region starts immediately after bind poses. For each
/// [`BlendshapeBufferDescriptor`], adds 12 × vertex_count bytes for positions, normals, and
/// tangents when the corresponding [`BlendshapeDataFlags`] bit is set.
///
/// Returns an error if the total buffer length would exceed 2 GB (MAX_BUFFER_SIZE).
pub fn compute_mesh_buffer_layout(
    vertex_stride: i32,
    vertex_count: i32,
    index_count: i32,
    index_bytes: i32,
    bone_count: i32,
    bone_weight_count: i32,
    blendshape_buffers: Option<&[BlendshapeBufferDescriptor]>,
) -> Result<MeshBufferLayout, &'static str> {
    let vertex_size = (vertex_stride * vertex_count) as usize;
    let index_buffer_length = (index_count * index_bytes) as usize;
    let index_buffer_start = vertex_size;
    let bone_counts_start = index_buffer_start + index_buffer_length;
    let bone_counts_length = vertex_count as usize;
    let bone_weights_start = bone_counts_start + bone_counts_length;
    let bone_weights_length = (bone_weight_count * 8) as usize; // BoneWeight = 8 bytes
    let bind_poses_start = bone_weights_start + bone_weights_length;
    let bind_poses_length = (bone_count * 64) as usize; // Matrix4x4 = 64 bytes
    let blendshape_data_start = bind_poses_start + bind_poses_length;
    let blendshape_data_length = blendshape_buffers
        .map(|b| compute_blendshape_data_length(b, vertex_count))
        .unwrap_or(0);
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

/// Extracts blendshape offsets from the raw buffer into GPU-friendly format.
///
/// Reads from `raw` starting at `layout.blendshape_data_start`. For each descriptor, reads
/// positions (if present), normals (if present), tangents (if present) as Vector3 chunks
/// (12 bytes each), matching Renderite.Shared.MeshBuffer.ComputeBufferLayout byte order.
/// Writes into output with [`BLENDSHAPE_OFFSET_GPU_STRIDE`] bytes per element: position at 0,
/// normal at 16, tangent at 32 (WGSL vec3 alignment). `num_blendshapes` is max(blendshape_index+1).
pub fn extract_blendshape_offsets(
    raw: &[u8],
    layout: &MeshBufferLayout,
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: i32,
) -> Option<(Vec<u8>, i32)> {
    if blendshape_buffers.is_empty() || vertex_count <= 0 {
        return None;
    }
    let vertex_count = vertex_count as usize;
    // Clamp each index to ≥0 before adding 1 so a negative i32 can't wrap to
    // a giant usize.  Also cap the result at a sanity limit — blendshape counts
    // in the thousands would require gigabytes and are clearly bad data.
    const MAX_BLENDSHAPES: usize = 4096;
    let num_blendshapes = blendshape_buffers
        .iter()
        .map(|d| d.blendshape_index.max(0) + 1)
        .max()
        .unwrap_or(0) as usize;
    if num_blendshapes == 0 {
        return None;
    }
    if num_blendshapes > MAX_BLENDSHAPES {
        logger::warn!(
            "extract_blendshape_offsets: num_blendshapes={} exceeds sanity cap {}; rejecting mesh",
            num_blendshapes, MAX_BLENDSHAPES
        );
        return None;
    }

    let required_len = layout.blendshape_data_start + layout.blendshape_data_length;
    if raw.len() < required_len {
        return None;
    }

    let alloc_size = num_blendshapes
        .checked_mul(vertex_count)
        .and_then(|n| n.checked_mul(BLENDSHAPE_OFFSET_GPU_STRIDE));
    let Some(alloc_size) = alloc_size else {
        logger::warn!(
            "extract_blendshape_offsets: allocation overflow (num_blendshapes={} vertex_count={} stride={})",
            num_blendshapes, vertex_count, BLENDSHAPE_OFFSET_GPU_STRIDE
        );
        return None;
    };
    let mut out = vec![0u8; alloc_size];
    const VECTOR3_BYTES: usize = 12;
    const POSITION_OFFSET: usize = 0;
    const NORMAL_OFFSET: usize = 16;
    const TANGENT_OFFSET: usize = 32;

    let mut byte_offset = layout.blendshape_data_start;

    for descriptor in blendshape_buffers {
        let bi = descriptor.blendshape_index.max(0) as usize;

        if descriptor.data_flags.positions() {
            let chunk_len = VECTOR3_BYTES * vertex_count;
            if byte_offset + chunk_len > raw.len() {
                return None;
            }
            for v in 0..vertex_count {
                let out_offset =
                    (bi * vertex_count + v) * BLENDSHAPE_OFFSET_GPU_STRIDE + POSITION_OFFSET;
                let src_offset = byte_offset + v * VECTOR3_BYTES;
                out[out_offset..out_offset + VECTOR3_BYTES]
                    .copy_from_slice(&raw[src_offset..src_offset + VECTOR3_BYTES]);
            }
            byte_offset += chunk_len;
        }

        if descriptor.data_flags.normals() {
            let chunk_len = VECTOR3_BYTES * vertex_count;
            if byte_offset + chunk_len > raw.len() {
                return None;
            }
            for v in 0..vertex_count {
                let out_offset =
                    (bi * vertex_count + v) * BLENDSHAPE_OFFSET_GPU_STRIDE + NORMAL_OFFSET;
                let src_offset = byte_offset + v * VECTOR3_BYTES;
                out[out_offset..out_offset + VECTOR3_BYTES]
                    .copy_from_slice(&raw[src_offset..src_offset + VECTOR3_BYTES]);
            }
            byte_offset += chunk_len;
        }

        if descriptor.data_flags.tangets() {
            let chunk_len = VECTOR3_BYTES * vertex_count;
            if byte_offset + chunk_len > raw.len() {
                return None;
            }
            for v in 0..vertex_count {
                let out_offset =
                    (bi * vertex_count + v) * BLENDSHAPE_OFFSET_GPU_STRIDE + TANGENT_OFFSET;
                let src_offset = byte_offset + v * VECTOR3_BYTES;
                out[out_offset..out_offset + VECTOR3_BYTES]
                    .copy_from_slice(&raw[src_offset..src_offset + VECTOR3_BYTES]);
            }
            byte_offset += chunk_len;
        }
    }

    Some((out, num_blendshapes as i32))
}

/// Identity matrix for synthetic bind pose (blendshape-only meshes). Column-major.
pub fn identity_bind_pose() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Builds synthetic bone data for blendshape-only meshes (bone_count = 0, blendshape_count > 0).
/// Host sends boneCount = -1 and one slot transform index; we use identity bind pose and full
/// weight on bone 0 so vertices are transformed by the slot only.
pub fn synthetic_bone_data_for_blendshape_only(
    vertex_count: i32,
) -> (Vec<[[f32; 4]; 4]>, Vec<u8>, Vec<u8>) {
    let vc = vertex_count.max(0) as usize;
    let bind_poses = vec![identity_bind_pose()];
    let bone_counts = vec![1u8; vc];
    let mut bone_weights = Vec::with_capacity(vc * 8);
    for _ in 0..vc {
        bone_weights.extend_from_slice(&1.0f32.to_le_bytes());
        bone_weights.extend_from_slice(&0i32.to_le_bytes());
    }
    (bind_poses, bone_counts, bone_weights)
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
