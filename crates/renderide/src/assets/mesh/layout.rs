//! Mesh packed-buffer layout matching `Renderite.Shared.MeshBuffer.ComputeBufferLayout`.
//!
//! Regions: vertices → indices → bone_counts → bone_weights → bind_poses → blendshape_data.

use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, SubmeshBufferDescriptor,
    VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

/// Stride in bytes for one blendshape offset tuple in GPU storage (WGSL `vec3` alignment).
pub const BLENDSHAPE_OFFSET_GPU_STRIDE: usize = 48;

fn vertex_format_size(format: VertexAttributeFormat) -> i32 {
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
#[derive(Clone, Copy, Debug)]
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
    let vertex_size = (vertex_stride * vertex_count) as usize;
    let index_buffer_length = (index_count * index_bytes) as usize;
    let index_buffer_start = vertex_size;
    let bone_counts_start = index_buffer_start + index_buffer_length;
    let bone_counts_length = vertex_count as usize;
    let bone_weights_start = bone_counts_start + bone_counts_length;
    let bone_weights_length = (bone_weight_count * 8) as usize;
    let bind_poses_start = bone_weights_start + bone_weights_length;
    let bind_poses_length = (bone_count * 64) as usize;
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

/// Identity matrix (column-major) for synthetic bind poses.
pub fn identity_bind_pose() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Synthetic bone streams for blendshape-only meshes (`bone_count == 0`, blendshapes present).
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

/// Repacks host blendshape deltas into GPU storage (`BLENDSHAPE_OFFSET_GPU_STRIDE` per vertex per shape).
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
            "extract_blendshape_offsets: num_blendshapes={num_blendshapes} exceeds cap {MAX_BLENDSHAPES}"
        );
        return None;
    }

    let required_len = layout.blendshape_data_start + layout.blendshape_data_length;
    if raw.len() < required_len {
        return None;
    }

    let alloc_size = num_blendshapes
        .checked_mul(vertex_count)?
        .checked_mul(BLENDSHAPE_OFFSET_GPU_STRIDE)?;
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

/// Returns byte offset and size of the first attribute of `target` type in the interleaved vertex.
pub fn attribute_offset_and_size(
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<(usize, usize)> {
    let mut offset: i32 = 0;
    for a in attrs {
        let size = (vertex_format_size(a.format) * a.dimensions) as usize;
        if (a.attribute as i16) == (target as i16) {
            return Some((offset as usize, size));
        }
        offset += size as i32;
    }
    None
}

/// Extracts float3 position and normal streams from interleaved vertices into dense `vec4<f32>`
/// storage (16 bytes each per vertex). Returns [`None`] when attributes are missing or not both
/// three-component `float32` (debug raster / compute path requirement).
pub fn extract_float3_position_normal_as_vec4_streams(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<(Vec<u8>, Vec<u8>)> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let pos = attribute_offset_and_size(attrs, VertexAttributeType::Position)?;
    let nrm = attribute_offset_and_size(attrs, VertexAttributeType::Normal)?;
    let pos_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Position as i16))?;
    let nrm_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Normal as i16))?;
    if pos_attr.format != VertexAttributeFormat::Float32 || pos_attr.dimensions != 3 {
        return None;
    }
    if nrm_attr.format != VertexAttributeFormat::Float32 || nrm_attr.dimensions != 3 {
        return None;
    }
    if pos.1 != 12 || nrm.1 != 12 {
        return None;
    }

    let mut pos_out = vec![0u8; vertex_count * 16];
    let mut nrm_out = vec![0u8; vertex_count * 16];
    let one = 1.0f32.to_le_bytes();
    for i in 0..vertex_count {
        let base = i * stride;
        let p0 = base + pos.0;
        let n0 = base + nrm.0;
        if p0 + 12 > vertex_data.len() || n0 + 12 > vertex_data.len() {
            return None;
        }
        let po = i * 16;
        pos_out[po..po + 12].copy_from_slice(&vertex_data[p0..p0 + 12]);
        pos_out[po + 12..po + 16].copy_from_slice(&one);

        let no = i * 16;
        nrm_out[no..no + 12].copy_from_slice(&vertex_data[n0..n0 + 12]);
        nrm_out[no + 12..no + 16].fill(0);
    }
    Some((pos_out, nrm_out))
}

/// Dense `vec2<f32>` UV stream (`8` bytes per vertex) for embedded materials (e.g. world Unlit).
///
/// When [`VertexAttributeType::UV0`] is missing or not `float32`×2, returns **zeros** so a vertex buffer
/// slot can always be bound.
pub fn uv0_float2_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 8];
    let Some((off, sz)) = attribute_offset_and_size(attrs, VertexAttributeType::UV0) else {
        return Some(out);
    };
    let uv_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::UV0 as i16))?;
    if uv_attr.format != VertexAttributeFormat::Float32 || uv_attr.dimensions != 2 {
        return Some(out);
    }
    if sz != 8 {
        return Some(out);
    }
    for i in 0..vertex_count {
        let base = i * stride + off;
        if base + 8 > vertex_data.len() {
            return None;
        }
        let o = i * 8;
        out[o..o + 8].copy_from_slice(&vertex_data[base..base + 8]);
    }
    Some(out)
}

/// Dense `vec4<f32>` color stream (`16` bytes per vertex) for UI / text embedded materials.
///
/// Missing or unsupported color attributes default to opaque white so non-colored meshes keep
/// rendering correctly while UI meshes can consume the host color stream when present.
pub fn color_float4_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 16];
    fill_color_stream_with_white(&mut out);

    let Some((off, _sz)) = attribute_offset_and_size(attrs, VertexAttributeType::Color) else {
        return Some(out);
    };
    let color_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Color as i16))?;

    for i in 0..vertex_count {
        let base = i * stride + off;
        if base >= vertex_data.len() {
            return None;
        }
        let rgba = match decode_vertex_color(vertex_data, base, color_attr) {
            Some(v) => v,
            None => return Some(out),
        };
        let o = i * 16;
        for (component, value) in rgba.into_iter().enumerate() {
            out[o + component * 4..o + component * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    Some(out)
}

fn fill_color_stream_with_white(out: &mut [u8]) {
    let one = 1.0f32.to_le_bytes();
    for chunk in out.chunks_exact_mut(16) {
        chunk[0..4].copy_from_slice(&one);
        chunk[4..8].copy_from_slice(&one);
        chunk[8..12].copy_from_slice(&one);
        chunk[12..16].copy_from_slice(&one);
    }
}

fn decode_vertex_color(
    vertex_data: &[u8],
    base: usize,
    attr: &VertexAttributeDescriptor,
) -> Option<[f32; 4]> {
    let dims = attr.dimensions.clamp(1, 4) as usize;
    let mut rgba = [1.0f32; 4];
    match attr.format {
        VertexAttributeFormat::UNorm8 => {
            let end = base.checked_add(dims)?;
            let src = vertex_data.get(base..end)?;
            for (i, byte) in src.iter().take(dims).enumerate() {
                rgba[i] = *byte as f32 / 255.0;
            }
        }
        VertexAttributeFormat::UNorm16 => {
            let end = base.checked_add(dims.checked_mul(2)?)?;
            let src = vertex_data.get(base..end)?;
            for (i, chunk) in src.chunks(2).take(dims).enumerate() {
                rgba[i] = u16::from_le_bytes(chunk.try_into().ok()?) as f32 / 65535.0;
            }
        }
        VertexAttributeFormat::Float32 => {
            let end = base.checked_add(dims.checked_mul(4)?)?;
            let src = vertex_data.get(base..end)?;
            for (i, chunk) in src.chunks(4).take(dims).enumerate() {
                rgba[i] = f32::from_le_bytes(chunk.try_into().ok()?);
            }
        }
        _ => return None,
    }
    Some(rgba)
}

/// Splits the mesh tail `bone_weights` region into GPU storage buffers for the skinning shader:
/// `array<vec4<u32>>` joint indices and `array<vec4<f32>>` weights per vertex.
///
/// Supports either **4 influences** (`32 * vertex_count` bytes as `(f32 weight, i32 index)` tuples)
/// or **1 influence** (`8 * vertex_count` bytes) as produced by [`super::synthetic_bone_data_for_blendshape_only`].
pub fn split_bone_weights_tail_for_gpu(
    bone_weights_tail: &[u8],
    vertex_count: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if vertex_count == 0 {
        return None;
    }
    let four_inf = vertex_count * 32;
    let one_inf = vertex_count * 8;
    let span = if bone_weights_tail.len() >= four_inf {
        4usize
    } else if bone_weights_tail.len() >= one_inf {
        1usize
    } else {
        return None;
    };

    let mut idx_bytes = vec![0u8; vertex_count * 16];
    let mut wt_bytes = vec![0u8; vertex_count * 16];

    for v in 0..vertex_count {
        for k in 0..4 {
            let (w, j) = if k < span {
                let off = v * (span * 8) + k * 8;
                if off + 8 > bone_weights_tail.len() {
                    return None;
                }
                let w_raw = f32::from_le_bytes(bone_weights_tail[off..off + 4].try_into().ok()?);
                let j = i32::from_le_bytes(bone_weights_tail[off + 4..off + 8].try_into().ok()?);
                // Match legacy skinned VB build: unmapped bones must not contribute (index 0 only if weight > 0).
                if j < 0 {
                    (0.0f32, 0u32)
                } else {
                    (w_raw, j as u32)
                }
            } else {
                (0.0f32, 0u32)
            };
            let wb = v * 16 + k * 4;
            wt_bytes[wb..wb + 4].copy_from_slice(&w.to_le_bytes());
            idx_bytes[wb..wb + 4].copy_from_slice(&j.to_le_bytes());
        }
    }
    Some((idx_bytes, wt_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{SubmeshBufferDescriptor, SubmeshTopology};
    use glam::Mat4;

    #[test]
    fn layout_no_bones_no_blend_matches_stride() {
        let sub = vec![SubmeshBufferDescriptor {
            topology: SubmeshTopology::default(),
            index_start: 0,
            index_count: 3,
            bounds: crate::shared::RenderBoundingBox::default(),
        }];
        let ic = compute_index_count(&sub);
        let l = compute_mesh_buffer_layout(32, 2, ic, 2, 0, 0, None).unwrap();
        assert_eq!(l.vertex_size, 64);
        assert_eq!(l.index_buffer_start, 64);
        assert_eq!(l.index_buffer_length, 6);
        assert_eq!(l.bone_counts_length, 2);
        assert_eq!(l.bone_weights_length, 0);
        assert_eq!(l.bind_poses_length, 0);
        assert_eq!(l.total_buffer_length, 64 + 6 + 2);
    }

    #[test]
    fn vertex_stride_sum() {
        use crate::shared::{VertexAttributeDescriptor, VertexAttributeType};
        let attrs = [
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
        ];
        assert_eq!(compute_vertex_stride(&attrs), 24);
    }

    #[test]
    fn split_bone_weights_four_influences_roundtrip() {
        let mut tail = Vec::new();
        for v in 0..2u8 {
            for k in 0..4u8 {
                let w = (v + k) as f32 * 0.1;
                let j = (k as i32) + (v as i32) * 10;
                tail.extend_from_slice(&w.to_le_bytes());
                tail.extend_from_slice(&j.to_le_bytes());
            }
        }
        let (idx, wt) = split_bone_weights_tail_for_gpu(&tail, 2).expect("split");
        let w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
        let i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
        assert!((w0 - 0.0).abs() < 1e-5);
        assert_eq!(i0, 0);

        // Vertex 1, first influence (k=0): w=0.1, j=10
        let w1_0 = f32::from_le_bytes(wt[16..20].try_into().unwrap());
        let i1_0 = u32::from_le_bytes(idx[16..20].try_into().unwrap());
        assert!((w1_0 - 0.1).abs() < 1e-5);
        assert_eq!(i1_0, 10);
    }

    #[test]
    fn split_bone_weights_negative_index_zeroes_weight() {
        let mut tail = Vec::new();
        tail.extend_from_slice(&0.5f32.to_le_bytes());
        tail.extend_from_slice(&(-1i32).to_le_bytes());
        tail.extend_from_slice(&0f32.to_le_bytes());
        tail.extend_from_slice(&0i32.to_le_bytes());
        tail.extend_from_slice(&0f32.to_le_bytes());
        tail.extend_from_slice(&0i32.to_le_bytes());
        tail.extend_from_slice(&0f32.to_le_bytes());
        tail.extend_from_slice(&0i32.to_le_bytes());
        let (idx, wt) = split_bone_weights_tail_for_gpu(&tail, 1).expect("split");
        let w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
        let i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
        assert!((w0 - 0.0).abs() < 1e-5);
        assert_eq!(i0, 0u32);
    }

    /// Unity uploads inverse bind matrices; the renderer stores them as [`Mat4::from_cols_array_2d`]
    /// without an extra `.inverse()` (see [`crate::assets::mesh::GpuMesh::skinning_bind_matrices`]).
    #[test]
    fn unity_bindpose_raw_matches_glam_columns_not_inverse() {
        let expected = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let a = expected.to_cols_array();
        let raw: [[f32; 4]; 4] = [
            [a[0], a[1], a[2], a[3]],
            [a[4], a[5], a[6], a[7]],
            [a[8], a[9], a[10], a[11]],
            [a[12], a[13], a[14], a[15]],
        ];
        let stored = Mat4::from_cols_array_2d(&raw);
        assert!(stored.abs_diff_eq(expected, 1e-5));
        assert!(!stored.abs_diff_eq(expected.inverse(), 1e-2));
    }

    #[test]
    fn uv0_float2_zeros_when_missing() {
        let attrs = [
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
        ];
        let stride = 24usize;
        let verts = 2usize;
        let raw = vec![0u8; verts * stride];
        let out = uv0_float2_stream_bytes(&raw, verts, stride, &attrs).expect("uv stream");
        assert_eq!(out.len(), verts * 8);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn color_stream_defaults_to_opaque_white_when_missing() {
        let attrs = [VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        }];
        let raw = vec![0u8; 12];
        let out = color_float4_stream_bytes(&raw, 1, 12, &attrs).expect("color stream");
        let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
        assert_eq!(rgba, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn color_stream_decodes_unorm8_rgba() {
        let attrs = [VertexAttributeDescriptor {
            attribute: VertexAttributeType::Color,
            format: VertexAttributeFormat::UNorm8,
            dimensions: 4,
        }];
        let raw = vec![255u8, 128u8, 0u8, 64u8];
        let out = color_float4_stream_bytes(&raw, 1, 4, &attrs).expect("color stream");
        let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
        assert!((rgba[0] - 1.0).abs() < 1e-6);
        assert!((rgba[1] - (128.0 / 255.0)).abs() < 1e-6);
        assert!((rgba[2] - 0.0).abs() < 1e-6);
        assert!((rgba[3] - (64.0 / 255.0)).abs() < 1e-6);
    }
}
