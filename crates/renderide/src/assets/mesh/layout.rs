//! Mesh packed-buffer layout matching `Renderite.Shared.MeshBuffer.ComputeBufferLayout`.
//!
//! Regions: vertices → indices → bone_counts → bone_weights → bind_poses → blendshape_data.

use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, SubmeshBufferDescriptor,
    VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

/// Bytes per sparse blendshape entry on the GPU:
/// `vertex_index: u32` + `delta.xyz: f32` (12) — matches [`blendshape_scatter_main`] struct layout.
pub const BLENDSHAPE_SPARSE_ENTRY_SIZE: usize = 16;

/// Bytes per shape range row: `first_entry: u32`, `entry_count: u32`.
pub const BLENDSHAPE_SHAPE_DESCRIPTOR_SIZE: usize = 8;

/// Deltas smaller than this magnitude (length squared) are dropped as non-influencing.
pub const BLENDSHAPE_POSITION_EPSILON_SQ: f32 = 1e-14;

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
    let blendshape_data_length = blendshape_buffers
        .map(|b| compute_blendshape_data_length(b, vertex_count as i32))
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

/// GPU-ready sparse position deltas and a small per-shape descriptor table (`first_entry`, `entry_count`).
///
/// Normal and tangent streams from the host are not stored; the current scatter pass applies position
/// deltas only.
pub struct BlendshapeGpuPack {
    /// Tightly packed rows of `vertex_index: u32` followed by `delta.xyz: f32` ([`BLENDSHAPE_SPARSE_ENTRY_SIZE`] bytes each).
    pub sparse_deltas: Vec<u8>,
    /// `num_blendshapes` rows of `(first_entry, entry_count)` as little-endian `u32` pairs ([`BLENDSHAPE_SHAPE_DESCRIPTOR_SIZE`] bytes per row).
    pub shape_descriptor_bytes: Vec<u8>,
    /// Same ranges as [`Self::shape_descriptor_bytes`], kept for CPU scatter planning without parsing bytes.
    pub shape_ranges: Vec<(u32, u32)>,
    /// Logical blendshape slot count (`max(blendshape_index) + 1`).
    pub num_blendshapes: i32,
}

/// Repacks host blendshape **position** deltas into sparse GPU storage (16 B per affected vertex),
/// a small per-shape descriptor table, and CPU-side ranges for scatter dispatches.
///
/// Only [`BlendshapeDataFlags::POSITIONS`] channels contribute. Normal/tangent streams are consumed
/// from the wire to advance offsets but are not uploaded.
pub fn extract_blendshape_offsets(
    raw: &[u8],
    layout: &MeshBufferLayout,
    blendshape_buffers: &[BlendshapeBufferDescriptor],
    vertex_count: i32,
) -> Option<BlendshapeGpuPack> {
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

    const VECTOR3_BYTES: usize = 12;
    let mut per_shape: Vec<Vec<(u32, [f32; 3])>> = vec![Vec::new(); num_blendshapes];

    let mut byte_offset = layout.blendshape_data_start;

    for descriptor in blendshape_buffers {
        let bi = descriptor.blendshape_index.max(0) as usize;
        if bi >= num_blendshapes {
            continue;
        }

        if descriptor.data_flags.positions() {
            let chunk_len = VECTOR3_BYTES * vertex_count;
            if byte_offset + chunk_len > raw.len() {
                return None;
            }
            for v in 0..vertex_count {
                let src_offset = byte_offset + v * VECTOR3_BYTES;
                let x = f32::from_le_bytes(raw[src_offset..src_offset + 4].try_into().ok()?);
                let y = f32::from_le_bytes(raw[src_offset + 4..src_offset + 8].try_into().ok()?);
                let z = f32::from_le_bytes(raw[src_offset + 8..src_offset + 12].try_into().ok()?);
                let mag_sq = z.mul_add(z, x.mul_add(x, y * y));
                if mag_sq > BLENDSHAPE_POSITION_EPSILON_SQ {
                    per_shape[bi].push((v as u32, [x, y, z]));
                }
            }
            byte_offset += chunk_len;
        }

        if descriptor.data_flags.normals() {
            let chunk_len = VECTOR3_BYTES * vertex_count;
            if byte_offset + chunk_len > raw.len() {
                return None;
            }
            byte_offset += chunk_len;
        }

        if descriptor.data_flags.tangets() {
            let chunk_len = VECTOR3_BYTES * vertex_count;
            if byte_offset + chunk_len > raw.len() {
                return None;
            }
            byte_offset += chunk_len;
        }
    }

    let mut sparse_deltas = Vec::new();
    let mut shape_descriptor_bytes = vec![0u8; num_blendshapes * BLENDSHAPE_SHAPE_DESCRIPTOR_SIZE];
    let mut shape_ranges: Vec<(u32, u32)> = Vec::with_capacity(num_blendshapes);

    for (s, entries) in per_shape.iter().enumerate() {
        let first_entry = (sparse_deltas.len() / BLENDSHAPE_SPARSE_ENTRY_SIZE) as u32;
        let count = entries.len() as u32;
        for (vi, d) in entries {
            sparse_deltas.extend_from_slice(&vi.to_le_bytes());
            sparse_deltas.extend_from_slice(&d[0].to_le_bytes());
            sparse_deltas.extend_from_slice(&d[1].to_le_bytes());
            sparse_deltas.extend_from_slice(&d[2].to_le_bytes());
        }
        let base = s.saturating_mul(BLENDSHAPE_SHAPE_DESCRIPTOR_SIZE);
        shape_descriptor_bytes[base..base + 4].copy_from_slice(&first_entry.to_le_bytes());
        shape_descriptor_bytes[base + 4..base + 8].copy_from_slice(&count.to_le_bytes());
        shape_ranges.push((first_entry, count));
    }

    Some(BlendshapeGpuPack {
        sparse_deltas,
        shape_descriptor_bytes,
        shape_ranges,
        num_blendshapes: num_blendshapes as i32,
    })
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

/// Extracts a float3 position stream and a normal stream from interleaved vertices into dense
/// `vec4<f32>` storage (16 bytes each per vertex).
///
/// Position must be at least three-component `float32`. Normal is allowed to be absent or
/// unsupported; in that case a stable +Z normal is synthesized so UI meshes that do not upload
/// normals still satisfy the shared raster vertex layout.
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
    let pos_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Position as i16))?;
    if pos_attr.format != VertexAttributeFormat::Float32 || pos_attr.dimensions < 3 {
        return None;
    }
    if pos.1 < 12 {
        return None;
    }

    let mut pos_out = vec![0u8; vertex_count * 16];
    let mut nrm_out = vec![0u8; vertex_count * 16];
    let one = 1.0f32.to_le_bytes();
    fill_normal_stream_with_forward_z(&mut nrm_out);

    let nrm = attribute_offset_and_size(attrs, VertexAttributeType::Normal);
    let nrm_attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (VertexAttributeType::Normal as i16));
    let nrm_offset = if matches!(
        (nrm, nrm_attr),
        (Some((_, sz)), Some(attr))
            if attr.format == VertexAttributeFormat::Float32 && attr.dimensions >= 3 && sz >= 12
    ) {
        nrm.map(|(off, _)| off)
    } else {
        None
    };

    for i in 0..vertex_count {
        let base = i * stride;
        let p0 = base + pos.0;
        if p0 + 12 > vertex_data.len() {
            return None;
        }
        let po = i * 16;
        pos_out[po..po + 12].copy_from_slice(&vertex_data[p0..p0 + 12]);
        pos_out[po + 12..po + 16].copy_from_slice(&one);

        if let Some(nrm_offset) = nrm_offset {
            let n0 = base + nrm_offset;
            if n0 + 12 > vertex_data.len() {
                return None;
            }
            let no = i * 16;
            nrm_out[no..no + 12].copy_from_slice(&vertex_data[n0..n0 + 12]);
        }
    }
    Some((pos_out, nrm_out))
}

fn fill_normal_stream_with_forward_z(out: &mut [u8]) {
    let zero = 0.0f32.to_le_bytes();
    let one = 1.0f32.to_le_bytes();
    for chunk in out.chunks_exact_mut(16) {
        chunk[0..4].copy_from_slice(&zero);
        chunk[4..8].copy_from_slice(&zero);
        chunk[8..12].copy_from_slice(&one);
        chunk[12..16].copy_from_slice(&zero);
    }
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
    vertex_float2_stream_bytes(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::UV0,
    )
}

/// Dense `vec2<f32>` vertex stream for an arbitrary float2 attribute.
///
/// Missing or unsupported attributes return zeros so optional embedded shader streams can still
/// bind a stable vertex buffer slot.
pub fn vertex_float2_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 8];
    let Some((off, sz)) = attribute_offset_and_size(attrs, target) else {
        return Some(out);
    };
    let attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (target as i16))?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 2 {
        return Some(out);
    }
    if sz < 8 {
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

/// Dense `vec4<f32>` vertex stream for an arbitrary float attribute.
///
/// Missing or unsupported attributes return `default` per vertex.
pub fn vertex_float4_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    default: [f32; 4],
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }
    let mut out = vec![0u8; vertex_count * 16];
    for chunk in out.chunks_exact_mut(16) {
        for (component, value) in default.iter().enumerate() {
            let o = component * 4;
            chunk[o..o + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    let Some((off, sz)) = attribute_offset_and_size(attrs, target) else {
        return Some(out);
    };
    let attr = attrs
        .iter()
        .find(|a| (a.attribute as i16) == (target as i16))?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 1 {
        return Some(out);
    }
    let dims = attr.dimensions.clamp(1, 4) as usize;
    if sz < dims * 4 {
        return Some(out);
    }
    for i in 0..vertex_count {
        let base = i * stride + off;
        if base + dims * 4 > vertex_data.len() {
            return None;
        }
        let o = i * 16;
        for c in 0..dims {
            let src = base + c * 4;
            out[o + c * 4..o + c * 4 + 4].copy_from_slice(&vertex_data[src..src + 4]);
        }
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
        let Some(rgba) = decode_vertex_color(vertex_data, base, color_attr) else {
            return Some(out);
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
        VertexAttributeFormat::UNorm8 | VertexAttributeFormat::UInt8 => {
            let end = base.checked_add(dims)?;
            let src = vertex_data.get(base..end)?;
            for (i, byte) in src.iter().take(dims).enumerate() {
                rgba[i] = *byte as f32 / 255.0;
            }
        }
        VertexAttributeFormat::UNorm16 | VertexAttributeFormat::UInt16 => {
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
