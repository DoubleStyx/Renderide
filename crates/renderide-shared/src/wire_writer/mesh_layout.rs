//! Host-side mesh payload encoder.
//!
//! Produces the byte buffer for `MeshUploadData.buffer`, matching the canonical region order from
//! `Renderite.Shared.MeshBuffer.ComputeBufferLayout`: vertices → indices → `bone_counts` →
//! `bone_weights` → `bind_poses` → `blendshape_data`. The renderer-side parser lives at
//! `crates/renderide/src/assets/mesh/layout.rs` and reads exactly this layout.
//!
//! For the integration test we only use vertices and indices: the sphere has no bones and no
//! blendshapes. Vertices are interleaved according to `vertex_attributes`; the encoder requires
//! that the caller pre-interleave the bytes since attribute layouts can be arbitrary.

use crate::shared::{IndexBufferFormat, VertexAttributeDescriptor, VertexAttributeFormat};

/// One vertex attribute's source bytes in **per-vertex** layout. The encoder concatenates them
/// across all attributes, in the order given by [`MeshLayoutInput::vertex_attributes`], to form
/// each interleaved vertex.
#[derive(Clone, Debug)]
pub struct InterleavedAttribute<'a> {
    /// Per-vertex bytes for this attribute. Length must equal `vertex_count * attribute_byte_size`.
    pub bytes: &'a [u8],
}

/// Inputs to [`write_mesh_payload`].
#[derive(Clone, Debug)]
pub struct MeshLayoutInput<'a> {
    /// Number of vertices in the mesh.
    pub vertex_count: i32,
    /// Vertex attribute layout. The encoder validates that `attributes.len() == sources.len()`.
    pub vertex_attributes: Vec<VertexAttributeDescriptor>,
    /// Per-attribute source bytes (same order as `vertex_attributes`, one entry per attribute).
    ///
    /// Each entry has `vertex_count` rows; each row is `vertex_format_size(format) * dimensions`
    /// bytes. The encoder packs row 0 of attribute 0, then row 0 of attribute 1, ..., then row 1
    /// of attribute 0, ... interleaving by row.
    pub sources: Vec<InterleavedAttribute<'a>>,
    /// Index data in the chosen format: u16 little-endian (`UInt16`) or u32 little-endian
    /// (`UInt32`).
    pub indices: &'a [u8],
    /// Format of the indices.
    pub index_buffer_format: IndexBufferFormat,
}

/// Output of [`write_mesh_payload`]: the packed byte buffer plus computed strides for verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshPayload {
    /// Bytes ready to be written to a shared-memory buffer and pointed at by
    /// `MeshUploadData.buffer`.
    pub bytes: Vec<u8>,
    /// Interleaved vertex stride (bytes per vertex) used while packing.
    pub vertex_stride_bytes: usize,
    /// Total length of the index region in bytes (`indices.len()`).
    pub index_region_bytes: usize,
}

/// Errors produced by [`write_mesh_payload`].
#[derive(Debug, thiserror::Error)]
pub enum MeshLayoutError {
    /// `vertex_attributes` and `sources` had different lengths.
    #[error("vertex_attributes ({attrs}) and sources ({srcs}) must have the same length")]
    AttributeSourceMismatch {
        /// Number of attribute descriptors.
        attrs: usize,
        /// Number of source byte slices.
        srcs: usize,
    },
    /// One of the attribute source slices had the wrong byte length for the declared
    /// `vertex_count` × per-vertex size.
    #[error(
        "attribute {index} (type={attribute_type:?}, format={format:?}, dims={dims}) source had \
         {got} bytes; expected {vertex_count} vertices × {row_bytes} bytes/vertex"
    )]
    SourceBytesMismatch {
        /// Index of the offending attribute.
        index: usize,
        /// Attribute type tag (mostly for debugging).
        attribute_type: crate::shared::VertexAttributeType,
        /// Attribute format.
        format: VertexAttributeFormat,
        /// Per-vertex component count.
        dims: i32,
        /// Number of vertices declared on the input.
        vertex_count: i32,
        /// Per-vertex byte count expected for this attribute.
        row_bytes: usize,
        /// Bytes actually supplied.
        got: usize,
    },
    /// The index region length did not match the expected `index_count * bytes_per_index` (when
    /// the caller supplied an index count via the index byte buffer that did not divide cleanly).
    #[error("index region length {got} is not a multiple of {bytes_per_index}")]
    IndexBytesMisaligned {
        /// Bytes supplied.
        got: usize,
        /// Bytes per index for the chosen format.
        bytes_per_index: usize,
    },
    /// `vertex_count` was negative.
    #[error("vertex_count must be non-negative (got {0})")]
    NegativeVertexCount(i32),
}

/// Returns bytes per scalar for the given vertex attribute format.
pub fn vertex_format_size(format: VertexAttributeFormat) -> usize {
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

/// Returns bytes per index element for the given index buffer format.
pub fn index_bytes_per_element(format: IndexBufferFormat) -> usize {
    match format {
        IndexBufferFormat::UInt16 => 2,
        IndexBufferFormat::UInt32 => 4,
    }
}

/// Computes the per-vertex stride (bytes) from the attribute list.
pub fn compute_vertex_stride(attrs: &[VertexAttributeDescriptor]) -> usize {
    attrs
        .iter()
        .map(|a| vertex_format_size(a.format) * a.dimensions.max(0) as usize)
        .sum()
}

/// Encodes the per-vertex bytes interleaved across attributes, then appends the index buffer.
///
/// No bone weights, no bind poses, no blendshape data are written. Those regions can be added
/// later by an extended writer; the renderer's parser handles their absence as zero-length.
pub fn write_mesh_payload(input: &MeshLayoutInput<'_>) -> Result<MeshPayload, MeshLayoutError> {
    if input.vertex_count < 0 {
        return Err(MeshLayoutError::NegativeVertexCount(input.vertex_count));
    }
    if input.vertex_attributes.len() != input.sources.len() {
        return Err(MeshLayoutError::AttributeSourceMismatch {
            attrs: input.vertex_attributes.len(),
            srcs: input.sources.len(),
        });
    }

    let bytes_per_index = index_bytes_per_element(input.index_buffer_format);
    if !input.indices.is_empty() && !input.indices.len().is_multiple_of(bytes_per_index) {
        return Err(MeshLayoutError::IndexBytesMisaligned {
            got: input.indices.len(),
            bytes_per_index,
        });
    }

    let vertex_count = input.vertex_count as usize;
    let stride: usize = compute_vertex_stride(&input.vertex_attributes);
    let row_sizes: Vec<usize> = input
        .vertex_attributes
        .iter()
        .map(|a| vertex_format_size(a.format) * a.dimensions.max(0) as usize)
        .collect();

    for (i, (src, attr)) in input
        .sources
        .iter()
        .zip(input.vertex_attributes.iter())
        .enumerate()
    {
        let expected = row_sizes[i].saturating_mul(vertex_count);
        if src.bytes.len() != expected {
            return Err(MeshLayoutError::SourceBytesMismatch {
                index: i,
                attribute_type: attr.attribute,
                format: attr.format,
                dims: attr.dimensions,
                vertex_count: input.vertex_count,
                row_bytes: row_sizes[i],
                got: src.bytes.len(),
            });
        }
    }

    let vertex_region_bytes = stride
        .checked_mul(vertex_count)
        .ok_or(MeshLayoutError::NegativeVertexCount(input.vertex_count))?;
    // Match the renderer-side `compute_mesh_buffer_layout`: even when bones/blendshapes are
    // absent, the bone_counts region is **always** sized at `vertex_count` bytes, so the renderer
    // expects the host buffer to extend at least that far. We zero-fill it here.
    let bone_counts_region_bytes = vertex_count;
    let total = vertex_region_bytes
        .checked_add(input.indices.len())
        .and_then(|n| n.checked_add(bone_counts_region_bytes))
        .ok_or(MeshLayoutError::NegativeVertexCount(input.vertex_count))?;
    let mut bytes = vec![0u8; total];

    if vertex_count > 0 {
        for v in 0..vertex_count {
            let mut col_offset = 0usize;
            for (src, &row_size) in input.sources.iter().zip(row_sizes.iter()) {
                let dst_start = v * stride + col_offset;
                let src_start = v * row_size;
                bytes[dst_start..dst_start + row_size]
                    .copy_from_slice(&src.bytes[src_start..src_start + row_size]);
                col_offset += row_size;
            }
        }
    }

    if !input.indices.is_empty() {
        bytes[vertex_region_bytes..vertex_region_bytes + input.indices.len()]
            .copy_from_slice(input.indices);
    }

    Ok(MeshPayload {
        bytes,
        vertex_stride_bytes: stride,
        index_region_bytes: input.indices.len(),
    })
}

/// Convenience helper: float3 positions in tightly packed `[f32; 3]` per vertex.
pub fn position_float3_attr() -> VertexAttributeDescriptor {
    VertexAttributeDescriptor {
        attribute: crate::shared::VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }
}

/// Convenience helper: float3 normals in tightly packed `[f32; 3]` per vertex.
pub fn normal_float3_attr() -> VertexAttributeDescriptor {
    VertexAttributeDescriptor {
        attribute: crate::shared::VertexAttributeType::Normal,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::IndexBufferFormat;

    fn pos_bytes(positions: &[[f32; 3]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(positions.len() * 12);
        for p in positions {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    }

    fn nrm_bytes(normals: &[[f32; 3]]) -> Vec<u8> {
        pos_bytes(normals)
    }

    fn idx_bytes_u16(idx: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(idx.len() * 2);
        for i in idx {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out
    }

    #[test]
    fn computes_stride_correctly_for_pos_normal() {
        let attrs = vec![position_float3_attr(), normal_float3_attr()];
        assert_eq!(compute_vertex_stride(&attrs), 12 + 12);
    }

    #[test]
    fn writes_layout_for_two_vertices_pos_normal_uint16_indices() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let normals = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let indices = [0u16, 1u16, 0u16];

        let pos_bytes = pos_bytes(&positions);
        let nrm_bytes = nrm_bytes(&normals);
        let idx_bytes = idx_bytes_u16(&indices);

        let input = MeshLayoutInput {
            vertex_count: 2,
            vertex_attributes: vec![position_float3_attr(), normal_float3_attr()],
            sources: vec![
                InterleavedAttribute { bytes: &pos_bytes },
                InterleavedAttribute { bytes: &nrm_bytes },
            ],
            indices: &idx_bytes,
            index_buffer_format: IndexBufferFormat::UInt16,
        };
        let payload = write_mesh_payload(&input).expect("encode");

        // vertices (2 × 24) + indices (3 × 2) + bone_counts region (vertex_count = 2 zero bytes)
        assert_eq!(payload.bytes.len(), 2 * 24 + 6 + 2);
        assert_eq!(payload.vertex_stride_bytes, 24);
        assert_eq!(payload.index_region_bytes, 6);

        // Verify vertex 0 starts at offset 0 with position then normal.
        let v0_pos = &payload.bytes[0..12];
        assert_eq!(v0_pos, [0u8; 12]);
        let v0_nrm = &payload.bytes[12..24];
        let mut expected_nrm = Vec::new();
        for c in normals[0] {
            expected_nrm.extend_from_slice(&c.to_le_bytes());
        }
        assert_eq!(v0_nrm, &expected_nrm[..]);

        // Verify vertex 1 starts at offset 24 with the new position bytes.
        let mut expected_v1_pos = Vec::new();
        for c in positions[1] {
            expected_v1_pos.extend_from_slice(&c.to_le_bytes());
        }
        assert_eq!(&payload.bytes[24..36], &expected_v1_pos[..]);

        // Index region begins at offset 48.
        assert_eq!(&payload.bytes[48..54], &idx_bytes[..]);
        // bone_counts region (zero-filled) follows the index region.
        assert_eq!(&payload.bytes[54..56], &[0u8; 2]);
    }

    #[test]
    fn rejects_mismatched_source_byte_count() {
        let attrs = vec![position_float3_attr()];
        let bad_pos = vec![0u8; 5]; // 5 bytes, not 24 expected for 2 vertices × 12.
        let input = MeshLayoutInput {
            vertex_count: 2,
            vertex_attributes: attrs,
            sources: vec![InterleavedAttribute { bytes: &bad_pos }],
            indices: &[],
            index_buffer_format: IndexBufferFormat::UInt32,
        };
        let err = write_mesh_payload(&input).expect_err("should reject");
        match err {
            MeshLayoutError::SourceBytesMismatch {
                got, vertex_count, ..
            } => {
                assert_eq!(got, 5);
                assert_eq!(vertex_count, 2);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn round_trip_pos_normal_streams_via_renderer_layout_helpers() {
        // This mirrors the renderer-side `extract_float3_position_normal_as_vec4_streams` reader.
        // We assert the encoder's output reads back correctly when interpreted with the same
        // attribute descriptors and stride.
        let positions = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let normals = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

        let attrs = vec![position_float3_attr(), normal_float3_attr()];
        let pos_buf = pos_bytes(&positions);
        let nrm_buf = nrm_bytes(&normals);
        let input = MeshLayoutInput {
            vertex_count: positions.len() as i32,
            vertex_attributes: attrs.clone(),
            sources: vec![
                InterleavedAttribute { bytes: &pos_buf },
                InterleavedAttribute { bytes: &nrm_buf },
            ],
            indices: &[],
            index_buffer_format: IndexBufferFormat::UInt32,
        };
        let payload = write_mesh_payload(&input).expect("encode");
        let stride = payload.vertex_stride_bytes;
        assert_eq!(stride, 24);

        // Manually re-extract the dense `vec4<f32>` streams the renderer would build.
        let mut decoded_positions = Vec::with_capacity(positions.len());
        let mut decoded_normals = Vec::with_capacity(positions.len());
        for v in 0..positions.len() {
            let base = v * stride;
            let px = f32::from_le_bytes(payload.bytes[base..base + 4].try_into().unwrap());
            let py = f32::from_le_bytes(payload.bytes[base + 4..base + 8].try_into().unwrap());
            let pz = f32::from_le_bytes(payload.bytes[base + 8..base + 12].try_into().unwrap());
            let nx = f32::from_le_bytes(payload.bytes[base + 12..base + 16].try_into().unwrap());
            let ny = f32::from_le_bytes(payload.bytes[base + 16..base + 20].try_into().unwrap());
            let nz = f32::from_le_bytes(payload.bytes[base + 20..base + 24].try_into().unwrap());
            decoded_positions.push([px, py, pz]);
            decoded_normals.push([nx, ny, nz]);
        }
        assert_eq!(decoded_positions, positions);
        assert_eq!(decoded_normals, normals);
    }
}
