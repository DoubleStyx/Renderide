//! CPU-side tangent stream extraction and MikkTSpace fallback generation.

use bevy_mikktspace::{Geometry, TangentSpace, generate_tangents};

use crate::shared::{
    IndexBufferFormat, SubmeshBufferDescriptor, SubmeshTopology, VertexAttributeDescriptor,
    VertexAttributeFormat, VertexAttributeType,
};

use super::layout::attribute_offset_and_size;

const DEFAULT_TANGENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const TANGENT_EPSILON_SQUARED: f32 = 1.0e-20;

/// Returns a dense `vec4<f32>` tangent stream, preferring host tangents and generating MikkTSpace
/// tangents when the host did not provide a usable tangent attribute.
pub(super) fn tangent_stream_bytes(
    vertex_data: &[u8],
    index_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    index_format: IndexBufferFormat,
    submeshes: &[SubmeshBufferDescriptor],
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }

    if let Some(host_tangents) = host_tangent_stream_bytes(vertex_data, vertex_count, stride, attrs)
    {
        return Some(host_tangents);
    }

    Some(
        generate_mikktspace_tangent_stream_bytes(
            vertex_data,
            index_data,
            vertex_count,
            stride,
            attrs,
            index_format,
            submeshes,
        )
        .unwrap_or_else(|| default_tangent_stream_bytes(vertex_count)),
    )
}

fn host_tangent_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    let attr = find_attribute(attrs, VertexAttributeType::Tangent)?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 3 {
        return None;
    }
    let (offset, size) = attribute_offset_and_size(attrs, VertexAttributeType::Tangent)?;
    let dimensions = attr.dimensions.clamp(3, 4) as usize;
    if size < dimensions * 4 {
        return None;
    }

    let mut out = default_tangent_stream_bytes(vertex_count);
    for vertex in 0..vertex_count {
        let base = vertex * stride + offset;
        if base + dimensions * 4 > vertex_data.len() {
            return None;
        }
        let dst = vertex * 16;
        for component in 0..dimensions {
            let src = base + component * 4;
            out[dst + component * 4..dst + component * 4 + 4]
                .copy_from_slice(&vertex_data[src..src + 4]);
        }
    }
    Some(out)
}

fn generate_mikktspace_tangent_stream_bytes(
    vertex_data: &[u8],
    index_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    index_format: IndexBufferFormat,
    submeshes: &[SubmeshBufferDescriptor],
) -> Option<Vec<u8>> {
    let positions = read_float3_vertex_stream(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Position,
    )?;
    let normals = read_float3_vertex_stream(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Normal,
    )?;
    let tex_coords = read_float2_vertex_stream(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::UV0,
    )?;
    let indices = decode_indices(index_data, index_format)?;
    let faces = collect_triangle_faces(&indices, vertex_count, submeshes)?;

    let mut geometry = MikkGeometry {
        positions,
        normals,
        tex_coords,
        faces,
        tangents: vec![DEFAULT_TANGENT; vertex_count],
    };
    if generate_tangents(&mut geometry).is_err() {
        return None;
    }
    Some(encode_tangents(&geometry.tangents))
}

fn find_attribute(
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<VertexAttributeDescriptor> {
    attrs
        .iter()
        .copied()
        .find(|attr| (attr.attribute as i16) == (target as i16))
}

fn read_float3_vertex_stream(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<Vec<[f32; 3]>> {
    let attr = find_attribute(attrs, target)?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 3 {
        return None;
    }
    let (offset, size) = attribute_offset_and_size(attrs, target)?;
    if size < 12 {
        return None;
    }

    let mut out = Vec::with_capacity(vertex_count);
    for vertex in 0..vertex_count {
        let base = vertex * stride + offset;
        if base + 12 > vertex_data.len() {
            return None;
        }
        out.push([
            read_f32(vertex_data, base)?,
            read_f32(vertex_data, base + 4)?,
            read_f32(vertex_data, base + 8)?,
        ]);
    }
    Some(out)
}

fn read_float2_vertex_stream(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
) -> Option<Vec<[f32; 2]>> {
    let attr = find_attribute(attrs, target)?;
    if attr.format != VertexAttributeFormat::Float32 || attr.dimensions < 2 {
        return None;
    }
    let (offset, size) = attribute_offset_and_size(attrs, target)?;
    if size < 8 {
        return None;
    }

    let mut out = Vec::with_capacity(vertex_count);
    for vertex in 0..vertex_count {
        let base = vertex * stride + offset;
        if base + 8 > vertex_data.len() {
            return None;
        }
        out.push([
            read_f32(vertex_data, base)?,
            read_f32(vertex_data, base + 4)?,
        ]);
    }
    Some(out)
}

fn read_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    let end = offset.checked_add(4)?;
    let raw: [u8; 4] = bytes.get(offset..end)?.try_into().ok()?;
    Some(f32::from_le_bytes(raw))
}

fn decode_indices(index_data: &[u8], index_format: IndexBufferFormat) -> Option<Vec<u32>> {
    match index_format {
        IndexBufferFormat::UInt16 => {
            if !index_data.len().is_multiple_of(2) {
                return None;
            }
            Some(
                index_data
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as u32)
                    .collect(),
            )
        }
        IndexBufferFormat::UInt32 => {
            if !index_data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                index_data
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            )
        }
    }
}

fn collect_triangle_faces(
    indices: &[u32],
    vertex_count: usize,
    submeshes: &[SubmeshBufferDescriptor],
) -> Option<Vec<[usize; 3]>> {
    let mut faces = Vec::new();
    for submesh in submeshes {
        if submesh.topology != SubmeshTopology::Triangles {
            continue;
        }
        let Ok(start) = usize::try_from(submesh.index_start) else {
            continue;
        };
        let Ok(count) = usize::try_from(submesh.index_count) else {
            continue;
        };
        let Some(end) = start.checked_add(count) else {
            continue;
        };
        let Some(submesh_indices) = indices.get(start..end) else {
            continue;
        };
        for triangle in submesh_indices.chunks_exact(3) {
            let face = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            if face.iter().any(|index| *index >= vertex_count) {
                continue;
            }
            if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                continue;
            }
            faces.push(face);
        }
    }
    (!faces.is_empty()).then_some(faces)
}

fn encode_tangents(tangents: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tangents.len() * 16);
    for tangent in tangents {
        for value in sanitize_tangent(*tangent) {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn default_tangent_stream_bytes(vertex_count: usize) -> Vec<u8> {
    encode_tangents(&vec![DEFAULT_TANGENT; vertex_count])
}

fn sanitize_tangent(tangent: [f32; 4]) -> [f32; 4] {
    if !tangent.iter().all(|component| component.is_finite()) {
        return DEFAULT_TANGENT;
    }
    let len_squared = tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2];
    if len_squared <= TANGENT_EPSILON_SQUARED {
        return DEFAULT_TANGENT;
    }
    let inv_len = len_squared.sqrt().recip();
    [
        tangent[0] * inv_len,
        tangent[1] * inv_len,
        tangent[2] * inv_len,
        if tangent[3] < 0.0 { -1.0 } else { 1.0 },
    ]
}

struct MikkGeometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tex_coords: Vec<[f32; 2]>,
    faces: Vec<[usize; 3]>,
    tangents: Vec<[f32; 4]>,
}

impl Geometry for MikkGeometry {
    fn num_faces(&self) -> usize {
        self.faces.len()
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.faces[face][vert]]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.faces[face][vert]]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.tex_coords[self.faces[face][vert]]
    }

    fn set_tangent(&mut self, tangent_space: Option<TangentSpace>, face: usize, vert: usize) {
        let Some(tangent_space) = tangent_space else {
            return;
        };
        let Some(face_indices) = self.faces.get(face) else {
            return;
        };
        let Some(vertex_index) = face_indices.get(vert).copied() else {
            return;
        };
        if let Some(slot) = self.tangents.get_mut(vertex_index) {
            *slot = sanitize_tangent(tangent_space.tangent_encoded());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attr(attribute: VertexAttributeType, dimensions: i32) -> VertexAttributeDescriptor {
        VertexAttributeDescriptor {
            attribute,
            format: VertexAttributeFormat::Float32,
            dimensions,
        }
    }

    fn triangle_submesh(index_count: i32) -> SubmeshBufferDescriptor {
        SubmeshBufferDescriptor {
            topology: SubmeshTopology::Triangles,
            index_start: 0,
            index_count,
            bounds: Default::default(),
        }
    }

    fn point_submesh(index_count: i32) -> SubmeshBufferDescriptor {
        SubmeshBufferDescriptor {
            topology: SubmeshTopology::Points,
            index_start: 0,
            index_count,
            bounds: Default::default(),
        }
    }

    fn push_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_vertex(bytes: &mut Vec<u8>, position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) {
        for value in position.into_iter().chain(normal).chain(uv) {
            push_f32(bytes, value);
        }
    }

    fn push_vertex_with_tangent(
        bytes: &mut Vec<u8>,
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
    ) {
        push_vertex(bytes, position, normal, uv);
        for value in tangent {
            push_f32(bytes, value);
        }
    }

    fn quad_vertices() -> Vec<u8> {
        let mut bytes = Vec::new();
        let normal = [0.0, 0.0, 1.0];
        push_vertex(&mut bytes, [-1.0, -1.0, 0.0], normal, [0.0, 0.0]);
        push_vertex(&mut bytes, [1.0, -1.0, 0.0], normal, [1.0, 0.0]);
        push_vertex(&mut bytes, [1.0, 1.0, 0.0], normal, [1.0, 1.0]);
        push_vertex(&mut bytes, [-1.0, 1.0, 0.0], normal, [0.0, 1.0]);
        bytes
    }

    fn quad_indices() -> Vec<u8> {
        [0u16, 1, 2, 0, 2, 3]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect()
    }

    fn read_tangent(bytes: &[u8], vertex: usize) -> [f32; 4] {
        let start = vertex * 16;
        [
            f32::from_le_bytes(bytes[start..start + 4].try_into().expect("x")),
            f32::from_le_bytes(bytes[start + 4..start + 8].try_into().expect("y")),
            f32::from_le_bytes(bytes[start + 8..start + 12].try_into().expect("z")),
            f32::from_le_bytes(bytes[start + 12..start + 16].try_into().expect("w")),
        ]
    }

    #[test]
    fn host_tangent_stream_is_preserved_when_valid() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
            attr(VertexAttributeType::Tangent, 4),
        ];
        let mut vertices = Vec::new();
        push_vertex_with_tangent(
            &mut vertices,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        );

        let tangents = tangent_stream_bytes(
            &vertices,
            &[],
            1,
            48,
            &attrs,
            IndexBufferFormat::UInt16,
            &[],
        )
        .expect("tangent stream");

        assert_eq!(read_tangent(&tangents, 0), [0.0, 1.0, 0.0, -1.0]);
    }

    #[test]
    fn missing_tangents_are_generated_for_indexed_textured_triangle_mesh() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
        ];
        let tangents = tangent_stream_bytes(
            &quad_vertices(),
            &quad_indices(),
            4,
            32,
            &attrs,
            IndexBufferFormat::UInt16,
            &[triangle_submesh(6)],
        )
        .expect("tangent stream");

        for vertex in 0..4 {
            assert_eq!(read_tangent(&tangents, vertex), [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn missing_uvs_fall_back_to_stable_default_tangents() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
        ];
        let tangents = tangent_stream_bytes(
            &quad_vertices(),
            &quad_indices(),
            4,
            32,
            &attrs,
            IndexBufferFormat::UInt16,
            &[triangle_submesh(6)],
        )
        .expect("tangent stream");

        for vertex in 0..4 {
            assert_eq!(read_tangent(&tangents, vertex), DEFAULT_TANGENT);
        }
    }

    #[test]
    fn point_submeshes_fall_back_to_stable_default_tangents() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
        ];
        let tangents = tangent_stream_bytes(
            &quad_vertices(),
            &quad_indices(),
            4,
            32,
            &attrs,
            IndexBufferFormat::UInt16,
            &[point_submesh(6)],
        )
        .expect("tangent stream");

        for vertex in 0..4 {
            assert_eq!(read_tangent(&tangents, vertex), DEFAULT_TANGENT);
        }
    }
}
