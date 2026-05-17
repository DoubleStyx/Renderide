//! CPU-side tangent stream extraction and MikkTSpace fallback generation.

use bevy_mikktspace::{Geometry, TangentSpace, generate_tangents};
use glam::{Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::shared::{
    IndexBufferFormat, SubmeshBufferDescriptor, SubmeshTopology, VertexAttributeDescriptor,
    VertexAttributeType,
};

use super::super::layout::VertexDecodeKind;
use super::super::layout::raw_float4_stream_bytes;
use super::attribute_reader::AttributeReader;

const _: () = assert!(
    cfg!(target_endian = "little"),
    "renderide assumes a little-endian target for vertex stream decode",
);

const DEFAULT_TANGENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const DEFAULT_TANGENT_VEC: Vec4 = Vec4::new(1.0, 0.0, 0.0, 1.0);
const DEFAULT_RAW_TANGENT_PAYLOAD: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const TANGENT_EPSILON_SQUARED: f32 = 1.0e-20;
/// Vertex count above which vertex-stream extraction and tangent encoding fan out across rayon.
///
/// Production meshes cluster around 1k-8k vertices, so a threshold of 2048 lets medium avatar and
/// prop meshes use the worker pool while tiny meshes stay serial.
const VERTEX_STREAM_PARALLEL_MIN: usize = 1_024;

/// CPU-side mesh source used to extract or generate tangent streams.
#[derive(Copy, Clone)]
pub(super) struct TangentStreamSource<'a> {
    /// Interleaved vertex bytes from the host mesh payload.
    pub vertex_data: &'a [u8],
    /// Index bytes from the host mesh payload.
    pub index_data: &'a [u8],
    /// Number of vertices in `vertex_data`.
    pub vertex_count: usize,
    /// Byte stride of one interleaved vertex.
    pub stride: usize,
    /// Host vertex attribute descriptors, in interleaved order.
    pub attrs: &'a [VertexAttributeDescriptor],
    /// Host index-buffer format.
    pub index_format: IndexBufferFormat,
    /// Host submesh descriptors.
    pub submeshes: &'a [SubmeshBufferDescriptor],
}

/// Returns a dense `vec4<f32>` tangent stream, preferring host tangents and generating MikkTSpace
/// tangents when the host did not provide a usable tangent attribute.
pub(super) fn tangent_stream_bytes(
    source: TangentStreamSource<'_>,
    generate_missing: bool,
) -> Option<Vec<u8>> {
    let TangentStreamSource {
        vertex_data,
        index_data,
        vertex_count,
        stride,
        attrs,
        index_format,
        submeshes,
    } = source;
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

    if !generate_missing {
        return Some(
            normal_based_tangent_stream_bytes(vertex_data, vertex_count, stride, attrs)
                .unwrap_or_else(|| default_tangent_stream_bytes(vertex_count)),
        );
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
        .or_else(|| normal_based_tangent_stream_bytes(vertex_data, vertex_count, stride, attrs))
        .unwrap_or_else(|| default_tangent_stream_bytes(vertex_count)),
    )
}

/// Returns a dense `vec4<f32>` tangent payload stream without geometric sanitization.
pub(super) fn raw_tangent_payload_stream_bytes(source: TangentStreamSource<'_>) -> Option<Vec<u8>> {
    raw_float4_stream_bytes(
        source.vertex_data,
        source.vertex_count,
        source.stride,
        source.attrs,
        VertexAttributeType::Tangent,
        DEFAULT_RAW_TANGENT_PAYLOAD,
    )
}

fn host_tangent_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    let reader = AttributeReader::from_attrs(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Tangent,
        VertexDecodeKind::Direction,
        3,
    )?;

    let mut out = default_tangent_stream_bytes(vertex_count);
    let copy_one = |dst: &mut [u8], vertex: usize| {
        if let Some(tangent) = reader.read_vec4(vertex, DEFAULT_TANGENT) {
            let sanitized = sanitize_tangent(Vec4::from_array(tangent));
            dst.copy_from_slice(bytemuck::bytes_of(&sanitized));
        }
    };
    if vertex_count >= VERTEX_STREAM_PARALLEL_MIN {
        out.par_chunks_exact_mut(16)
            .enumerate()
            .for_each(|(vertex, slot)| copy_one(slot, vertex));
    } else {
        for (vertex, slot) in out.chunks_exact_mut(16).enumerate() {
            copy_one(slot, vertex);
        }
    }
    Some(out)
}

fn normal_based_tangent_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    let normals = read_vertex_stream3(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Normal,
        VertexDecodeKind::Direction,
    )?;
    let tangents: Vec<Vec4> = normals
        .iter()
        .map(|normal| tangent_from_normal(*normal))
        .collect();
    Some(encode_tangents(&tangents))
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
    let positions = read_vertex_stream3(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Position,
        VertexDecodeKind::Position,
    )?;
    let normals = read_vertex_stream3(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Normal,
        VertexDecodeKind::Direction,
    )?;
    let tex_coords = read_vertex_stream2(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::UV0,
        VertexDecodeKind::TexCoord,
    )?;
    let indices = decode_indices(index_data, index_format)?;
    let faces = collect_triangle_faces(&indices, vertex_count, submeshes)?;

    let mut geometry = MikkGeometry {
        positions,
        normals,
        tex_coords,
        faces,
        tangents: vec![DEFAULT_TANGENT_VEC; vertex_count],
    };
    if generate_tangents(&mut geometry).is_err() {
        return None;
    }
    Some(encode_tangents(&geometry.tangents))
}

fn read_vertex_stream3(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    kind: VertexDecodeKind,
) -> Option<Vec<Vec3>> {
    let reader =
        AttributeReader::from_attrs(vertex_data, vertex_count, stride, attrs, target, kind, 3)?;
    let read_one = |vertex: usize| -> Vec3 {
        reader
            .read_vec3(vertex)
            .map(Vec3::from_array)
            .unwrap_or(Vec3::ZERO)
    };
    let out: Vec<Vec3> = if vertex_count >= VERTEX_STREAM_PARALLEL_MIN {
        (0..vertex_count).into_par_iter().map(read_one).collect()
    } else {
        (0..vertex_count).map(read_one).collect()
    };
    Some(out)
}

fn read_vertex_stream2(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    kind: VertexDecodeKind,
) -> Option<Vec<Vec2>> {
    let reader =
        AttributeReader::from_attrs(vertex_data, vertex_count, stride, attrs, target, kind, 2)?;
    let read_one = |vertex: usize| -> Vec2 {
        reader
            .read_vec2(vertex)
            .map(Vec2::from_array)
            .unwrap_or(Vec2::ZERO)
    };
    let out: Vec<Vec2> = if vertex_count >= VERTEX_STREAM_PARALLEL_MIN {
        (0..vertex_count).into_par_iter().map(read_one).collect()
    } else {
        (0..vertex_count).map(read_one).collect()
    };
    Some(out)
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

fn encode_tangents(tangents: &[Vec4]) -> Vec<u8> {
    let mut out = vec![0u8; tangents.len() * 16];
    let write_one = |slot: &mut [u8], tangent: &Vec4| {
        let sanitized = sanitize_tangent(*tangent);
        slot.copy_from_slice(bytemuck::bytes_of(&sanitized));
    };
    if tangents.len() >= VERTEX_STREAM_PARALLEL_MIN {
        out.par_chunks_exact_mut(16)
            .zip(tangents.par_iter())
            .for_each(|(slot, tangent)| write_one(slot, tangent));
    } else {
        for (slot, tangent) in out.chunks_exact_mut(16).zip(tangents.iter()) {
            write_one(slot, tangent);
        }
    }
    out
}

fn default_tangent_stream_bytes(vertex_count: usize) -> Vec<u8> {
    encode_tangents(&vec![DEFAULT_TANGENT_VEC; vertex_count])
}

fn sanitize_tangent(tangent: Vec4) -> Vec4 {
    if !tangent.is_finite() {
        return DEFAULT_TANGENT_VEC;
    }
    let xyz = tangent.truncate();
    let len_squared = xyz.length_squared();
    if len_squared <= TANGENT_EPSILON_SQUARED {
        return DEFAULT_TANGENT_VEC;
    }
    let unit = xyz * len_squared.sqrt().recip();
    Vec4::new(
        unit.x,
        unit.y,
        unit.z,
        if tangent[3] < 0.0 { -1.0 } else { 1.0 },
    )
}

fn tangent_from_normal(normal: Vec3) -> Vec4 {
    let Some(n) = normalize3(normal) else {
        return DEFAULT_TANGENT_VEC;
    };
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let Some(t) = normalize3(tangent) else {
        return DEFAULT_TANGENT_VEC;
    };
    t.extend(1.0)
}

fn normalize3(v: Vec3) -> Option<Vec3> {
    if !v.is_finite() {
        return None;
    }
    let len_squared = v.length_squared();
    if len_squared <= TANGENT_EPSILON_SQUARED {
        return None;
    }
    Some(v * len_squared.sqrt().recip())
}

struct MikkGeometry {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    tex_coords: Vec<Vec2>,
    faces: Vec<[usize; 3]>,
    tangents: Vec<Vec4>,
}

impl Geometry for MikkGeometry {
    fn num_faces(&self) -> usize {
        self.faces.len()
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.faces[face][vert]].to_array()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.faces[face][vert]].to_array()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.tex_coords[self.faces[face][vert]].to_array()
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
            *slot = sanitize_tangent(Vec4::from_array(tangent_space.tangent_encoded()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::VertexAttributeFormat;

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

    fn quad_vertices_with_y_tangent_uvs() -> Vec<u8> {
        let mut bytes = Vec::new();
        let normal = [0.0, 0.0, 1.0];
        push_vertex(&mut bytes, [-1.0, -1.0, 0.0], normal, [0.0, 0.0]);
        push_vertex(&mut bytes, [1.0, -1.0, 0.0], normal, [0.0, 1.0]);
        push_vertex(&mut bytes, [1.0, 1.0, 0.0], normal, [1.0, 1.0]);
        push_vertex(&mut bytes, [-1.0, 1.0, 0.0], normal, [1.0, 0.0]);
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

    fn tangent_source<'a>(
        vertex_data: &'a [u8],
        index_data: &'a [u8],
        vertex_count: usize,
        stride: usize,
        attrs: &'a [VertexAttributeDescriptor],
        submeshes: &'a [SubmeshBufferDescriptor],
    ) -> TangentStreamSource<'a> {
        TangentStreamSource {
            vertex_data,
            index_data,
            vertex_count,
            stride,
            attrs,
            index_format: IndexBufferFormat::UInt16,
            submeshes,
        }
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

        let tangents =
            tangent_stream_bytes(tangent_source(&vertices, &[], 1, 48, &attrs, &[]), true)
                .expect("tangent stream");

        assert_eq!(read_tangent(&tangents, 0), [0.0, 1.0, 0.0, -1.0]);
    }

    #[test]
    fn raw_tangent_payload_stream_preserves_color_data() {
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
            [0.0, 0.0, 0.0, 0.75],
        );

        let geometric =
            tangent_stream_bytes(tangent_source(&vertices, &[], 1, 48, &attrs, &[]), false)
                .expect("geometric tangent stream");
        let raw =
            raw_tangent_payload_stream_bytes(tangent_source(&vertices, &[], 1, 48, &attrs, &[]))
                .expect("raw tangent stream");

        assert_eq!(read_tangent(&geometric, 0), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(read_tangent(&raw, 0), [0.0, 0.0, 0.0, 0.75]);
    }

    #[test]
    fn missing_tangents_are_generated_for_indexed_textured_triangle_mesh() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
        ];
        let vertices = quad_vertices();
        let indices = quad_indices();
        let submeshes = [triangle_submesh(6)];
        let tangents = tangent_stream_bytes(
            tangent_source(&vertices, &indices, 4, 32, &attrs, &submeshes),
            true,
        )
        .expect("tangent stream");

        for vertex in 0..4 {
            assert_eq!(read_tangent(&tangents, vertex), [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn missing_tangent_generation_flag_controls_fallback() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
        ];
        let vertices = quad_vertices_with_y_tangent_uvs();
        let indices = quad_indices();
        let submeshes = [triangle_submesh(6)];
        let source = TangentStreamSource {
            vertex_data: &vertices,
            index_data: &indices,
            vertex_count: 4,
            stride: 32,
            attrs: &attrs,
            index_format: IndexBufferFormat::UInt16,
            submeshes: &submeshes,
        };

        let generated = tangent_stream_bytes(source, true).expect("generated tangent stream");
        let defaulted = tangent_stream_bytes(source, false).expect("default tangent stream");

        assert_ne!(read_tangent(&generated, 0), DEFAULT_TANGENT);
        assert_eq!(read_tangent(&defaulted, 0), DEFAULT_TANGENT);
    }

    #[test]
    fn missing_uvs_fall_back_to_stable_default_tangents() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
        ];
        let vertices = quad_vertices();
        let indices = quad_indices();
        let submeshes = [triangle_submesh(6)];
        let tangents = tangent_stream_bytes(
            tangent_source(&vertices, &indices, 4, 32, &attrs, &submeshes),
            true,
        )
        .expect("tangent stream");

        for vertex in 0..4 {
            assert_eq!(read_tangent(&tangents, vertex), DEFAULT_TANGENT);
        }
    }

    #[test]
    fn host_tangent_stream_parallel_path_matches_serial() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
            attr(VertexAttributeType::Tangent, 4),
        ];
        let stride = 48usize;
        let vertex_count = VERTEX_STREAM_PARALLEL_MIN + 17;
        let mut vertices = Vec::with_capacity(stride * vertex_count);
        for v in 0..vertex_count {
            push_vertex_with_tangent(
                &mut vertices,
                [v as f32, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [v as f32 * 0.1, 0.0],
                [
                    1.0,
                    (v % 5) as f32 * 0.2,
                    0.0,
                    if v % 2 == 0 { 1.0 } else { -1.0 },
                ],
            );
        }
        let parallel_out = tangent_stream_bytes(
            tangent_source(&vertices, &[], vertex_count, stride, &attrs, &[]),
            true,
        )
        .expect("tangent stream");
        let mut serial_out = vec![0u8; vertex_count * 16];
        let tangent_offset = 12 + 12 + 8;
        for v in 0..vertex_count {
            let base = v * stride + tangent_offset;
            let tangent = bytemuck::pod_read_unaligned::<[f32; 4]>(&vertices[base..base + 16]);
            let tangent = sanitize_tangent(Vec4::from_array(tangent));
            serial_out[v * 16..v * 16 + 16].copy_from_slice(bytemuck::bytes_of(&tangent));
        }
        assert_eq!(parallel_out, serial_out);
    }

    #[test]
    fn point_submeshes_fall_back_to_stable_default_tangents() {
        let attrs = [
            attr(VertexAttributeType::Position, 3),
            attr(VertexAttributeType::Normal, 3),
            attr(VertexAttributeType::UV0, 2),
        ];
        let vertices = quad_vertices();
        let indices = quad_indices();
        let submeshes = [point_submesh(6)];
        let tangents = tangent_stream_bytes(
            tangent_source(&vertices, &indices, 4, 32, &attrs, &submeshes),
            true,
        )
        .expect("tangent stream");

        for vertex in 0..4 {
            assert_eq!(read_tangent(&tangents, vertex), DEFAULT_TANGENT);
        }
    }
}
