//! Dense vertex stream extraction for embedded material vertex buffers.

use crate::shared::{VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType};

use super::super::gpu_mesh::attribute_reader::AttributeReader;
use super::buffer_layout::vertex_format_size;

/// Host UV channels exposed through the mesh-forward vertex path.
pub const UV_VERTEX_ATTRIBUTE_TYPES: [VertexAttributeType; 8] = [
    VertexAttributeType::UV0,
    VertexAttributeType::UV1,
    VertexAttributeType::UV2,
    VertexAttributeType::UV3,
    VertexAttributeType::UV4,
    VertexAttributeType::UV5,
    VertexAttributeType::UV6,
    VertexAttributeType::UV7,
];

/// Bytes per vertex in the wide UV pack: eight `vec4<f32>` rows.
pub const WIDE_UV_VERTEX_STRIDE_BYTES: usize = UV_VERTEX_ATTRIBUTE_TYPES.len() * 16;

/// Attribute semantic used when expanding host vertex scalars into float streams.
#[derive(Clone, Copy)]
pub(in crate::assets::mesh) enum VertexDecodeKind {
    /// Data payloads that should preserve the host-authored scalar values.
    Raw,
    /// Object-space position data.
    Position,
    /// Direction vectors such as normals and tangents.
    Direction,
    /// Texture coordinate data.
    TexCoord,
    /// Vertex color channels.
    Color,
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
/// Position must be at least three-component numeric data. Normal is allowed to be absent or
/// unsupported; in that case a stable +Z normal is synthesized so UI meshes that do not upload
/// normals still satisfy the shared raster vertex layout.
pub fn extract_float3_position_normal_as_vec4_streams(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<(Vec<u8>, Vec<u8>)> {
    let position_reader = AttributeReader::from_attrs(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Position,
        VertexDecodeKind::Position,
        3,
    )?;
    let normal_reader = AttributeReader::from_attrs(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Normal,
        VertexDecodeKind::Direction,
        3,
    );

    let mut pos_out = vec![0u8; vertex_count * 16];
    let mut nrm_out = vec![0u8; vertex_count * 16];
    let one = 1.0f32.to_le_bytes();
    fill_normal_stream_with_forward_z(&mut nrm_out);

    for i in 0..vertex_count {
        let position = position_reader.read_vec3(i)?;
        let po = i * 16;
        write_f32s(&mut pos_out[po..po + 12], &position);
        pos_out[po + 12..po + 16].copy_from_slice(&one);

        if let Some(nr) = &normal_reader {
            let normal = nr.read_vec3(i)?;
            let no = i * 16;
            write_f32s(&mut nrm_out[no..no + 12], &normal);
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
/// When [`VertexAttributeType::UV0`] is missing or has fewer than two components, returns **zeros**
/// so a vertex buffer slot can always be bound.
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

/// Dense `vec2<f32>` vertex stream for an arbitrary two-component attribute.
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
    let Some(reader) = AttributeReader::from_attrs(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        target,
        VertexDecodeKind::TexCoord,
        2,
    ) else {
        return Some(out);
    };
    for i in 0..vertex_count {
        let uv = reader.read_vec2(i)?;
        let o = i * 8;
        write_f32s(&mut out[o..o + 8], &uv);
    }
    Some(out)
}

/// Dense wide UV stream for shaders that consume UV4-UV7 or 3D/4D UV channels.
///
/// Each vertex stores eight consecutive `vec4<f32>` rows (`UV0` through `UV7`). Missing UV
/// channels and missing z/w components are zero-filled.
pub fn wide_uv_stream_bytes(
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

    let mut out = vec![0u8; vertex_count.checked_mul(WIDE_UV_VERTEX_STRIDE_BYTES)?];
    let readers: [Option<AttributeReader<'_>>; 8] = UV_VERTEX_ATTRIBUTE_TYPES.map(|target| {
        AttributeReader::from_attrs(
            vertex_data,
            vertex_count,
            stride,
            attrs,
            target,
            VertexDecodeKind::TexCoord,
            2,
        )
    });

    for vertex in 0..vertex_count {
        for (channel, reader) in readers.iter().enumerate() {
            let Some(reader) = reader else {
                continue;
            };
            let uv = reader.read_vec4(vertex, [0.0; 4])?;
            let offset = vertex * WIDE_UV_VERTEX_STRIDE_BYTES + channel * 16;
            write_f32s(&mut out[offset..offset + 16], &uv);
        }
    }
    Some(out)
}

/// Dense `vec4<f32>` vertex stream for an arbitrary float attribute.
///
/// Missing or unsupported attributes return `default` per vertex.
fn vertex_float4_stream_bytes_with_kind(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    default: [f32; 4],
    kind: VertexDecodeKind,
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

    let Some(reader) =
        AttributeReader::from_attrs(vertex_data, vertex_count, stride, attrs, target, kind, 1)
    else {
        return Some(out);
    };
    for i in 0..vertex_count {
        let values = reader.read_vec4(i, default)?;
        let o = i * 16;
        write_f32s(&mut out[o..o + 16], &values);
    }

    Some(out)
}

/// Dense raw `vec4<f32>` payload stream for attributes used as shader data instead of geometry.
///
/// Missing or unsupported attributes return `default` per vertex.
pub fn raw_float4_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    default: [f32; 4],
) -> Option<Vec<u8>> {
    vertex_float4_stream_bytes_with_kind(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        target,
        default,
        VertexDecodeKind::Raw,
    )
}

/// Dense `vec4<f32>` vertex stream for an arbitrary float attribute.
///
/// Missing or unsupported attributes return `default` per vertex.
#[cfg(test)]
pub fn vertex_float4_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    target: VertexAttributeType,
    default: [f32; 4],
) -> Option<Vec<u8>> {
    vertex_float4_stream_bytes_with_kind(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        target,
        default,
        VertexDecodeKind::Position,
    )
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

    let Some(reader) = AttributeReader::from_attrs(
        vertex_data,
        vertex_count,
        stride,
        attrs,
        VertexAttributeType::Color,
        VertexDecodeKind::Color,
        1,
    ) else {
        return Some(out);
    };

    for i in 0..vertex_count {
        let rgba = reader.read_vec4(i, [1.0; 4])?;
        let o = i * 16;
        write_f32s(&mut out[o..o + 16], &rgba);
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

fn decode_vertex_scalar(
    vertex_data: &[u8],
    base: usize,
    component: usize,
    format: VertexAttributeFormat,
    kind: VertexDecodeKind,
) -> Option<f32> {
    let scalar_size = vertex_format_size(format) as usize;
    let offset = base.checked_add(component.checked_mul(scalar_size)?)?;
    match format {
        VertexAttributeFormat::Float32 => {
            let src = vertex_data.get(offset..offset + 4)?;
            Some(f32::from_le_bytes(src.try_into().ok()?))
        }
        VertexAttributeFormat::Half16 => {
            let src = vertex_data.get(offset..offset + 2)?;
            Some(f16_to_f32(u16::from_le_bytes(src.try_into().ok()?)))
        }
        VertexAttributeFormat::UNorm8 => {
            let value = f32::from(*vertex_data.get(offset)?) / 255.0;
            Some(apply_unsigned_normalized(value, kind))
        }
        VertexAttributeFormat::UNorm16 => {
            let src = vertex_data.get(offset..offset + 2)?;
            let value = f32::from(u16::from_le_bytes(src.try_into().ok()?)) / 65535.0;
            Some(apply_unsigned_normalized(value, kind))
        }
        VertexAttributeFormat::UInt8 => {
            let value = f32::from(*vertex_data.get(offset)?);
            Some(apply_unsigned_integer(value, 255.0, kind))
        }
        VertexAttributeFormat::UInt16 => {
            let src = vertex_data.get(offset..offset + 2)?;
            let value = f32::from(u16::from_le_bytes(src.try_into().ok()?));
            Some(apply_unsigned_integer(value, 65535.0, kind))
        }
        VertexAttributeFormat::UInt32 => {
            let src = vertex_data.get(offset..offset + 4)?;
            Some(u32::from_le_bytes(src.try_into().ok()?) as f32)
        }
        VertexAttributeFormat::SInt8 => {
            let value = i8::from_le_bytes([*vertex_data.get(offset)?]);
            Some(apply_signed_integer(f32::from(value), 127.0, kind))
        }
        VertexAttributeFormat::SInt16 => {
            let src = vertex_data.get(offset..offset + 2)?;
            let value = i16::from_le_bytes(src.try_into().ok()?);
            Some(apply_signed_integer(f32::from(value), 32767.0, kind))
        }
        VertexAttributeFormat::SInt32 => {
            let src = vertex_data.get(offset..offset + 4)?;
            let value = i32::from_le_bytes(src.try_into().ok()?);
            Some(apply_signed_integer(value as f32, i32::MAX as f32, kind))
        }
    }
}

/// Decodes a two-component vertex attribute into `f32` values.
pub(in crate::assets::mesh) fn decode_vertex_vec2(
    vertex_data: &[u8],
    base: usize,
    attr: VertexAttributeDescriptor,
    kind: VertexDecodeKind,
) -> Option<[f32; 2]> {
    if attr.dimensions < 2 {
        return None;
    }
    Some([
        decode_vertex_scalar(vertex_data, base, 0, attr.format, kind)?,
        decode_vertex_scalar(vertex_data, base, 1, attr.format, kind)?,
    ])
}

/// Decodes a three-component vertex attribute into `f32` values.
pub(in crate::assets::mesh) fn decode_vertex_vec3(
    vertex_data: &[u8],
    base: usize,
    attr: VertexAttributeDescriptor,
    kind: VertexDecodeKind,
) -> Option<[f32; 3]> {
    if attr.dimensions < 3 {
        return None;
    }
    Some([
        decode_vertex_scalar(vertex_data, base, 0, attr.format, kind)?,
        decode_vertex_scalar(vertex_data, base, 1, attr.format, kind)?,
        decode_vertex_scalar(vertex_data, base, 2, attr.format, kind)?,
    ])
}

/// Decodes up to four vertex attribute components into `f32` values.
pub(in crate::assets::mesh) fn decode_vertex_vec4(
    vertex_data: &[u8],
    base: usize,
    attr: VertexAttributeDescriptor,
    kind: VertexDecodeKind,
    default: [f32; 4],
) -> Option<[f32; 4]> {
    if attr.dimensions < 1 {
        return None;
    }
    let dims = attr.dimensions.clamp(1, 4) as usize;
    let mut out = default;
    for (component, value) in out.iter_mut().enumerate().take(dims) {
        *value = decode_vertex_scalar(vertex_data, base, component, attr.format, kind)?;
    }
    Some(out)
}

fn write_f32s<const N: usize>(dst: &mut [u8], values: &[f32; N]) {
    for (component, value) in values.iter().enumerate() {
        let offset = component * 4;
        dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn apply_unsigned_normalized(value: f32, kind: VertexDecodeKind) -> f32 {
    if matches!(kind, VertexDecodeKind::Direction) {
        value.mul_add(2.0, -1.0)
    } else {
        value
    }
}

fn apply_unsigned_integer(value: f32, max_value: f32, kind: VertexDecodeKind) -> f32 {
    match kind {
        VertexDecodeKind::Color => value / max_value,
        VertexDecodeKind::Direction => (value / max_value).mul_add(2.0, -1.0),
        VertexDecodeKind::Raw | VertexDecodeKind::Position | VertexDecodeKind::TexCoord => value,
    }
}

fn apply_signed_integer(value: f32, max_abs_value: f32, kind: VertexDecodeKind) -> f32 {
    if matches!(kind, VertexDecodeKind::Direction) {
        (value / max_abs_value).max(-1.0)
    } else {
        value
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let out = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mant = mantissa;
                let mut exp = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    exp -= 1;
                }
                mant &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(out)
}
