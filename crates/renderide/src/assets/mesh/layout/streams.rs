//! Dense vertex stream extraction for embedded material vertex buffers.

use rayon::prelude::*;

use crate::shared::{VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType};

use super::super::gpu_mesh::attribute_reader::AttributeReader;
use super::buffer_layout::vertex_format_size;

/// Vertices assigned to one stream expansion worker chunk.
const VERTEX_STREAM_PARALLEL_CHUNK_VERTICES: usize = 256;
/// Vertex count above which stream expansion fans out across Rayon workers.
const VERTEX_STREAM_PARALLEL_MIN: usize = VERTEX_STREAM_PARALLEL_CHUNK_VERTICES * 2;

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

/// UV channels stored in one wide UV page.
pub const WIDE_UV_PAGE_CHANNEL_COUNT: usize = 4;
/// Bytes per vertex in one wide UV page: four `vec4<f32>` rows.
pub const WIDE_UV_PAGE_VERTEX_STRIDE_BYTES: usize = WIDE_UV_PAGE_CHANNEL_COUNT * 16;
/// Bytes per vertex in the low wide UV page containing UV0 through UV3.
pub const WIDE_UV_LOW_VERTEX_STRIDE_BYTES: usize = WIDE_UV_PAGE_VERTEX_STRIDE_BYTES;
/// Bytes per vertex in the high wide UV page containing UV4 through UV7.
pub const WIDE_UV_HIGH_VERTEX_STRIDE_BYTES: usize = WIDE_UV_PAGE_VERTEX_STRIDE_BYTES;

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
    if normal_reader.is_none() {
        fill_normal_stream_with_forward_z(&mut nrm_out);
    }
    let normal_reader = normal_reader.as_ref();

    if should_parallelize_vertex_stream(vertex_count) {
        pos_out
            .par_chunks_exact_mut(16)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .zip(
                nrm_out
                    .par_chunks_exact_mut(16)
                    .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES),
            )
            .enumerate()
            .try_for_each(|(i, (pos_slot, nrm_slot))| {
                write_position_normal_vertex(
                    i,
                    &position_reader,
                    normal_reader,
                    pos_slot,
                    nrm_slot,
                    one,
                )
            })?;
    } else {
        for (i, (pos_slot, nrm_slot)) in pos_out
            .chunks_exact_mut(16)
            .zip(nrm_out.chunks_exact_mut(16))
            .enumerate()
        {
            write_position_normal_vertex(
                i,
                &position_reader,
                normal_reader,
                pos_slot,
                nrm_slot,
                one,
            )?;
        }
    }
    Some((pos_out, nrm_out))
}

fn should_parallelize_vertex_stream(vertex_count: usize) -> bool {
    vertex_count >= VERTEX_STREAM_PARALLEL_MIN
}

fn write_position_normal_vertex(
    vertex: usize,
    position_reader: &AttributeReader<'_>,
    normal_reader: Option<&AttributeReader<'_>>,
    pos_slot: &mut [u8],
    nrm_slot: &mut [u8],
    one: [u8; 4],
) -> Option<()> {
    let position = position_reader.read_vec3(vertex)?;
    write_f32s(&mut pos_slot[..12], &position);
    pos_slot[12..16].copy_from_slice(&one);

    if let Some(reader) = normal_reader {
        let normal = reader.read_vec3(vertex)?;
        write_f32s(&mut nrm_slot[..12], &normal);
    }

    Some(())
}

fn fill_normal_stream_with_forward_z(out: &mut [u8]) {
    let zero = 0.0f32.to_le_bytes();
    let one = 1.0f32.to_le_bytes();
    let write_chunk = |chunk: &mut [u8]| {
        chunk[0..4].copy_from_slice(&zero);
        chunk[4..8].copy_from_slice(&zero);
        chunk[8..12].copy_from_slice(&one);
        chunk[12..16].copy_from_slice(&zero);
    };
    if should_parallelize_vertex_stream(out.len() / 16) {
        out.par_chunks_exact_mut(16)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .for_each(write_chunk);
    } else {
        out.chunks_exact_mut(16).for_each(write_chunk);
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
    if should_parallelize_vertex_stream(vertex_count) {
        out.par_chunks_exact_mut(8)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .enumerate()
            .try_for_each(|(i, slot)| write_vertex_float2(&reader, i, slot))?;
    } else {
        for (i, slot) in out.chunks_exact_mut(8).enumerate() {
            write_vertex_float2(&reader, i, slot)?;
        }
    }
    Some(out)
}

fn write_vertex_float2(reader: &AttributeReader<'_>, vertex: usize, slot: &mut [u8]) -> Option<()> {
    let uv = reader.read_vec2(vertex)?;
    write_f32s(slot, &uv);
    Some(())
}

/// Dense low wide UV stream for shaders that consume 3D/4D UV0-UV3 channels.
///
/// Each vertex stores four consecutive `vec4<f32>` rows (`UV0` through `UV3`). Missing UV
/// channels and missing z/w components are zero-filled.
pub fn wide_low_uv_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    wide_uv_page_stream_bytes(vertex_data, vertex_count, stride, attrs, 0)
}

/// Dense high wide UV stream for shaders that consume UV4-UV7 channels.
///
/// Each vertex stores four consecutive `vec4<f32>` rows (`UV4` through `UV7`). Missing UV
/// channels and missing z/w components are zero-filled.
pub fn wide_high_uv_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
) -> Option<Vec<u8>> {
    wide_uv_page_stream_bytes(vertex_data, vertex_count, stride, attrs, 4)
}

fn wide_uv_page_stream_bytes(
    vertex_data: &[u8],
    vertex_count: usize,
    stride: usize,
    attrs: &[VertexAttributeDescriptor],
    start_channel: usize,
) -> Option<Vec<u8>> {
    if vertex_count == 0 || stride == 0 {
        return None;
    }
    let need = vertex_count.checked_mul(stride)?;
    if vertex_data.len() < need {
        return None;
    }

    let end_channel = start_channel.checked_add(WIDE_UV_PAGE_CHANNEL_COUNT)?;
    let page_channels = UV_VERTEX_ATTRIBUTE_TYPES.get(start_channel..end_channel)?;
    let mut out = vec![0u8; vertex_count.checked_mul(WIDE_UV_PAGE_VERTEX_STRIDE_BYTES)?];
    let readers: [Option<AttributeReader<'_>>; WIDE_UV_PAGE_CHANNEL_COUNT] =
        core::array::from_fn(|channel| {
            AttributeReader::from_attrs(
                vertex_data,
                vertex_count,
                stride,
                attrs,
                page_channels[channel],
                VertexDecodeKind::TexCoord,
                2,
            )
        });

    if should_parallelize_vertex_stream(vertex_count) {
        out.par_chunks_exact_mut(WIDE_UV_PAGE_VERTEX_STRIDE_BYTES)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .enumerate()
            .try_for_each(|(vertex, slot)| write_wide_uv_page_vertex(&readers, vertex, slot))?;
    } else {
        for (vertex, slot) in out
            .chunks_exact_mut(WIDE_UV_PAGE_VERTEX_STRIDE_BYTES)
            .enumerate()
        {
            write_wide_uv_page_vertex(&readers, vertex, slot)?;
        }
    }
    Some(out)
}

fn write_wide_uv_page_vertex(
    readers: &[Option<AttributeReader<'_>>; WIDE_UV_PAGE_CHANNEL_COUNT],
    vertex: usize,
    slot: &mut [u8],
) -> Option<()> {
    for (page_channel, reader) in readers.iter().enumerate() {
        let Some(reader) = reader else {
            continue;
        };
        let uv = reader.read_vec4(vertex, [0.0; 4])?;
        let offset = page_channel * 16;
        write_f32s(&mut slot[offset..offset + 16], &uv);
    }
    Some(())
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
    fill_float4_stream_with_default(&mut out, default);

    let Some(reader) =
        AttributeReader::from_attrs(vertex_data, vertex_count, stride, attrs, target, kind, 1)
    else {
        return Some(out);
    };
    if should_parallelize_vertex_stream(vertex_count) {
        out.par_chunks_exact_mut(16)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .enumerate()
            .try_for_each(|(i, slot)| write_vertex_float4(&reader, i, slot, default))?;
    } else {
        for (i, slot) in out.chunks_exact_mut(16).enumerate() {
            write_vertex_float4(&reader, i, slot, default)?;
        }
    }

    Some(out)
}

fn write_vertex_float4(
    reader: &AttributeReader<'_>,
    vertex: usize,
    slot: &mut [u8],
    default: [f32; 4],
) -> Option<()> {
    let values = reader.read_vec4(vertex, default)?;
    write_f32s(slot, &values);
    Some(())
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

    if should_parallelize_vertex_stream(vertex_count) {
        out.par_chunks_exact_mut(16)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .enumerate()
            .try_for_each(|(i, slot)| write_vertex_float4(&reader, i, slot, [1.0; 4]))?;
    } else {
        for (i, slot) in out.chunks_exact_mut(16).enumerate() {
            write_vertex_float4(&reader, i, slot, [1.0; 4])?;
        }
    }

    Some(out)
}

fn fill_float4_stream_with_default(out: &mut [u8], default: [f32; 4]) {
    let default = default.map(|value| value.to_le_bytes());
    let write_chunk = |chunk: &mut [u8]| {
        for (component, value) in default.iter().enumerate() {
            let offset = component * 4;
            chunk[offset..offset + 4].copy_from_slice(value);
        }
    };
    if should_parallelize_vertex_stream(out.len() / 16) {
        out.par_chunks_exact_mut(16)
            .with_min_len(VERTEX_STREAM_PARALLEL_CHUNK_VERTICES)
            .for_each(write_chunk);
    } else {
        out.chunks_exact_mut(16).for_each(write_chunk);
    }
}

fn fill_color_stream_with_white(out: &mut [u8]) {
    fill_float4_stream_with_default(out, [1.0; 4]);
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
