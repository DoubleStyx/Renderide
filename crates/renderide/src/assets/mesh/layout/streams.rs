//! Dense vertex stream extraction for embedded material vertex buffers.

use crate::shared::{VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType};

use super::buffer_layout::vertex_format_size;

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
/// When [`VertexAttributeType::UV0`] is missing or not `float32`x2, returns **zeros** so a vertex buffer
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
#[cfg(test)]
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
        let Some(rgba) = decode_vertex_color(vertex_data, base, *color_attr) else {
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
    attr: VertexAttributeDescriptor,
) -> Option<[f32; 4]> {
    let dims = attr.dimensions.clamp(1, 4) as usize;
    let mut rgba = [1.0f32; 4];
    match attr.format {
        VertexAttributeFormat::UNorm8 | VertexAttributeFormat::UInt8 => {
            let end = base.checked_add(dims)?;
            let src = vertex_data.get(base..end)?;
            for (i, byte) in src.iter().take(dims).enumerate() {
                rgba[i] = f32::from(*byte) / 255.0;
            }
        }
        VertexAttributeFormat::UNorm16 | VertexAttributeFormat::UInt16 => {
            let end = base.checked_add(dims.checked_mul(2)?)?;
            let src = vertex_data.get(base..end)?;
            for (i, chunk) in src.chunks(2).take(dims).enumerate() {
                rgba[i] = f32::from(u16::from_le_bytes(chunk.try_into().ok()?)) / 65535.0;
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
