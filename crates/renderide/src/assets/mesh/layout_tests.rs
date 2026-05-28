//! Unit tests for [`super::layout`] (host mesh buffer layout and stream extraction).

mod blendshape;
mod buffer_layout;
mod skinning;
mod streams;

use super::layout::WIDE_UV_VERTEX_STRIDE_BYTES;

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_f32x2_stream(bytes: &[u8], vertex: usize) -> [f32; 2] {
    let offset = vertex * 8;
    bytemuck::pod_read_unaligned(&bytes[offset..offset + 8])
}

fn read_f32x4_stream(bytes: &[u8], vertex: usize) -> [f32; 4] {
    let offset = vertex * 16;
    bytemuck::pod_read_unaligned(&bytes[offset..offset + 16])
}

fn read_wide_uv_stream(bytes: &[u8], vertex: usize, channel: usize) -> [f32; 4] {
    let offset = vertex * WIDE_UV_VERTEX_STRIDE_BYTES + channel * 16;
    bytemuck::pod_read_unaligned(&bytes[offset..offset + 16])
}
