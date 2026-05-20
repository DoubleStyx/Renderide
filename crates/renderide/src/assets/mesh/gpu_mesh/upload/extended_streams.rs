//! Lazy extended vertex-stream upload helpers.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::shared::{
    IndexBufferFormat, SubmeshBufferDescriptor, VertexAttributeDescriptor, VertexAttributeType,
};

use super::super::super::layout::{
    WIDE_UV_VERTEX_STRIDE_BYTES, vertex_float2_stream_bytes, wide_uv_stream_bytes,
};
use super::super::tangent_generation::{
    TangentStreamSource, raw_tangent_payload_stream_bytes, tangent_stream_bytes,
};

/// Tangent plus UV1-UV3 optional vertex buffers from extended stream upload.
pub(in crate::assets::mesh::gpu_mesh) type ExtendedVertexStreams = (
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
    Option<Arc<wgpu::Buffer>>,
);

/// CPU-side mesh source required to build lazy tangent and UV1-UV3 vertex streams.
pub(in crate::assets::mesh::gpu_mesh) struct ExtendedVertexUploadSource<'a> {
    /// Interleaved vertex bytes from the host mesh payload.
    pub vertex_slice: &'a [u8],
    /// Index bytes from the host mesh payload.
    pub index_slice: &'a [u8],
    /// Number of vertices in `vertex_slice`.
    pub vertex_count: usize,
    /// Byte stride of one interleaved vertex.
    pub vertex_stride: usize,
    /// Host vertex attribute descriptors, in interleaved order.
    pub vertex_attributes: &'a [VertexAttributeDescriptor],
    /// Host index-buffer format.
    pub index_format: IndexBufferFormat,
    /// Host submesh descriptors.
    pub submeshes: &'a [SubmeshBufferDescriptor],
}

impl<'a> ExtendedVertexUploadSource<'a> {
    fn tangent_source(&self) -> TangentStreamSource<'a> {
        TangentStreamSource {
            vertex_data: self.vertex_slice,
            index_data: self.index_slice,
            vertex_count: self.vertex_count,
            stride: self.vertex_stride,
            attrs: self.vertex_attributes,
            index_format: self.index_format,
            submeshes: self.submeshes,
        }
    }
}

/// CPU-side mesh source required to build one lazy UV vertex stream.
#[derive(Copy, Clone)]
pub(in crate::assets::mesh::gpu_mesh) struct UvVertexUploadSource<'a> {
    /// Interleaved vertex bytes from the host mesh payload.
    pub vertex_slice: &'a [u8],
    /// Number of vertices in `vertex_slice`.
    pub vertex_count: usize,
    /// Byte stride of one interleaved vertex.
    pub vertex_stride: usize,
    /// Host vertex attribute descriptors, in interleaved order.
    pub vertex_attributes: &'a [VertexAttributeDescriptor],
    /// UV attribute to extract.
    pub target: VertexAttributeType,
    /// Debug label suffix for the generated GPU buffer.
    pub label: &'a str,
}

fn float4_default_stream_bytes(vertex_count: usize, default: [f32; 4]) -> Vec<u8> {
    let mut out = vec![0u8; vertex_count * 16];
    for chunk in out.chunks_exact_mut(16) {
        for (component, value) in default.iter().enumerate() {
            let o = component * 4;
            chunk[o..o + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn float2_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * 8]
}

fn wide_uv_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * WIDE_UV_VERTEX_STRIDE_BYTES]
}

#[inline]
pub(in crate::assets::mesh::gpu_mesh) fn vertex_stream_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
}

#[inline]
pub(in crate::assets::mesh::gpu_mesh) fn tangent_stream_usage() -> wgpu::BufferUsages {
    vertex_stream_usage() | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
}

fn create_vertex_stream_buffer(
    device: &wgpu::Device,
    asset_id: i32,
    label: &str,
    bytes: &[u8],
) -> Arc<wgpu::Buffer> {
    let buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} {label}_stream")),
            contents: bytes,
            usage: vertex_stream_usage(),
        }),
    );
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_vertex_stream");
    buffer
}

fn create_tangent_stream_buffer(
    device: &wgpu::Device,
    asset_id: i32,
    bytes: &[u8],
) -> Arc<wgpu::Buffer> {
    let buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} tangent_stream")),
            contents: bytes,
            usage: tangent_stream_usage(),
        }),
    );
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_tangent_stream");
    buffer
}

/// Uploads tangent and UV1-UV3 streams from host vertex data.
pub(in crate::assets::mesh::gpu_mesh) fn upload_extended_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
    generate_missing_tangents: bool,
) -> ExtendedVertexStreams {
    let vc_usize = source.vertex_count;
    if vc_usize == 0 {
        return (None, None, None, None);
    }

    let tangent_bytes = tangent_stream_bytes(source.tangent_source(), generate_missing_tangents)
        .unwrap_or_else(|| float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]));

    let make_uv = |target: VertexAttributeType, label: &str| {
        let bytes = vertex_float2_stream_bytes(
            source.vertex_slice,
            vc_usize,
            source.vertex_stride,
            source.vertex_attributes,
            target,
        )
        .unwrap_or_else(|| float2_zero_stream_bytes(vc_usize));
        create_vertex_stream_buffer(device, asset_id, label, &bytes)
    };

    (
        Some(create_tangent_stream_buffer(
            device,
            asset_id,
            &tangent_bytes,
        )),
        Some(make_uv(VertexAttributeType::UV1, "uv1")),
        Some(make_uv(VertexAttributeType::UV2, "uv2")),
        Some(make_uv(VertexAttributeType::UV3, "uv3")),
    )
}

/// Uploads a generated tangent stream from host vertex data.
pub(in crate::assets::mesh::gpu_mesh) fn upload_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
    generate_missing_tangents: bool,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let tangent_bytes = tangent_stream_bytes(source.tangent_source(), generate_missing_tangents)
        .unwrap_or_else(|| float4_default_stream_bytes(source.vertex_count, [1.0, 0.0, 0.0, 1.0]));
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

/// Uploads the raw tangent payload expected by raw-tangent shaders.
pub(in crate::assets::mesh::gpu_mesh) fn upload_raw_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: ExtendedVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let tangent_bytes = raw_tangent_payload_stream_bytes(source.tangent_source())
        .unwrap_or_else(|| float4_default_stream_bytes(source.vertex_count, [1.0; 4]));
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

/// Uploads a raw tangent payload stream filled with default values.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_raw_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0; 4]);
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

/// Uploads a tangent stream filled with default tangent values.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_tangent_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]);
    Some(create_tangent_stream_buffer(
        device,
        asset_id,
        &tangent_bytes,
    ))
}

/// Uploads one UV stream from host vertex data.
pub(in crate::assets::mesh::gpu_mesh) fn upload_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = vertex_float2_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
        source.target,
    )
    .unwrap_or_else(|| float2_zero_stream_bytes(source.vertex_count));
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        source.label,
        &uv_bytes,
    ))
}

/// Uploads the wide UV payload stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_wide_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = wide_uv_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    )
    .unwrap_or_else(|| wide_uv_zero_stream_bytes(source.vertex_count));
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        source.label,
        &uv_bytes,
    ))
}

/// Uploads a zero-filled wide UV stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_wide_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv_bytes = wide_uv_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device, asset_id, "wide_uv", &uv_bytes,
    ))
}

/// Uploads a zero-filled float2 UV stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
    label: &str,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv_bytes = float2_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device, asset_id, label, &uv_bytes,
    ))
}

/// Uploads default tangent and UV1-UV3 streams.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_extended_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> ExtendedVertexStreams {
    if vc_usize == 0 {
        return (None, None, None, None);
    }
    let tangent_bytes = float4_default_stream_bytes(vc_usize, [1.0, 0.0, 0.0, 1.0]);
    let uv_bytes = float2_zero_stream_bytes(vc_usize);
    (
        Some(create_tangent_stream_buffer(
            device,
            asset_id,
            &tangent_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv1", &uv_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv2", &uv_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv3", &uv_bytes,
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::{tangent_stream_usage, vertex_stream_usage};

    #[test]
    fn stream_usage_flags_match_binding_contracts() {
        assert!(vertex_stream_usage().contains(wgpu::BufferUsages::VERTEX));
        assert!(vertex_stream_usage().contains(wgpu::BufferUsages::COPY_DST));
        assert!(!vertex_stream_usage().contains(wgpu::BufferUsages::STORAGE));
        assert!(tangent_stream_usage().contains(wgpu::BufferUsages::STORAGE));
        assert!(tangent_stream_usage().contains(wgpu::BufferUsages::COPY_SRC));
    }
}
