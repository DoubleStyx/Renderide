//! Lazy extended vertex-stream upload helpers.

use std::sync::Arc;

use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::cpu_parallelism::{
    admit_mesh_stream_jobs, current_reference_worker_count, record_parallel_admission,
};
use crate::shared::{
    IndexBufferFormat, SubmeshBufferDescriptor, VertexAttributeDescriptor, VertexAttributeType,
};

use super::super::super::layout::{
    WIDE_UV_HIGH_VERTEX_STRIDE_BYTES, WIDE_UV_LOW_VERTEX_STRIDE_BYTES, color_float4_stream_bytes,
    extract_float3_position_normal_as_vec4_streams, uv0_float2_stream_bytes,
    vertex_float2_stream_bytes, wide_high_uv_stream_bytes, wide_low_uv_stream_bytes,
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

fn wide_low_uv_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * WIDE_UV_LOW_VERTEX_STRIDE_BYTES]
}

fn wide_high_uv_zero_stream_bytes(vertex_count: usize) -> Vec<u8> {
    vec![0u8; vertex_count * WIDE_UV_HIGH_VERTEX_STRIDE_BYTES]
}

#[derive(Clone, Copy, Debug)]
enum ExtendedStreamJob {
    Tangent,
    Uv1,
    Uv2,
    Uv3,
}

enum ExtendedStreamJobResult {
    Tangent(Vec<u8>),
    Uv1(Vec<u8>),
    Uv2(Vec<u8>),
    Uv3(Vec<u8>),
}

impl ExtendedStreamJob {
    fn compute(
        self,
        source: &ExtendedVertexUploadSource<'_>,
        generate_missing_tangents: bool,
    ) -> ExtendedStreamJobResult {
        match self {
            Self::Tangent => ExtendedStreamJobResult::Tangent(
                tangent_stream_bytes(source.tangent_source(), generate_missing_tangents)
                    .unwrap_or_else(|| {
                        float4_default_stream_bytes(source.vertex_count, [1.0, 0.0, 0.0, 1.0])
                    }),
            ),
            Self::Uv1 => {
                ExtendedStreamJobResult::Uv1(extended_uv_bytes(source, VertexAttributeType::UV1))
            }
            Self::Uv2 => {
                ExtendedStreamJobResult::Uv2(extended_uv_bytes(source, VertexAttributeType::UV2))
            }
            Self::Uv3 => {
                ExtendedStreamJobResult::Uv3(extended_uv_bytes(source, VertexAttributeType::UV3))
            }
        }
    }
}

fn extended_uv_bytes(
    source: &ExtendedVertexUploadSource<'_>,
    target: VertexAttributeType,
) -> Vec<u8> {
    vertex_float2_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
        target,
    )
    .unwrap_or_else(|| float2_zero_stream_bytes(source.vertex_count))
}

fn compute_extended_stream_bytes(
    source: &ExtendedVertexUploadSource<'_>,
    generate_missing_tangents: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let jobs = [
        ExtendedStreamJob::Tangent,
        ExtendedStreamJob::Uv1,
        ExtendedStreamJob::Uv2,
        ExtendedStreamJob::Uv3,
    ];
    let admission = admit_mesh_stream_jobs(
        jobs.len(),
        source.vertex_count,
        current_reference_worker_count(),
    );
    record_parallel_admission(
        "mesh_lazy_extended_streams",
        source.vertex_count,
        jobs.len(),
        admission,
    );
    let results = if let Some(chunk_size) = admission.chunk_size() {
        jobs.par_iter()
            .copied()
            .with_min_len(chunk_size)
            .map(|job| job.compute(source, generate_missing_tangents))
            .collect::<Vec<_>>()
    } else {
        jobs.iter()
            .copied()
            .map(|job| job.compute(source, generate_missing_tangents))
            .collect::<Vec<_>>()
    };

    let mut tangent = Vec::new();
    let mut uv1 = Vec::new();
    let mut uv2 = Vec::new();
    let mut uv3 = Vec::new();
    for result in results {
        match result {
            ExtendedStreamJobResult::Tangent(bytes) => tangent = bytes,
            ExtendedStreamJobResult::Uv1(bytes) => uv1 = bytes,
            ExtendedStreamJobResult::Uv2(bytes) => uv2 = bytes,
            ExtendedStreamJobResult::Uv3(bytes) => uv3 = bytes,
        }
    }
    (tangent, uv1, uv2, uv3)
}

fn compute_default_extended_stream_bytes(
    vertex_count: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let jobs = [0usize, 1, 2, 3];
    let admission =
        admit_mesh_stream_jobs(jobs.len(), vertex_count, current_reference_worker_count());
    record_parallel_admission(
        "mesh_default_extended_streams",
        vertex_count,
        jobs.len(),
        admission,
    );
    let compute = |job| match job {
        0 => float4_default_stream_bytes(vertex_count, [1.0, 0.0, 0.0, 1.0]),
        _ => float2_zero_stream_bytes(vertex_count),
    };
    let mut results = if let Some(chunk_size) = admission.chunk_size() {
        jobs.par_iter()
            .copied()
            .with_min_len(chunk_size)
            .map(compute)
            .collect::<Vec<_>>()
    } else {
        jobs.iter().copied().map(compute).collect::<Vec<_>>()
    };
    let uv3 = results.pop().unwrap_or_default();
    let uv2 = results.pop().unwrap_or_default();
    let uv1 = results.pop().unwrap_or_default();
    let tangent = results.pop().unwrap_or_default();
    (tangent, uv1, uv2, uv3)
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

fn create_primary_stream_buffer(
    device: &wgpu::Device,
    asset_id: i32,
    label: &str,
    bytes: &[u8],
) -> Arc<wgpu::Buffer> {
    let buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {asset_id} {label}_stream")),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        }),
    );
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_primary_stream");
    buffer
}

/// Uploads position and normal streams from retained host vertex data.
pub(in crate::assets::mesh::gpu_mesh) fn upload_position_normal_vertex_streams(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> (Option<Arc<wgpu::Buffer>>, Option<Arc<wgpu::Buffer>>) {
    if source.vertex_count == 0 {
        return (None, None);
    }
    let Some((positions, normals)) = extract_float3_position_normal_as_vec4_streams(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    ) else {
        return (None, None);
    };
    (
        Some(create_primary_stream_buffer(
            device,
            asset_id,
            "positions",
            &positions,
        )),
        Some(create_primary_stream_buffer(
            device, asset_id, "normals", &normals,
        )),
    )
}

/// Uploads a UV0 stream from retained host vertex data.
pub(in crate::assets::mesh::gpu_mesh) fn upload_uv0_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = uv0_float2_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    )?;
    Some(create_vertex_stream_buffer(
        device, asset_id, "uv0", &uv_bytes,
    ))
}

/// Uploads a color stream from retained host vertex data.
pub(in crate::assets::mesh::gpu_mesh) fn upload_color_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let color_bytes = color_float4_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    )?;
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        "color",
        &color_bytes,
    ))
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

    let (tangent_bytes, uv1_bytes, uv2_bytes, uv3_bytes) =
        compute_extended_stream_bytes(&source, generate_missing_tangents);

    (
        Some(create_tangent_stream_buffer(
            device,
            asset_id,
            &tangent_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv1", &uv1_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv2", &uv2_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv3", &uv3_bytes,
        )),
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

/// Uploads the low wide UV payload stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_wide_low_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = wide_low_uv_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    )
    .unwrap_or_else(|| wide_low_uv_zero_stream_bytes(source.vertex_count));
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        source.label,
        &uv_bytes,
    ))
}

/// Uploads the high wide UV payload stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_wide_high_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    source: UvVertexUploadSource<'_>,
) -> Option<Arc<wgpu::Buffer>> {
    if source.vertex_count == 0 {
        return None;
    }
    let uv_bytes = wide_high_uv_stream_bytes(
        source.vertex_slice,
        source.vertex_count,
        source.vertex_stride,
        source.vertex_attributes,
    )
    .unwrap_or_else(|| wide_high_uv_zero_stream_bytes(source.vertex_count));
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        source.label,
        &uv_bytes,
    ))
}

/// Uploads a zero-filled low wide UV stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_wide_low_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv_bytes = wide_low_uv_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        "wide_low_uv",
        &uv_bytes,
    ))
}

/// Uploads a zero-filled high wide UV stream.
pub(in crate::assets::mesh::gpu_mesh) fn upload_default_wide_high_uv_vertex_stream(
    device: &wgpu::Device,
    asset_id: i32,
    vc_usize: usize,
) -> Option<Arc<wgpu::Buffer>> {
    if vc_usize == 0 {
        return None;
    }
    let uv_bytes = wide_high_uv_zero_stream_bytes(vc_usize);
    Some(create_vertex_stream_buffer(
        device,
        asset_id,
        "wide_high_uv",
        &uv_bytes,
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
    let (tangent_bytes, uv1_bytes, uv2_bytes, uv3_bytes) =
        compute_default_extended_stream_bytes(vc_usize);
    (
        Some(create_tangent_stream_buffer(
            device,
            asset_id,
            &tangent_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv1", &uv1_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv2", &uv2_bytes,
        )),
        Some(create_vertex_stream_buffer(
            device, asset_id, "uv3", &uv3_bytes,
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
