//! CPU preparation of derived mesh vertex-stream byte payloads.

use rayon::prelude::*;

use crate::cpu_parallelism::admit_current_mesh_stream_jobs;
use crate::shared::{MeshUploadData, VertexAttributeType};

use super::super::super::layout::{
    MeshBufferLayout, color_float4_stream_bytes, compute_vertex_stride,
    extract_float3_position_normal_as_vec4_streams, uv0_float2_stream_bytes,
    vertex_float2_stream_bytes, wide_high_uv_stream_bytes, wide_low_uv_stream_bytes,
};
use super::super::demand::{MeshDerivedStreamDemand, MeshDerivedStreamMask};
use super::super::tangent_generation::{
    TangentStreamSource, raw_tangent_payload_stream_bytes, tangent_stream_bytes,
};

/// CPU-prepared derived vertex stream bytes for a full mesh upload.
#[derive(Clone, Debug, Default)]
pub(crate) struct PreparedDerivedStreams {
    /// Decomposed position stream bytes.
    pub(crate) positions: Option<Vec<u8>>,
    /// Decomposed normal stream bytes.
    pub(crate) normals: Option<Vec<u8>>,
    /// UV0 stream bytes.
    pub(crate) uv0: Option<Vec<u8>>,
    /// Vertex color stream bytes.
    pub(crate) color: Option<Vec<u8>>,
    /// Geometric tangent stream bytes.
    pub(crate) tangent: Option<Vec<u8>>,
    /// Raw tangent payload bytes.
    pub(crate) raw_tangent: Option<Vec<u8>>,
    /// UV1 stream bytes.
    pub(crate) uv1: Option<Vec<u8>>,
    /// UV2 stream bytes.
    pub(crate) uv2: Option<Vec<u8>>,
    /// UV3 stream bytes.
    pub(crate) uv3: Option<Vec<u8>>,
    /// Low wide-UV stream bytes.
    pub(crate) wide_low_uv: Option<Vec<u8>>,
    /// High wide-UV stream bytes.
    pub(crate) wide_high_uv: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
enum DerivedStreamJob {
    PositionNormal,
    Uv0,
    Color,
    Tangent,
    RawTangent,
    Uv1,
    Uv2,
    Uv3,
    WideLowUv,
    WideHighUv,
}

enum DerivedStreamJobResult {
    PositionNormal(Option<(Vec<u8>, Vec<u8>)>),
    Uv0(Option<Vec<u8>>),
    Color(Option<Vec<u8>>),
    Tangent(Option<Vec<u8>>),
    RawTangent(Option<Vec<u8>>),
    Uv1(Option<Vec<u8>>),
    Uv2(Option<Vec<u8>>),
    Uv3(Option<Vec<u8>>),
    WideLowUv(Option<Vec<u8>>),
    WideHighUv(Option<Vec<u8>>),
}

impl DerivedStreamJob {
    fn compute(
        self,
        vertex_slice: &[u8],
        index_slice: &[u8],
        data: &MeshUploadData,
        vc_usize: usize,
        vertex_stride_us: usize,
        demand: MeshDerivedStreamDemand,
    ) -> DerivedStreamJobResult {
        match self {
            Self::PositionNormal => DerivedStreamJobResult::PositionNormal(
                extract_float3_position_normal_as_vec4_streams(
                    vertex_slice,
                    vc_usize,
                    vertex_stride_us,
                    &data.vertex_attributes,
                ),
            ),
            Self::Uv0 => DerivedStreamJobResult::Uv0(uv0_float2_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
            )),
            Self::Color => DerivedStreamJobResult::Color(color_float4_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
            )),
            Self::Tangent => DerivedStreamJobResult::Tangent(tangent_stream_bytes(
                tangent_source(vertex_slice, index_slice, data, vc_usize, vertex_stride_us),
                demand.tangent_fallback_mode.generate_missing(),
            )),
            Self::RawTangent => {
                DerivedStreamJobResult::RawTangent(raw_tangent_payload_stream_bytes(
                    tangent_source(vertex_slice, index_slice, data, vc_usize, vertex_stride_us),
                ))
            }
            Self::Uv1 => DerivedStreamJobResult::Uv1(vertex_float2_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
                VertexAttributeType::UV1,
            )),
            Self::Uv2 => DerivedStreamJobResult::Uv2(vertex_float2_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
                VertexAttributeType::UV2,
            )),
            Self::Uv3 => DerivedStreamJobResult::Uv3(vertex_float2_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
                VertexAttributeType::UV3,
            )),
            Self::WideLowUv => DerivedStreamJobResult::WideLowUv(wide_low_uv_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
            )),
            Self::WideHighUv => DerivedStreamJobResult::WideHighUv(wide_high_uv_stream_bytes(
                vertex_slice,
                vc_usize,
                vertex_stride_us,
                &data.vertex_attributes,
            )),
        }
    }
}

impl PreparedDerivedStreams {
    fn apply_job_result(&mut self, result: DerivedStreamJobResult) {
        match result {
            DerivedStreamJobResult::PositionNormal(Some((positions, normals))) => {
                self.positions = Some(positions);
                self.normals = Some(normals);
            }
            DerivedStreamJobResult::PositionNormal(None) => {}
            DerivedStreamJobResult::Uv0(bytes) => self.uv0 = bytes,
            DerivedStreamJobResult::Color(bytes) => self.color = bytes,
            DerivedStreamJobResult::Tangent(bytes) => self.tangent = bytes,
            DerivedStreamJobResult::RawTangent(bytes) => self.raw_tangent = bytes,
            DerivedStreamJobResult::Uv1(bytes) => self.uv1 = bytes,
            DerivedStreamJobResult::Uv2(bytes) => self.uv2 = bytes,
            DerivedStreamJobResult::Uv3(bytes) => self.uv3 = bytes,
            DerivedStreamJobResult::WideLowUv(bytes) => self.wide_low_uv = bytes,
            DerivedStreamJobResult::WideHighUv(bytes) => self.wide_high_uv = bytes,
        }
    }
}

/// Prepares derived stream bytes requested by the current demand mask.
pub(crate) fn prepare_derived_stream_bytes(
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    demand: MeshDerivedStreamDemand,
) -> PreparedDerivedStreams {
    profiling::scope!("asset::mesh_prepare_derived_streams");
    let vc_usize = data.vertex_count.max(0) as usize;
    let vertex_stride_us = compute_vertex_stride(&data.vertex_attributes).max(1) as usize;
    let vertex_slice = &raw[..layout.vertex_size];
    let index_slice =
        &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
    let mut prepared = PreparedDerivedStreams::default();
    let mut jobs = Vec::with_capacity(9);

    if demand
        .mask
        .intersects(MeshDerivedStreamMask::POSITION | MeshDerivedStreamMask::NORMAL)
    {
        jobs.push(DerivedStreamJob::PositionNormal);
    }
    if demand.mask.contains(MeshDerivedStreamMask::UV0) {
        jobs.push(DerivedStreamJob::Uv0);
    }
    if demand.mask.contains(MeshDerivedStreamMask::COLOR) {
        jobs.push(DerivedStreamJob::Color);
    }
    if demand.mask.contains(MeshDerivedStreamMask::TANGENT) {
        jobs.push(DerivedStreamJob::Tangent);
    }
    if demand.mask.contains(MeshDerivedStreamMask::RAW_TANGENT) {
        jobs.push(DerivedStreamJob::RawTangent);
    }
    if demand.mask.contains(MeshDerivedStreamMask::UV1) {
        jobs.push(DerivedStreamJob::Uv1);
    }
    if demand.mask.contains(MeshDerivedStreamMask::UV2) {
        jobs.push(DerivedStreamJob::Uv2);
    }
    if demand.mask.contains(MeshDerivedStreamMask::UV3) {
        jobs.push(DerivedStreamJob::Uv3);
    }
    if demand.mask.contains(MeshDerivedStreamMask::WIDE_UV_LOW) {
        jobs.push(DerivedStreamJob::WideLowUv);
    }
    if demand.mask.contains(MeshDerivedStreamMask::WIDE_UV_HIGH) {
        jobs.push(DerivedStreamJob::WideHighUv);
    }

    let admission =
        admit_current_mesh_stream_jobs("mesh_prepare_derived_streams", jobs.len(), vc_usize);
    let results = if let Some(chunk_size) = admission.chunk_size() {
        jobs.par_iter()
            .copied()
            .with_min_len(chunk_size)
            .map(|job| {
                job.compute(
                    vertex_slice,
                    index_slice,
                    data,
                    vc_usize,
                    vertex_stride_us,
                    demand,
                )
            })
            .collect::<Vec<_>>()
    } else {
        jobs.iter()
            .copied()
            .map(|job| {
                job.compute(
                    vertex_slice,
                    index_slice,
                    data,
                    vc_usize,
                    vertex_stride_us,
                    demand,
                )
            })
            .collect::<Vec<_>>()
    };
    for result in results {
        prepared.apply_job_result(result);
    }

    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!(
            "mesh_upload::background_prepared_streams",
            prepared.available_mask().bits().count_ones() as f64
        );
    }
    prepared
}

fn tangent_source<'a>(
    vertex_slice: &'a [u8],
    index_slice: &'a [u8],
    data: &'a MeshUploadData,
    vc_usize: usize,
    vertex_stride_us: usize,
) -> TangentStreamSource<'a> {
    TangentStreamSource {
        vertex_data: vertex_slice,
        index_data: index_slice,
        vertex_count: vc_usize,
        stride: vertex_stride_us,
        attrs: &data.vertex_attributes,
        index_format: data.index_buffer_format,
        submeshes: &data.submeshes,
    }
}

#[cfg(feature = "tracy")]
impl PreparedDerivedStreams {
    /// Returns the streams with prepared byte payloads.
    pub(crate) fn available_mask(&self) -> MeshDerivedStreamMask {
        let mut mask = MeshDerivedStreamMask::EMPTY;
        if self.positions.is_some() {
            mask |= MeshDerivedStreamMask::POSITION;
        }
        if self.normals.is_some() {
            mask |= MeshDerivedStreamMask::NORMAL;
        }
        if self.uv0.is_some() {
            mask |= MeshDerivedStreamMask::UV0;
        }
        if self.color.is_some() {
            mask |= MeshDerivedStreamMask::COLOR;
        }
        if self.tangent.is_some() {
            mask |= MeshDerivedStreamMask::TANGENT;
        }
        if self.raw_tangent.is_some() {
            mask |= MeshDerivedStreamMask::RAW_TANGENT;
        }
        if self.uv1.is_some() {
            mask |= MeshDerivedStreamMask::UV1;
        }
        if self.uv2.is_some() {
            mask |= MeshDerivedStreamMask::UV2;
        }
        if self.uv3.is_some() {
            mask |= MeshDerivedStreamMask::UV3;
        }
        if self.wide_low_uv.is_some() {
            mask |= MeshDerivedStreamMask::WIDE_UV_LOW;
        }
        if self.wide_high_uv.is_some() {
            mask |= MeshDerivedStreamMask::WIDE_UV_HIGH;
        }
        mask
    }
}
