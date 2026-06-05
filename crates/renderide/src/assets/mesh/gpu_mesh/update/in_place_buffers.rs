//! Buffer compatibility checks and queue writes used by in-place mesh updates.

use rayon::prelude::*;

use crate::cpu_parallelism::admit_current_mesh_stream_jobs;
use crate::shared::{
    IndexBufferFormat, MeshUploadData, SubmeshBufferDescriptor, VertexAttributeDescriptor,
    VertexAttributeType,
};

use super::super::super::layout::{
    MeshBufferLayout, color_float4_stream_bytes, extract_bind_poses, extract_blendshape_offsets,
    extract_float3_position_normal_as_vec4_streams, split_bone_weights_tail_for_gpu,
    uv0_float2_stream_bytes, vertex_float2_stream_bytes, wide_high_uv_stream_bytes,
    wide_low_uv_stream_bytes,
};
use super::super::hints::{blendshape_descriptor_count, derived_streams_compatible_for_in_place};
use super::super::tangent_generation::{TangentStreamSource, tangent_stream_bytes};
use super::super::upload::{
    padded_sparse_bytes, queue_init_buffer_size_matches, write_mesh_upload_buffer,
};
use super::super::{GpuMesh, MeshBufferUploadSink, MeshDerivedStreamMask, PreparedDerivedStreams};

/// Validates sparse blendshape GPU buffers and scatter ranges against a fresh [`extract_blendshape_offsets`] pass.
pub(super) fn blendshape_and_deform_buffers_match_for_in_place(
    mesh: &GpuMesh,
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    raw: &[u8],
    use_blendshapes: bool,
) -> bool {
    let n_blend = blendshape_descriptor_count(&data.blendshape_buffers);
    if use_blendshapes && n_blend > 0 {
        let Some(extracted) =
            extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count)
        else {
            return false;
        };
        if extracted.num_blendshapes.max(0) as u32 != n_blend {
            return false;
        }
        let sparse_expect = padded_sparse_bytes(&extracted.sparse_deltas);
        let Some(sb) = mesh.blendshape_sparse_buffer.as_ref() else {
            return false;
        };
        if !queue_init_buffer_size_matches(sb.size(), sparse_expect.len()) {
            return false;
        }
        if mesh.blendshape_frame_ranges != extracted.frame_ranges {
            return false;
        }
        if mesh.blendshape_shape_frame_spans != extracted.shape_frame_spans {
            return false;
        }
        if mesh.num_blendshapes != n_blend {
            return false;
        }
        if mesh.blendshape_has_position_deltas != extracted.has_position_deltas
            || mesh.blendshape_has_normal_deltas != extracted.has_normal_deltas
            || mesh.blendshape_has_tangent_deltas != extracted.has_tangent_deltas
        {
            return false;
        }
    } else if mesh.num_blendshapes > 0
        || mesh.blendshape_sparse_buffer.is_some()
        || !mesh.blendshape_frame_ranges.is_empty()
        || !mesh.blendshape_shape_frame_spans.is_empty()
        || mesh.blendshape_has_position_deltas
        || mesh.blendshape_has_normal_deltas
        || mesh.blendshape_has_tangent_deltas
    {
        return false;
    }

    true
}

/// Real skeleton (`bone_count > 0`): validates bone buffer sizes against `layout` / split weights.
pub(super) fn compatible_for_in_place_real_skeleton(
    mesh: &GpuMesh,
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    raw: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
    vertex_slice: &[u8],
) -> bool {
    let bc = &raw[layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
    let bw =
        &raw[layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
    match split_bone_weights_tail_for_gpu(bc, bw, vc_usize, data.bone_count as usize) {
        Some(ref streams) => {
            if !mesh.bone_indices_buffer.as_ref().is_some_and(|b| {
                queue_init_buffer_size_matches(b.size(), streams.bone_indices_vec4.len())
            }) {
                return false;
            }
            if !mesh.bone_weights_vec4_buffer.as_ref().is_some_and(|b| {
                queue_init_buffer_size_matches(b.size(), streams.bone_weights_vec4.len())
            }) {
                return false;
            }
            if !mesh
                .bone_influence_offsets_buffer
                .as_ref()
                .is_some_and(|b| {
                    queue_init_buffer_size_matches(b.size(), streams.influence_offsets.len())
                })
            {
                return false;
            }
            if !mesh.bone_influences_buffer.as_ref().is_some_and(|b| {
                queue_init_buffer_size_matches(b.size(), influence_buffer_len(&streams.influences))
            }) {
                return false;
            }
        }
        None => {
            if mesh.bone_indices_buffer.is_some()
                || mesh.bone_weights_vec4_buffer.is_some()
                || mesh.bone_influence_offsets_buffer.is_some()
                || mesh.bone_influences_buffer.is_some()
            {
                return false;
            }
        }
    }
    if !mesh
        .bone_counts_buffer
        .as_ref()
        .is_some_and(|b| queue_init_buffer_size_matches(b.size(), layout.bone_counts_length))
    {
        return false;
    }
    if !mesh
        .bind_poses_buffer
        .as_ref()
        .is_some_and(|b| queue_init_buffer_size_matches(b.size(), layout.bind_poses_length))
    {
        return false;
    }
    if mesh.skinning_bind_matrices.len() != data.bone_count.max(0) as usize {
        return false;
    }
    derived_streams_compatible_for_in_place(mesh, vertex_slice, data, vc_usize, vertex_stride_us)
}

/// Shared host layout and GPU mesh handles for in-place mesh buffer writes (VB, IB, bones).
pub(super) struct MeshInPlaceWriteContext<'a> {
    /// Mesh being rewritten.
    pub(super) mesh: &'a GpuMesh,
    /// Sink receiving buffer writes.
    pub(super) upload_sink: &'a dyn MeshBufferUploadSink,
    /// Raw host payload covering the validated layout.
    pub(super) raw: &'a [u8],
    /// Parsed byte layout over [`Self::raw`].
    pub(super) layout: &'a MeshBufferLayout,
    /// Host mesh metadata for this upload.
    pub(super) data: &'a MeshUploadData,
    /// Vertex count as `usize`.
    pub(super) vertex_count: usize,
    /// Interleaved vertex stride as `usize`.
    pub(super) vertex_stride: usize,
    /// Derived streams requested for this write.
    pub(super) demand_mask: MeshDerivedStreamMask,
    /// CPU-prepared derived stream bytes available for this write.
    pub(super) prepared_derived_streams: Option<&'a PreparedDerivedStreams>,
}

#[derive(Clone, Copy, Debug)]
enum InPlaceDerivedStreamJob {
    PositionNormal,
    Uv0,
    Color,
    WideLowUv,
    WideHighUv,
    Tangent,
    Uv1,
    Uv2,
    Uv3,
}

enum InPlaceDerivedStreamResult {
    PositionNormal(Option<(Vec<u8>, Vec<u8>)>),
    Uv0(Option<Vec<u8>>),
    Color(Option<Vec<u8>>),
    WideLowUv(Option<Vec<u8>>),
    WideHighUv(Option<Vec<u8>>),
    Tangent(Option<Vec<u8>>),
    Uv1(Option<Vec<u8>>),
    Uv2(Option<Vec<u8>>),
    Uv3(Option<Vec<u8>>),
}

struct InPlaceDerivedStreamSource<'a> {
    vertex_slice: &'a [u8],
    index_slice: &'a [u8],
    vertex_count: usize,
    vertex_stride: usize,
    vertex_attributes: &'a [VertexAttributeDescriptor],
    index_format: IndexBufferFormat,
    submeshes: &'a [SubmeshBufferDescriptor],
    generate_missing_tangents: bool,
}

impl InPlaceDerivedStreamJob {
    fn compute(self, source: &InPlaceDerivedStreamSource<'_>) -> InPlaceDerivedStreamResult {
        match self {
            Self::PositionNormal => InPlaceDerivedStreamResult::PositionNormal(
                extract_float3_position_normal_as_vec4_streams(
                    source.vertex_slice,
                    source.vertex_count,
                    source.vertex_stride,
                    source.vertex_attributes,
                ),
            ),
            Self::Uv0 => InPlaceDerivedStreamResult::Uv0(uv0_float2_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
            )),
            Self::Color => InPlaceDerivedStreamResult::Color(color_float4_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
            )),
            Self::WideLowUv => InPlaceDerivedStreamResult::WideLowUv(wide_low_uv_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
            )),
            Self::WideHighUv => InPlaceDerivedStreamResult::WideHighUv(wide_high_uv_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
            )),
            Self::Tangent => InPlaceDerivedStreamResult::Tangent(tangent_stream_bytes(
                in_place_tangent_source(source),
                source.generate_missing_tangents,
            )),
            Self::Uv1 => InPlaceDerivedStreamResult::Uv1(vertex_float2_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
                VertexAttributeType::UV1,
            )),
            Self::Uv2 => InPlaceDerivedStreamResult::Uv2(vertex_float2_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
                VertexAttributeType::UV2,
            )),
            Self::Uv3 => InPlaceDerivedStreamResult::Uv3(vertex_float2_stream_bytes(
                source.vertex_slice,
                source.vertex_count,
                source.vertex_stride,
                source.vertex_attributes,
                VertexAttributeType::UV3,
            )),
        }
    }
}

fn in_place_tangent_source<'a>(
    source: &'a InPlaceDerivedStreamSource<'_>,
) -> TangentStreamSource<'a> {
    TangentStreamSource {
        vertex_data: source.vertex_slice,
        index_data: source.index_slice,
        vertex_count: source.vertex_count,
        stride: source.vertex_stride,
        attrs: source.vertex_attributes,
        index_format: source.index_format,
        submeshes: source.submeshes,
    }
}

/// Writes interleaved VB then optional derived position/normal/uv/color streams.
pub(super) fn write_in_place_vertex_and_derived_streams(
    ctx: &MeshInPlaceWriteContext<'_>,
    write_vertex: bool,
    write_index: bool,
) {
    profiling::scope!("asset::mesh_write_in_place::vertex_derived_streams");
    if write_vertex {
        {
            profiling::scope!("asset::mesh_write_in_place::write_interleaved_vertex");
            write_mesh_upload_buffer(
                ctx.upload_sink,
                ctx.mesh.vertex_buffer.as_ref(),
                0,
                &ctx.raw[..ctx.layout.vertex_size],
            );
        }
    }
    let vertex_slice = &ctx.raw[..ctx.layout.vertex_size];
    if !write_vertex && !write_index {
        return;
    }
    if try_write_in_place_derived_streams_parallel(ctx, vertex_slice, write_vertex) {
        return;
    }
    if write_vertex {
        if ctx
            .demand_mask
            .intersects(MeshDerivedStreamMask::POSITION | MeshDerivedStreamMask::NORMAL)
        {
            write_in_place_position_normal_streams(ctx, vertex_slice);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::UV0) {
            write_in_place_uv0_stream(ctx, vertex_slice);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::COLOR) {
            write_in_place_color_stream(ctx, vertex_slice);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::WIDE_UV_LOW) {
            write_in_place_wide_low_uv_stream(ctx, vertex_slice);
        }
        if ctx
            .demand_mask
            .contains(MeshDerivedStreamMask::WIDE_UV_HIGH)
        {
            write_in_place_wide_high_uv_stream(ctx, vertex_slice);
        }
    }

    if ctx
        .demand_mask
        .intersects(MeshDerivedStreamMask::TANGENT | MeshDerivedStreamMask::BILLBOARD)
    {
        profiling::scope!("asset::mesh_write_in_place::write_tangent_stream");
        let source = TangentStreamSource {
            vertex_data: vertex_slice,
            index_data: &ctx.raw[ctx.layout.index_buffer_start
                ..ctx.layout.index_buffer_start + ctx.layout.index_buffer_length],
            vertex_count: ctx.vertex_count,
            stride: ctx.vertex_stride,
            attrs: &ctx.data.vertex_attributes,
            index_format: ctx.data.index_buffer_format,
            submeshes: &ctx.data.submeshes,
        };
        if let Some(tb) = ctx.mesh.tangent_buffer.as_ref()
            && let Some(t) = ctx
                .prepared_derived_streams
                .and_then(|prepared| prepared.tangent.as_deref())
                .map(std::borrow::Cow::Borrowed)
                .or_else(|| {
                    tangent_stream_bytes(source, ctx.mesh.tangent_fallback_mode.generate_missing())
                        .map(std::borrow::Cow::Owned)
                })
        {
            write_mesh_upload_buffer(ctx.upload_sink, tb.as_ref(), 0, t.as_ref());
        }
    }

    if !write_vertex {
        return;
    }

    if ctx.demand_mask.intersects(
        MeshDerivedStreamMask::UV1
            | MeshDerivedStreamMask::UV2
            | MeshDerivedStreamMask::UV3
            | MeshDerivedStreamMask::BILLBOARD,
    ) {
        write_in_place_uv1_to_uv3_streams(ctx, vertex_slice);
    }
}

fn try_write_in_place_derived_streams_parallel(
    ctx: &MeshInPlaceWriteContext<'_>,
    vertex_slice: &[u8],
    write_vertex: bool,
) -> bool {
    let mut jobs = Vec::with_capacity(9);
    if write_vertex {
        if ctx
            .demand_mask
            .intersects(MeshDerivedStreamMask::POSITION | MeshDerivedStreamMask::NORMAL)
            && ctx.mesh.positions_buffer.is_some()
            && ctx.mesh.normals_buffer.is_some()
        {
            jobs.push(InPlaceDerivedStreamJob::PositionNormal);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::UV0) && ctx.mesh.uv0_buffer.is_some() {
            jobs.push(InPlaceDerivedStreamJob::Uv0);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::COLOR) && ctx.mesh.color_buffer.is_some()
        {
            jobs.push(InPlaceDerivedStreamJob::Color);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::WIDE_UV_LOW)
            && ctx.mesh.wide_low_uv_buffer.is_some()
        {
            jobs.push(InPlaceDerivedStreamJob::WideLowUv);
        }
        if ctx
            .demand_mask
            .contains(MeshDerivedStreamMask::WIDE_UV_HIGH)
            && ctx.mesh.wide_high_uv_buffer.is_some()
        {
            jobs.push(InPlaceDerivedStreamJob::WideHighUv);
        }
    }
    if ctx.demand_mask.contains(MeshDerivedStreamMask::TANGENT) && ctx.mesh.tangent_buffer.is_some()
    {
        jobs.push(InPlaceDerivedStreamJob::Tangent);
    }
    if write_vertex {
        if ctx.demand_mask.contains(MeshDerivedStreamMask::UV1) && ctx.mesh.uv1_buffer.is_some() {
            jobs.push(InPlaceDerivedStreamJob::Uv1);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::UV2) && ctx.mesh.uv2_buffer.is_some() {
            jobs.push(InPlaceDerivedStreamJob::Uv2);
        }
        if ctx.demand_mask.contains(MeshDerivedStreamMask::UV3) && ctx.mesh.uv3_buffer.is_some() {
            jobs.push(InPlaceDerivedStreamJob::Uv3);
        }
    }
    let admission = admit_current_mesh_stream_jobs(
        "mesh_write_in_place_derived_streams",
        jobs.len(),
        ctx.vertex_count,
    );
    let Some(chunk_size) = admission.chunk_size() else {
        return false;
    };
    let source = InPlaceDerivedStreamSource {
        vertex_slice,
        index_slice: &ctx.raw[ctx.layout.index_buffer_start
            ..ctx.layout.index_buffer_start + ctx.layout.index_buffer_length],
        vertex_count: ctx.vertex_count,
        vertex_stride: ctx.vertex_stride,
        vertex_attributes: &ctx.data.vertex_attributes,
        index_format: ctx.data.index_buffer_format,
        submeshes: &ctx.data.submeshes,
        generate_missing_tangents: ctx.mesh.tangent_fallback_mode.generate_missing(),
    };
    let results = jobs
        .par_iter()
        .copied()
        .with_min_len(chunk_size)
        .map(|job| job.compute(&source))
        .collect::<Vec<_>>();
    for result in results {
        write_in_place_derived_stream_result(ctx, result);
    }
    true
}

fn write_in_place_derived_stream_result(
    ctx: &MeshInPlaceWriteContext<'_>,
    result: InPlaceDerivedStreamResult,
) {
    match result {
        InPlaceDerivedStreamResult::PositionNormal(Some((positions, normals))) => {
            if let (Some(pb), Some(nb)) = (
                ctx.mesh.positions_buffer.as_ref(),
                ctx.mesh.normals_buffer.as_ref(),
            ) {
                write_mesh_upload_buffer(ctx.upload_sink, pb.as_ref(), 0, &positions);
                write_mesh_upload_buffer(ctx.upload_sink, nb.as_ref(), 0, &normals);
            }
        }
        InPlaceDerivedStreamResult::PositionNormal(None) => {}
        InPlaceDerivedStreamResult::Uv0(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.uv0_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::Uv0(None) => {}
        InPlaceDerivedStreamResult::Color(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.color_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::Color(None) => {}
        InPlaceDerivedStreamResult::WideLowUv(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.wide_low_uv_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::WideLowUv(None) => {}
        InPlaceDerivedStreamResult::WideHighUv(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.wide_high_uv_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::WideHighUv(None) => {}
        InPlaceDerivedStreamResult::Tangent(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.tangent_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::Tangent(None) => {}
        InPlaceDerivedStreamResult::Uv1(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.uv1_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::Uv1(None) => {}
        InPlaceDerivedStreamResult::Uv2(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.uv2_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::Uv2(None) => {}
        InPlaceDerivedStreamResult::Uv3(Some(bytes)) => {
            if let Some(buffer) = ctx.mesh.uv3_buffer.as_ref() {
                write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &bytes);
            }
        }
        InPlaceDerivedStreamResult::Uv3(None) => {}
    }
}

fn write_in_place_position_normal_streams(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_position_normal_streams");
    let (Some(pb), Some(nb)) = (
        ctx.mesh.positions_buffer.as_ref(),
        ctx.mesh.normals_buffer.as_ref(),
    ) else {
        return;
    };
    if let Some(prepared) = ctx.prepared_derived_streams
        && let (Some(positions), Some(normals)) =
            (prepared.positions.as_deref(), prepared.normals.as_deref())
    {
        write_mesh_upload_buffer(ctx.upload_sink, pb.as_ref(), 0, positions);
        write_mesh_upload_buffer(ctx.upload_sink, nb.as_ref(), 0, normals);
        return;
    }
    if let Some((positions, normals)) = extract_float3_position_normal_as_vec4_streams(
        vertex_slice,
        ctx.vertex_count,
        ctx.vertex_stride,
        &ctx.data.vertex_attributes,
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, pb.as_ref(), 0, &positions);
        write_mesh_upload_buffer(ctx.upload_sink, nb.as_ref(), 0, &normals);
    }
}

fn write_in_place_uv0_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_uv0_stream");
    if let (Some(uvb), Some(prepared)) = (
        ctx.mesh.uv0_buffer.as_ref(),
        ctx.prepared_derived_streams
            .and_then(|prepared| prepared.uv0.as_deref()),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, uvb.as_ref(), 0, prepared);
        return;
    }
    if let (Some(uvb), Some(uv)) = (
        ctx.mesh.uv0_buffer.as_ref(),
        uv0_float2_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, uvb.as_ref(), 0, &uv);
    }
}

fn write_in_place_color_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_color_stream");
    if let (Some(cb), Some(prepared)) = (
        ctx.mesh.color_buffer.as_ref(),
        ctx.prepared_derived_streams
            .and_then(|prepared| prepared.color.as_deref()),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, cb.as_ref(), 0, prepared);
        return;
    }
    if let (Some(cb), Some(c)) = (
        ctx.mesh.color_buffer.as_ref(),
        color_float4_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, cb.as_ref(), 0, &c);
    }
}

fn write_in_place_wide_low_uv_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_wide_low_uv_stream");
    if let (Some(uvb), Some(prepared)) = (
        ctx.mesh.wide_low_uv_buffer.as_ref(),
        ctx.prepared_derived_streams
            .and_then(|prepared| prepared.wide_low_uv.as_deref()),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, uvb.as_ref(), 0, prepared);
        return;
    }
    if let (Some(uvb), Some(uv)) = (
        ctx.mesh.wide_low_uv_buffer.as_ref(),
        wide_low_uv_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, uvb.as_ref(), 0, &uv);
    }
}

fn write_in_place_wide_high_uv_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_wide_high_uv_stream");
    if let (Some(uvb), Some(prepared)) = (
        ctx.mesh.wide_high_uv_buffer.as_ref(),
        ctx.prepared_derived_streams
            .and_then(|prepared| prepared.wide_high_uv.as_deref()),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, uvb.as_ref(), 0, prepared);
        return;
    }
    if let (Some(uvb), Some(uv)) = (
        ctx.mesh.wide_high_uv_buffer.as_ref(),
        wide_high_uv_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_upload_buffer(ctx.upload_sink, uvb.as_ref(), 0, &uv);
    }
}

fn write_in_place_uv1_to_uv3_streams(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_uv1_to_uv3_streams");
    for (buffer, target, mask) in [
        (
            &ctx.mesh.uv1_buffer,
            VertexAttributeType::UV1,
            MeshDerivedStreamMask::UV1,
        ),
        (
            &ctx.mesh.uv2_buffer,
            VertexAttributeType::UV2,
            MeshDerivedStreamMask::UV2,
        ),
        (
            &ctx.mesh.uv3_buffer,
            VertexAttributeType::UV3,
            MeshDerivedStreamMask::UV3,
        ),
    ] {
        if !ctx.demand_mask.contains(mask) {
            continue;
        }
        let prepared = match target {
            VertexAttributeType::UV1 => ctx
                .prepared_derived_streams
                .and_then(|prepared| prepared.uv1.as_deref()),
            VertexAttributeType::UV2 => ctx
                .prepared_derived_streams
                .and_then(|prepared| prepared.uv2.as_deref()),
            VertexAttributeType::UV3 => ctx
                .prepared_derived_streams
                .and_then(|prepared| prepared.uv3.as_deref()),
            _ => None,
        };
        if let (Some(buffer), Some(uv)) = (buffer.as_ref(), prepared) {
            write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, uv);
            continue;
        }
        if let (Some(buffer), Some(uv)) = (
            buffer.as_ref(),
            vertex_float2_stream_bytes(
                vertex_slice,
                ctx.vertex_count,
                ctx.vertex_stride,
                &ctx.data.vertex_attributes,
                target,
            ),
        ) {
            write_mesh_upload_buffer(ctx.upload_sink, buffer.as_ref(), 0, &uv);
        }
    }
}

/// Writes index buffer slice when `write_ib` is set.
pub(super) fn write_in_place_index_buffer(
    mesh: &GpuMesh,
    upload_sink: &dyn MeshBufferUploadSink,
    raw: &[u8],
    layout: &MeshBufferLayout,
    write_ib: bool,
) {
    profiling::scope!("asset::mesh_write_in_place::index_buffer");
    if !write_ib {
        return;
    }
    let ib_slice =
        &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
    write_mesh_upload_buffer(upload_sink, mesh.index_buffer.as_ref(), 0, ib_slice);
}

/// Per-buffer hint flags driving [`write_in_place_bone_buffers`].
///
/// Each field maps one-to-one to a host upload hint and selects which bone-related buffers are
/// rewritten in this in-place update.
#[derive(Clone, Copy, Debug)]
pub(super) struct BoneBufferWriteHints {
    /// Whole upload involves bone buffers; if `false`, the call is a no-op.
    pub(super) needs_bone_buffers: bool,
    /// Full upload: every bone buffer should be rewritten irrespective of the per-buffer flags.
    pub(super) full: bool,
    /// Bone counts and bone weights/indices should be rewritten.
    pub(super) write_bone_weights: bool,
    /// Bind poses should be rewritten.
    pub(super) write_bind_poses: bool,
}

/// Writes bone/synthetic bone buffers from `raw` according to `hints`.
pub(super) fn write_in_place_bone_buffers(
    ctx: &MeshInPlaceWriteContext<'_>,
    hints: BoneBufferWriteHints,
) -> Option<()> {
    profiling::scope!("asset::mesh_write_in_place::bone_buffers");
    let BoneBufferWriteHints {
        needs_bone_buffers,
        full,
        write_bone_weights,
        write_bind_poses,
    } = hints;
    if !needs_bone_buffers {
        return Some(());
    }
    if ctx.data.bone_count > 0 {
        profiling::scope!("asset::mesh_write_in_place::real_bone_buffers");
        if full || write_bone_weights {
            let bc = &ctx.raw[ctx.layout.bone_counts_start
                ..ctx.layout.bone_counts_start + ctx.layout.bone_counts_length];
            let bw = &ctx.raw[ctx.layout.bone_weights_start
                ..ctx.layout.bone_weights_start + ctx.layout.bone_weights_length];
            if let Some(bcb) = &ctx.mesh.bone_counts_buffer {
                write_mesh_upload_buffer(ctx.upload_sink, bcb.as_ref(), 0, bc);
            }
            if let Some(streams) = split_bone_weights_tail_for_gpu(
                bc,
                bw,
                ctx.vertex_count,
                ctx.data.bone_count as usize,
            ) {
                if let Some(bi) = &ctx.mesh.bone_indices_buffer {
                    write_mesh_upload_buffer(
                        ctx.upload_sink,
                        bi.as_ref(),
                        0,
                        &streams.bone_indices_vec4,
                    );
                }
                if let Some(bwt) = &ctx.mesh.bone_weights_vec4_buffer {
                    write_mesh_upload_buffer(
                        ctx.upload_sink,
                        bwt.as_ref(),
                        0,
                        &streams.bone_weights_vec4,
                    );
                }
                if let Some(offsets) = &ctx.mesh.bone_influence_offsets_buffer {
                    write_mesh_upload_buffer(
                        ctx.upload_sink,
                        offsets.as_ref(),
                        0,
                        &streams.influence_offsets,
                    );
                }
                if let Some(influences) = &ctx.mesh.bone_influences_buffer {
                    let influence_bytes = influence_buffer_bytes(&streams.influences);
                    write_mesh_upload_buffer(
                        ctx.upload_sink,
                        influences.as_ref(),
                        0,
                        &influence_bytes,
                    );
                }
            }
        }
        if full || write_bind_poses {
            let bp_raw = &ctx.raw[ctx.layout.bind_poses_start
                ..ctx.layout.bind_poses_start + ctx.layout.bind_poses_length];
            if let Some(bp) = &ctx.mesh.bind_poses_buffer {
                let bind_poses_arr = extract_bind_poses(bp_raw, ctx.data.bone_count as usize)?;
                let bp_bytes: Vec<u8> = bind_poses_arr
                    .iter()
                    .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                    .collect();
                write_mesh_upload_buffer(ctx.upload_sink, bp.as_ref(), 0, &bp_bytes);
            }
        }
    }
    Some(())
}

fn influence_buffer_len(influences: &[u8]) -> usize {
    influences.len().max(8)
}

fn influence_buffer_bytes(influences: &[u8]) -> Vec<u8> {
    if influences.is_empty() {
        vec![0; 8]
    } else {
        influences.to_vec()
    }
}

/// Sparse blendshape GPU buffer and CPU ranges.
pub(super) fn write_in_place_blendshape_buffer(
    mesh: &GpuMesh,
    upload_sink: &dyn MeshBufferUploadSink,
    raw: &[u8],
    layout: &MeshBufferLayout,
    data: &MeshUploadData,
    write_blend: bool,
) -> Option<()> {
    profiling::scope!("asset::mesh_write_in_place::blendshape_buffer");
    if !write_blend {
        return Some(());
    }
    let Some(sb) = mesh.blendshape_sparse_buffer.as_ref() else {
        return Some(());
    };
    let extracted = {
        profiling::scope!("asset::mesh_write_in_place::extract_blendshape_offsets");
        extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count)?
    };
    let sparse = {
        profiling::scope!("asset::mesh_write_in_place::pad_sparse_blendshapes");
        padded_sparse_bytes(&extracted.sparse_deltas)
    };
    {
        profiling::scope!("asset::mesh_write_in_place::write_blendshape_gpu_buffers");
        write_mesh_upload_buffer(upload_sink, sb.as_ref(), 0, &sparse);
    }
    Some(())
}
