//! Buffer compatibility checks and queue writes used by in-place mesh updates.

use crate::shared::{MeshUploadData, VertexAttributeType};

use super::super::super::layout::{
    MeshBufferLayout, color_float4_stream_bytes, extract_bind_poses, extract_blendshape_offsets,
    extract_float3_position_normal_as_vec4_streams, split_bone_weights_tail_for_gpu,
    uv0_float2_stream_bytes, vertex_float2_stream_bytes, wide_uv_stream_bytes,
};
use super::super::GpuMesh;
use super::super::hints::{blendshape_descriptor_count, derived_streams_compatible_for_in_place};
use super::super::tangent_generation::{
    TangentStreamSource, raw_tangent_payload_stream_bytes, tangent_stream_bytes,
};
use super::super::upload::{
    padded_sparse_bytes, queue_init_buffer_size_matches, write_mesh_queue_buffer,
};

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
    match split_bone_weights_tail_for_gpu(bc, bw, vc_usize) {
        Some((ref ib, ref wb)) => {
            if !mesh
                .bone_indices_buffer
                .as_ref()
                .is_some_and(|b| queue_init_buffer_size_matches(b.size(), ib.len()))
            {
                return false;
            }
            if !mesh
                .bone_weights_vec4_buffer
                .as_ref()
                .is_some_and(|b| queue_init_buffer_size_matches(b.size(), wb.len()))
            {
                return false;
            }
        }
        None => {
            if mesh.bone_indices_buffer.is_some() || mesh.bone_weights_vec4_buffer.is_some() {
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
    /// Queue receiving buffer writes.
    pub(super) queue: &'a wgpu::Queue,
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
            write_mesh_queue_buffer(
                ctx.queue,
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
    if write_vertex {
        write_in_place_position_normal_streams(ctx, vertex_slice);
        write_in_place_uv0_stream(ctx, vertex_slice);
        write_in_place_color_stream(ctx, vertex_slice);
        write_in_place_wide_uv_stream(ctx, vertex_slice);
    }

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
        if let (Some(tb), Some(t)) = (
            ctx.mesh.tangent_buffer.as_ref(),
            tangent_stream_bytes(source, ctx.mesh.tangent_fallback_mode.generate_missing()),
        ) {
            write_mesh_queue_buffer(ctx.queue, tb.as_ref(), 0, &t);
        }
        if let (Some(tb), Some(t)) = (
            ctx.mesh.raw_tangent_buffer.as_ref(),
            raw_tangent_payload_stream_bytes(source),
        ) {
            write_mesh_queue_buffer(ctx.queue, tb.as_ref(), 0, &t);
        }
    }

    if !write_vertex {
        return;
    }

    write_in_place_uv1_to_uv3_streams(ctx, vertex_slice);
}

fn write_in_place_position_normal_streams(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_position_normal_streams");
    if let (Some(pb), Some(nb), Some((pvec, nvec))) = (
        ctx.mesh.positions_buffer.as_ref(),
        ctx.mesh.normals_buffer.as_ref(),
        extract_float3_position_normal_as_vec4_streams(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        )
        .as_ref(),
    ) {
        write_mesh_queue_buffer(ctx.queue, pb.as_ref(), 0, pvec);
        write_mesh_queue_buffer(ctx.queue, nb.as_ref(), 0, nvec);
    }
}

fn write_in_place_uv0_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_uv0_stream");
    if let (Some(uvb), Some(uv)) = (
        ctx.mesh.uv0_buffer.as_ref(),
        uv0_float2_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_queue_buffer(ctx.queue, uvb.as_ref(), 0, &uv);
    }
}

fn write_in_place_color_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_color_stream");
    if let (Some(cb), Some(c)) = (
        ctx.mesh.color_buffer.as_ref(),
        color_float4_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_queue_buffer(ctx.queue, cb.as_ref(), 0, &c);
    }
}

fn write_in_place_wide_uv_stream(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_wide_uv_stream");
    if let (Some(uvb), Some(uv)) = (
        ctx.mesh.wide_uv_buffer.as_ref(),
        wide_uv_stream_bytes(
            vertex_slice,
            ctx.vertex_count,
            ctx.vertex_stride,
            &ctx.data.vertex_attributes,
        ),
    ) {
        write_mesh_queue_buffer(ctx.queue, uvb.as_ref(), 0, &uv);
    }
}

fn write_in_place_uv1_to_uv3_streams(ctx: &MeshInPlaceWriteContext<'_>, vertex_slice: &[u8]) {
    profiling::scope!("asset::mesh_write_in_place::write_uv1_to_uv3_streams");
    for (buffer, target) in [
        (&ctx.mesh.uv1_buffer, VertexAttributeType::UV1),
        (&ctx.mesh.uv2_buffer, VertexAttributeType::UV2),
        (&ctx.mesh.uv3_buffer, VertexAttributeType::UV3),
    ] {
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
            write_mesh_queue_buffer(ctx.queue, buffer.as_ref(), 0, &uv);
        }
    }
}

/// Writes index buffer slice when `write_ib` is set.
pub(super) fn write_in_place_index_buffer(
    mesh: &GpuMesh,
    queue: &wgpu::Queue,
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
    write_mesh_queue_buffer(queue, mesh.index_buffer.as_ref(), 0, ib_slice);
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
                write_mesh_queue_buffer(ctx.queue, bcb.as_ref(), 0, bc);
            }
            if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(bc, bw, ctx.vertex_count) {
                if let Some(bi) = &ctx.mesh.bone_indices_buffer {
                    write_mesh_queue_buffer(ctx.queue, bi.as_ref(), 0, &ib);
                }
                if let Some(bwt) = &ctx.mesh.bone_weights_vec4_buffer {
                    write_mesh_queue_buffer(ctx.queue, bwt.as_ref(), 0, &wb);
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
                write_mesh_queue_buffer(ctx.queue, bp.as_ref(), 0, &bp_bytes);
            }
        }
    }
    Some(())
}

/// Sparse blendshape GPU buffer and CPU ranges.
pub(super) fn write_in_place_blendshape_buffer(
    mesh: &GpuMesh,
    queue: &wgpu::Queue,
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
        write_mesh_queue_buffer(queue, sb.as_ref(), 0, &sparse);
    }
    Some(())
}
