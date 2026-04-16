//! GPU-resident mesh: wgpu buffers only; host layout preserved in one interleaved vertex buffer.

use std::sync::Arc;

use crate::shared::{MeshUploadData, MeshUploadHintFlag, RenderBoundingBox};
use glam::Mat4;

use super::layout::{
    color_float4_stream_bytes, compute_index_count, compute_vertex_stride, extract_bind_poses,
    extract_blendshape_offsets, extract_float3_position_normal_as_vec4_streams,
    split_bone_weights_tail_for_gpu, synthetic_bone_data_for_blendshape_only,
    uv0_float2_stream_bytes, MeshBufferLayout,
};

use crate::gpu::GpuLimits;

use super::upload_impl::{
    allocate_deform_outputs, create_core_vertex_index_buffers, extract_derived_vertex_streams,
    padded_sparse_bytes, resident_bytes_for_mesh_upload, upload_blendshape_buffer,
    upload_bone_and_skin_buffers, validate_mesh_upload_layout,
};

use super::gpu_mesh_hints::{
    blendshape_descriptor_count, derived_streams_compatible_for_in_place,
    mesh_upload_hint_any_selective, mesh_upload_hint_touches_vertex_streams,
    validated_submesh_ranges, wgpu_index_format,
};

/// Resident mesh on GPU: no CPU geometry retained.
///
/// **Vertex groups** in Renderite are expressed through per-vertex bone influence streams
/// (`bone_counts` + bone weight tail) when the host provides skeleton data.
#[derive(Debug)]
pub struct GpuMesh {
    /// Host mesh asset id (`MeshUploadData.asset_id`).
    pub asset_id: i32,
    /// Full interleaved vertices as sent by the host (`vertex_attributes` order).
    pub vertex_buffer: Arc<wgpu::Buffer>,
    /// GPU index buffer (contents match host [`IndexBufferFormat`]).
    pub index_buffer: Arc<wgpu::Buffer>,
    /// Element size for `index_buffer` (`Uint16` vs `Uint32`).
    pub index_format: wgpu::IndexFormat,
    /// Total index elements across all submeshes.
    pub index_count: u32,
    /// Per-submesh `(first_index, index_count)` in elements of `index_format`.
    pub submeshes: Vec<(u32, u32)>,
    /// Vertex count from the host upload (used for deform and draw ranges).
    pub vertex_count: u32,
    /// Byte stride of one interleaved vertex in `vertex_buffer`.
    pub vertex_stride: u32,
    /// Axis-aligned bounds in mesh space (from host).
    pub bounds: RenderBoundingBox,
    /// Optional 1 byte per vertex (skinned / synthetic for blendshape-only).
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex joint indices as `vec4<u32>` (16 bytes / vertex) for skinning compute.
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex bone weights as `vec4<f32>` for skinning compute.
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    /// Column-major `float4x4` bind poses (64 bytes per bone).
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    /// Sparse packed position deltas (`vertex_index`, `delta.xyz`) for all shapes ([`crate::assets::mesh::BLENDSHAPE_SPARSE_ENTRY_SIZE`] bytes/entry).
    pub blendshape_sparse_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-shape `(first_entry, entry_count)` rows (`u32` pairs) mirroring [`Self::blendshape_sparse_ranges`].
    pub blendshape_shape_descriptor_buffer: Option<Arc<wgpu::Buffer>>,
    /// CPU copy of each shape’s sparse range for scatter dispatch (length equals [`Self::num_blendshapes`] when blendshapes are present).
    pub blendshape_sparse_ranges: Vec<(u32, u32)>,
    /// Number of logical blendshape slots (`max(blendshape_index)+1`).
    pub num_blendshapes: u32,
    /// Decomposed position stream (`vec4<f32>` per vertex) for compute + debug raster.
    pub positions_buffer: Option<Arc<wgpu::Buffer>>,
    /// Bind-pose normal stream (`vec4<f32>` per vertex; xyz used). Serves as the skinning compute
    /// input; the forward pass binds [`Self::deformed_normals_buffer`] when skinning is active.
    pub normals_buffer: Option<Arc<wgpu::Buffer>>,
    /// Blendshape output and/or skinning input ping buffer (`vec4<f32>` per vertex).
    pub deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    /// Skinning output positions (`vec4<f32>` per vertex).
    pub deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    /// Skinning output normals in world space (`vec4<f32>` per vertex; xyz used), inverse-transpose
    /// LBS of bind-pose normals. Present when [`Self::has_skeleton`].
    pub deformed_normals_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec2<f32>` UV0 stream (`8` bytes/vertex) for embedded raster materials; zeros when uv0 is absent.
    pub uv0_buffer: Option<Arc<wgpu::Buffer>>,
    /// `vec4<f32>` color stream for UI/text embedded materials; defaults to opaque white when absent.
    pub color_buffer: Option<Arc<wgpu::Buffer>>,
    /// True when the host uploaded a real skeleton (`bone_count > 0`).
    pub has_skeleton: bool,
    /// Unity [`Mesh.bindposes`](https://docs.unity3d.com/ScriptReference/Mesh-bindposes.html):
    /// inverse bind matrices (mesh space → bone bind space). Per-frame palette is
    /// `world_bone * skinning_bind_matrices[i]`.
    pub skinning_bind_matrices: Vec<Mat4>,
    /// Approximate VRAM (bytes), used by [`crate::resources::VramAccounting`].
    pub resident_bytes: u64,
}

/// Validates sparse blendshape GPU buffers and scatter ranges against a fresh [`extract_blendshape_offsets`] pass.
fn blendshape_and_deform_buffers_match_for_in_place(
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
        if sb.size() != sparse_expect.len() as u64 {
            return false;
        }
        let Some(db) = mesh.blendshape_shape_descriptor_buffer.as_ref() else {
            return false;
        };
        if db.size() != extracted.shape_descriptor_bytes.len() as u64 {
            return false;
        }
        if mesh.blendshape_sparse_ranges != extracted.shape_ranges {
            return false;
        }
        if mesh.num_blendshapes != n_blend {
            return false;
        }
    } else if mesh.num_blendshapes > 0
        || mesh.blendshape_sparse_buffer.is_some()
        || mesh.blendshape_shape_descriptor_buffer.is_some()
        || !mesh.blendshape_sparse_ranges.is_empty()
    {
        return false;
    }

    if mesh.num_blendshapes > 0 {
        let need = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
        if mesh.deform_temp_buffer.as_ref().map(|b| b.size()) != Some(need) {
            return false;
        }
    }
    if mesh.has_skeleton {
        let need = (data.vertex_count.max(0) as u64).saturating_mul(16).max(16);
        if mesh.deformed_positions_buffer.as_ref().map(|b| b.size()) != Some(need) {
            return false;
        }
        if mesh.deformed_normals_buffer.as_ref().map(|b| b.size()) != Some(need) {
            return false;
        }
    }
    true
}

/// Real skeleton (`bone_count > 0`): validates bone buffer sizes against `layout` / split weights.
fn compatible_for_in_place_real_skeleton(
    mesh: &GpuMesh,
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    raw: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
    vertex_slice: &[u8],
) -> bool {
    let bw =
        &raw[layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
    match split_bone_weights_tail_for_gpu(bw, vc_usize) {
        Some((ref ib, ref wb)) => {
            if mesh.bone_indices_buffer.as_ref().map(|b| b.size()) != Some(ib.len() as u64) {
                return false;
            }
            if mesh.bone_weights_vec4_buffer.as_ref().map(|b| b.size()) != Some(wb.len() as u64) {
                return false;
            }
        }
        None => {
            if mesh.bone_indices_buffer.is_some() || mesh.bone_weights_vec4_buffer.is_some() {
                return false;
            }
        }
    }
    if mesh.bone_counts_buffer.as_ref().map(|b| b.size()) != Some(layout.bone_counts_length as u64)
    {
        return false;
    }
    if mesh.bind_poses_buffer.as_ref().map(|b| b.size()) != Some(layout.bind_poses_length as u64) {
        return false;
    }
    if mesh.skinning_bind_matrices.len() != data.bone_count.max(0) as usize {
        return false;
    }
    derived_streams_compatible_for_in_place(mesh, vertex_slice, data, vc_usize, vertex_stride_us)
}

/// Blendshape-only synthetic bone layout: single bind pose + split indices/weights.
fn compatible_for_in_place_synthetic_blendshape_skeleton(
    mesh: &GpuMesh,
    data: &MeshUploadData,
    vertex_slice: &[u8],
    vc_usize: usize,
    vertex_stride_us: usize,
) -> bool {
    let (bind_poses_arr, bone_counts, bone_weights) =
        synthetic_bone_data_for_blendshape_only(data.vertex_count);
    if mesh.bone_counts_buffer.as_ref().map(|b| b.size()) != Some(bone_counts.len() as u64) {
        return false;
    }
    if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(&bone_weights, vc_usize) {
        if mesh.bone_indices_buffer.as_ref().map(|b| b.size()) != Some(ib.len() as u64) {
            return false;
        }
        if mesh.bone_weights_vec4_buffer.as_ref().map(|b| b.size()) != Some(wb.len() as u64) {
            return false;
        }
    } else {
        return false;
    }
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    if mesh.bind_poses_buffer.as_ref().map(|b| b.size()) != Some(bp_bytes.len() as u64) {
        return false;
    }
    if mesh.skinning_bind_matrices.len() != 1 {
        return false;
    }
    derived_streams_compatible_for_in_place(mesh, vertex_slice, data, vc_usize, vertex_stride_us)
}

/// Shared host layout and GPU mesh handles for in-place mesh buffer writes (VB, IB, bones).
struct MeshInPlaceWriteContext<'a> {
    mesh: &'a GpuMesh,
    queue: &'a wgpu::Queue,
    raw: &'a [u8],
    layout: &'a MeshBufferLayout,
    data: &'a MeshUploadData,
    vertex_count: usize,
    vertex_stride: usize,
}

/// Writes interleaved VB then optional derived position/normal/uv/color streams.
fn write_in_place_vertex_and_derived_streams(
    ctx: &MeshInPlaceWriteContext<'_>,
    write_vertex: bool,
) {
    if write_vertex {
        ctx.queue.write_buffer(
            ctx.mesh.vertex_buffer.as_ref(),
            0,
            &ctx.raw[..ctx.layout.vertex_size],
        );
    }
    let vertex_slice = &ctx.raw[..ctx.layout.vertex_size];
    if !write_vertex {
        return;
    }
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
        ctx.queue.write_buffer(pb.as_ref(), 0, pvec);
        ctx.queue.write_buffer(nb.as_ref(), 0, nvec);
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
        ctx.queue.write_buffer(uvb.as_ref(), 0, &uv);
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
        ctx.queue.write_buffer(cb.as_ref(), 0, &c);
    }
}

/// Writes index buffer slice when `write_ib` is set.
fn write_in_place_index_buffer(
    mesh: &GpuMesh,
    queue: &wgpu::Queue,
    raw: &[u8],
    layout: &MeshBufferLayout,
    write_ib: bool,
) {
    if !write_ib {
        return;
    }
    let ib_slice =
        &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
    queue.write_buffer(mesh.index_buffer.as_ref(), 0, ib_slice);
}

/// Writes bone/synthetic bone buffers from `raw` according to hint flags.
fn write_in_place_bone_buffers(
    ctx: &MeshInPlaceWriteContext<'_>,
    needs_bone_buffers: bool,
    synthetic_bones: bool,
    full: bool,
    write_bone_weights: bool,
    write_bind_poses: bool,
) -> Option<()> {
    if !needs_bone_buffers {
        return Some(());
    }
    if synthetic_bones && (full || write_bone_weights || write_bind_poses) {
        let (bind_poses_arr, bone_counts, bone_weights) =
            synthetic_bone_data_for_blendshape_only(ctx.data.vertex_count);
        if let Some(bc) = &ctx.mesh.bone_counts_buffer {
            ctx.queue.write_buffer(bc.as_ref(), 0, &bone_counts);
        }
        if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(&bone_weights, ctx.vertex_count) {
            if let Some(bi) = &ctx.mesh.bone_indices_buffer {
                ctx.queue.write_buffer(bi.as_ref(), 0, &ib);
            }
            if let Some(bwt) = &ctx.mesh.bone_weights_vec4_buffer {
                ctx.queue.write_buffer(bwt.as_ref(), 0, &wb);
            }
        }
        let bp_bytes: Vec<u8> = bind_poses_arr
            .iter()
            .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
            .collect();
        if let Some(bp) = &ctx.mesh.bind_poses_buffer {
            ctx.queue.write_buffer(bp.as_ref(), 0, &bp_bytes);
        }
    } else if ctx.data.bone_count > 0 {
        if full || write_bone_weights {
            let bc = &ctx.raw[ctx.layout.bone_counts_start
                ..ctx.layout.bone_counts_start + ctx.layout.bone_counts_length];
            let bw = &ctx.raw[ctx.layout.bone_weights_start
                ..ctx.layout.bone_weights_start + ctx.layout.bone_weights_length];
            if let Some(bcb) = &ctx.mesh.bone_counts_buffer {
                ctx.queue.write_buffer(bcb.as_ref(), 0, bc);
            }
            if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(bw, ctx.vertex_count) {
                if let Some(bi) = &ctx.mesh.bone_indices_buffer {
                    ctx.queue.write_buffer(bi.as_ref(), 0, &ib);
                }
                if let Some(bwt) = &ctx.mesh.bone_weights_vec4_buffer {
                    ctx.queue.write_buffer(bwt.as_ref(), 0, &wb);
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
                ctx.queue.write_buffer(bp.as_ref(), 0, &bp_bytes);
            }
        }
    }
    Some(())
}

/// Sparse blendshape GPU buffers and CPU ranges (`write_buffer` for both storage blobs).
fn write_in_place_blendshape_buffer(
    mesh: &GpuMesh,
    queue: &wgpu::Queue,
    raw: &[u8],
    layout: &MeshBufferLayout,
    data: &MeshUploadData,
    write_blend: bool,
) -> Option<()> {
    if !write_blend {
        return Some(());
    }
    let Some(sb) = mesh.blendshape_sparse_buffer.as_ref() else {
        return Some(());
    };
    let Some(db) = mesh.blendshape_shape_descriptor_buffer.as_ref() else {
        return Some(());
    };
    let extracted =
        extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count)?;
    let sparse = padded_sparse_bytes(&extracted.sparse_deltas);
    queue.write_buffer(sb.as_ref(), 0, &sparse);
    queue.write_buffer(db.as_ref(), 0, &extracted.shape_descriptor_bytes);
    Some(())
}

impl GpuMesh {
    /// Uploads mesh data from a raw byte slice covering at least `layout.total_buffer_length`.
    ///
    /// `raw` must be the mapping for `data.buffer` only for the duration of this call.
    pub fn upload(
        device: &wgpu::Device,
        gpu_limits: &GpuLimits,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
    ) -> Option<Self> {
        let max_buf = gpu_limits.max_buffer_size();
        if !validate_mesh_upload_layout(raw, data, layout, max_buf) {
            return None;
        }

        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();

        let core = create_core_vertex_index_buffers(device, raw, data, layout);
        let vc_usize = data.vertex_count.max(0) as usize;

        let derived = extract_derived_vertex_streams(device, raw, data, layout, &core);

        let bone_skin =
            upload_bone_and_skin_buffers(device, raw, data, layout, use_blendshapes, vc_usize)?;

        let blend_up = upload_blendshape_buffer(
            device,
            gpu_limits,
            raw,
            data,
            layout,
            use_blendshapes,
            max_buf,
        );
        let num_blendshapes = blend_up.num_blendshapes;

        let has_skeleton = data.bone_count > 0;
        let needs_blend_compute = num_blendshapes > 0;
        let needs_skin_compute = has_skeleton;

        let deform = allocate_deform_outputs(device, data, needs_blend_compute, needs_skin_compute);

        let submeshes = validated_submesh_ranges(&data.submeshes, core.index_count_u32);

        let resident_bytes = resident_bytes_for_mesh_upload(
            &core.vb,
            &core.ib,
            &derived,
            &bone_skin,
            &blend_up.sparse_buffer,
            &blend_up.shape_descriptor_buffer,
            &deform,
        );

        Some(Self {
            asset_id: data.asset_id,
            vertex_buffer: Arc::new(core.vb),
            index_buffer: Arc::new(core.ib),
            index_format: core.index_format,
            index_count: core.index_count_u32,
            submeshes,
            vertex_count: data.vertex_count.max(0) as u32,
            vertex_stride: core.vertex_stride,
            bounds: data.bounds,
            bone_counts_buffer: bone_skin.bone_counts_buffer,
            bone_indices_buffer: bone_skin.bone_indices_buffer,
            bone_weights_vec4_buffer: bone_skin.bone_weights_vec4_buffer,
            bind_poses_buffer: bone_skin.bind_poses_buffer,
            blendshape_sparse_buffer: blend_up.sparse_buffer,
            blendshape_shape_descriptor_buffer: blend_up.shape_descriptor_buffer,
            blendshape_sparse_ranges: blend_up.sparse_ranges,
            num_blendshapes,
            positions_buffer: derived.positions_buffer,
            normals_buffer: derived.normals_buffer,
            deform_temp_buffer: deform.deform_temp_buffer,
            deformed_positions_buffer: deform.deformed_positions_buffer,
            deformed_normals_buffer: deform.deformed_normals_buffer,
            uv0_buffer: derived.uv0_buffer,
            color_buffer: derived.color_buffer,
            has_skeleton,
            skinning_bind_matrices: bone_skin.skinning_bind_matrices,
            resident_bytes,
        })
    }

    /// `true` when [`Self::positions_buffer`] and [`Self::normals_buffer`] exist for the debug mesh path.
    pub fn debug_streams_ready(&self) -> bool {
        self.positions_buffer.is_some() && self.normals_buffer.is_some()
    }

    /// Whether `data`/`layout` match this mesh’s buffer sizes and optional derived streams so we can
    /// [`Self::write_in_place`] instead of allocating new buffers.
    pub(crate) fn compatible_for_in_place_update(
        &self,
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        raw: &[u8],
    ) -> bool {
        if raw.len() < layout.total_buffer_length {
            return false;
        }
        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();
        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let index_count = compute_index_count(&data.submeshes);
        let index_count_u32 = index_count.max(0) as u32;
        if self.vertex_stride != vertex_stride
            || self.vertex_count != data.vertex_count.max(0) as u32
            || self.index_count != index_count_u32
            || self.index_format != wgpu_index_format(data.index_buffer_format)
        {
            return false;
        }
        if self.vertex_buffer.size() != layout.vertex_size as u64
            || self.index_buffer.size() != layout.index_buffer_length as u64
        {
            return false;
        }

        let vc_usize = data.vertex_count.max(0) as usize;
        let vertex_stride_us = vertex_stride as usize;
        let vertex_slice = &raw[..layout.vertex_size];

        let needs_bone_buffers = data.bone_count > 0 || (use_blendshapes && data.vertex_count > 0);

        let no_gpu_bones = self.bone_counts_buffer.is_none()
            && self.bone_indices_buffer.is_none()
            && self.bone_weights_vec4_buffer.is_none()
            && self.bind_poses_buffer.is_none();
        let no_gpu_blend = self.blendshape_sparse_buffer.is_none()
            && self.blendshape_shape_descriptor_buffer.is_none()
            && self.num_blendshapes == 0
            && self.blendshape_sparse_ranges.is_empty();

        let data_static = data.bone_count == 0 && !use_blendshapes;
        let gpu_static =
            !self.has_skeleton && self.num_blendshapes == 0 && no_gpu_bones && no_gpu_blend;

        if data_static && gpu_static {
            return derived_streams_compatible_for_in_place(
                self,
                vertex_slice,
                data,
                vc_usize,
                vertex_stride_us,
            );
        }

        if self.has_skeleton != (data.bone_count > 0) {
            return false;
        }

        if !blendshape_and_deform_buffers_match_for_in_place(
            self,
            data,
            layout,
            raw,
            use_blendshapes,
        ) {
            return false;
        }

        if !needs_bone_buffers {
            if self.bone_counts_buffer.is_some()
                || self.bind_poses_buffer.is_some()
                || self.bone_indices_buffer.is_some()
                || self.bone_weights_vec4_buffer.is_some()
            {
                return false;
            }
            return derived_streams_compatible_for_in_place(
                self,
                vertex_slice,
                data,
                vc_usize,
                vertex_stride_us,
            );
        }

        if data.bone_count > 0 {
            return compatible_for_in_place_real_skeleton(
                self,
                data,
                layout,
                raw,
                vc_usize,
                vertex_stride_us,
                vertex_slice,
            );
        }

        if use_blendshapes && data.vertex_count > 0 {
            return compatible_for_in_place_synthetic_blendshape_skeleton(
                self,
                data,
                vertex_slice,
                vc_usize,
                vertex_stride_us,
            );
        }

        false
    }

    /// Overwrites vertex, index, and optional bone/blendshape/derived stream data using
    /// [`wgpu::Queue::write_buffer`], honoring [`MeshUploadHintFlag`] when set (otherwise full writes).
    pub(crate) fn write_in_place(
        &self,
        queue: &wgpu::Queue,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
        hint: MeshUploadHintFlag,
    ) -> Option<GpuMesh> {
        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let vc_usize = data.vertex_count.max(0) as usize;
        let vertex_stride_us = vertex_stride as usize;

        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();
        let needs_bone_buffers = data.bone_count > 0 || (use_blendshapes && data.vertex_count > 0);
        let synthetic_bones = data.bone_count == 0 && use_blendshapes && data.vertex_count > 0;

        let full = !mesh_upload_hint_any_selective(hint);
        let write_vertex = full || mesh_upload_hint_touches_vertex_streams(hint);
        let write_ib = full || hint.geometry();
        let write_bone_weights = full || hint.bone_weights();
        let write_bind_poses = full || hint.bind_poses();
        let write_blend = full || hint.blendshapes();

        let want_submeshes = validated_submesh_ranges(&data.submeshes, self.index_count);

        write_in_place_vertex_and_derived_streams(
            &MeshInPlaceWriteContext {
                mesh: self,
                queue,
                raw,
                layout,
                data,
                vertex_count: vc_usize,
                vertex_stride: vertex_stride_us,
            },
            write_vertex,
        );
        write_in_place_index_buffer(self, queue, raw, layout, write_ib);
        write_in_place_bone_buffers(
            &MeshInPlaceWriteContext {
                mesh: self,
                queue,
                raw,
                layout,
                data,
                vertex_count: vc_usize,
                vertex_stride: vertex_stride_us,
            },
            needs_bone_buffers,
            synthetic_bones,
            full,
            write_bone_weights,
            write_bind_poses,
        )?;
        write_in_place_blendshape_buffer(self, queue, raw, layout, data, write_blend)?;

        let mut skinning = self.skinning_bind_matrices.clone();
        if data.bone_count > 0 && (full || write_bind_poses) {
            let bp_raw =
                &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
            if let Some(arr) = extract_bind_poses(bp_raw, data.bone_count as usize) {
                skinning = arr.iter().map(Mat4::from_cols_array_2d).collect();
            }
        } else if synthetic_bones && (full || write_bone_weights || write_bind_poses) {
            let (bind_poses_arr, _, _) = synthetic_bone_data_for_blendshape_only(data.vertex_count);
            skinning = bind_poses_arr
                .iter()
                .map(Mat4::from_cols_array_2d)
                .collect();
        }

        Some(Self {
            asset_id: self.asset_id,
            vertex_buffer: Arc::clone(&self.vertex_buffer),
            index_buffer: Arc::clone(&self.index_buffer),
            index_format: self.index_format,
            index_count: self.index_count,
            submeshes: want_submeshes,
            vertex_count: self.vertex_count,
            vertex_stride: self.vertex_stride,
            bounds: data.bounds,
            bone_counts_buffer: self.bone_counts_buffer.clone(),
            bone_indices_buffer: self.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: self.bone_weights_vec4_buffer.clone(),
            bind_poses_buffer: self.bind_poses_buffer.clone(),
            blendshape_sparse_buffer: self.blendshape_sparse_buffer.clone(),
            blendshape_shape_descriptor_buffer: self.blendshape_shape_descriptor_buffer.clone(),
            blendshape_sparse_ranges: self.blendshape_sparse_ranges.clone(),
            num_blendshapes: self.num_blendshapes,
            positions_buffer: self.positions_buffer.clone(),
            normals_buffer: self.normals_buffer.clone(),
            deform_temp_buffer: self.deform_temp_buffer.clone(),
            deformed_positions_buffer: self.deformed_positions_buffer.clone(),
            deformed_normals_buffer: self.deformed_normals_buffer.clone(),
            uv0_buffer: self.uv0_buffer.clone(),
            color_buffer: self.color_buffer.clone(),
            has_skeleton: self.has_skeleton,
            skinning_bind_matrices: skinning,
            resident_bytes: self.resident_bytes,
        })
    }
}
