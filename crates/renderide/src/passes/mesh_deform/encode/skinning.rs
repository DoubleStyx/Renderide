//! Linear blend skinning compute encoding.
//!
//! Builds the bone palette for the work item, uploads it into the frame-global slab, then
//! issues a single skinning dispatch that consumes the deformed positions / normals.

use std::num::NonZeroU64;

use glam::Mat4;

use crate::mesh_deform::SkinCacheEntry;
use crate::mesh_deform::advance_slab_cursor;
use crate::mesh_deform::{SkinningPaletteParams, build_skinning_palette};
use crate::scene::RenderSpaceId;

use super::super::snapshot::MeshDeformSnapshot;
use super::MeshDeformEncodeGpu;

/// Skinning path inputs after blendshape (optional) has run.
pub(super) struct SkinningDeformContext<'a, 'b> {
    pub scene: &'a crate::scene::SceneCoordinator,
    pub space_id: RenderSpaceId,
    pub mesh: &'a MeshDeformSnapshot,
    pub bone_transform_indices: Option<&'a [i32]>,
    pub smr_node_id: i32,
    pub render_context: crate::shared::RenderingContext,
    pub head_output_transform: Mat4,
    pub bone_cursor: &'b mut u64,
    pub needs_blend: bool,
    pub wg: u32,
    pub cache_entry: &'a SkinCacheEntry,
    pub positions_arena: &'a wgpu::Buffer,
    pub normals_arena: &'a wgpu::Buffer,
    pub temp_arena: &'a wgpu::Buffer,
    pub skin_dispatch_cursor: &'b mut u64,
}

/// Linear blend skinning compute after optional blendshape pass.
pub(super) fn record_skinning_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    ctx: SkinningDeformContext<'_, '_>,
) -> bool {
    profiling::scope!("mesh_deform::record_skinning");
    let Some(ref positions) = ctx.mesh.positions_buffer else {
        return false;
    };
    let Some(ref src_n) = ctx.mesh.normals_buffer else {
        return false;
    };
    let Some(ref bone_idx) = ctx.mesh.bone_indices_buffer else {
        return false;
    };
    let Some(ref bone_wt) = ctx.mesh.bone_weights_vec4_buffer else {
        return false;
    };
    let Some(indices) = ctx.bone_transform_indices else {
        return false;
    };
    let Some(nrm_range) = ctx.cache_entry.normals.as_ref() else {
        return false;
    };

    let bone_count_u = ctx.mesh.skinning_bind_matrices.len() as u32;
    gpu.scratch.ensure_bone_capacity(gpu.device, bone_count_u);
    let Some(palette_mats) = build_skinning_palette(SkinningPaletteParams {
        scene: ctx.scene,
        space_id: ctx.space_id,
        skinning_bind_matrices: &ctx.mesh.skinning_bind_matrices,
        has_skeleton: ctx.mesh.has_skeleton,
        bone_transform_indices: indices,
        smr_node_id: ctx.smr_node_id,
        render_context: ctx.render_context,
        head_output_transform: ctx.head_output_transform,
    }) else {
        return false;
    };
    let mut palette: Vec<u8> = vec![0u8; palette_mats.len() * 64];
    for (bi, pal) in palette_mats.iter().enumerate() {
        let cols = pal.to_cols_array();
        palette[bi * 64..bi * 64 + 64].copy_from_slice(bytemuck::cast_slice(&cols));
    }

    let palette_len = palette.len() as u64;
    gpu.scratch
        .ensure_bone_byte_capacity(gpu.device, ctx.bone_cursor.saturating_add(palette_len));
    gpu.upload_batch
        .write_buffer(&gpu.scratch.bone_matrices, *ctx.bone_cursor, &palette);

    let Some(bone_binding_size) = NonZeroU64::new(palette_len) else {
        return false;
    };

    let (src_for_skin, base_src_pos_e) = if ctx.needs_blend {
        let Some(t) = ctx.cache_entry.temp.as_ref() else {
            return false;
        };
        (ctx.temp_arena, t.first_element_index(16))
    } else {
        (positions.as_ref(), 0u32)
    };

    let skin_params = pack_skin_dispatch_params(
        ctx.mesh.vertex_count,
        base_src_pos_e,
        0,
        ctx.cache_entry.positions.first_element_index(16),
        nrm_range.first_element_index(16),
    );
    let sd_cursor = *ctx.skin_dispatch_cursor;
    gpu.scratch
        .ensure_skin_dispatch_byte_capacity(gpu.device, sd_cursor.saturating_add(32));
    gpu.upload_batch
        .write_buffer(&gpu.scratch.skin_dispatch, sd_cursor, &skin_params);

    skinning_dispatch_with_uploaded_palette(SkinningPaletteDispatch {
        device: gpu.device,
        encoder: gpu.encoder,
        pre: gpu.pre,
        scratch: gpu.scratch,
        src_positions: src_for_skin,
        bone_idx,
        bone_wt,
        dst_pos: ctx.positions_arena,
        src_n: src_n.as_ref(),
        dst_n: ctx.normals_arena,
        bone_cursor: *ctx.bone_cursor,
        bone_binding_size,
        wg: ctx.wg,
        skin_dispatch_offset: sd_cursor,
        profiler: gpu.profiler,
    });

    *ctx.bone_cursor = advance_slab_cursor(*ctx.bone_cursor, palette_len);
    *ctx.skin_dispatch_cursor = advance_slab_cursor(sd_cursor, 32);
    true
}

/// Buffers and offsets for one skinning dispatch after the bone palette is uploaded to `scratch`.
struct SkinningPaletteDispatch<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut wgpu::CommandEncoder,
    pre: &'a crate::mesh_deform::MeshPreprocessPipelines,
    scratch: &'a crate::mesh_deform::MeshDeformScratch,
    src_positions: &'a wgpu::Buffer,
    bone_idx: &'a wgpu::Buffer,
    bone_wt: &'a wgpu::Buffer,
    dst_pos: &'a wgpu::Buffer,
    src_n: &'a wgpu::Buffer,
    dst_n: &'a wgpu::Buffer,
    bone_cursor: u64,
    bone_binding_size: NonZeroU64,
    wg: u32,
    /// Byte offset into [`crate::mesh_deform::MeshDeformScratch::skin_dispatch`] for this dispatch's `SkinDispatchParams`.
    skin_dispatch_offset: u64,
    /// GPU profiler for the pass-level timestamp query on the skinning compute pass.
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Builds skinning bind group (bone slab + attributes) and dispatches the skinning shader.
fn skinning_dispatch_with_uploaded_palette(dispatch: SkinningPaletteDispatch<'_>) {
    let Some(skin_u_size) = NonZeroU64::new(32) else {
        return;
    };
    let skin_bg = dispatch
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bg"),
            layout: &dispatch.pre.skinning_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dispatch.scratch.bone_matrices,
                        offset: dispatch.bone_cursor,
                        size: Some(dispatch.bone_binding_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dispatch.src_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dispatch.bone_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dispatch.bone_wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dispatch.dst_pos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dispatch.src_n.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dispatch.dst_n.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dispatch.scratch.skin_dispatch,
                        offset: dispatch.skin_dispatch_offset,
                        size: Some(skin_u_size),
                    }),
                },
            ],
        });

    let pass_query = dispatch
        .profiler
        .map(|p| p.begin_pass_query("skinning", dispatch.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut cpass = dispatch
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skinning"),
                timestamp_writes,
            });
        cpass.set_pipeline(&dispatch.pre.skinning_pipeline);
        cpass.set_bind_group(0, &skin_bg, &[]);
        cpass.dispatch_workgroups(dispatch.wg, 1, 1);
    };
    if let (Some(p), Some(q)) = (dispatch.profiler, pass_query) {
        p.end_query(dispatch.encoder, q);
    }
}

/// `shaders/passes/compute/mesh_skinning.wgsl` `SkinDispatchParams` (32 bytes).
fn pack_skin_dispatch_params(
    vertex_count: u32,
    base_src_pos_e: u32,
    base_src_nrm_e: u32,
    base_dst_pos_e: u32,
    base_dst_nrm_e: u32,
) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&base_src_pos_e.to_le_bytes());
    o[8..12].copy_from_slice(&base_src_nrm_e.to_le_bytes());
    o[12..16].copy_from_slice(&base_dst_pos_e.to_le_bytes());
    o[16..20].copy_from_slice(&base_dst_nrm_e.to_le_bytes());
    o
}
