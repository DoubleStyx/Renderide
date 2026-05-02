//! Linear blend skinning compute encoding.
//!
//! Builds the bone palette for the work item, uploads it into the frame-global slab, then
//! issues a single skinning dispatch that consumes the deformed positions / normals.

use std::num::NonZeroU64;

use glam::Mat4;

use crate::mesh_deform::advance_slab_cursor;
use crate::mesh_deform::{Range, SkinCacheEntry};
use crate::mesh_deform::{SkinningPaletteParams, write_skinning_palette_bytes};
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
    pub tangents_arena: &'a wgpu::Buffer,
    pub temp_arena: &'a wgpu::Buffer,
    pub skin_dispatch_cursor: &'b mut u64,
}

/// Skinning `SkinDispatchParams.flags` bit for writing tangents.
const SKIN_DISPATCH_APPLY_TANGENTS: u32 = 1;

/// Linear blend skinning compute after optional blendshape pass.
pub(super) fn record_skinning_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    ctx: SkinningDeformContext<'_, '_>,
) -> bool {
    profiling::scope!("mesh_deform::record_skinning");
    let Some(required) = required_skinning_inputs(&ctx) else {
        return false;
    };
    let Some((bone_binding_size, palette_len)) =
        upload_skinning_palette(gpu, &ctx, required.bone_transform_indices)
    else {
        return false;
    };

    let (src_for_skin, base_src_pos_e) = if ctx.needs_blend {
        let Some(t) = ctx.cache_entry.temp.as_ref() else {
            return false;
        };
        (ctx.temp_arena, t.first_element_index(16))
    } else {
        (required.positions, 0u32)
    };
    let (src_n_for_skin, base_src_nrm_e) =
        if let Some(temp_normals) = ctx.cache_entry.temp_normals.as_ref() {
            (ctx.temp_arena, temp_normals.first_element_index(16))
        } else {
            (required.src_n, 0u32)
        };
    let tangent_dispatch = resolve_tangent_dispatch(
        ctx.mesh,
        ctx.cache_entry,
        ctx.tangents_arena,
        ctx.temp_arena,
    );

    let skin_params = pack_skin_dispatch_params(SkinDispatchParamFields {
        vertex_count: ctx.mesh.vertex_count,
        base_src_pos_e,
        base_src_nrm_e,
        base_src_tan_e: tangent_dispatch.base_src_tan_e,
        base_dst_pos_e: ctx.cache_entry.positions.first_element_index(16),
        base_dst_nrm_e: required.nrm_range.first_element_index(16),
        base_dst_tan_e: tangent_dispatch.base_dst_tan_e,
        flags: tangent_dispatch.flags,
    });
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
        bone_idx: required.bone_idx,
        bone_wt: required.bone_wt,
        dst_pos: ctx.positions_arena,
        src_n: src_n_for_skin,
        dst_n: ctx.normals_arena,
        src_tangent: tangent_dispatch.src_buffer,
        dst_tangent: tangent_dispatch.dst_buffer,
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

struct RequiredSkinningInputs<'a> {
    positions: &'a wgpu::Buffer,
    src_n: &'a wgpu::Buffer,
    bone_idx: &'a wgpu::Buffer,
    bone_wt: &'a wgpu::Buffer,
    bone_transform_indices: &'a [i32],
    nrm_range: &'a Range,
}

fn required_skinning_inputs<'a>(
    ctx: &'a SkinningDeformContext<'_, '_>,
) -> Option<RequiredSkinningInputs<'a>> {
    Some(RequiredSkinningInputs {
        positions: ctx.mesh.positions_buffer.as_ref()?.as_ref(),
        src_n: ctx.mesh.normals_buffer.as_ref()?.as_ref(),
        bone_idx: ctx.mesh.bone_indices_buffer.as_ref()?.as_ref(),
        bone_wt: ctx.mesh.bone_weights_vec4_buffer.as_ref()?.as_ref(),
        bone_transform_indices: ctx.bone_transform_indices?,
        nrm_range: ctx.cache_entry.normals.as_ref()?,
    })
}

fn upload_skinning_palette(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    ctx: &SkinningDeformContext<'_, '_>,
    bone_transform_indices: &[i32],
) -> Option<(NonZeroU64, u64)> {
    let bone_count_u = ctx.mesh.skinning_bind_matrices.len() as u32;
    gpu.scratch.ensure_bone_capacity(gpu.device, bone_count_u);
    write_skinning_palette_bytes(
        SkinningPaletteParams {
            scene: ctx.scene,
            space_id: ctx.space_id,
            skinning_bind_matrices: &ctx.mesh.skinning_bind_matrices,
            has_skeleton: ctx.mesh.has_skeleton,
            bone_transform_indices,
            smr_node_id: ctx.smr_node_id,
            render_context: ctx.render_context,
            head_output_transform: ctx.head_output_transform,
        },
        &mut gpu.scratch.bone_palette_bytes,
    )?;

    let palette_len = gpu.scratch.bone_palette_bytes.len() as u64;
    gpu.scratch
        .ensure_bone_byte_capacity(gpu.device, ctx.bone_cursor.saturating_add(palette_len));
    gpu.upload_batch.write_buffer(
        &gpu.scratch.bone_matrices,
        *ctx.bone_cursor,
        gpu.scratch.bone_palette_bytes.as_slice(),
    );
    NonZeroU64::new(palette_len).map(|binding_size| (binding_size, palette_len))
}

struct TangentSkinDispatch<'a> {
    src_buffer: Option<&'a wgpu::Buffer>,
    dst_buffer: Option<&'a wgpu::Buffer>,
    base_src_tan_e: u32,
    base_dst_tan_e: u32,
    flags: u32,
}

fn resolve_tangent_dispatch<'a>(
    mesh: &'a MeshDeformSnapshot,
    cache_entry: &'a SkinCacheEntry,
    tangents_arena: &'a wgpu::Buffer,
    temp_arena: &'a wgpu::Buffer,
) -> TangentSkinDispatch<'a> {
    let Some(dst_range) = cache_entry.tangents.as_ref() else {
        return TangentSkinDispatch {
            src_buffer: None,
            dst_buffer: None,
            base_src_tan_e: 0,
            base_dst_tan_e: 0,
            flags: 0,
        };
    };
    let Some(base_tangent_buffer) = mesh.tangent_buffer.as_ref() else {
        return TangentSkinDispatch {
            src_buffer: None,
            dst_buffer: None,
            base_src_tan_e: 0,
            base_dst_tan_e: 0,
            flags: 0,
        };
    };
    let (src_buffer, base_src_tan_e) =
        if let Some(temp_tangents) = cache_entry.temp_tangents.as_ref() {
            (temp_arena, temp_tangents.first_element_index(16))
        } else {
            (base_tangent_buffer.as_ref(), 0)
        };
    TangentSkinDispatch {
        src_buffer: Some(src_buffer),
        dst_buffer: Some(tangents_arena),
        base_src_tan_e,
        base_dst_tan_e: dst_range.first_element_index(16),
        flags: SKIN_DISPATCH_APPLY_TANGENTS,
    }
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
    src_tangent: Option<&'a wgpu::Buffer>,
    dst_tangent: Option<&'a wgpu::Buffer>,
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
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dispatch
                        .src_tangent
                        .unwrap_or(&dispatch.scratch.dummy_vec4)
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: dispatch
                        .dst_tangent
                        .unwrap_or(&dispatch.scratch.dummy_vec4)
                        .as_entire_binding(),
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

/// CPU-side field layout for `mesh_skinning.wgsl` `SkinDispatchParams`.
struct SkinDispatchParamFields {
    vertex_count: u32,
    base_src_pos_e: u32,
    base_src_nrm_e: u32,
    base_src_tan_e: u32,
    base_dst_pos_e: u32,
    base_dst_nrm_e: u32,
    base_dst_tan_e: u32,
    flags: u32,
}

/// `shaders/passes/compute/mesh_skinning.wgsl` `SkinDispatchParams` (32 bytes).
fn pack_skin_dispatch_params(fields: SkinDispatchParamFields) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&fields.vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&fields.base_src_pos_e.to_le_bytes());
    o[8..12].copy_from_slice(&fields.base_src_nrm_e.to_le_bytes());
    o[12..16].copy_from_slice(&fields.base_src_tan_e.to_le_bytes());
    o[16..20].copy_from_slice(&fields.base_dst_pos_e.to_le_bytes());
    o[20..24].copy_from_slice(&fields.base_dst_nrm_e.to_le_bytes());
    o[24..28].copy_from_slice(&fields.base_dst_tan_e.to_le_bytes());
    o[28..32].copy_from_slice(&fields.flags.to_le_bytes());
    o
}
