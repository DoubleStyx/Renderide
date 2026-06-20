//! Linear blend skinning compute encoding.
//!
//! Builds the bone palette for the work item, uploads it into the frame-global slab, then
//! issues a single skinning dispatch that consumes the deformed positions / normals.

use std::num::NonZeroU64;
use std::sync::Arc;

use glam::Mat4;

use crate::mesh_deform::{Range, SkinCacheEntry, SkinningBindGroupKey};
use crate::mesh_deform::{SkinningPaletteParams, write_skinning_palette_bytes};
use crate::mesh_deform::{advance_slab_cursor, buffer_identity};
use crate::scene::{RenderSpaceId, SceneTransformRead};
use crate::shared::SkinWeightMode;

use super::super::snapshot::MeshDeformSnapshot;
use super::{MeshDeformEncodeGpu, MeshDeformRecordStats};

const SKIN_DISPATCH_PARAM_BYTES: u64 = 48;

/// Scene inputs needed to build one skinning palette.
pub(super) struct SkinningPaletteBuildContext<'a, S: SceneTransformRead + ?Sized> {
    pub scene: &'a S,
    pub space_id: RenderSpaceId,
    pub mesh: &'a MeshDeformSnapshot,
    pub bone_transform_indices: Option<&'a [i32]>,
    pub smr_node_id: i32,
    pub render_context: crate::shared::RenderingContext,
    pub head_output_transform: Mat4,
}

/// Skinning path inputs after blendshape (optional) has run.
pub(super) struct SkinningDeformContext<'a, 'b> {
    pub mesh: &'a MeshDeformSnapshot,
    pub bone_cursor: &'b mut u64,
    pub needs_blend: bool,
    pub wg: u32,
    pub cache_entry: &'a SkinCacheEntry,
    pub positions_arena: &'a wgpu::Buffer,
    pub normals_arena: &'a wgpu::Buffer,
    pub tangents_arena: &'a wgpu::Buffer,
    pub temp_arena: &'a wgpu::Buffer,
    pub skin_dispatch_cursor: &'b mut u64,
    pub prepared_palette_len: u64,
    pub skin_weight_mode: SkinWeightMode,
}

/// Skinning `SkinDispatchParams.flags` bit for writing tangents.
const SKIN_DISPATCH_APPLY_TANGENTS: u32 = 1;

/// One skinning dispatch inside the frame batch.
pub(super) struct SkinningDispatchJob {
    bind_group: Arc<wgpu::BindGroup>,
    params_offset: u32,
    wg: u32,
}

/// Builds one palette into scratch CPU bytes so the caller can hash it before deciding to skip.
pub(super) fn prepare_skinning_palette_bytes<S>(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    ctx: SkinningPaletteBuildContext<'_, S>,
) -> Option<u64>
where
    S: SceneTransformRead + Sync + ?Sized,
{
    profiling::scope!("mesh_deform::skinning_palette");
    let bone_transform_indices = ctx.bone_transform_indices?;
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
    Some(gpu.scratch.bone_palette_bytes.len() as u64)
}

/// Linear blend skinning compute after optional blendshape pass.
pub(super) fn record_skinning_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    ctx: SkinningDeformContext<'_, '_>,
    jobs: &mut Vec<SkinningDispatchJob>,
) -> MeshDeformRecordStats {
    profiling::scope!("mesh_deform::record_skinning");
    let mut stats = MeshDeformRecordStats::default();
    let Some(base_bone_e) =
        upload_prepared_skinning_palette(gpu, *ctx.bone_cursor, ctx.prepared_palette_len)
    else {
        return stats;
    };
    let Some(required) = required_skinning_inputs(&ctx) else {
        return stats;
    };

    let (src_for_skin, base_src_pos_e) = if ctx.needs_blend {
        let Some(t) = ctx.cache_entry.temp.as_ref() else {
            return stats;
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
        base_bone_e,
        base_src_pos_e,
        base_src_nrm_e,
        base_src_tan_e: tangent_dispatch.base_src_tan_e,
        base_dst_pos_e: ctx.cache_entry.positions.first_element_index(16),
        base_dst_nrm_e: required.nrm_range.first_element_index(16),
        base_dst_tan_e: tangent_dispatch.base_dst_tan_e,
        flags: tangent_dispatch.flags,
        skin_weight_limit: skin_weight_limit(ctx.skin_weight_mode),
    });
    let sd_cursor = *ctx.skin_dispatch_cursor;
    let Some(params_offset) = dynamic_uniform_offset(sd_cursor) else {
        return stats;
    };
    gpu.scratch.ensure_skin_dispatch_byte_capacity(
        gpu.device,
        sd_cursor.saturating_add(SKIN_DISPATCH_PARAM_BYTES),
    );
    gpu.uploads
        .write_buffer(&gpu.scratch.skin_dispatch, sd_cursor, &skin_params);

    let (bind_group, reused) = skinning_bind_group(
        gpu,
        SkinningPaletteDispatch {
            src_positions: src_for_skin,
            bone_idx: required.bone_idx,
            bone_wt: required.bone_wt,
            bone_influence_offsets: required.bone_influence_offsets,
            bone_influences: required.bone_influences,
            dst_pos: ctx.positions_arena,
            src_n: src_n_for_skin,
            dst_n: ctx.normals_arena,
            src_tangent: tangent_dispatch.src_buffer,
            dst_tangent: tangent_dispatch.dst_buffer,
        },
    );
    if reused {
        stats.bind_group_cache_reuses = stats.bind_group_cache_reuses.saturating_add(1);
    } else {
        stats.bind_groups_created = stats.bind_groups_created.saturating_add(1);
    }
    jobs.push(SkinningDispatchJob {
        bind_group,
        params_offset,
        wg: ctx.wg,
    });

    *ctx.bone_cursor = advance_slab_cursor(*ctx.bone_cursor, ctx.prepared_palette_len);
    *ctx.skin_dispatch_cursor = advance_slab_cursor(sd_cursor, SKIN_DISPATCH_PARAM_BYTES);
    stats.skin_dispatches = stats.skin_dispatches.saturating_add(1);
    stats
}

struct RequiredSkinningInputs<'a> {
    positions: &'a wgpu::Buffer,
    src_n: &'a wgpu::Buffer,
    bone_idx: &'a wgpu::Buffer,
    bone_wt: &'a wgpu::Buffer,
    bone_influence_offsets: &'a wgpu::Buffer,
    bone_influences: &'a wgpu::Buffer,
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
        bone_influence_offsets: ctx.mesh.bone_influence_offsets_buffer.as_ref()?.as_ref(),
        bone_influences: ctx.mesh.bone_influences_buffer.as_ref()?.as_ref(),
        nrm_range: ctx.cache_entry.normals.as_ref()?,
    })
}

fn skin_weight_limit(mode: SkinWeightMode) -> u32 {
    match mode {
        SkinWeightMode::OneBone => 1,
        SkinWeightMode::TwoBones => 2,
        SkinWeightMode::FourBones => 4,
        SkinWeightMode::Unlimited => 0,
    }
}

fn upload_prepared_skinning_palette(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    bone_cursor: u64,
    palette_len: u64,
) -> Option<u32> {
    gpu.scratch
        .ensure_bone_byte_capacity(gpu.device, bone_cursor.saturating_add(palette_len));
    gpu.uploads.write_buffer(
        &gpu.scratch.bone_matrices,
        bone_cursor,
        gpu.scratch.bone_palette_bytes.as_slice(),
    );
    let base_bone_e = bone_cursor.checked_div(64)?;
    let base_bone_e = u32::try_from(base_bone_e).ok();
    if base_bone_e.is_none() {
        logger::warn!("mesh deform: bone palette offset exceeded skinning shader range");
    }
    base_bone_e
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
    src_positions: &'a wgpu::Buffer,
    bone_idx: &'a wgpu::Buffer,
    bone_wt: &'a wgpu::Buffer,
    bone_influence_offsets: &'a wgpu::Buffer,
    bone_influences: &'a wgpu::Buffer,
    dst_pos: &'a wgpu::Buffer,
    src_n: &'a wgpu::Buffer,
    dst_n: &'a wgpu::Buffer,
    src_tangent: Option<&'a wgpu::Buffer>,
    dst_tangent: Option<&'a wgpu::Buffer>,
}

/// Builds or reuses a skinning bind group.
fn skinning_bind_group(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    dispatch: SkinningPaletteDispatch<'_>,
) -> (Arc<wgpu::BindGroup>, bool) {
    let src_tangent = dispatch.src_tangent.unwrap_or(&gpu.scratch.dummy_vec4_read);
    let dst_tangent = dispatch
        .dst_tangent
        .unwrap_or(&gpu.scratch.dummy_vec4_write);
    let key = SkinningBindGroupKey {
        scratch_generation: gpu.scratch.resource_generation(),
        src_positions: buffer_identity(dispatch.src_positions),
        bone_indices: buffer_identity(dispatch.bone_idx),
        bone_weights: buffer_identity(dispatch.bone_wt),
        bone_influence_offsets: buffer_identity(dispatch.bone_influence_offsets),
        bone_influences: buffer_identity(dispatch.bone_influences),
        dst_positions: buffer_identity(dispatch.dst_pos),
        src_normals: buffer_identity(dispatch.src_n),
        dst_normals: buffer_identity(dispatch.dst_n),
        src_tangents: buffer_identity(src_tangent),
        dst_tangents: buffer_identity(dst_tangent),
    };
    if let Some(bind_group) = gpu.scratch.skinning_bind_group(key) {
        return (bind_group, true);
    }
    let skin_u_size = NonZeroU64::new(SKIN_DISPATCH_PARAM_BYTES).unwrap_or(NonZeroU64::MIN);
    let bind_group = {
        profiling::scope!("mesh_deform::skinning_create_bg");
        Arc::new(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bg"),
            layout: &gpu.pre.skinning_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.scratch.bone_matrices.as_entire_binding(),
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
                        buffer: &gpu.scratch.skin_dispatch,
                        offset: 0,
                        size: Some(skin_u_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: src_tangent.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: dst_tangent.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: dispatch.bone_influence_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dispatch.bone_influences.as_entire_binding(),
                },
            ],
        }))
    };
    crate::profiling::note_resource_churn!(BindGroup, "mesh_deform::skinning_bind_group");
    gpu.scratch
        .insert_skinning_bind_group(key, Arc::clone(&bind_group));
    (bind_group, false)
}

fn dynamic_uniform_offset(offset: u64) -> Option<u32> {
    let offset = u32::try_from(offset).ok();
    if offset.is_none() {
        logger::warn!("mesh deform: skinning param offset exceeded WebGPU dynamic-offset range");
    }
    offset
}

/// Dispatches all queued skinning jobs in a single compute pass.
pub(super) fn flush_skinning_jobs(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    jobs: &[SkinningDispatchJob],
) -> MeshDeformRecordStats {
    let mut stats = MeshDeformRecordStats::default();
    if jobs.is_empty() {
        return stats;
    }
    let pass_query = gpu
        .profiler
        .map(|p| p.begin_pass_query("skinning_batch", gpu.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        profiling::scope!("mesh_deform::skinning_dispatch_batch");
        let mut cpass = gpu
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("skinning_batch"),
                timestamp_writes,
            });
        cpass.set_pipeline(&gpu.pre.skinning_pipeline);
        for job in jobs {
            cpass.set_bind_group(0, job.bind_group.as_ref(), &[job.params_offset]);
            cpass.dispatch_workgroups(job.wg, 1, 1);
        }
    }
    stats.compute_passes = 1;
    if let (Some(p), Some(q)) = (gpu.profiler, pass_query) {
        p.end_query(gpu.encoder, q);
    }
    stats
}

/// CPU-side field layout for `mesh_skinning.wgsl` `SkinDispatchParams`.
struct SkinDispatchParamFields {
    vertex_count: u32,
    base_bone_e: u32,
    base_src_pos_e: u32,
    base_src_nrm_e: u32,
    base_src_tan_e: u32,
    base_dst_pos_e: u32,
    base_dst_nrm_e: u32,
    base_dst_tan_e: u32,
    flags: u32,
    skin_weight_limit: u32,
}

/// `shaders/passes/compute/mesh_skinning.wgsl` `SkinDispatchParams` (48 bytes).
fn pack_skin_dispatch_params(fields: SkinDispatchParamFields) -> [u8; 48] {
    let mut o = [0u8; 48];
    o[0..4].copy_from_slice(&fields.vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&fields.base_bone_e.to_le_bytes());
    o[8..12].copy_from_slice(&fields.base_src_pos_e.to_le_bytes());
    o[12..16].copy_from_slice(&fields.base_src_nrm_e.to_le_bytes());
    o[16..20].copy_from_slice(&fields.base_src_tan_e.to_le_bytes());
    o[20..24].copy_from_slice(&fields.base_dst_pos_e.to_le_bytes());
    o[24..28].copy_from_slice(&fields.base_dst_nrm_e.to_le_bytes());
    o[28..32].copy_from_slice(&fields.base_dst_tan_e.to_le_bytes());
    o[32..36].copy_from_slice(&fields.flags.to_le_bytes());
    o[36..40].copy_from_slice(&fields.skin_weight_limit.to_le_bytes());
    o
}
