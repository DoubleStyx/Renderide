//! Command encoding for blendshape and skinning compute dispatches.

use std::num::NonZeroU64;

use glam::Mat4;

use crate::backend::advance_slab_cursor;
use crate::backend::mesh_deform::plan_blendshape_scatter_chunks;
use crate::gpu::GpuLimits;
use crate::render_graph::skinning_palette::{build_skinning_palette, SkinningPaletteParams};
use crate::scene::RenderSpaceId;

use super::snapshot::{
    deform_needs_blend_snapshot, deform_needs_skin_snapshot, MeshDeformSnapshot,
};

/// GPU handles and scratch used while recording mesh deform compute on one encoder.
pub(super) struct MeshDeformEncodeGpu<'a> {
    /// Submission queue for buffer writes and compute.
    pub queue: &'a wgpu::Queue,
    /// Device for bind groups and pipelines.
    pub device: &'a wgpu::Device,
    /// Limits checked before dispatch.
    pub gpu_limits: &'a GpuLimits,
    /// Encoder receiving compute passes.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Preprocess pipelines (blendshape + skinning).
    pub pre: &'a crate::backend::mesh_deform::MeshPreprocessPipelines,
    /// Scratch buffers and slab cursors backing.
    pub scratch: &'a mut crate::backend::MeshDeformScratch,
}

/// Scene, mesh snapshot, and slab cursors for one deform work item.
pub(super) struct MeshDeformRecordInputs<'a, 'b> {
    /// Scene graph for bone palette resolution.
    pub scene: &'a crate::scene::SceneCoordinator,
    /// Active render space for the mesh.
    pub space_id: RenderSpaceId,
    /// GPU snapshot of mesh buffers and skinning metadata.
    pub mesh: &'a MeshDeformSnapshot,
    /// Per-bone scene transform indices (skinned meshes).
    pub bone_transform_indices: Option<&'a [i32]>,
    /// SMR node id for skinning fallbacks.
    pub smr_node_id: i32,
    /// Host render context (mono vs stereo clip).
    pub render_context: crate::shared::RenderingContext,
    /// Head / HMD output transform for palette construction.
    pub head_output_transform: Mat4,
    /// Blendshape weights (parallel to mesh blendshape count).
    pub blend_weights: &'a [f32],
    /// Running offset into the bone matrix slab.
    pub bone_cursor: &'b mut u64,
    /// Running offset into the blend weight staging slab.
    pub blend_weight_cursor: &'b mut u64,
}

pub(super) fn record_mesh_deform(
    mut gpu: MeshDeformEncodeGpu<'_>,
    inputs: MeshDeformRecordInputs<'_, '_>,
) {
    let Some(deform_guard) =
        validate_deform_preconditions(inputs.mesh, inputs.bone_transform_indices, gpu.gpu_limits)
    else {
        return;
    };

    if deform_guard.needs_blend {
        record_blendshape_deform(
            &mut gpu,
            inputs.mesh,
            inputs.blend_weights,
            inputs.blend_weight_cursor,
        );
    }

    if deform_guard.needs_skin {
        record_skinning_deform(
            &mut gpu,
            SkinningDeformContext {
                scene: inputs.scene,
                space_id: inputs.space_id,
                mesh: inputs.mesh,
                bone_transform_indices: inputs.bone_transform_indices,
                smr_node_id: inputs.smr_node_id,
                render_context: inputs.render_context,
                head_output_transform: inputs.head_output_transform,
                bone_cursor: inputs.bone_cursor,
                needs_blend: deform_guard.needs_blend,
                wg: deform_guard.skin_wg,
            },
        );
    }
}

/// Early-out state for [`record_mesh_deform`].
struct DeformValidate {
    needs_blend: bool,
    needs_skin: bool,
    /// Workgroups for skinning (`mesh_skinning.wgsl`), one thread per vertex.
    skin_wg: u32,
}

/// Returns `None` when there is no deform work or dispatch would exceed GPU limits.
fn validate_deform_preconditions(
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    gpu_limits: &GpuLimits,
) -> Option<DeformValidate> {
    mesh.positions_buffer.as_ref()?;
    let vc = mesh.vertex_count;
    if vc == 0 {
        return None;
    }
    let needs_blend = deform_needs_blend_snapshot(mesh);
    let needs_skin = deform_needs_skin_snapshot(mesh, bone_transform_indices);

    if !needs_blend && !needs_skin {
        return None;
    }

    let skin_wg = workgroup_count(vc);
    if needs_skin && !gpu_limits.compute_dispatch_fits(skin_wg, 1, 1) {
        logger::warn!(
            "mesh deform: skinning dispatch {}×1×1 exceeds max_compute_workgroups_per_dimension ({})",
            skin_wg,
            gpu_limits.max_compute_workgroups_per_dimension()
        );
        return None;
    }

    Some(DeformValidate {
        needs_blend,
        needs_skin,
        skin_wg,
    })
}

/// Skinning path inputs after blendshape (optional) has run.
///
/// Carries scene graph, mesh buffers, and slab cursors for [`record_skinning_deform`].
struct SkinningDeformContext<'a, 'b> {
    scene: &'a crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &'a MeshDeformSnapshot,
    bone_transform_indices: Option<&'a [i32]>,
    smr_node_id: i32,
    render_context: crate::shared::RenderingContext,
    head_output_transform: Mat4,
    bone_cursor: &'b mut u64,
    needs_blend: bool,
    wg: u32,
}

/// Sparse blendshape scatter: copy bind poses → temp, then one scatter dispatch per weighted shape chunk.
fn record_blendshape_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    mesh: &MeshDeformSnapshot,
    blend_weights: &[f32],
    blend_weight_cursor: &mut u64,
) {
    let Some(ref positions) = mesh.positions_buffer else {
        return;
    };
    let Some(ref temp) = mesh.deform_temp_buffer else {
        return;
    };
    let Some(ref sparse) = mesh.blendshape_sparse_buffer else {
        return;
    };
    let vc = mesh.vertex_count;
    let shape_count = mesh.num_blendshapes;
    if shape_count == 0 {
        return;
    }
    if mesh.blendshape_sparse_ranges.len() != shape_count as usize {
        logger::warn!(
            "mesh deform: blendshape_sparse_ranges len {} != num_blendshapes {}",
            mesh.blendshape_sparse_ranges.len(),
            shape_count
        );
        return;
    }

    let copy_len = u64::from(vc).saturating_mul(16).max(16);
    gpu.encoder
        .copy_buffer_to_buffer(positions.as_ref(), 0, temp.as_ref(), 0, copy_len);

    gpu.scratch
        .ensure_shape_weight_capacity(gpu.device, shape_count);
    let mut wbytes = vec![0u8; (shape_count as usize) * 4];
    for s in 0..shape_count as usize {
        let w = blend_weights.get(s).copied().unwrap_or(0.0);
        wbytes[s * 4..s * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }

    let weight_binding_len = wbytes.len() as u64;
    gpu.scratch.ensure_blend_weight_byte_capacity(
        gpu.device,
        (*blend_weight_cursor).saturating_add(weight_binding_len),
    );
    gpu.queue.write_buffer(
        &gpu.scratch.blendshape_weights,
        *blend_weight_cursor,
        &wbytes,
    );

    let max_wg = gpu.gpu_limits.max_compute_workgroups_per_dimension();

    let mut packed_params: Vec<u8> = Vec::new();
    let mut dispatch_wgs: Vec<u32> = Vec::new();

    for s in 0..shape_count {
        let w = blend_weights.get(s as usize).copied().unwrap_or(0.0);
        if w == 0.0 {
            continue;
        }
        let (first, cnt) = mesh.blendshape_sparse_ranges[s as usize];
        if cnt == 0 {
            continue;
        }
        for (sparse_base, sparse_count) in plan_blendshape_scatter_chunks(first, cnt, max_wg) {
            let wg = workgroup_count(sparse_count);
            if !gpu.gpu_limits.compute_dispatch_fits(wg, 1, 1) {
                logger::warn!(
                    "mesh deform: blendshape scatter dispatch {}×1×1 exceeds max_compute_workgroups_per_dimension ({})",
                    wg,
                    max_wg
                );
                return;
            }
            packed_params.extend_from_slice(&build_scatter_params(
                vc,
                s,
                sparse_base,
                sparse_count,
            ));
            dispatch_wgs.push(wg);
        }
    }

    if packed_params.is_empty() {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
        return;
    }

    gpu.scratch
        .ensure_blendshape_params_staging(gpu.device, packed_params.len() as u64);
    gpu.queue
        .write_buffer(&gpu.scratch.blendshape_params_staging, 0, &packed_params);

    let Some(weight_size) = NonZeroU64::new(weight_binding_len) else {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
        return;
    };

    for (i, &scatter_wg) in dispatch_wgs.iter().enumerate() {
        let src_off = (i as u64).saturating_mul(32);
        gpu.encoder.copy_buffer_to_buffer(
            &gpu.scratch.blendshape_params_staging,
            src_off,
            &gpu.scratch.blendshape_params,
            0,
            32,
        );

        let blend_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blendshape_scatter_bg"),
            layout: &gpu.pre.blendshape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.scratch.blendshape_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sparse.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu.scratch.blendshape_weights,
                        offset: *blend_weight_cursor,
                        size: Some(weight_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp.as_entire_binding(),
                },
            ],
        });

        let mut cpass = gpu
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blendshape_scatter"),
                timestamp_writes: None,
            });
        cpass.set_pipeline(&gpu.pre.blendshape_pipeline);
        cpass.set_bind_group(0, &blend_bg, &[]);
        cpass.dispatch_workgroups(scatter_wg, 1, 1);
    }

    *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
}

/// Linear blend skinning compute after optional blendshape pass.
fn record_skinning_deform(gpu: &mut MeshDeformEncodeGpu<'_>, ctx: SkinningDeformContext<'_, '_>) {
    let Some(ref positions) = ctx.mesh.positions_buffer else {
        return;
    };
    let Some(ref dst) = ctx.mesh.deformed_positions_buffer else {
        return;
    };
    let Some(ref dst_n) = ctx.mesh.deformed_normals_buffer else {
        return;
    };
    let Some(ref src_n) = ctx.mesh.normals_buffer else {
        return;
    };
    let Some(ref bone_idx) = ctx.mesh.bone_indices_buffer else {
        return;
    };
    let Some(ref bone_wt) = ctx.mesh.bone_weights_vec4_buffer else {
        return;
    };
    let Some(indices) = ctx.bone_transform_indices else {
        return;
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
        return;
    };
    let mut palette: Vec<u8> = vec![0u8; palette_mats.len() * 64];
    for (bi, pal) in palette_mats.iter().enumerate() {
        let cols = pal.to_cols_array();
        palette[bi * 64..bi * 64 + 64].copy_from_slice(bytemuck::cast_slice(&cols));
    }

    let palette_len = palette.len() as u64;
    gpu.scratch
        .ensure_bone_byte_capacity(gpu.device, ctx.bone_cursor.saturating_add(palette_len));
    gpu.queue
        .write_buffer(&gpu.scratch.bone_matrices, *ctx.bone_cursor, &palette);

    let Some(bone_binding_size) = NonZeroU64::new(palette_len) else {
        return;
    };

    let src_for_skin: &wgpu::Buffer = if ctx.needs_blend {
        let Some(buf) = ctx.mesh.deform_temp_buffer.as_deref() else {
            return;
        };
        buf
    } else {
        positions.as_ref()
    };

    skinning_dispatch_with_uploaded_palette(SkinningPaletteDispatch {
        device: gpu.device,
        encoder: gpu.encoder,
        pre: gpu.pre,
        scratch: gpu.scratch,
        src_positions: src_for_skin,
        bone_idx,
        bone_wt,
        dst_pos: dst,
        src_n,
        dst_n,
        bone_cursor: *ctx.bone_cursor,
        bone_binding_size,
        wg: ctx.wg,
    });

    *ctx.bone_cursor = advance_slab_cursor(*ctx.bone_cursor, palette_len);
}

/// Buffers and offsets for one skinning dispatch after the bone palette is uploaded to `scratch`.
///
/// Used by [`skinning_dispatch_with_uploaded_palette`] to build the skinning bind group and dispatch.
struct SkinningPaletteDispatch<'a> {
    device: &'a wgpu::Device,
    encoder: &'a mut wgpu::CommandEncoder,
    pre: &'a crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &'a crate::backend::MeshDeformScratch,
    src_positions: &'a wgpu::Buffer,
    bone_idx: &'a wgpu::Buffer,
    bone_wt: &'a wgpu::Buffer,
    dst_pos: &'a wgpu::Buffer,
    src_n: &'a wgpu::Buffer,
    dst_n: &'a wgpu::Buffer,
    bone_cursor: u64,
    bone_binding_size: NonZeroU64,
    wg: u32,
}

/// Builds skinning bind group (bone slab + attributes) and dispatches the skinning shader.
fn skinning_dispatch_with_uploaded_palette(dispatch: SkinningPaletteDispatch<'_>) {
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
            ],
        });

    let mut cpass = dispatch
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skinning"),
            timestamp_writes: None,
        });
    cpass.set_pipeline(&dispatch.pre.skinning_pipeline);
    cpass.set_bind_group(0, &skin_bg, &[]);
    cpass.dispatch_workgroups(dispatch.wg, 1, 1);
}

fn workgroup_count(count: u32) -> u32 {
    (count.saturating_add(63)) / 64
}

/// Encodes `source/compute/mesh_blendshape.wgsl` `Params` (32 bytes).
fn build_scatter_params(
    vertex_count: u32,
    shape_index: u32,
    sparse_base: u32,
    sparse_count: u32,
) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&shape_index.to_le_bytes());
    o[8..12].copy_from_slice(&sparse_base.to_le_bytes());
    o[12..16].copy_from_slice(&sparse_count.to_le_bytes());
    o
}
