//! Blendshape and skinning compute dispatches before the main forward pass.

use std::num::NonZeroU64;
use std::sync::Arc;

use glam::Mat4;

use crate::assets::mesh::BLENDSHAPE_OFFSET_GPU_STRIDE;
use crate::backend::advance_slab_cursor;
use crate::gpu::plan_blendshape_bind_chunks;

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::PassResources;
use crate::scene::RenderSpaceId;

/// Encodes mesh deformation compute for all active render spaces.
#[derive(Debug, Default)]
pub struct MeshDeformPass;

impl MeshDeformPass {
    pub fn new() -> Self {
        Self
    }
}

/// GPU buffer handles + metadata copied from [`crate::assets::mesh::GpuMesh`] so we can hold
/// deform state without borrowing the mesh pool across preprocess/scratch access.
struct MeshDeformSnapshot {
    vertex_count: u32,
    num_blendshapes: u32,
    has_skeleton: bool,
    positions_buffer: Option<Arc<wgpu::Buffer>>,
    blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    skinning_bind_matrices: Vec<Mat4>,
}

impl MeshDeformSnapshot {
    fn from_mesh(m: &crate::assets::mesh::GpuMesh) -> Self {
        Self {
            vertex_count: m.vertex_count,
            num_blendshapes: m.num_blendshapes,
            has_skeleton: m.has_skeleton,
            positions_buffer: m.positions_buffer.clone(),
            blendshape_buffer: m.blendshape_buffer.clone(),
            deform_temp_buffer: m.deform_temp_buffer.clone(),
            deformed_positions_buffer: m.deformed_positions_buffer.clone(),
            bone_indices_buffer: m.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: m.bone_weights_vec4_buffer.clone(),
            skinning_bind_matrices: m.skinning_bind_matrices.clone(),
        }
    }
}

struct DeformWorkItem {
    space_id: RenderSpaceId,
    mesh: MeshDeformSnapshot,
    skinned: Option<Vec<i32>>,
    /// [`StaticMeshRenderer::node_id`] (SMR) for skinning fallbacks when a bone is unmapped.
    smr_node_id: i32,
    blend_weights: Vec<f32>,
}

impl RenderPass for MeshDeformPass {
    fn name(&self) -> &str {
        "MeshDeform"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        let mut work: Vec<DeformWorkItem> = Vec::new();
        for space_id in frame.scene.render_space_ids() {
            let Some(space) = frame.scene.space(space_id) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for r in &space.static_mesh_renderers {
                if r.mesh_asset_id < 0 {
                    continue;
                }
                let Some(m) = frame.backend.mesh_pool().get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                work.push(DeformWorkItem {
                    space_id,
                    mesh: MeshDeformSnapshot::from_mesh(m),
                    skinned: None,
                    smr_node_id: -1,
                    blend_weights: r.blend_shape_weights.clone(),
                });
            }
            for skinned in &space.skinned_mesh_renderers {
                let r = &skinned.base;
                if r.mesh_asset_id < 0 {
                    continue;
                }
                let Some(m) = frame.backend.mesh_pool().get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                work.push(DeformWorkItem {
                    space_id,
                    mesh: MeshDeformSnapshot::from_mesh(m),
                    skinned: Some(skinned.bone_transform_indices.clone()),
                    smr_node_id: r.node_id,
                    blend_weights: r.blend_shape_weights.clone(),
                });
            }
        }

        let Some((pre, scratch)) = frame.backend.mesh_deform_pre_and_scratch() else {
            return Ok(());
        };

        let queue = match ctx.queue.lock() {
            Ok(q) => q,
            Err(poisoned) => poisoned.into_inner(),
        };

        let mut bone_cursor = 0u64;
        let mut blend_weight_cursor = 0u64;

        for item in work {
            record_mesh_deform(
                &queue,
                ctx.device,
                ctx.encoder,
                pre,
                scratch,
                frame.scene,
                item.space_id,
                &item.mesh,
                item.skinned.as_deref(),
                item.smr_node_id,
                &item.blend_weights,
                &mut bone_cursor,
                &mut blend_weight_cursor,
            );
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn record_mesh_deform(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::gpu::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    scene: &crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    smr_node_id: i32,
    blend_weights: &[f32],
    bone_cursor: &mut u64,
    blend_weight_cursor: &mut u64,
) {
    let Some(ref positions) = mesh.positions_buffer else {
        return;
    };
    let vc = mesh.vertex_count;
    if vc == 0 {
        return;
    }
    let wg = workgroup_count(vc);

    let needs_blend = mesh.num_blendshapes > 0
        && mesh.blendshape_buffer.is_some()
        && mesh.deform_temp_buffer.is_some();
    let needs_skin = bone_transform_indices.is_some()
        && mesh.has_skeleton
        && mesh.deformed_positions_buffer.is_some()
        && mesh.bone_indices_buffer.is_some()
        && mesh.bone_weights_vec4_buffer.is_some()
        && !mesh.skinning_bind_matrices.is_empty();

    if !needs_blend && !needs_skin {
        return;
    }

    if needs_blend {
        let Some(ref temp) = mesh.deform_temp_buffer else {
            return;
        };
        let Some(ref deltas) = mesh.blendshape_buffer else {
            return;
        };
        let shape_count = mesh.num_blendshapes;
        if shape_count == 0 {
            return;
        }
        scratch.ensure_shape_weight_capacity(device, shape_count);
        let mut wbytes = vec![0u8; (shape_count as usize) * 4];
        for s in 0..shape_count as usize {
            let w = blend_weights.get(s).copied().unwrap_or(0.0);
            wbytes[s * 4..s * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }

        let weight_binding_len = wbytes.len() as u64;
        scratch.ensure_blend_weight_byte_capacity(
            device,
            (*blend_weight_cursor).saturating_add(weight_binding_len),
        );
        queue.write_buffer(&scratch.blendshape_weights, *blend_weight_cursor, &wbytes);

        let limits = device.limits();
        let Some(chunks) = plan_blendshape_bind_chunks(
            shape_count,
            vc,
            limits.max_storage_buffer_binding_size,
            limits.min_storage_buffer_offset_alignment,
        ) else {
            logger::warn!(
                "mesh deform: blendshape bind chunks unavailable (vertex_count={vc} shape_count={shape_count} max_bind={})",
                limits.max_storage_buffer_binding_size
            );
            return;
        };

        let stride = u64::from(vc) * u64::from(BLENDSHAPE_OFFSET_GPU_STRIDE as u32);

        let mut packed_params = Vec::with_capacity(chunks.len() * 32);
        for (chunk_i, (shape_start, chunk_shapes)) in chunks.iter().enumerate() {
            let params = build_blend_params(vc, *chunk_shapes, *shape_start, chunk_i == 0);
            packed_params.extend_from_slice(&params);
        }
        scratch.ensure_blendshape_params_staging(device, packed_params.len() as u64);
        queue.write_buffer(&scratch.blendshape_params_staging, 0, &packed_params);

        for (chunk_i, (shape_start, chunk_shapes)) in chunks.iter().enumerate() {
            let src_off = (chunk_i as u64).saturating_mul(32);
            encoder.copy_buffer_to_buffer(
                &scratch.blendshape_params_staging,
                src_off,
                &scratch.blendshape_params,
                0,
                32,
            );

            let offset = u64::from(*shape_start).saturating_mul(stride);
            let size = u64::from(*chunk_shapes).saturating_mul(stride);
            let Some(size_nz) = NonZeroU64::new(size) else {
                continue;
            };

            let Some(weight_size) = NonZeroU64::new(weight_binding_len) else {
                continue;
            };

            let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blendshape_bg"),
                layout: &pre.blendshape_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scratch.blendshape_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: positions.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: deltas.as_ref(),
                            offset,
                            size: Some(size_nz),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &scratch.blendshape_weights,
                            offset: *blend_weight_cursor,
                            size: Some(weight_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: temp.as_entire_binding(),
                    },
                ],
            });

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blendshape"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pre.blendshape_pipeline);
            cpass.set_bind_group(0, &blend_bg, &[]);
            cpass.dispatch_workgroups(wg, 1, 1);
        }

        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
    }

    if needs_skin {
        let Some(ref dst) = mesh.deformed_positions_buffer else {
            return;
        };
        let Some(ref bone_idx) = mesh.bone_indices_buffer else {
            return;
        };
        let Some(ref bone_wt) = mesh.bone_weights_vec4_buffer else {
            return;
        };
        let Some(indices) = bone_transform_indices else {
            return;
        };

        let bone_count_u = mesh.skinning_bind_matrices.len() as u32;
        scratch.ensure_bone_capacity(device, bone_count_u);
        let smr_world = (smr_node_id >= 0)
            .then(|| scene.world_matrix(space_id, smr_node_id as usize))
            .flatten()
            .unwrap_or(Mat4::IDENTITY);

        let mut palette: Vec<u8> = vec![0u8; (bone_count_u as usize) * 64];
        for bi in 0..bone_count_u as usize {
            let bind_mat = mesh.skinning_bind_matrices[bi];
            let tid = indices.get(bi).copied().unwrap_or(-1);
            let pal = if tid < 0 {
                smr_world
            } else {
                match scene.world_matrix(space_id, tid as usize) {
                    Some(world) => world * bind_mat,
                    None => smr_world,
                }
            };
            let cols = pal.to_cols_array();
            palette[bi * 64..bi * 64 + 64].copy_from_slice(bytemuck::cast_slice(&cols));
        }

        let palette_len = palette.len() as u64;
        scratch.ensure_bone_byte_capacity(device, bone_cursor.saturating_add(palette_len));
        queue.write_buffer(&scratch.bone_matrices, *bone_cursor, &palette);

        let Some(bone_binding_size) = NonZeroU64::new(palette_len) else {
            return;
        };

        let src_for_skin: &wgpu::Buffer = if needs_blend {
            mesh.deform_temp_buffer.as_deref().expect("blend temp")
        } else {
            positions.as_ref()
        };

        let skin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bg"),
            layout: &pre.skinning_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scratch.bone_matrices,
                        offset: *bone_cursor,
                        size: Some(bone_binding_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_for_skin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bone_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bone_wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skinning"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pre.skinning_pipeline);
        cpass.set_bind_group(0, &skin_bg, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
        drop(cpass);

        *bone_cursor = advance_slab_cursor(*bone_cursor, palette_len);
    }
}

fn workgroup_count(vertex_count: u32) -> u32 {
    (vertex_count.saturating_add(63)) / 64
}

/// Encodes `mesh_blendshape.wgsl` `Params` (32 bytes).
fn build_blend_params(
    vertex_count: u32,
    chunk_shape_count: u32,
    weight_base: u32,
    first_chunk: bool,
) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&chunk_shape_count.to_le_bytes());
    o[8..12].copy_from_slice(&vertex_count.to_le_bytes());
    o[12..16].copy_from_slice(&weight_base.to_le_bytes());
    let fc = u32::from(first_chunk);
    o[16..20].copy_from_slice(&fc.to_le_bytes());
    o
}

#[cfg(test)]
mod palette_tests {
    use glam::{Mat4, Vec3};

    #[test]
    fn palette_is_world_times_bind() {
        let world = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let bind = Mat4::from_scale(Vec3::splat(2.0));
        let pal = world * bind;
        let expected = world * bind;
        assert!(pal.abs_diff_eq(expected, 1e-5));
    }
}
