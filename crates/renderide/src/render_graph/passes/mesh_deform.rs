//! Blendshape and skinning compute dispatches before the main forward pass.

use std::num::NonZeroU64;
use std::sync::Arc;

use glam::Mat4;

use crate::assets::mesh::{GpuMesh, BLENDSHAPE_OFFSET_GPU_STRIDE};
use crate::backend::advance_slab_cursor;
use crate::backend::mesh_deform::plan_blendshape_bind_chunks;

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::PassResources;
use crate::render_graph::skinning_palette::build_skinning_palette;
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
    normals_buffer: Option<Arc<wgpu::Buffer>>,
    blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    deformed_normals_buffer: Option<Arc<wgpu::Buffer>>,
    bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    skinning_bind_matrices: Vec<Mat4>,
}

impl MeshDeformSnapshot {
    /// Copies GPU resources from `m`. When `clone_skinning_bind_matrices` is `false`, bind matrices
    /// are omitted (blendshape-only path never reads them).
    fn from_mesh(m: &GpuMesh, clone_skinning_bind_matrices: bool) -> Self {
        Self {
            vertex_count: m.vertex_count,
            num_blendshapes: m.num_blendshapes,
            has_skeleton: m.has_skeleton,
            positions_buffer: m.positions_buffer.clone(),
            normals_buffer: m.normals_buffer.clone(),
            blendshape_buffer: m.blendshape_buffer.clone(),
            deform_temp_buffer: m.deform_temp_buffer.clone(),
            deformed_positions_buffer: m.deformed_positions_buffer.clone(),
            deformed_normals_buffer: m.deformed_normals_buffer.clone(),
            bone_indices_buffer: m.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: m.bone_weights_vec4_buffer.clone(),
            skinning_bind_matrices: if clone_skinning_bind_matrices {
                m.skinning_bind_matrices.clone()
            } else {
                Vec::new()
            },
        }
    }
}

/// Returns whether blendshape compute should run for `m` (parity with [`record_mesh_deform`]).
fn deform_needs_blend_mesh(m: &GpuMesh) -> bool {
    m.num_blendshapes > 0 && m.blendshape_buffer.is_some() && m.deform_temp_buffer.is_some()
}

/// Returns whether skinning compute should run for `m` (parity with [`record_mesh_deform`]).
fn deform_needs_skin_mesh(m: &GpuMesh, bone_transform_indices: Option<&[i32]>) -> bool {
    bone_transform_indices.is_some()
        && m.has_skeleton
        && m.deformed_positions_buffer.is_some()
        && m.deformed_normals_buffer.is_some()
        && m.normals_buffer.is_some()
        && m.bone_indices_buffer.is_some()
        && m.bone_weights_vec4_buffer.is_some()
        && !m.skinning_bind_matrices.is_empty()
}

/// Returns `true` when [`record_mesh_deform`] would run blend and/or skin dispatches (not an early no-op).
fn gpu_mesh_needs_deform_dispatch(m: &GpuMesh, bone_transform_indices: Option<&[i32]>) -> bool {
    if m.positions_buffer.is_none() || m.vertex_count == 0 {
        return false;
    }
    deform_needs_blend_mesh(m) || deform_needs_skin_mesh(m, bone_transform_indices)
}

/// Snapshot variant of [`deform_needs_blend_mesh`].
fn deform_needs_blend_snapshot(mesh: &MeshDeformSnapshot) -> bool {
    mesh.num_blendshapes > 0
        && mesh.blendshape_buffer.is_some()
        && mesh.deform_temp_buffer.is_some()
}

/// Snapshot variant of [`deform_needs_skin_mesh`].
fn deform_needs_skin_snapshot(
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
) -> bool {
    bone_transform_indices.is_some()
        && mesh.has_skeleton
        && mesh.deformed_positions_buffer.is_some()
        && mesh.deformed_normals_buffer.is_some()
        && mesh.normals_buffer.is_some()
        && mesh.bone_indices_buffer.is_some()
        && mesh.bone_weights_vec4_buffer.is_some()
        && !mesh.skinning_bind_matrices.is_empty()
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

        let mut est = 0usize;
        for space_id in frame.scene.render_space_ids() {
            let Some(space) = frame.scene.space(space_id) else {
                continue;
            };
            if space.is_active {
                est = est
                    .saturating_add(space.static_mesh_renderers.len())
                    .saturating_add(space.skinned_mesh_renderers.len());
            }
        }
        let mut work: Vec<DeformWorkItem> = Vec::with_capacity(est);

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
                if !gpu_mesh_needs_deform_dispatch(m, None) {
                    continue;
                }
                work.push(DeformWorkItem {
                    space_id,
                    mesh: MeshDeformSnapshot::from_mesh(m, false),
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
                let bone_ix = skinned.bone_transform_indices.as_slice();
                if !gpu_mesh_needs_deform_dispatch(m, Some(bone_ix)) {
                    continue;
                }
                let clone_bind = deform_needs_skin_mesh(m, Some(bone_ix));
                work.push(DeformWorkItem {
                    space_id,
                    mesh: MeshDeformSnapshot::from_mesh(m, clone_bind),
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
        let render_context = frame.scene.active_main_render_context();
        let head_output_transform = frame.host_camera.head_output_transform;

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
                render_context,
                head_output_transform,
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
    pre: &crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    scene: &crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    smr_node_id: i32,
    render_context: crate::shared::RenderingContext,
    head_output_transform: Mat4,
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

    let needs_blend = deform_needs_blend_snapshot(mesh);
    let needs_skin = deform_needs_skin_snapshot(mesh, bone_transform_indices);

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
        let Some(ref dst_n) = mesh.deformed_normals_buffer else {
            return;
        };
        let Some(ref src_n) = mesh.normals_buffer else {
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
        let Some(palette_mats) = build_skinning_palette(
            scene,
            space_id,
            &mesh.skinning_bind_matrices,
            mesh.has_skeleton,
            indices,
            smr_node_id,
            render_context,
            head_output_transform,
        ) else {
            return;
        };
        let mut palette: Vec<u8> = vec![0u8; palette_mats.len() * 64];
        for (bi, pal) in palette_mats.iter().enumerate() {
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
            let Some(buf) = mesh.deform_temp_buffer.as_deref() else {
                return;
            };
            buf
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: src_n.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dst_n.as_entire_binding(),
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

/// Encodes `source/compute/mesh_blendshape.wgsl` `Params` (32 bytes).
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
    use glam::{Mat3, Mat4, Vec3};

    #[test]
    fn palette_is_world_times_bind() {
        let world = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let bind = Mat4::from_scale(Vec3::splat(2.0));
        let pal = world * bind;
        let expected = world * bind;
        assert!(pal.abs_diff_eq(expected, 1e-5));
    }

    /// Matches WGSL `transpose(inverse(mat3_linear(M)))` for rigid rotations: equals the linear part.
    #[test]
    fn normal_matrix_inverse_transpose_is_rotation_for_orthogonal() {
        let m3 = Mat3::from_axis_angle(Vec3::Z, 1.15);
        let inv_t = m3.inverse().transpose();
        assert!(inv_t.abs_diff_eq(m3, 1e-5));
    }
}
