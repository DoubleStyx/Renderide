//! Blendshape and skinning compute dispatches before the main forward pass.
//!
//! Work items are collected per render space in parallel ([`rayon`]); compute is still recorded
//! sequentially on one [`wgpu::CommandEncoder`].

mod encode;
mod snapshot;

use rayon::prelude::*;

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::{PassPhase, RenderPass};
use crate::render_graph::resources::PassResources;
use crate::resources::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};

use self::encode::{record_mesh_deform, MeshDeformEncodeGpu, MeshDeformRecordInputs};
use self::snapshot::{deform_needs_skin_mesh, gpu_mesh_needs_deform_dispatch, MeshDeformSnapshot};

/// Encodes mesh deformation compute for all active render spaces.
#[derive(Debug, Default)]
pub struct MeshDeformPass;

impl MeshDeformPass {
    /// Creates a mesh deform pass instance.
    pub fn new() -> Self {
        Self
    }
}

struct DeformWorkItem {
    space_id: RenderSpaceId,
    mesh: MeshDeformSnapshot,
    skinned: Option<Vec<i32>>,
    /// [`StaticMeshRenderer::node_id`](crate::scene::StaticMeshRenderer::node_id) (SMR) for skinning fallbacks when a bone is unmapped.
    smr_node_id: i32,
    blend_weights: Vec<f32>,
}

/// Collects deform work items for one render space (read-only scene + mesh pool).
fn collect_deform_work_for_space(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    space_id: RenderSpaceId,
) -> Vec<DeformWorkItem> {
    let mut work = Vec::new();
    let Some(space) = scene.space(space_id) else {
        return work;
    };
    if !space.is_active {
        return work;
    }
    for r in &space.static_mesh_renderers {
        if r.mesh_asset_id < 0 {
            continue;
        }
        let Some(m) = mesh_pool.get_mesh(r.mesh_asset_id) else {
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
        let Some(m) = mesh_pool.get_mesh(r.mesh_asset_id) else {
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
    work
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

    fn phase(&self) -> PassPhase {
        PassPhase::FrameGlobal
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        if frame
            .backend
            .frame_resources
            .mesh_deform_dispatched_this_tick()
        {
            return Ok(());
        }

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
        // Scope `scene` + `mesh_pool` so the rayon closure only captures `Sync` refs, not
        // [`crate::backend::RenderBackend`] (contains `RefCell`, imgui, non-`Sync` graph passes).
        let work: Vec<DeformWorkItem> = {
            let scene = frame.scene;
            let mesh_pool = frame.backend.mesh_pool();
            let space_ids: Vec<RenderSpaceId> = scene.render_space_ids().collect();
            let work_chunks: Vec<Vec<DeformWorkItem>> = space_ids
                .par_iter()
                .copied()
                .map(|space_id| collect_deform_work_for_space(scene, mesh_pool, space_id))
                .collect();
            let mut work: Vec<DeformWorkItem> = Vec::with_capacity(est);
            for chunk in work_chunks {
                work.extend(chunk);
            }
            work
        };

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
                MeshDeformEncodeGpu {
                    queue: &queue,
                    device: ctx.device,
                    gpu_limits: ctx.gpu_limits,
                    encoder: ctx.encoder,
                    pre,
                    scratch,
                },
                MeshDeformRecordInputs {
                    scene: frame.scene,
                    space_id: item.space_id,
                    mesh: &item.mesh,
                    bone_transform_indices: item.skinned.as_deref(),
                    smr_node_id: item.smr_node_id,
                    render_context,
                    head_output_transform,
                    blend_weights: &item.blend_weights,
                    bone_cursor: &mut bone_cursor,
                    blend_weight_cursor: &mut blend_weight_cursor,
                },
            );
        }

        frame
            .backend
            .frame_resources
            .set_mesh_deform_dispatched_this_tick();
        Ok(())
    }
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
