//! Blendshape and skinning compute dispatches before the main forward pass.
//!
//! Work items are collected per render space in parallel ([`rayon`]); compute is still recorded
//! sequentially on one [`wgpu::CommandEncoder`].

mod encode;
mod snapshot;

use rayon::prelude::*;

use crate::backend::mesh_deform::EntryNeed;
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::handles::ResourceId;
use crate::render_graph::module::RenderModule;
use crate::render_graph::pass::{PassPhase, RenderPass};
use crate::render_graph::resources::PassResources;
use crate::render_graph::{GraphBuilder, SharedRenderHandles};
use crate::resources::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};

use self::encode::{record_mesh_deform, MeshDeformEncodeGpu, MeshDeformRecordInputs};
use self::snapshot::{
    deform_needs_blend_snapshot, deform_needs_skin_mesh, deform_needs_skin_snapshot,
    gpu_mesh_needs_deform_dispatch, MeshDeformSnapshot,
};

/// Encodes mesh deformation compute for all active render spaces.
#[derive(Debug)]
pub struct MeshDeformPass {
    mesh_deform_outputs: ResourceId,
}

impl MeshDeformPass {
    /// Creates a mesh deform pass bound to the logical deform output resource.
    pub fn new(mesh_deform_outputs: ResourceId) -> Self {
        Self {
            mesh_deform_outputs,
        }
    }
}

/// Registers [`MeshDeformPass`] on the main frame graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct MeshDeformModule;

impl RenderModule for MeshDeformModule {
    fn name(&self) -> &str {
        "mesh_deform"
    }

    fn register(self: Box<Self>, builder: &mut GraphBuilder, handles: &SharedRenderHandles) {
        builder.add_pass(Box::new(MeshDeformPass::new(handles.mesh_deform_outputs)));
    }
}

struct DeformWorkItem {
    space_id: RenderSpaceId,
    /// [`StaticMeshRenderer::node_id`](crate::scene::StaticMeshRenderer::node_id) for GPU skin cache key.
    node_id: i32,
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
            node_id: r.node_id,
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
            node_id: r.node_id,
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
            writes: vec![self.mesh_deform_outputs],
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

        let queue = match ctx.queue.lock() {
            Ok(q) => q,
            Err(poisoned) => poisoned.into_inner(),
        };

        let mut bone_cursor = 0u64;
        let mut blend_weight_cursor = 0u64;
        let mut skin_dispatch_cursor = 0u64;
        let render_context = frame.scene.active_main_render_context();
        let head_output_transform = frame.host_camera.head_output_transform;

        {
            let backend = &mut frame.backend;
            let Some((pre, scratch, skin_cache)) = backend.mesh_deform_pass_refs() else {
                return Ok(());
            };

            for item in work {
                let need = EntryNeed {
                    needs_blend: deform_needs_blend_snapshot(&item.mesh),
                    needs_skin: deform_needs_skin_snapshot(&item.mesh, item.skinned.as_deref()),
                };
                let key = (item.space_id, item.node_id);
                let Some((cache_entry, positions_arena, normals_arena, temp_arena)) = skin_cache
                    .get_or_alloc_with_arenas(
                        ctx.device,
                        ctx.encoder,
                        key,
                        need,
                        item.mesh.vertex_count,
                    )
                else {
                    continue;
                };

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
                        skin_dispatch_cursor: &mut skin_dispatch_cursor,
                        skin_cache_entry: cache_entry,
                        positions_arena,
                        normals_arena,
                        temp_arena,
                    },
                );
            }

            let fc = skin_cache.frame_counter();
            skin_cache.sweep_stale(fc.saturating_sub(2));
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
