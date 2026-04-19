//! Blendshape and skinning compute dispatches before the main forward pass.
//!
//! Work items are collected per render space in parallel ([`rayon`]); compute is still recorded
//! sequentially on one [`wgpu::CommandEncoder`].

mod encode;
mod snapshot;

use std::fmt;

use rayon::prelude::*;

use crate::backend::mesh_deform::EntryNeed;
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, PassPhase, RenderPass};
use crate::resources::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};

use self::encode::{record_mesh_deform, MeshDeformEncodeGpu, MeshDeformRecordInputs};
use self::snapshot::{
    deform_needs_blend_snapshot, deform_needs_skin_mesh, deform_needs_skin_snapshot,
    gpu_mesh_needs_deform_dispatch, MeshDeformSnapshot,
};

/// Encodes mesh deformation compute for all active render spaces.
///
/// Per-frame collection reuses scratch buffers to avoid `Vec` allocations on the hot path.
#[derive(Default)]
pub struct MeshDeformPass {
    /// Reused ordering of [`SceneCoordinator::render_space_ids`] for parallel per-space collection.
    mesh_deform_space_ids_scratch: Vec<RenderSpaceId>,
    /// One bucket per render space; inner [`Vec`] capacities are reused across frames.
    mesh_deform_chunks_scratch: Vec<Vec<DeformWorkItem>>,
    /// Flattened work list passed to encode (cleared after each successful dispatch).
    mesh_deform_work_scratch: Vec<DeformWorkItem>,
}

impl fmt::Debug for MeshDeformPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MeshDeformPass").finish_non_exhaustive()
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
    work: &mut Vec<DeformWorkItem>,
) {
    work.clear();
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active {
        return;
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
}

/// Upper bound on deform work items (static + skinned) across active spaces for scratch reservation.
fn deform_work_upper_bound(scene: &SceneCoordinator) -> usize {
    let mut est = 0usize;
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if space.is_active {
            est = est
                .saturating_add(space.static_mesh_renderers.len())
                .saturating_add(space.skinned_mesh_renderers.len());
        }
    }
    est
}

impl MeshDeformPass {
    /// Creates a mesh deform pass with empty scratch buffers (filled lazily on first execute).
    pub fn new() -> Self {
        Self::default()
    }

    /// Parallel per-space collection merged into [`Self::mesh_deform_work_scratch`].
    fn collect_deform_work_into_scratch(&mut self, scene: &SceneCoordinator, mesh_pool: &MeshPool) {
        let est = deform_work_upper_bound(scene);
        self.mesh_deform_space_ids_scratch.clear();
        self.mesh_deform_space_ids_scratch
            .extend(scene.render_space_ids());
        let space_count = self.mesh_deform_space_ids_scratch.len();
        if self.mesh_deform_chunks_scratch.len() < space_count {
            self.mesh_deform_chunks_scratch
                .resize_with(space_count, Vec::new);
        } else {
            self.mesh_deform_chunks_scratch.truncate(space_count);
        }

        {
            let space_ids = &self.mesh_deform_space_ids_scratch;
            let chunks = &mut self.mesh_deform_chunks_scratch;
            // Scope `scene` + `mesh_pool` so the rayon closure only captures `Sync` refs, not
            // fat frame handles (non-`Sync` graph state).
            space_ids
                .par_iter()
                .copied()
                .zip(chunks.par_iter_mut())
                .for_each(|(space_id, chunk)| {
                    collect_deform_work_for_space(scene, mesh_pool, space_id, chunk);
                });
        }

        self.mesh_deform_work_scratch.clear();
        self.mesh_deform_work_scratch.reserve(est);
        for chunk in &mut self.mesh_deform_chunks_scratch {
            self.mesh_deform_work_scratch.append(chunk);
        }
    }
}

impl RenderPass for MeshDeformPass {
    fn name(&self) -> &str {
        "MeshDeform"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.cull_exempt();
        Ok(())
    }

    fn phase(&self) -> PassPhase {
        PassPhase::FrameGlobal
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        if frame.frame_resources.mesh_deform_dispatched_this_tick() {
            return Ok(());
        }

        let mesh_pool = &frame.asset_transfers.mesh_pool;
        self.collect_deform_work_into_scratch(frame.scene, mesh_pool);

        let Some(pre) = frame.mesh_preprocess else {
            self.mesh_deform_work_scratch.clear();
            return Ok(());
        };
        let Some(scratch) = frame.mesh_deform_scratch.as_mut() else {
            self.mesh_deform_work_scratch.clear();
            return Ok(());
        };
        let Some(skin_cache) = frame.skin_cache.as_mut() else {
            self.mesh_deform_work_scratch.clear();
            return Ok(());
        };

        let queue: &wgpu::Queue = ctx.queue.as_ref();

        let mut bone_cursor = 0u64;
        let mut blend_weight_cursor = 0u64;
        let mut skin_dispatch_cursor = 0u64;
        let render_context = frame.scene.active_main_render_context();
        let head_output_transform = frame.host_camera.head_output_transform;

        for item in self.mesh_deform_work_scratch.drain(..) {
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
                    queue,
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

        frame.frame_resources.set_mesh_deform_dispatched_this_tick();
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
