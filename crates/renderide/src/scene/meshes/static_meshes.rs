//! Static mesh renderable updates: shared-memory extraction, dense apply, and tests.

use rayon::prelude::*;

use crate::cpu_parallelism::{
    admit_renderable_update_items, current_reference_worker_count, record_parallel_admission,
};
use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::{non_negative_i32s, swap_remove_dense_indices};
use crate::scene::error::SceneError;
use crate::scene::meshes::types::{
    MeshRendererStateApplyPlan, StaticMeshRenderer, decode_mesh_renderer_state_plan,
};
use crate::scene::render_space::RenderSpaceState;
use crate::shared::{
    LayerType, MESH_RENDERER_STATE_HOST_ROW_BYTES, MeshRenderablesUpdate, MeshRendererState,
};

use super::diagnostics::{STATIC_MESH_OOB_WARNED_SCENES, warn_oob_renderable_index_once};

/// Owned per-space static mesh-renderable update payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedMeshRenderablesUpdate {
    /// Static-mesh renderable removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New static-mesh renderable transform ids (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-renderer mesh state rows (terminated by `renderable_index < 0`).
    pub mesh_states: Vec<MeshRendererState>,
    /// Optional packed material/property-block id slab (`None` when host omitted the buffer).
    pub mesh_materials_and_property_blocks: Option<Vec<i32>>,
}

/// Reads every shared-memory buffer referenced by [`MeshRenderablesUpdate`] into owned vectors.
pub(crate) fn extract_mesh_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &MeshRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedMeshRenderablesUpdate, SceneError> {
    let mut out = ExtractedMeshRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("mesh removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("mesh additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("mesh mesh_states scene_id={scene_id}");
        out.mesh_states = shm
            .access_copy_memory_packable_rows::<MeshRendererState>(
                &update.mesh_states,
                MESH_RENDERER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!("mesh mesh_materials_and_property_blocks scene_id={scene_id}");
            out.mesh_materials_and_property_blocks = Some(
                shm.access_copy_diagnostic_with_context::<i32>(
                    &update.mesh_materials_and_property_blocks,
                    Some(&ctx_m),
                )
                .map_err(SceneError::SharedMemoryAccess)?,
            );
        }
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState::static_mesh_renderers`] using a pre-extracted payload.
pub(crate) fn apply_mesh_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedMeshRenderablesUpdate,
    scene_id: i32,
) {
    profiling::scope!("scene::apply_meshes");
    swap_remove_dense_indices(&mut space.static_mesh_renderers, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        let instance_id = space.allocate_mesh_renderer_instance_id();
        space.static_mesh_renderers.push(StaticMeshRenderer {
            instance_id,
            node_id,
            layer: LayerType::Hidden,
            ..Default::default()
        });
    }
    let packed_ref = extracted.mesh_materials_and_property_blocks.as_deref();
    let mut packed_cursor = 0usize;
    let len = space.static_mesh_renderers.len();
    let mut plans: Vec<Option<MeshRendererStateApplyPlan>> = vec![None; len];
    let mut active_rows = 0usize;
    let mut valid_plan_count = 0usize;
    for state in &extracted.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        active_rows += 1;
        let idx = state.renderable_index as usize;
        let plan = decode_mesh_renderer_state_plan(state, packed_ref, &mut packed_cursor);
        if let Some(slot) = plans.get_mut(idx) {
            if let Some(existing) = slot {
                existing.merge_later_row(plan);
            } else {
                *slot = Some(plan);
                valid_plan_count += 1;
            }
        } else {
            warn_oob_renderable_index_once(
                scene_id,
                "static",
                idx,
                len,
                &STATIC_MESH_OOB_WARNED_SCENES,
            );
        }
    }
    apply_static_mesh_state_plans(
        &mut space.static_mesh_renderers,
        plans,
        active_rows,
        valid_plan_count,
    );
}

fn apply_static_mesh_state_plans(
    renderers: &mut [StaticMeshRenderer],
    plans: Vec<Option<MeshRendererStateApplyPlan>>,
    active_rows: usize,
    valid_plan_count: usize,
) {
    if valid_plan_count == 0 {
        return;
    }
    let admission =
        admit_renderable_update_items(valid_plan_count, current_reference_worker_count());
    record_parallel_admission(
        "static_mesh_state_apply",
        active_rows,
        valid_plan_count,
        admission,
    );
    if let Some(chunk_size) = admission.chunk_size() {
        renderers
            .par_iter_mut()
            .zip(plans.into_par_iter())
            .with_min_len(chunk_size)
            .for_each(|(renderer, plan)| {
                if let Some(plan) = plan {
                    plan.apply_to(renderer);
                }
            });
    } else {
        for (renderer, plan) in renderers.iter_mut().zip(plans) {
            if let Some(plan) = plan {
                plan.apply_to(renderer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::meshes::types::MeshRendererInstanceId;
    use crate::scene::render_space::RenderSpaceState;
    use crate::shared::MeshRendererState;

    use super::{ExtractedMeshRenderablesUpdate, apply_mesh_renderables_update_extracted};

    #[test]
    fn static_instance_ids_are_fresh_and_survive_swap_remove() {
        let mut space = RenderSpaceState::default();
        apply_mesh_renderables_update_extracted(
            &mut space,
            &ExtractedMeshRenderablesUpdate {
                additions: vec![10, 11, 12, -1],
                ..Default::default()
            },
            1,
        );
        let ids: Vec<_> = space
            .static_mesh_renderers
            .iter()
            .map(|renderer| renderer.instance_id)
            .collect();
        assert_eq!(
            ids,
            vec![
                MeshRendererInstanceId(1),
                MeshRendererInstanceId(2),
                MeshRendererInstanceId(3),
            ]
        );

        apply_mesh_renderables_update_extracted(
            &mut space,
            &ExtractedMeshRenderablesUpdate {
                removals: vec![1, -1],
                additions: vec![13, -1],
                ..Default::default()
            },
            1,
        );
        let ids: Vec<_> = space
            .static_mesh_renderers
            .iter()
            .map(|renderer| renderer.instance_id)
            .collect();
        assert_eq!(
            ids,
            vec![
                MeshRendererInstanceId(1),
                MeshRendererInstanceId(3),
                MeshRendererInstanceId(4),
            ]
        );
    }

    #[test]
    fn parallel_state_apply_preserves_prior_material_update_when_later_row_skips_materials() {
        const RENDERER_COUNT: usize = 128;
        let mut space = RenderSpaceState::default();
        let additions = (0..RENDERER_COUNT as i32).chain([-1]).collect::<Vec<_>>();
        apply_mesh_renderables_update_extracted(
            &mut space,
            &ExtractedMeshRenderablesUpdate {
                additions,
                ..Default::default()
            },
            2,
        );

        let mut mesh_states = (0..RENDERER_COUNT)
            .map(|index| MeshRendererState {
                renderable_index: index as i32,
                mesh_asset_id: index as i32,
                material_count: 1,
                material_property_block_count: -1,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        mesh_states.push(MeshRendererState {
            renderable_index: 0,
            mesh_asset_id: 999,
            material_count: -1,
            material_property_block_count: -1,
            ..Default::default()
        });
        mesh_states.push(MeshRendererState {
            renderable_index: -1,
            ..Default::default()
        });
        let material_ids = (0..RENDERER_COUNT as i32)
            .map(|index| 10_000 + index)
            .collect::<Vec<_>>();

        apply_mesh_renderables_update_extracted(
            &mut space,
            &ExtractedMeshRenderablesUpdate {
                mesh_states,
                mesh_materials_and_property_blocks: Some(material_ids),
                ..Default::default()
            },
            2,
        );

        let first = &space.static_mesh_renderers[0];
        assert_eq!(first.mesh_asset_id, 999);
        assert_eq!(first.primary_material_asset_id, Some(10_000));
        assert_eq!(first.material_slots[0].material_asset_id, 10_000);
        assert_eq!(
            space.static_mesh_renderers[RENDERER_COUNT - 1].primary_material_asset_id,
            Some(10_000 + RENDERER_COUNT as i32 - 1)
        );
    }
}
