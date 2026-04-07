//! Static and skinned mesh renderable updates from shared memory (Unity `RenderableManager` parity).

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    BlendshapeUpdate, BlendshapeUpdateBatch, BoneAssignment, LayerType, MeshRenderablesUpdate,
    MeshRendererState, SkinnedMeshRenderablesUpdate,
};

use super::error::SceneError;
use super::mesh_material_row::apply_mesh_renderer_state_row;
use super::mesh_renderable::{SkinnedMeshRenderer, StaticMeshRenderer};
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

static BONE_INDEX_EMPTY_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn fixup_skinned_bones_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for ev in removals {
        let removed_id = ev.removed_index;
        let last_index = ev.last_index_before_swap;
        for entry in &mut space.skinned_mesh_renderers {
            entry.base.node_id = fixup_transform_id(entry.base.node_id, removed_id, last_index);
            for id in &mut entry.bone_transform_indices {
                *id = fixup_transform_id(*id, removed_id, last_index);
            }
            if let Some(rid) = entry.root_bone_transform_id {
                entry.root_bone_transform_id =
                    Some(fixup_transform_id(rid, removed_id, last_index));
            }
        }
    }
}

/// Applies [`MeshRenderablesUpdate`]: removals → additions → mesh states + packed materials.
pub(crate) fn apply_mesh_renderables_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &MeshRenderablesUpdate,
    _frame_index: i32,
    scene_id: i32,
) -> Result<(), SceneError> {
    if update.removals.length > 0 {
        let ctx = format!("mesh removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &raw in removals.iter().take_while(|&&i| i >= 0) {
            let idx = raw as usize;
            if idx < space.static_mesh_renderers.len() {
                space.static_mesh_renderers.swap_remove(idx);
            }
        }
    }
    if update.additions.length > 0 {
        let ctx = format!("mesh additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        let added_node_ids: Vec<i32> = additions.iter().take_while(|&&i| i >= 0).copied().collect();
        for &node_id in &added_node_ids {
            space.static_mesh_renderers.push(StaticMeshRenderer {
                node_id,
                layer: LayerType::hidden,
                ..Default::default()
            });
        }
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("mesh mesh_states scene_id={scene_id}");
        let states = shm
            .access_copy_diagnostic_with_context::<MeshRendererState>(
                &update.mesh_states,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let packed_ids = if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!("mesh mesh_materials_and_property_blocks scene_id={scene_id}");
            Some(
                shm.access_copy_diagnostic_with_context::<i32>(
                    &update.mesh_materials_and_property_blocks,
                    Some(&ctx_m),
                )
                .map_err(SceneError::SharedMemoryAccess)?,
            )
        } else {
            None
        };
        let packed_ref = packed_ids.as_deref();
        let mut packed_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let drawable = space.static_mesh_renderers.get_mut(idx);
            apply_mesh_renderer_state_row(drawable, &state, packed_ref, &mut packed_cursor);
        }
    }
    Ok(())
}

/// Applies [`SkinnedMeshRenderablesUpdate`] after optional transform removals were applied.
pub(crate) fn apply_skinned_mesh_renderables_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &SkinnedMeshRenderablesUpdate,
    _frame_index: i32,
    scene_id: i32,
    transform_removals: &[TransformRemovalEvent],
) -> Result<(), SceneError> {
    fixup_skinned_bones_for_transform_removals(space, transform_removals);

    if update.removals.length > 0 {
        let ctx = format!("skinned removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &raw in removals.iter().take_while(|&&i| i >= 0) {
            let idx = raw as usize;
            if idx < space.skinned_mesh_renderers.len() {
                space.skinned_mesh_renderers.swap_remove(idx);
            }
        }
    }
    if update.additions.length > 0 {
        let ctx = format!("skinned additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        let added_node_ids: Vec<i32> = additions.iter().take_while(|&&i| i >= 0).copied().collect();
        for &node_id in &added_node_ids {
            space.skinned_mesh_renderers.push(SkinnedMeshRenderer {
                base: StaticMeshRenderer {
                    node_id,
                    layer: LayerType::hidden,
                    ..Default::default()
                },
                ..Default::default()
            });
        }
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("skinned mesh_states scene_id={scene_id}");
        let states = shm
            .access_copy_diagnostic_with_context::<MeshRendererState>(
                &update.mesh_states,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let packed_ids = if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!("skinned mesh_materials_and_property_blocks scene_id={scene_id}");
            Some(
                shm.access_copy_diagnostic_with_context::<i32>(
                    &update.mesh_materials_and_property_blocks,
                    Some(&ctx_m),
                )
                .map_err(SceneError::SharedMemoryAccess)?,
            )
        } else {
            None
        };
        let packed_ref = packed_ids.as_deref();
        let mut packed_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let drawable = space.skinned_mesh_renderers.get_mut(idx);
            apply_mesh_renderer_state_row(drawable, &state, packed_ref, &mut packed_cursor);
        }
    }
    if update.bone_assignments.length > 0 {
        if update.bone_transform_indexes.length <= 0 {
            if let Ok(mut warned) = BONE_INDEX_EMPTY_WARNED_SCENES.lock() {
                if warned.insert(scene_id) {
                    logger::warn!(
                        "Skinned update: bone_assignments present but bone_transform_indexes empty (scene_id={scene_id}); skipping bone index application"
                    );
                }
            }
        } else {
            let ctx_assign = format!("skinned bone_assignments scene_id={scene_id}");
            let assignments = shm
                .access_copy_diagnostic_with_context::<BoneAssignment>(
                    &update.bone_assignments,
                    Some(&ctx_assign),
                )
                .map_err(SceneError::SharedMemoryAccess)?;
            let ctx_idx = format!("skinned bone_transform_indexes scene_id={scene_id}");
            let indexes = shm
                .access_copy_diagnostic_with_context::<i32>(
                    &update.bone_transform_indexes,
                    Some(&ctx_idx),
                )
                .map_err(SceneError::SharedMemoryAccess)?;
            let mut index_offset = 0;
            for assignment in &assignments {
                if assignment.renderable_index < 0 {
                    break;
                }
                let idx = assignment.renderable_index as usize;
                let bone_count = assignment.bone_count.max(0) as usize;
                if idx < space.skinned_mesh_renderers.len()
                    && index_offset + bone_count <= indexes.len()
                {
                    let ids: Vec<i32> = indexes[index_offset..index_offset + bone_count].to_vec();
                    space.skinned_mesh_renderers[idx].bone_transform_indices = ids;
                    space.skinned_mesh_renderers[idx].root_bone_transform_id =
                        if assignment.root_bone_transform_id >= 0 {
                            Some(assignment.root_bone_transform_id)
                        } else {
                            None
                        };
                }
                index_offset += bone_count;
            }
        }
    }

    if update.blendshape_update_batches.length > 0 && update.blendshape_updates.length > 0 {
        let ctx_batch = format!("skinned blendshape_update_batches scene_id={scene_id}");
        let batches = shm
            .access_copy_diagnostic_with_context::<BlendshapeUpdateBatch>(
                &update.blendshape_update_batches,
                Some(&ctx_batch),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let ctx_upd = format!("skinned blendshape_updates scene_id={scene_id}");
        let updates = shm
            .access_copy_diagnostic_with_context::<BlendshapeUpdate>(
                &update.blendshape_updates,
                Some(&ctx_upd),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let mut update_offset = 0;
        for batch in &batches {
            if batch.renderable_index < 0 {
                break;
            }
            let idx = batch.renderable_index as usize;
            let count = batch.blendshape_update_count.max(0) as usize;
            if idx < space.skinned_mesh_renderers.len() && update_offset + count <= updates.len() {
                let weights = &mut space.skinned_mesh_renderers[idx].base.blend_shape_weights;
                for upd in &updates[update_offset..update_offset + count] {
                    let bi = upd.blendshape_index.max(0) as usize;
                    let needed = bi + 1;
                    if weights.len() < needed {
                        weights.resize(needed, 0.0);
                    }
                    weights[bi] = upd.weight;
                }
            }
            update_offset += count;
        }
    }

    Ok(())
}
