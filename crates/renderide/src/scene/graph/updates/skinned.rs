//! Skinned mesh renderable updates from host.
//!
//! Renderable removals use **buffer order** and swap-with-last, matching Unity `RenderableManager`.

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Drawable, Scene};
use crate::shared::{
    BlendshapeUpdate, BlendshapeUpdateBatch, BoneAssignment, LayerType, ShadowCastMode,
    SkinnedMeshRenderablesUpdate,
};

use super::super::error::SceneError;
use super::super::pods::MeshRendererStatePod;
use super::super::world_matrices::fixup_transform_id;
use super::mesh_material_slots::apply_mesh_renderer_state_row;

/// Scene IDs for which we logged a bone_assignments / bone_transform_indexes length mismatch.
static BONE_INDEX_EMPTY_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Applies skinned mesh renderable updates. Bone transform IDs are fixup'd when transforms
/// are removed via swap_remove: references to the removed ID become -1; references to the
/// last index (now swapped into the removed slot) become the removed ID.
pub(crate) fn apply_skinned_mesh_renderables_update(
    scene: &mut Scene,
    shm: &mut SharedMemoryAccessor,
    update: &SkinnedMeshRenderablesUpdate,
    _frame_index: i32,
    transform_removals: &[(i32, usize)],
) -> Result<(), SceneError> {
    for &(removed_id, last_index) in transform_removals {
        for entry in &mut scene.skinned_drawables {
            entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
            if let Some(ref mut ids) = entry.bone_transform_ids {
                for id in ids.iter_mut() {
                    *id = fixup_transform_id(*id, removed_id, last_index);
                }
            }
            if let Some(rid) = entry.root_bone_transform_id {
                entry.root_bone_transform_id =
                    Some(fixup_transform_id(rid, removed_id, last_index));
            }
        }
    }

    if update.removals.length > 0 {
        let ctx = format!("skinned removals scene_id={}", scene.id);
        let removals = shm
            .access_with_context::<i32>(&update.removals, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        for &raw in removals.iter().take_while(|&&i| i >= 0) {
            let idx = raw as usize;
            if idx < scene.skinned_drawables.len() {
                scene.skinned_drawables.swap_remove(idx);
            }
        }
    }
    if update.additions.length > 0 {
        let ctx = format!("skinned additions scene_id={}", scene.id);
        let additions = shm
            .access_with_context::<i32>(&update.additions, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        let added_node_ids: Vec<i32> = additions.iter().take_while(|&&i| i >= 0).copied().collect();
        for &node_id in &added_node_ids {
            let layer = scene
                .layer_assignments
                .get(&node_id)
                .copied()
                .unwrap_or(LayerType::overlay);
            scene.skinned_drawables.push(Drawable {
                node_id,
                layer,
                mesh_handle: -1,
                material_handle: None,
                sort_key: 0,
                is_skinned: true,
                bone_transform_ids: None,
                root_bone_transform_id: None,
                blend_shape_weights: Some(vec![]),
                stencil_state: None,
                material_override_block_id: None,
                mesh_renderer_property_block_slot0_id: None,
                material_slots: Vec::new(),
                render_transform_override: None,
                shadow_cast_mode: ShadowCastMode::on,
            });
        }
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("skinned mesh_states scene_id={}", scene.id);
        let states = shm
            .access_with_context::<MeshRendererStatePod>(&update.mesh_states, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        let packed_ids = if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!(
                "skinned mesh_materials_and_property_blocks scene_id={}",
                scene.id
            );
            Some(
                shm.access_with_context::<i32>(&update.mesh_materials_and_property_blocks, &ctx_m)
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
            let drawable = scene.skinned_drawables.get_mut(idx);
            apply_mesh_renderer_state_row(drawable, &state, packed_ref, &mut packed_cursor);
        }
    }
    if update.bone_assignments.length > 0 {
        if update.bone_transform_indexes.length <= 0 {
            if let Ok(mut warned) = BONE_INDEX_EMPTY_WARNED_SCENES.lock()
                && warned.insert(scene.id)
            {
                logger::warn!(
                    "Skinned update: bone_assignments present but bone_transform_indexes empty (scene_id={}); skipping bone index application",
                    scene.id
                );
            }
        } else {
            let ctx_assign = format!("skinned bone_assignments scene_id={}", scene.id);
            let assignments = shm
                .access_with_context::<BoneAssignment>(&update.bone_assignments, &ctx_assign)
                .map_err(SceneError::SharedMemoryAccess)?;
            let ctx_idx = format!("skinned bone_transform_indexes scene_id={}", scene.id);
            let indexes = shm
                .access_with_context::<i32>(&update.bone_transform_indexes, &ctx_idx)
                .map_err(SceneError::SharedMemoryAccess)?;
            let mut index_offset = 0;
            for assignment in &assignments {
                if assignment.renderable_index < 0 {
                    break;
                }
                let idx = assignment.renderable_index as usize;
                let bone_count = assignment.bone_count.max(0) as usize;
                if idx < scene.skinned_drawables.len() && index_offset + bone_count <= indexes.len()
                {
                    let ids: Vec<i32> = indexes[index_offset..index_offset + bone_count].to_vec();
                    scene.skinned_drawables[idx].bone_transform_ids = Some(ids);
                    scene.skinned_drawables[idx].root_bone_transform_id =
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

    // Apply blendshape weight updates from host. Matches SkinnedMeshRendererManager:
    // iterate BlendshapeUpdateBatch, apply BlendshapeUpdate entries to each drawable.
    // Skip when either buffer is empty (host may not send updates every frame);
    // drawables retain their previous blend_shape_weights. Never call access_copy_diagnostic
    // with length-0 descriptors.
    if !update.blendshape_update_batches.is_empty() && !update.blendshape_updates.is_empty() {
        let ctx_batch = format!("skinned blendshape_update_batches scene_id={}", scene.id);
        let batches = shm
            .access_with_context::<BlendshapeUpdateBatch>(
                &update.blendshape_update_batches,
                &ctx_batch,
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let ctx_upd = format!("skinned blendshape_updates scene_id={}", scene.id);
        let updates = shm
            .access_with_context::<BlendshapeUpdate>(&update.blendshape_updates, &ctx_upd)
            .map_err(SceneError::SharedMemoryAccess)?;
        let mut update_offset = 0;
        for batch in &batches {
            if batch.renderable_index < 0 {
                break;
            }
            let idx = batch.renderable_index as usize;
            let count = batch.blendshape_update_count.max(0) as usize;
            if idx < scene.skinned_drawables.len() && update_offset + count <= updates.len() {
                let drawable = &mut scene.skinned_drawables[idx];
                let weights = drawable.blend_shape_weights.get_or_insert_with(Vec::new);
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
