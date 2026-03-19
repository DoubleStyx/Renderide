//! Transform hierarchy updates from host.

use std::collections::HashSet;

use glam::Mat4;

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Scene, SceneId};
use crate::shared::{TransformParentUpdate, TransformPoseUpdate, TransformsUpdate};

use super::super::error::SceneError;
use super::super::pose::{PoseValidation, render_transform_identity};
use super::super::world_matrices::{SceneCache, fixup_transform_id, mark_descendants_uncomputed};

/// Applies transform updates: removals, parent changes, pose updates.
/// Returns transform removals (removed_id, last_index) for skinned mesh fixup.
pub(crate) fn apply_transforms_update(
    scene: &mut Scene,
    cache: &mut SceneCache,
    world_matrices_dirty: &mut HashSet<SceneId>,
    scene_id: SceneId,
    shm: &mut SharedMemoryAccessor,
    update: &TransformsUpdate,
    frame_index: i32,
) -> Result<Vec<(i32, usize)>, SceneError> {
    let mut transform_removals = Vec::new();

    if cache.world_matrices.len() != scene.nodes.len() {
        cache
            .world_matrices
            .resize(scene.nodes.len(), Mat4::IDENTITY);
        cache.computed.resize(scene.nodes.len(), false);
        cache
            .local_matrices
            .resize(scene.nodes.len(), Mat4::IDENTITY);
        cache.local_dirty.resize(scene.nodes.len(), true);
    }

    if update.removals.length > 0 {
        let ctx = format!("transforms removals scene_id={}", scene_id);
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
        let mut indices: Vec<usize> = removals
            .iter()
            .take_while(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();
        indices.sort_by(|a, b| b.cmp(a));
        for &idx in &indices {
            if idx >= scene.nodes.len() {
                continue;
            }
            let removed_id = idx as i32;
            let last_index = scene.nodes.len() - 1;

            for (i, parent) in scene.node_parents.iter_mut().enumerate() {
                if *parent == removed_id {
                    *parent = -1;
                    if i < cache.computed.len() {
                        cache.computed[i] = false;
                    }
                } else if *parent == last_index as i32 {
                    *parent = removed_id;
                }
            }
            for entry in &mut scene.drawables {
                entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index);
            }
            if idx != last_index
                && let Some(layer) = scene.layer_assignments.remove(&(last_index as i32))
            {
                scene.layer_assignments.insert(removed_id, layer);
            }
            scene.layer_assignments.remove(&removed_id);
            transform_removals.push((removed_id, last_index));

            scene.nodes.swap_remove(idx);
            scene.node_parents.swap_remove(idx);
            if idx < cache.world_matrices.len() {
                cache.world_matrices.swap_remove(idx);
                cache.computed.swap_remove(idx);
                cache.local_matrices.swap_remove(idx);
                cache.local_dirty.swap_remove(idx);
            }
        }
    }

    while (scene.nodes.len() as i32) < update.target_transform_count {
        scene.nodes.push(render_transform_identity());
        scene.node_parents.push(-1);
        cache.world_matrices.push(Mat4::IDENTITY);
        cache.computed.push(false);
        cache.local_matrices.push(Mat4::IDENTITY);
        cache.local_dirty.push(true);
    }

    let mut changed_indices = std::collections::HashSet::new();

    if update.parent_updates.length > 0 {
        let ctx = format!("transforms parent_updates scene_id={}", scene_id);
        let parents = shm
            .access_copy_diagnostic_with_context::<TransformParentUpdate>(
                &update.parent_updates,
                Some(&ctx),
            )
            .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
        for pu in parents {
            if pu.transform_id < 0 {
                break;
            }
            if (pu.transform_id as usize) < scene.node_parents.len() {
                scene.node_parents[pu.transform_id as usize] = pu.new_parent_id;
                changed_indices.insert(pu.transform_id as usize);
            }
        }
    }

    if update.pose_updates.length > 0 {
        let ctx = format!("transforms pose_updates scene_id={}", scene_id);
        let poses = shm
            .access_copy_diagnostic_with_context::<TransformPoseUpdate>(
                &update.pose_updates,
                Some(&ctx),
            )
            .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
        for pu in &poses {
            if pu.transform_id < 0 {
                break;
            }
            if (pu.transform_id as usize) < scene.nodes.len() {
                let validation = PoseValidation {
                    pose: &pu.pose,
                    frame_index,
                    scene_id: scene.id,
                    transform_id: pu.transform_id,
                };
                if validation.is_valid() {
                    scene.nodes[pu.transform_id as usize] = pu.pose;
                } else {
                    logger::error!(
                        "Invalid pose scene={} transform={} frame={}: using identity",
                        scene.id,
                        pu.transform_id,
                        frame_index
                    );
                    scene.nodes[pu.transform_id as usize] = render_transform_identity();
                }
                changed_indices.insert(pu.transform_id as usize);
            }
        }
    }

    for i in &changed_indices {
        if *i < cache.computed.len() {
            cache.computed[*i] = false;
        }
        if *i < cache.local_dirty.len() {
            cache.local_dirty[*i] = true;
        }
    }
    mark_descendants_uncomputed(&scene.node_parents, &mut cache.computed);
    world_matrices_dirty.insert(scene_id);

    Ok(transform_removals)
}
