//! Transform hierarchy updates from host.
//!
//! Removal indices are applied in **shared-memory buffer order** (first entry first, `-1`
//! terminates), matching FrooxEngine `RenderTransformManager.FillUpdate` and Unity
//! `TransformManager.HandleUpdate` swap-with-last semantics. Do not reorder removals (for example
//! descending sort): the host records each id after prior removals have already been applied on
//! its side.
//!
//! When a [`TransformsUpdate`] carries no structural or pose changes (no cache resize, no
//! removals, no growth to `target_transform_count`, no parent or pose entries applied), the
//! world-matrix cache is left undisturbed: [`mark_descendants_uncomputed`] is skipped and the
//! scene id is not inserted into `world_matrices_dirty`, so [`SceneGraph::compute_world_matrices`]
//! can early-out on the next frame.

use std::collections::HashSet;

use glam::Mat4;

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Scene, SceneId};
use crate::shared::{TransformParentUpdate, TransformPoseUpdate, TransformsUpdate};

use super::super::error::SceneError;
use super::super::pose::{PoseValidation, render_transform_identity};
use super::super::world_matrices::{
    SceneCache, fixup_transform_id, mark_descendants_uncomputed, rebuild_children,
};

/// Applies swap-with-last transform removals in **host buffer order** (iterate until `id < 0`).
///
/// Each entry is an index valid for the scene **at the moment that removal is applied**, matching
/// FrooxEngine and Unity. Returns `(removal_log, had_any_removal)` for skinned-mesh fixup and
/// cache invalidation; caller should set `cache.children_dirty` when `had_any_removal` is true.
fn apply_transform_removals_ordered(
    scene: &mut Scene,
    cache: &mut SceneCache,
    removals: &[i32],
) -> (Vec<(i32, usize)>, bool) {
    let mut transform_removals = Vec::new();
    let mut had_removal = false;
    for &raw in removals.iter().take_while(|&&i| i >= 0) {
        let idx = raw as usize;
        if idx >= scene.nodes.len() {
            continue;
        }
        let removed_id = raw;
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
        had_removal = true;
    }
    (transform_removals, had_removal)
}

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
    // When true, world matrix cache must be recomputed and `mark_descendants_uncomputed` must run.
    let mut invalidate_world_matrices = false;

    if cache.world_matrices.len() != scene.nodes.len() {
        cache
            .world_matrices
            .resize(scene.nodes.len(), Mat4::IDENTITY);
        cache.computed.resize(scene.nodes.len(), false);
        cache
            .local_matrices
            .resize(scene.nodes.len(), Mat4::IDENTITY);
        cache.local_dirty.resize(scene.nodes.len(), true);
        invalidate_world_matrices = true;
    }

    if update.removals.length > 0 {
        let ctx = format!("transforms removals scene_id={}", scene_id);
        let removals = shm
            .access_with_context::<i32>(&update.removals, &ctx)
            .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
        let (removal_pairs, had_removal) =
            apply_transform_removals_ordered(scene, cache, removals.as_slice());
        transform_removals.extend(removal_pairs);
        if had_removal {
            // Structure changed: children cache needs rebuild before next mark_descendants call.
            cache.children_dirty = true;
            invalidate_world_matrices = true;
        }
    }

    let nodes_before_grow = scene.nodes.len();
    while (scene.nodes.len() as i32) < update.target_transform_count {
        scene.nodes.push(render_transform_identity());
        scene.node_parents.push(-1);
        cache.world_matrices.push(Mat4::IDENTITY);
        cache.computed.push(false);
        cache.local_matrices.push(Mat4::IDENTITY);
        cache.local_dirty.push(true);
    }
    if scene.nodes.len() != nodes_before_grow {
        invalidate_world_matrices = true;
    }

    let mut changed_indices = std::collections::HashSet::new();

    if update.parent_updates.length > 0 {
        let ctx = format!("transforms parent_updates scene_id={}", scene_id);
        let parents = shm
            .access_with_context::<TransformParentUpdate>(&update.parent_updates, &ctx)
            .map_err(|e| SceneError::SharedMemoryAccess(e.to_string()))?;
        let mut had_parent_update = false;
        for pu in parents {
            if pu.transform_id < 0 {
                break;
            }
            if (pu.transform_id as usize) < scene.node_parents.len() {
                scene.node_parents[pu.transform_id as usize] = pu.new_parent_id;
                changed_indices.insert(pu.transform_id as usize);
                had_parent_update = true;
            }
        }
        if had_parent_update {
            cache.children_dirty = true;
            invalidate_world_matrices = true;
        }
    }

    if update.pose_updates.length > 0 {
        let ctx = format!("transforms pose_updates scene_id={}", scene_id);
        let poses = shm
            .access_with_context::<TransformPoseUpdate>(&update.pose_updates, &ctx)
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

    if !changed_indices.is_empty() {
        invalidate_world_matrices = true;
    }

    for i in &changed_indices {
        if *i < cache.computed.len() {
            cache.computed[*i] = false;
        }
        if *i < cache.local_dirty.len() {
            cache.local_dirty[*i] = true;
        }
    }
    // Rebuild children index if structure changed, then propagate dirty flags down.
    if cache.children_dirty {
        rebuild_children(&scene.node_parents, scene.nodes.len(), &mut cache.children);
        cache.children_dirty = false;
    }
    if invalidate_world_matrices {
        mark_descendants_uncomputed(&cache.children, &mut cache.computed);
        world_matrices_dirty.insert(scene_id);
    }

    Ok(transform_removals)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Quaternion, Vector3};

    use super::*;
    use crate::scene::Scene;
    use crate::shared::RenderTransform;

    fn node_tagged(i: f32) -> RenderTransform {
        RenderTransform {
            position: Vector3::new(i, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::identity(),
        }
    }

    fn cache_matching(nodes_len: usize) -> SceneCache {
        SceneCache {
            world_matrices: vec![Mat4::IDENTITY; nodes_len],
            computed: vec![false; nodes_len],
            local_matrices: vec![Mat4::IDENTITY; nodes_len],
            local_dirty: vec![true; nodes_len],
            visit_epoch: vec![0; nodes_len],
            walk_epoch: 0,
            children: vec![],
            children_dirty: true,
        }
    }

    /// Two removal orders must yield different surviving nodes; descending sort would collapse them.
    #[test]
    fn transform_removals_buffer_order_zero_then_one_vs_one_then_zero() {
        let mut scene_a = Scene::default();
        for i in 0..4 {
            scene_a.nodes.push(node_tagged(i as f32));
            scene_a.node_parents.push(-1);
        }
        let mut cache_a = cache_matching(4);
        let (_log, _) = apply_transform_removals_ordered(&mut scene_a, &mut cache_a, &[0, 1, -1]);
        assert_eq!(scene_a.nodes.len(), 2);
        assert!((scene_a.nodes[0].position.x - 3.0).abs() < 1e-5);
        assert!((scene_a.nodes[1].position.x - 2.0).abs() < 1e-5);

        let mut scene_b = Scene::default();
        for i in 0..4 {
            scene_b.nodes.push(node_tagged(i as f32));
            scene_b.node_parents.push(-1);
        }
        let mut cache_b = cache_matching(4);
        let (_log, _) = apply_transform_removals_ordered(&mut scene_b, &mut cache_b, &[1, 0, -1]);
        assert_eq!(scene_b.nodes.len(), 2);
        assert!((scene_b.nodes[0].position.x - 2.0).abs() < 1e-5);
        assert!((scene_b.nodes[1].position.x - 3.0).abs() < 1e-5);
    }

    #[test]
    fn transform_removals_negative_terminates() {
        let mut scene = Scene::default();
        for i in 0..3 {
            scene.nodes.push(node_tagged(i as f32));
            scene.node_parents.push(-1);
        }
        let mut cache = cache_matching(3);
        let (_log, _) = apply_transform_removals_ordered(&mut scene, &mut cache, &[0, -1, 1]);
        assert_eq!(scene.nodes.len(), 2);
        assert!((scene.nodes[0].position.x - 2.0).abs() < 1e-5);
        assert!((scene.nodes[1].position.x - 1.0).abs() < 1e-5);
    }
}
