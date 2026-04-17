//! Transform hierarchy updates from host shared memory (dense indices, ordered removals).
//!
//! Removal indices are applied in **buffer order** (first entry first, `-1` terminates), matching host
//! swap-with-last semantics. **Do not** sort removals.
//!
//! Static mesh [`node_id`](crate::scene::mesh_renderable::StaticMeshRenderer::node_id) values are
//! remapped before each `swap_remove`, matching Unity `Transform` identity semantics and the same
//! `fixup_transform_id` rule used for skinned mesh bone indices in `mesh_apply`.

use std::collections::HashSet;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    TransformParentUpdate, TransformPoseUpdate, TransformsUpdate,
    TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
};

use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::pose::{render_transform_identity, PoseValidation};
use super::render_space::RenderSpaceState;
use super::world::{
    fixup_transform_id, mark_descendants_uncomputed, rebuild_children, WorldTransformCache,
};

/// Shared-memory accessor and host transform payload for [`apply_transforms_update`].
///
/// Groups the IPC read surfaces used with every transform apply so call sites stay compact.
pub struct TransformsUpdateBuffers<'a> {
    /// Ring-buffer accessor for removal and pose rows.
    pub shm: &'a mut SharedMemoryAccessor,
    /// Host transform delta for this render space.
    pub update: &'a TransformsUpdate,
    /// Host frame index for pose validation.
    pub frame_index: i32,
}

/// One successful transform removal: dense index removed and last valid index before `swap_remove`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformRemovalEvent {
    /// Removed dense transform index (`i32`, same as host removal buffer entry).
    pub removed_index: i32,
    /// Last valid index in `nodes` before the slot was removed (swapped-into source).
    pub last_index_before_swap: usize,
}

fn apply_transform_removals_ordered(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    removals: &[i32],
    out: &mut Vec<TransformRemovalEvent>,
) -> bool {
    let mut had_removal = false;
    for &raw in removals.iter().take_while(|&&i| i >= 0) {
        let idx = raw as usize;
        if idx >= space.nodes.len() {
            continue;
        }
        let removed_id = raw;
        let last_index_before_swap = space.nodes.len() - 1;

        for (i, parent) in space.node_parents.iter_mut().enumerate() {
            if *parent == removed_id {
                *parent = -1;
                if i < cache.computed.len() {
                    cache.computed[i] = false;
                }
            } else if *parent == last_index_before_swap as i32 {
                *parent = removed_id;
            }
        }

        for entry in &mut space.static_mesh_renderers {
            entry.node_id = fixup_transform_id(entry.node_id, removed_id, last_index_before_swap);
        }

        space.nodes.swap_remove(idx);
        space.node_parents.swap_remove(idx);
        if idx < cache.world_matrices.len() {
            cache.world_matrices.swap_remove(idx);
            cache.computed.swap_remove(idx);
            cache.local_matrices.swap_remove(idx);
            cache.local_dirty.swap_remove(idx);
            if idx < cache.visit_epoch.len() {
                cache.visit_epoch.swap_remove(idx);
            }
        }
        out.push(TransformRemovalEvent {
            removed_index: removed_id,
            last_index_before_swap,
        });
        had_removal = true;
    }
    had_removal
}

/// Resizes world/cache sidecars when the node table grew or shrank on host.
fn ensure_world_cache_matches_node_count(
    space: &RenderSpaceState,
    cache: &mut WorldTransformCache,
    invalidate_world: &mut bool,
) {
    if cache.world_matrices.len() == space.nodes.len() {
        return;
    }
    cache
        .world_matrices
        .resize(space.nodes.len(), glam::Mat4::IDENTITY);
    cache.computed.resize(space.nodes.len(), false);
    cache
        .local_matrices
        .resize(space.nodes.len(), glam::Mat4::IDENTITY);
    cache.local_dirty.resize(space.nodes.len(), true);
    cache.visit_epoch.resize(space.nodes.len(), 0);
    *invalidate_world = true;
}

/// Extends dense transform buffers up to `target_transform_count` with identity locals.
fn grow_transform_buffers_to_target(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    update: &TransformsUpdate,
    invalidate_world: &mut bool,
) {
    let nodes_before = space.nodes.len();
    while (space.nodes.len() as i32) < update.target_transform_count {
        space.nodes.push(render_transform_identity());
        space.node_parents.push(-1);
        cache.world_matrices.push(glam::Mat4::IDENTITY);
        cache.computed.push(false);
        cache.local_matrices.push(glam::Mat4::IDENTITY);
        cache.local_dirty.push(true);
        cache.visit_epoch.push(0);
    }
    if space.nodes.len() != nodes_before {
        *invalidate_world = true;
    }
}

/// Applies parent pointer deltas from shared memory.
fn apply_transform_parent_updates(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    shm: &mut SharedMemoryAccessor,
    update: &TransformsUpdate,
    sid: i32,
    changed: &mut HashSet<usize>,
    invalidate_world: &mut bool,
) -> Result<(), SceneError> {
    if update.parent_updates.length <= 0 {
        return Ok(());
    }
    let ctx = format!("transforms parent_updates scene_id={sid}");
    let parents = shm
        .access_copy_diagnostic_with_context::<TransformParentUpdate>(
            &update.parent_updates,
            Some(&ctx),
        )
        .map_err(SceneError::SharedMemoryAccess)?;
    let mut had_parent = false;
    for pu in parents {
        if pu.transform_id < 0 {
            break;
        }
        if (pu.transform_id as usize) < space.node_parents.len() {
            space.node_parents[pu.transform_id as usize] = pu.new_parent_id;
            changed.insert(pu.transform_id as usize);
            had_parent = true;
        }
    }
    if had_parent {
        cache.children_dirty = true;
        *invalidate_world = true;
    }
    Ok(())
}

/// Applies pose rows, validating each against [`PoseValidation`].
fn apply_transform_pose_updates(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &TransformsUpdate,
    frame_index: i32,
    sid: i32,
    changed: &mut HashSet<usize>,
) -> Result<(), SceneError> {
    if update.pose_updates.length <= 0 {
        return Ok(());
    }
    let ctx = format!("transforms pose_updates scene_id={sid}");
    let poses = shm
        .access_copy_memory_packable_rows::<TransformPoseUpdate>(
            &update.pose_updates,
            TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
            Some(&ctx),
        )
        .map_err(SceneError::SharedMemoryAccess)?;
    for pu in &poses {
        if pu.transform_id < 0 {
            break;
        }
        if (pu.transform_id as usize) < space.nodes.len() {
            let validation = PoseValidation { pose: &pu.pose };
            if validation.is_valid() {
                space.nodes[pu.transform_id as usize] = pu.pose;
            } else {
                logger::error!(
                    "invalid pose scene={sid} transform={} frame={frame_index}: identity",
                    pu.transform_id
                );
                space.nodes[pu.transform_id as usize] = render_transform_identity();
            }
            changed.insert(pu.transform_id as usize);
        }
    }
    Ok(())
}

/// Marks per-node dirty flags after local transform edits.
fn propagate_transform_change_dirty_flags(
    cache: &mut WorldTransformCache,
    changed: &HashSet<usize>,
) {
    for i in changed {
        if *i < cache.computed.len() {
            cache.computed[*i] = false;
        }
        if *i < cache.local_dirty.len() {
            cache.local_dirty[*i] = true;
        }
    }
}

/// Applies removals, growth, parent updates, and pose updates for one space.
///
/// Appends transform removal events in buffer order into `removal_events_out` (cleared at entry)
/// for consumers (e.g. skinned bone index fixup).
pub fn apply_transforms_update(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    world_dirty: &mut HashSet<RenderSpaceId>,
    space_id: RenderSpaceId,
    buffers: TransformsUpdateBuffers<'_>,
    removal_events_out: &mut Vec<TransformRemovalEvent>,
) -> Result<(), SceneError> {
    let TransformsUpdateBuffers {
        shm,
        update,
        frame_index,
    } = buffers;
    let sid = space_id.0;
    let mut invalidate_world = false;
    removal_events_out.clear();

    ensure_world_cache_matches_node_count(space, cache, &mut invalidate_world);

    if update.removals.length > 0 {
        let ctx = format!("transforms removals scene_id={sid}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        let had_removal =
            apply_transform_removals_ordered(space, cache, removals.as_slice(), removal_events_out);
        if had_removal {
            cache.children_dirty = true;
            invalidate_world = true;
        }
    }

    grow_transform_buffers_to_target(space, cache, update, &mut invalidate_world);

    let mut changed = HashSet::new();

    apply_transform_parent_updates(
        space,
        cache,
        shm,
        update,
        sid,
        &mut changed,
        &mut invalidate_world,
    )?;
    apply_transform_pose_updates(space, shm, update, frame_index, sid, &mut changed)?;

    if !changed.is_empty() {
        invalidate_world = true;
    }

    propagate_transform_change_dirty_flags(cache, &changed);

    if cache.children_dirty {
        rebuild_children(&space.node_parents, space.nodes.len(), &mut cache.children);
        cache.children_dirty = false;
    }
    if invalidate_world {
        mark_descendants_uncomputed(&cache.children, &mut cache.computed);
        world_dirty.insert(space_id);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;
    use crate::scene::mesh_renderable::StaticMeshRenderer;
    use crate::shared::RenderTransform;

    fn node_tagged(i: f32) -> RenderTransform {
        RenderTransform {
            position: Vec3::new(i, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    fn empty_cache(nodes_len: usize) -> WorldTransformCache {
        WorldTransformCache {
            world_matrices: vec![glam::Mat4::IDENTITY; nodes_len],
            computed: vec![false; nodes_len],
            local_matrices: vec![glam::Mat4::IDENTITY; nodes_len],
            local_dirty: vec![true; nodes_len],
            visit_epoch: vec![0; nodes_len],
            walk_epoch: 0,
            children: Vec::new(),
            children_dirty: true,
        }
    }

    #[test]
    fn removal_order_zero_then_one_vs_one_then_zero() {
        let mut space = RenderSpaceState::default();
        for i in 0..4 {
            space.nodes.push(node_tagged(i as f32));
            space.node_parents.push(-1);
        }
        let mut cache = empty_cache(4);
        let mut ev = Vec::new();
        let _ = apply_transform_removals_ordered(&mut space, &mut cache, &[0, 1, -1], &mut ev);
        assert_eq!(ev.len(), 2);
        assert_eq!(space.nodes.len(), 2);
        assert!((space.nodes[0].position.x - 3.0).abs() < 1e-5);
        assert!((space.nodes[1].position.x - 2.0).abs() < 1e-5);

        let mut space_b = RenderSpaceState::default();
        for i in 0..4 {
            space_b.nodes.push(node_tagged(i as f32));
            space_b.node_parents.push(-1);
        }
        let mut cache_b = empty_cache(4);
        let mut ev_b = Vec::new();
        let _ =
            apply_transform_removals_ordered(&mut space_b, &mut cache_b, &[1, 0, -1], &mut ev_b);
        assert_eq!(ev_b.len(), 2);
        assert_eq!(space_b.nodes.len(), 2);
        assert!((space_b.nodes[0].position.x - 2.0).abs() < 1e-5);
        assert!((space_b.nodes[1].position.x - 3.0).abs() < 1e-5);
    }

    #[test]
    fn removal_negative_one_terminates() {
        let mut space = RenderSpaceState::default();
        for i in 0..3 {
            space.nodes.push(node_tagged(i as f32));
            space.node_parents.push(-1);
        }
        let mut cache = empty_cache(3);
        let mut ev = Vec::new();
        let _ = apply_transform_removals_ordered(&mut space, &mut cache, &[0, -1, 1], &mut ev);
        assert_eq!(ev.len(), 1);
        assert_eq!(space.nodes.len(), 2);
        assert!((space.nodes[0].position.x - 2.0).abs() < 1e-5);
        assert!((space.nodes[1].position.x - 1.0).abs() < 1e-5);
    }

    /// When transform `last_index` is swapped into `removed_id`, static mesh `node_id` must follow.
    #[test]
    fn static_mesh_node_id_remapped_on_swap_with_last() {
        let mut space = RenderSpaceState::default();
        for i in 0..4 {
            space.nodes.push(node_tagged(i as f32));
            space.node_parents.push(-1);
        }
        space.static_mesh_renderers.push(StaticMeshRenderer {
            node_id: 3,
            ..Default::default()
        });
        let mut cache = empty_cache(4);
        let mut ev = Vec::new();
        let _ = apply_transform_removals_ordered(&mut space, &mut cache, &[0, -1], &mut ev);
        assert_eq!(ev.len(), 1);
        assert_eq!(space.nodes.len(), 3);
        assert_eq!(
            space.static_mesh_renderers[0].node_id, 0,
            "transform 3 moved to slot 0 after removing 0"
        );
    }

    #[test]
    fn static_mesh_node_id_cleared_when_mesh_was_on_removed_transform() {
        let mut space = RenderSpaceState::default();
        for i in 0..3 {
            space.nodes.push(node_tagged(i as f32));
            space.node_parents.push(-1);
        }
        space.static_mesh_renderers.push(StaticMeshRenderer {
            node_id: 1,
            ..Default::default()
        });
        let mut cache = empty_cache(3);
        let mut ev = Vec::new();
        let _ = apply_transform_removals_ordered(&mut space, &mut cache, &[1, -1], &mut ev);
        assert_eq!(ev.len(), 1);
        assert_eq!(space.static_mesh_renderers[0].node_id, -1);
    }
}
