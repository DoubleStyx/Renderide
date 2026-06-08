//! Transform hierarchy updates from host shared memory (dense indices, ordered removals).
//!
//! Removal indices are applied in **buffer order** (first entry first, `-1` terminates), matching host
//! swap-with-last semantics. **Do not** sort removals.
//!
//! After removals run, the per-space orchestrator
//! ([`crate::scene::coordinator::apply::apply_extracted_render_space_update`]) re-runs the
//! [`fixup_transform_id`](super::world::fixup_transform_id) sweep across cameras, static and skinned
//! mesh renderables, layer assignments, render overrides, and lights using the captured
//! [`TransformRemovalEvent`]s. Removal handling here therefore performs only the parent-pointer
//! repair that needs to happen before [`Vec::swap_remove`].
//!
//! The dense apply path is split across two submodules: [`removals`] handles transform removals
//! and parent-pointer deltas, [`poses`] handles pose row validation, commit, and dirty-flag
//! propagation. This barrel module owns the cross-phase [`NodeDirtyMask`], the buffer-growth
//! helpers, the shared-memory [`extract_transforms_update`] entry point, and the
//! [`apply_transforms_update_extracted`] orchestrator that runs the two phases in order.

mod poses;
mod removals;

pub use removals::{TransformRemovalEvent, apply_transform_removals_ordered};

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES, TransformParentUpdate, TransformPoseUpdate,
    TransformsUpdate,
};

use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::pose::render_transform_identity;
use super::render_space::RenderSpaceState;
use super::world::{
    WorldTransformCache, mark_descendants_uncomputed, mark_descendants_uncomputed_from_roots,
    rebuild_children,
};

const TRANSFORM_POSE_UPDATE_TERMINATOR_ROWS: usize = 1;
const MAX_TRANSFORM_ROWS: usize = 1_000_000;
const MAX_TRANSFORM_COPY_BYTES: usize = (MAX_TRANSFORM_ROWS
    + TRANSFORM_POSE_UPDATE_TERMINATOR_ROWS)
    * TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES;

/// Per-node dirty mask for one [`apply_transforms_update_extracted`] call.
///
/// Replaces the previous [`std::collections::HashSet<usize>`] tracker so pose / parent updates
/// can flip flags by index without hashing or rehash-driven reallocation. Values are aligned to
/// [`RenderSpaceState::nodes`] length so dirty-flag propagation can iterate in dense index order.
#[derive(Debug, Default)]
struct NodeDirtyMask {
    /// `true` at index `i` when transform `i` had its parent or pose mutated this call.
    flags: Vec<bool>,
    /// Dense list of indices marked dirty this call, without duplicates.
    indices: Vec<usize>,
    /// Dense transform index to pose-plan row while pose updates collapse duplicate rows.
    pose_plan_indices: Vec<usize>,
    /// `true` when at least one entry was set this call.
    any: bool,
}

impl NodeDirtyMask {
    /// Allocates an empty mask with enough dedup capacity for `node_count` nodes.
    #[cfg(test)]
    fn new(node_count: usize) -> Self {
        Self {
            flags: Vec::new(),
            indices: Vec::with_capacity(node_count.min(64)),
            pose_plan_indices: vec![usize::MAX; node_count],
            any: false,
        }
    }

    /// Takes the reusable scratch vectors from `cache` and resets the marks from the previous
    /// transform update.
    fn take_from_cache(cache: &mut WorldTransformCache, node_count: usize) -> Self {
        let mut flags = std::mem::take(&mut cache.transform_dirty_flags);
        let mut indices = std::mem::take(&mut cache.transform_dirty_indices);
        for &index in &indices {
            if let Some(flag) = flags.get_mut(index) {
                *flag = false;
            }
        }
        indices.clear();
        if flags.len() < node_count {
            flags.resize(node_count, false);
        }
        let mut pose_plan_indices = std::mem::take(&mut cache.transform_pose_plan_indices);
        if pose_plan_indices.len() < node_count {
            pose_plan_indices.resize(node_count, usize::MAX);
        }
        pose_plan_indices[..node_count].fill(usize::MAX);
        Self {
            flags,
            indices,
            pose_plan_indices,
            any: false,
        }
    }

    /// Restores the scratch vectors to `cache` for reuse by the next transform update.
    fn restore_into(self, cache: &mut WorldTransformCache) {
        cache.transform_dirty_flags = self.flags;
        cache.transform_dirty_indices = self.indices;
        cache.transform_pose_plan_indices = self.pose_plan_indices;
    }

    /// Sets the dirty flag for `index`, growing the mask if a host row referenced an index past
    /// the node table that has not yet been ensured by [`grow_transform_buffers_to_target`].
    #[inline]
    fn mark(&mut self, index: usize) {
        if index >= self.flags.len() {
            self.flags.resize(index + 1, false);
        }
        if self.flags[index] {
            return;
        }
        self.flags[index] = true;
        self.indices.push(index);
        self.any = true;
    }

    /// Whether any dirty flag was set.
    #[inline]
    fn any(&self) -> bool {
        self.any
    }

    /// Read-only access to the flag vector for dirty-flag propagation.
    #[cfg(test)]
    #[inline]
    fn flags(&self) -> &[bool] {
        &self.flags
    }

    /// Dirty transform indices marked during this apply call.
    #[inline]
    fn indices(&self) -> &[usize] {
        &self.indices
    }
}

/// Owned per-space transform-update payload extracted from shared memory.
///
/// Produced by [`extract_transforms_update`] in the serial pre-extract phase so the per-space
/// apply step (see [`apply_transforms_update_extracted`]) can run on a rayon worker without
/// holding a mutable borrow on [`SharedMemoryAccessor`].
#[derive(Default, Debug)]
pub struct ExtractedTransformsUpdate {
    /// Dense transform removal indices (terminated by `< 0`); applied in buffer order.
    pub removals: Vec<i32>,
    /// Parent pointer deltas for the dense transform table.
    pub parent_updates: Vec<TransformParentUpdate>,
    /// Pose rows (terminated by `transform_id < 0`).
    pub pose_updates: Vec<TransformPoseUpdate>,
    /// Target dense transform count (mirrors [`TransformsUpdate::target_transform_count`]).
    pub target_transform_count: i32,
    /// Host frame index for diagnostics.
    pub frame_index: i32,
}

/// Resizes world/cache sidecars when the node table grew or shrank on host.
fn ensure_world_cache_matches_node_count(
    space: &RenderSpaceState,
    cache: &mut WorldTransformCache,
    invalidate_world: &mut bool,
) {
    if cache.world_matrices.len() == space.nodes.len()
        && cache.degenerate_scales.len() == space.nodes.len()
    {
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
    cache.degenerate_scales.resize(space.nodes.len(), false);
    cache.visit_epoch.resize(space.nodes.len(), 0);
    *invalidate_world = true;
}

/// Extends dense transform buffers up to `target_transform_count` with identity locals.
fn grow_transform_buffers_to_target(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    target_transform_count: i32,
    invalidate_world: &mut bool,
) {
    let nodes_before = space.nodes.len();
    while (space.nodes.len() as i32) < target_transform_count {
        space.nodes.push(render_transform_identity());
        space.node_parents.push(-1);
        cache.world_matrices.push(glam::Mat4::IDENTITY);
        cache.computed.push(false);
        cache.local_matrices.push(glam::Mat4::IDENTITY);
        cache.local_dirty.push(true);
        cache.degenerate_scales.push(false);
        cache.visit_epoch.push(0);
    }
    if space.nodes.len() != nodes_before {
        *invalidate_world = true;
        space.hierarchy_dirty = true;
    }
}

fn transform_pose_update_copy_max_bytes(update: &TransformsUpdate) -> i32 {
    let target_rows = update.target_transform_count.max(0) as usize;
    let target_sized_bytes = target_rows
        .saturating_add(TRANSFORM_POSE_UPDATE_TERMINATOR_ROWS)
        .saturating_mul(TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES);
    let descriptor_sized_bytes = update
        .pose_updates
        .buffer_capacity
        .try_into()
        .ok()
        .and_then(|capacity: usize| {
            let offset = usize::try_from(update.pose_updates.offset).ok()?;
            capacity.checked_sub(offset)
        })
        .unwrap_or(0);
    target_sized_bytes
        .max(descriptor_sized_bytes.min(MAX_TRANSFORM_COPY_BYTES))
        .min(MAX_TRANSFORM_COPY_BYTES)
        .min(i32::MAX as usize) as i32
}

/// Reads every shared-memory buffer referenced by [`TransformsUpdate`] into owned vectors.
pub fn extract_transforms_update(
    shm: &mut SharedMemoryAccessor,
    update: &TransformsUpdate,
    frame_index: i32,
    sid: i32,
) -> Result<ExtractedTransformsUpdate, SceneError> {
    if update.target_transform_count < 0
        || update.target_transform_count as usize > MAX_TRANSFORM_ROWS
    {
        return Err(SceneError::SharedMemoryAccess(format!(
            "transforms scene_id={sid}: target_transform_count {} exceeds cap {}",
            update.target_transform_count, MAX_TRANSFORM_ROWS
        )));
    }
    let mut out = ExtractedTransformsUpdate {
        target_transform_count: update.target_transform_count,
        frame_index,
        ..Default::default()
    };
    if update.removals.length > 0 {
        profiling::scope!("scene::extract_transforms::removals");
        let ctx = format!("transforms removals scene_id={sid}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.parent_updates.length > 0 {
        profiling::scope!("scene::extract_transforms::parent_updates");
        let ctx = format!("transforms parent_updates scene_id={sid}");
        out.parent_updates = shm
            .access_copy_diagnostic_with_context::<TransformParentUpdate>(
                &update.parent_updates,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.pose_updates.length > 0 {
        profiling::scope!("scene::extract_transforms::pose_updates");
        let ctx = format!("transforms pose_updates scene_id={sid}");
        let max_bytes = transform_pose_update_copy_max_bytes(update);
        out.pose_updates = shm
            .access_copy_memory_packable_rows_until_with_max::<TransformPoseUpdate, _>(
                &update.pose_updates,
                TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
                max_bytes,
                Some(&ctx),
                |row| row.transform_id < 0,
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Applies removals, growth, parent updates, and pose updates for one space using a pre-extracted payload.
///
/// Returns `true` when the world cache for this space must be invalidated. The orchestrator
/// keeps the per-space dirty bookkeeping; downstream callers should treat this return value as
/// the signal to mark `space_id` dirty in their merged set.
pub fn apply_transforms_update_extracted(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    space_id: RenderSpaceId,
    extracted: &ExtractedTransformsUpdate,
    removal_events_out: &mut Vec<TransformRemovalEvent>,
) -> bool {
    profiling::scope!("scene::apply_transforms");
    removal_events_out.clear();
    let sid = space_id.0;
    let mut invalidate_world = false;
    let mut full_invalidate_world = false;

    {
        profiling::scope!("scene::apply_transforms::sync_cache_size");
        ensure_world_cache_matches_node_count(space, cache, &mut invalidate_world);
    }
    if invalidate_world {
        full_invalidate_world = true;
    }

    if !extracted.removals.is_empty() {
        profiling::scope!("scene::apply_transforms::removals");
        let had_removal = apply_transform_removals_ordered(
            space,
            cache,
            extracted.removals.as_slice(),
            removal_events_out,
        );
        if had_removal {
            cache.children_dirty = true;
            invalidate_world = true;
            full_invalidate_world = true;
            space.hierarchy_dirty = true;
        }
    }

    let before_grow = invalidate_world;
    {
        profiling::scope!("scene::apply_transforms::grow_buffers");
        grow_transform_buffers_to_target(
            space,
            cache,
            extracted.target_transform_count,
            &mut invalidate_world,
        );
    }
    if invalidate_world && !before_grow {
        full_invalidate_world = true;
    }

    let mut changed = {
        profiling::scope!("scene::apply_transforms::dirty_mask");
        NodeDirtyMask::take_from_cache(cache, space.nodes.len())
    };

    {
        profiling::scope!("scene::apply_transforms::parent_updates");
        removals::apply_transform_parent_updates_extracted(
            space,
            cache,
            &extracted.parent_updates,
            &mut changed,
            &mut invalidate_world,
        );
    }
    {
        profiling::scope!("scene::apply_transforms::pose_updates");
        poses::apply_transform_pose_updates_extracted(
            space,
            &extracted.pose_updates,
            extracted.frame_index,
            sid,
            &mut changed,
        );
    }

    if changed.any() {
        invalidate_world = true;
    }

    {
        profiling::scope!("scene::apply_transforms::propagate_dirty_flags");
        poses::propagate_transform_change_dirty_flags(cache, &changed);
    }

    if cache.children_dirty {
        profiling::scope!("scene::apply_transforms::rebuild_children");
        rebuild_children(&space.node_parents, space.nodes.len(), &mut cache.children);
        cache.children_dirty = false;
    }
    if full_invalidate_world {
        profiling::scope!("scene::apply_transforms::invalidate_all_descendants");
        mark_descendants_uncomputed(&cache.children, &mut cache.computed);
    } else if invalidate_world {
        profiling::scope!("scene::apply_transforms::invalidate_changed_descendants");
        mark_descendants_uncomputed_from_roots(
            &cache.children,
            &mut cache.computed,
            changed.indices(),
        );
    }
    {
        profiling::scope!("scene::apply_transforms::restore_dirty_mask");
        changed.restore_into(cache);
    }
    invalidate_world
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;
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
            degenerate_scales: vec![false; nodes_len],
            visit_epoch: vec![0; nodes_len],
            ..Default::default()
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

    #[test]
    fn pose_update_copy_max_caps_target_sized_oversized_pose_slabs() {
        let rows_over_cap = MAX_TRANSFORM_ROWS + 1;
        let update = TransformsUpdate {
            target_transform_count: rows_over_cap as i32,
            ..Default::default()
        };
        let uncapped_bytes = (rows_over_cap + TRANSFORM_POSE_UPDATE_TERMINATOR_ROWS)
            * TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES;

        assert!(uncapped_bytes > MAX_TRANSFORM_COPY_BYTES);
        assert_eq!(
            transform_pose_update_copy_max_bytes(&update),
            MAX_TRANSFORM_COPY_BYTES as i32
        );
    }

    #[test]
    fn pose_update_copy_max_uses_target_sized_guard_for_small_targets() {
        let update = TransformsUpdate {
            target_transform_count: 4,
            ..Default::default()
        };
        let expected =
            (4 + TRANSFORM_POSE_UPDATE_TERMINATOR_ROWS) * TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES;

        assert_eq!(
            transform_pose_update_copy_max_bytes(&update),
            expected as i32
        );
    }

    #[test]
    fn pose_update_copy_max_caps_descriptor_sized_oversized_pose_slabs() {
        let length = MAX_TRANSFORM_COPY_BYTES as i32 + TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES as i32;
        let update = TransformsUpdate {
            target_transform_count: 4,
            pose_updates: crate::shared::buffer::SharedMemoryBufferDescriptor {
                buffer_id: 9,
                buffer_capacity: length + 128,
                offset: 128,
                length,
            },
            ..Default::default()
        };

        assert_eq!(
            transform_pose_update_copy_max_bytes(&update),
            MAX_TRANSFORM_COPY_BYTES as i32
        );
    }

    #[test]
    fn pose_update_copy_max_does_not_trust_length_without_admitted_capacity() {
        let length = MAX_TRANSFORM_COPY_BYTES as i32 + TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES as i32;
        let update = TransformsUpdate {
            target_transform_count: 4,
            pose_updates: crate::shared::buffer::SharedMemoryBufferDescriptor {
                buffer_id: 9,
                buffer_capacity: 0,
                offset: 0,
                length,
            },
            ..Default::default()
        };
        let expected =
            (4 + TRANSFORM_POSE_UPDATE_TERMINATOR_ROWS) * TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES;

        assert_eq!(
            transform_pose_update_copy_max_bytes(&update),
            expected as i32
        );
    }

    /// [`NodeDirtyMask::mark`] grows the underlying `Vec<bool>` to fit indices that exceed the
    /// initial node-table size (e.g. when a host pose row references a slot just allocated by
    /// [`grow_transform_buffers_to_target`]).
    #[test]
    fn node_dirty_mask_grows_on_out_of_bounds_index() {
        let mut mask = NodeDirtyMask::new(2);
        mask.mark(5);
        assert!(mask.any());
        assert!(mask.flags()[5]);
        assert_eq!(mask.flags().len(), 6);
    }
}
