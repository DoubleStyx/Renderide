//! Transform hierarchy updates from host shared memory (dense indices, ordered removals).
//!
//! Removal indices are applied in **buffer order** (first entry first, `-1` terminates), matching host
//! swap-with-last semantics. **Do not** sort removals.
//!
//! Static mesh [`node_id`](crate::scene::mesh_renderable::StaticMeshRenderer::node_id) values are
//! remapped before each `swap_remove`, matching Unity `Transform` identity semantics and the same
//! `fixup_transform_id` rule used for skinned mesh bone indices in `mesh_apply`.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    RenderTransform, TransformParentUpdate, TransformPoseUpdate, TransformsUpdate,
    TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
};

use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::pose::{render_transform_identity, PoseValidation};
use super::render_space::RenderSpaceState;
use super::world::{
    fixup_transform_id, mark_descendants_uncomputed, rebuild_children, WorldTransformCache,
};

/// Minimum pose-update count before [`apply_transform_pose_updates`] fans out validation across
/// rayon workers. Below this threshold the scalar loop is faster than rayon dispatch overhead.
const POSE_UPDATE_PARALLEL_MIN_ROWS: usize = 1024;

/// Per-node dirty mask for one [`apply_transforms_update`] call.
///
/// Replaces the previous [`HashSet<usize>`] tracker so pose / parent updates can flip flags by
/// index without hashing or rehash-driven reallocation. Values are aligned to
/// [`RenderSpaceState::nodes`] length so [`propagate_transform_change_dirty_flags`] can iterate
/// in dense index order.
#[derive(Debug, Default)]
struct NodeDirtyMask {
    /// `true` at index `i` when transform `i` had its parent or pose mutated this call.
    flags: Vec<bool>,
    /// `true` when at least one entry was set this call.
    any: bool,
}

impl NodeDirtyMask {
    /// Allocates a fresh mask sized for `node_count` nodes.
    fn new(node_count: usize) -> Self {
        Self {
            flags: vec![false; node_count],
            any: false,
        }
    }

    /// Sets the dirty flag for `index`, growing the mask if a host row referenced an index past
    /// the node table that has not yet been ensured by [`grow_transform_buffers_to_target`].
    #[inline]
    fn mark(&mut self, index: usize) {
        if index >= self.flags.len() {
            self.flags.resize(index + 1, false);
        }
        self.flags[index] = true;
        self.any = true;
    }

    /// Whether any dirty flag was set.
    #[inline]
    fn any(&self) -> bool {
        self.any
    }
}

/// One successful transform removal: dense index removed and last valid index before `swap_remove`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformRemovalEvent {
    /// Removed dense transform index (`i32`, same as host removal buffer entry).
    pub removed_index: i32,
    /// Last valid index in `nodes` before the slot was removed (swapped-into source).
    pub last_index_before_swap: usize,
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

/// Applies removals in buffer order; writes events into `out` (cleared first).
pub fn apply_transform_removals_ordered(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    removals: &[i32],
    out: &mut Vec<TransformRemovalEvent>,
) -> bool {
    out.clear();
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
fn grow_transform_buffers_to_target_target(
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
        cache.visit_epoch.push(0);
    }
    if space.nodes.len() != nodes_before {
        *invalidate_world = true;
    }
}

/// Applies parent pointer deltas from a pre-extracted slice.
fn apply_transform_parent_updates_extracted(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    parents: &[TransformParentUpdate],
    changed: &mut NodeDirtyMask,
    invalidate_world: &mut bool,
) {
    if parents.is_empty() {
        return;
    }
    let mut had_parent = false;
    for pu in parents {
        if pu.transform_id < 0 {
            break;
        }
        if (pu.transform_id as usize) < space.node_parents.len() {
            space.node_parents[pu.transform_id as usize] = pu.new_parent_id;
            changed.mark(pu.transform_id as usize);
            had_parent = true;
        }
    }
    if had_parent {
        cache.children_dirty = true;
        *invalidate_world = true;
    }
}

/// Validated pose row ready for serial commit into [`RenderSpaceState::nodes`].
///
/// Produced by [`validate_pose_rows`] (serial or rayon) before the single-threaded apply phase so
/// per-row [`PoseValidation::is_valid`] checks can fan out without contending on `space.nodes`.
struct ValidatedPoseRow {
    /// Dense transform index into [`RenderSpaceState::nodes`].
    transform_index: usize,
    /// Pose to commit (already substituted with [`render_transform_identity`] when the host row was rejected).
    pose: RenderTransform,
    /// `true` when the host row failed [`PoseValidation::is_valid`] (caller logs the rejection).
    rejected: bool,
    /// Original [`TransformPoseUpdate::transform_id`] for the rejection log line.
    raw_transform_id: i32,
}

/// Index of the first sentinel `transform_id < 0` row, or `poses.len()` if no terminator is present.
#[inline]
fn pose_terminator_index(poses: &[TransformPoseUpdate]) -> usize {
    poses
        .iter()
        .position(|pu| pu.transform_id < 0)
        .unwrap_or(poses.len())
}

/// Walks the active prefix of `poses` once and produces one [`ValidatedPoseRow`] per in-bounds entry.
///
/// Above [`POSE_UPDATE_PARALLEL_MIN_ROWS`] the per-row validation fans out across rayon workers.
/// Output order matches input order so the caller's serial commit preserves last-write-wins
/// semantics for any duplicate transform indices in the host batch.
fn validate_pose_rows(poses: &[TransformPoseUpdate], node_count: usize) -> Vec<ValidatedPoseRow> {
    let active_len = pose_terminator_index(poses);
    let active = &poses[..active_len];
    let row_for = |pu: &TransformPoseUpdate| -> Option<ValidatedPoseRow> {
        let idx = pu.transform_id as usize;
        if idx >= node_count {
            return None;
        }
        let valid = PoseValidation { pose: &pu.pose }.is_valid();
        Some(ValidatedPoseRow {
            transform_index: idx,
            pose: if valid {
                pu.pose
            } else {
                render_transform_identity()
            },
            rejected: !valid,
            raw_transform_id: pu.transform_id,
        })
    };

    if active.len() >= POSE_UPDATE_PARALLEL_MIN_ROWS {
        use rayon::prelude::*;
        active.par_iter().filter_map(row_for).collect()
    } else {
        active.iter().filter_map(row_for).collect()
    }
}

/// Applies pose rows from a pre-extracted slice, validating each against [`PoseValidation`].
fn apply_transform_pose_updates_extracted(
    space: &mut RenderSpaceState,
    poses: &[TransformPoseUpdate],
    frame_index: i32,
    sid: i32,
    changed: &mut NodeDirtyMask,
) {
    if poses.is_empty() {
        return;
    }
    let validated = validate_pose_rows(poses, space.nodes.len());
    for row in validated {
        if row.rejected {
            logger::error!(
                "invalid pose scene={sid} transform={} frame={frame_index}: identity",
                row.raw_transform_id
            );
        }
        space.nodes[row.transform_index] = row.pose;
        changed.mark(row.transform_index);
    }
}

/// Marks per-node dirty flags after local transform edits.
fn propagate_transform_change_dirty_flags(
    cache: &mut WorldTransformCache,
    changed: &NodeDirtyMask,
) {
    if !changed.any() {
        return;
    }
    let n = changed
        .flags
        .len()
        .min(cache.computed.len().max(cache.local_dirty.len()));
    for (i, &dirty) in changed.flags[..n].iter().enumerate() {
        if !dirty {
            continue;
        }
        if i < cache.computed.len() {
            cache.computed[i] = false;
        }
        if i < cache.local_dirty.len() {
            cache.local_dirty[i] = true;
        }
    }
}

/// Reads every shared-memory buffer referenced by [`TransformsUpdate`] into owned vectors.
pub fn extract_transforms_update(
    shm: &mut SharedMemoryAccessor,
    update: &TransformsUpdate,
    frame_index: i32,
    sid: i32,
) -> Result<ExtractedTransformsUpdate, SceneError> {
    let mut out = ExtractedTransformsUpdate {
        target_transform_count: update.target_transform_count,
        frame_index,
        ..Default::default()
    };
    if update.removals.length > 0 {
        let ctx = format!("transforms removals scene_id={sid}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.parent_updates.length > 0 {
        let ctx = format!("transforms parent_updates scene_id={sid}");
        out.parent_updates = shm
            .access_copy_diagnostic_with_context::<TransformParentUpdate>(
                &update.parent_updates,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.pose_updates.length > 0 {
        let ctx = format!("transforms pose_updates scene_id={sid}");
        out.pose_updates = shm
            .access_copy_memory_packable_rows::<TransformPoseUpdate>(
                &update.pose_updates,
                TRANSFORM_POSE_UPDATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Applies removals, growth, parent updates, and pose updates for one space using a pre-extracted payload.
///
/// Writes `world_dirty` for `space_id` when any change invalidates the world cache. Returns
/// `true` when the caller should mark `space_id` dirty in its own merged set (used by the parallel
/// per-space apply pipeline that cannot share a `&mut HashSet`).
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

    ensure_world_cache_matches_node_count(space, cache, &mut invalidate_world);

    if !extracted.removals.is_empty() {
        let had_removal = apply_transform_removals_ordered(
            space,
            cache,
            extracted.removals.as_slice(),
            removal_events_out,
        );
        if had_removal {
            cache.children_dirty = true;
            invalidate_world = true;
        }
    }

    grow_transform_buffers_to_target_target(
        space,
        cache,
        extracted.target_transform_count,
        &mut invalidate_world,
    );

    let mut changed = NodeDirtyMask::new(space.nodes.len());

    apply_transform_parent_updates_extracted(
        space,
        cache,
        &extracted.parent_updates,
        &mut changed,
        &mut invalidate_world,
    );
    apply_transform_pose_updates_extracted(
        space,
        &extracted.pose_updates,
        extracted.frame_index,
        sid,
        &mut changed,
    );

    if changed.any() {
        invalidate_world = true;
    }

    propagate_transform_change_dirty_flags(cache, &changed);

    if cache.children_dirty {
        rebuild_children(&space.node_parents, space.nodes.len(), &mut cache.children);
        cache.children_dirty = false;
    }
    if invalidate_world {
        mark_descendants_uncomputed(&cache.children, &mut cache.computed);
    }
    invalidate_world
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

    /// [`pose_terminator_index`] returns the index of the first sentinel `transform_id < 0`,
    /// matching the early-break semantics of the host pose-update buffer protocol.
    #[test]
    fn pose_terminator_index_finds_first_sentinel() {
        let pose = node_tagged(0.0);
        let rows = vec![
            crate::shared::TransformPoseUpdate {
                transform_id: 0,
                pose,
            },
            crate::shared::TransformPoseUpdate {
                transform_id: 1,
                pose,
            },
            crate::shared::TransformPoseUpdate {
                transform_id: -1,
                pose,
            },
            crate::shared::TransformPoseUpdate {
                transform_id: 2,
                pose,
            },
        ];
        assert_eq!(pose_terminator_index(&rows), 2);
    }

    /// [`pose_terminator_index`] returns `len` when no sentinel is present so the validation pass
    /// processes every row.
    #[test]
    fn pose_terminator_index_no_sentinel_returns_len() {
        let pose = node_tagged(0.0);
        let rows = vec![crate::shared::TransformPoseUpdate {
            transform_id: 0,
            pose,
        }];
        assert_eq!(pose_terminator_index(&rows), rows.len());
    }

    /// [`validate_pose_rows`] preserves input order, drops out-of-range transform indices, and
    /// substitutes [`render_transform_identity`] for invalid poses while flagging them for log.
    #[test]
    fn validate_pose_rows_preserves_order_and_substitutes_invalid() {
        let valid = node_tagged(2.0);
        let mut bad = node_tagged(0.0);
        bad.position.x = f32::NAN;
        let rows = vec![
            crate::shared::TransformPoseUpdate {
                transform_id: 0,
                pose: valid,
            },
            crate::shared::TransformPoseUpdate {
                transform_id: 7,
                pose: valid,
            },
            crate::shared::TransformPoseUpdate {
                transform_id: 1,
                pose: bad,
            },
            crate::shared::TransformPoseUpdate {
                transform_id: -1,
                pose: valid,
            },
        ];
        let out = validate_pose_rows(&rows, 3);
        assert_eq!(
            out.len(),
            2,
            "out-of-range and sentinel rows must be dropped"
        );
        assert_eq!(out[0].transform_index, 0);
        assert!(!out[0].rejected);
        assert_eq!(out[1].transform_index, 1);
        assert!(out[1].rejected);
        let identity = render_transform_identity();
        assert_eq!(out[1].pose.position, identity.position);
        assert_eq!(out[1].pose.scale, identity.scale);
        assert_eq!(out[1].pose.rotation, identity.rotation);
    }

    /// [`validate_pose_rows`] above [`POSE_UPDATE_PARALLEL_MIN_ROWS`] still preserves input order
    /// (rayon `par_iter().filter_map().collect()` is order-preserving by index).
    #[test]
    fn validate_pose_rows_parallel_path_preserves_order() {
        let pose = node_tagged(1.0);
        let n = POSE_UPDATE_PARALLEL_MIN_ROWS + 16;
        let mut rows = Vec::with_capacity(n + 1);
        for i in 0..n {
            rows.push(crate::shared::TransformPoseUpdate {
                transform_id: i as i32,
                pose,
            });
        }
        rows.push(crate::shared::TransformPoseUpdate {
            transform_id: -1,
            pose,
        });
        let out = validate_pose_rows(&rows, n);
        assert_eq!(out.len(), n);
        for (i, row) in out.iter().enumerate() {
            assert_eq!(row.transform_index, i);
            assert!(!row.rejected);
        }
    }

    /// [`NodeDirtyMask::mark`] grows the underlying `Vec<bool>` to fit indices that exceed the
    /// initial node-table size (e.g. when a host pose row references a slot just allocated by
    /// [`grow_transform_buffers_to_target`]).
    #[test]
    fn node_dirty_mask_grows_on_out_of_bounds_index() {
        let mut mask = NodeDirtyMask::new(2);
        mask.mark(5);
        assert!(mask.any());
        assert!(mask.flags[5]);
        assert_eq!(mask.flags.len(), 6);
    }
}
