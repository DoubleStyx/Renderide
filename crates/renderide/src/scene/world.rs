//! Incremental world-matrix propagation and child index for transform hierarchies.

use glam::Mat4;
use rayon::prelude::*;

use crate::shared::RenderTransform;

use super::error::SceneError;
use super::math::{render_transform_has_degenerate_scale, render_transform_to_matrix};

/// Node count above which a bulk rebuild fans out across rayon by hierarchy depth level.
const WORLD_BULK_REBUILD_PARALLEL_MIN: usize = 1024;

/// Per-space cache: world matrices and incremental recompute bookkeeping.
#[derive(Debug)]
pub struct WorldTransformCache {
    /// World matrix per dense transform index (parent chain only; no [`RenderSpaceState::root_transform`](super::render_space::RenderSpaceState) multiply).
    pub world_matrices: Vec<Mat4>,
    /// `true` when `world_matrices[i]` is valid for the current local poses.
    pub computed: Vec<bool>,
    /// Cached local TRS matrices.
    pub local_matrices: Vec<Mat4>,
    /// Stale local TRS when pose changed.
    pub local_dirty: Vec<bool>,
    /// `true` when this node or any ancestor has a raw zero / near-zero object scale axis.
    pub degenerate_scales: Vec<bool>,
    /// Epoch per node for O(1) cycle detection during upward walks.
    pub(super) visit_epoch: Vec<u32>,
    /// Incremented before each upward walk.
    pub(super) walk_epoch: u32,
    /// Parent -> children lists; rebuilt when structure changes.
    pub(super) children: Vec<Vec<usize>>,
    /// `children` must be rebuilt before descendant marking.
    pub(super) children_dirty: bool,
    /// Bulk-rebuild scratch: depth per node (`u32::MAX` for cycle nodes). Reused across frames.
    pub(super) bfs_depth: Vec<u32>,
    /// Bulk-rebuild scratch: per-depth-level node-index buckets. Inner Vec capacities persist
    /// across frames so a stable scene re-uses the same storage every solve.
    pub(super) bfs_levels: Vec<Vec<usize>>,
    /// Bulk-rebuild scratch: indices of nodes whose ancestor chain forms a cycle.
    pub(super) bfs_cycle_nodes: Vec<usize>,
    /// Bulk-rebuild scratch: per-level computed (world matrix, degenerate flag) pairs before
    /// they are written back to [`Self::world_matrices`] and [`Self::degenerate_scales`].
    pub(super) bfs_writes: Vec<(Mat4, bool)>,
}

impl Default for WorldTransformCache {
    fn default() -> Self {
        Self {
            world_matrices: Vec::new(),
            computed: Vec::new(),
            local_matrices: Vec::new(),
            local_dirty: Vec::new(),
            degenerate_scales: Vec::new(),
            visit_epoch: Vec::new(),
            walk_epoch: 0,
            children: Vec::new(),
            children_dirty: true,
            bfs_depth: Vec::new(),
            bfs_levels: Vec::new(),
            bfs_cycle_nodes: Vec::new(),
            bfs_writes: Vec::new(),
        }
    }
}

/// After `swap_remove` at `removed_id` index, remaps a stored transform reference.
pub(crate) fn fixup_transform_id(old: i32, removed_id: i32, last_index: usize) -> i32 {
    if old == removed_id {
        -1
    } else if old == last_index as i32 {
        removed_id
    } else {
        old
    }
}

/// Rebuilds parent -> children adjacency.
pub(super) fn rebuild_children(node_parents: &[i32], n: usize, children: &mut Vec<Vec<usize>>) {
    children.resize_with(n, Vec::new);
    for c in children.iter_mut() {
        c.clear();
    }
    for (i, &p) in node_parents.iter().take(n).enumerate() {
        if p >= 0 && (p as usize) < n && p != i as i32 {
            children[p as usize].push(i);
        }
    }
}

/// Marks descendants of any node with `computed[i] == false` as uncomputed.
pub(super) fn mark_descendants_uncomputed(children: &[Vec<usize>], computed: &mut [bool]) {
    let n = computed.len();
    if n == 0 {
        return;
    }
    let mut stack: Vec<usize> = Vec::with_capacity(64.min(n));
    for i in 0..n {
        if computed[i] {
            continue;
        }
        stack.clear();
        let child_list: &[usize] = children.get(i).map_or(&[], Vec::as_slice);
        stack.extend_from_slice(child_list);
        while let Some(child) = stack.pop() {
            computed[child] = false;
            let child_list: &[usize] = children.get(child).map_or(&[], Vec::as_slice);
            stack.extend_from_slice(child_list);
        }
    }
}

/// Marks descendants of the provided dirty roots as uncomputed.
pub(super) fn mark_descendants_uncomputed_from_roots(
    children: &[Vec<usize>],
    computed: &mut [bool],
    roots: &[usize],
) {
    if computed.is_empty() || roots.is_empty() {
        return;
    }
    let mut stack: Vec<usize> = Vec::with_capacity(64.min(computed.len()));
    for &root in roots {
        if root < computed.len() {
            computed[root] = false;
        }
        stack.clear();
        stack.extend_from_slice(children.get(root).map_or(&[], Vec::as_slice));
        while let Some(child) = stack.pop() {
            if child >= computed.len() {
                continue;
            }
            computed[child] = false;
            stack.extend_from_slice(children.get(child).map_or(&[], Vec::as_slice));
        }
    }
}

#[inline]
fn get_local_matrix(
    nodes: &[RenderTransform],
    local_matrices: &mut [Mat4],
    local_dirty: &mut [bool],
    i: usize,
) -> Mat4 {
    if i < local_dirty.len() && local_dirty[i] {
        let m = render_transform_to_matrix(&nodes[i]);
        local_matrices[i] = m;
        local_dirty[i] = false;
        m
    } else if i < local_matrices.len() {
        local_matrices[i]
    } else {
        render_transform_to_matrix(&nodes[i])
    }
}

/// Returns `true` when every entry of `computed` is `false`, signalling a bulk rebuild.
#[inline]
fn cache_is_fully_dirty(computed: &[bool]) -> bool {
    !computed.iter().any(|c| *c)
}

/// Fills `depth` / `levels` / `cycle_nodes` from a parent-array tree, reusing the supplied
/// buffers' allocations. After this call `levels[d]` lists every node at depth `d` whose
/// ancestor chain reaches a root, and `cycle_nodes` lists every node whose ancestor chain
/// forms a cycle. Cycle nodes get a local-only fallback matrix (matching
/// [`WorldTransformCache::compute_world_matrices_incremental`]).
fn classify_nodes_by_depth(
    node_parents: &[i32],
    n: usize,
    depth: &mut Vec<u32>,
    levels: &mut Vec<Vec<usize>>,
    cycle_nodes: &mut Vec<usize>,
) {
    depth.clear();
    depth.resize(n, u32::MAX);
    cycle_nodes.clear();
    for inner in levels.iter_mut() {
        inner.clear();
    }

    // Repeatedly resolve depths in a sweep order: a node knows its depth once its parent does.
    // For a typical mostly-sorted host order this converges in 1-2 sweeps; cap at n iterations to
    // exit deterministically when a cycle leaves a residue of unresolved nodes.
    for (i, slot) in depth.iter_mut().enumerate().take(n) {
        let p = node_parents.get(i).copied().unwrap_or(-1);
        if p < 0 || (p as usize) >= n || p == i as i32 {
            *slot = 0;
        }
    }
    let mut changed = true;
    let mut sweep = 0usize;
    while changed && sweep < n {
        changed = false;
        for i in 0..n {
            if depth[i] != u32::MAX {
                continue;
            }
            let p = node_parents[i];
            if p >= 0 && (p as usize) < n {
                let pd = depth[p as usize];
                if pd != u32::MAX {
                    depth[i] = pd + 1;
                    changed = true;
                }
            }
        }
        sweep += 1;
    }

    let max_depth = depth
        .iter()
        .filter(|&&d| d != u32::MAX)
        .copied()
        .max()
        .unwrap_or(0) as usize;
    if levels.len() < max_depth + 1 {
        levels.resize_with(max_depth + 1, Vec::new);
    }
    for (i, &d) in depth.iter().enumerate() {
        if d == u32::MAX {
            cycle_nodes.push(i);
        } else {
            levels[d as usize].push(i);
        }
    }
}

impl WorldTransformCache {
    /// Bulk rebuild: recomputes every world matrix from scratch using a level-synchronous BFS
    /// over the transform hierarchy. Each level is independent across siblings, so the inner
    /// matrix multiplications fan out across rayon.
    ///
    /// Cycle nodes (transforms whose ancestor chain loops back) fall back to a local-only matrix,
    /// matching [`Self::compute_world_matrices_incremental`].
    fn compute_world_matrices_bulk_rebuild(
        &mut self,
        scene_id: i32,
        nodes: &[RenderTransform],
        node_parents: &[i32],
    ) {
        profiling::scope!("scene::world_bulk_rebuild");
        let n = nodes.len();
        if self.visit_epoch.len() < n {
            self.visit_epoch.resize(n, 0);
        }
        if self.degenerate_scales.len() < n {
            self.degenerate_scales.resize(n, false);
        }

        // Materialize every local matrix once. Read-only after this point so the parallel level
        // sweep below can capture &[Mat4] instead of fighting borrow rules around lazy materialise.
        for ((local, dirty), node) in self
            .local_matrices
            .iter_mut()
            .zip(self.local_dirty.iter_mut())
            .zip(nodes.iter())
            .take(n)
        {
            *local = render_transform_to_matrix(node);
            *dirty = false;
        }

        // Reused scratch state lives on the cache so the bulk-rebuild path only allocates on
        // the first call and grows lazily as the scene gets bigger.
        let WorldTransformCache {
            world_matrices,
            computed,
            local_matrices,
            degenerate_scales,
            bfs_depth,
            bfs_levels,
            bfs_cycle_nodes,
            bfs_writes,
            ..
        } = self;

        classify_nodes_by_depth(node_parents, n, bfs_depth, bfs_levels, bfs_cycle_nodes);

        for cycle_id in bfs_cycle_nodes.iter() {
            logger::trace!(
                "parent cycle at scene {} transform {} -- local-only fallback",
                scene_id,
                cycle_id
            );
            let local = local_matrices[*cycle_id];
            world_matrices[*cycle_id] = local;
            degenerate_scales[*cycle_id] = render_transform_has_degenerate_scale(&nodes[*cycle_id]);
            computed[*cycle_id] = true;
        }

        // Roots (depth 0) have no parent contribution; world = local.
        if let Some(roots) = bfs_levels.first()
            && !roots.is_empty()
        {
            let local_ro: &[Mat4] = local_matrices;
            bfs_writes.clear();
            roots
                .par_iter()
                .map(|&i| {
                    (
                        local_ro[i],
                        render_transform_has_degenerate_scale(&nodes[i]),
                    )
                })
                .collect_into_vec(bfs_writes);
            for (slot, &i) in bfs_writes.iter().zip(roots.iter()) {
                world_matrices[i] = slot.0;
                degenerate_scales[i] = slot.1;
                computed[i] = true;
            }
        }

        // Subsequent levels read the previous level's world / degenerate state, multiply locally,
        // and write back disjoint indices. The collect-then-apply split avoids needing unsafe
        // disjoint-index access into world_matrices.
        for level in bfs_levels.iter().skip(1) {
            if level.is_empty() {
                continue;
            }
            let local_ro: &[Mat4] = local_matrices;
            let world_ro: &[Mat4] = world_matrices;
            let degen_ro: &[bool] = degenerate_scales;
            bfs_writes.clear();
            level
                .par_iter()
                .map(|&i| {
                    let p = node_parents[i] as usize;
                    let parent_world = world_ro[p];
                    let parent_degen = degen_ro[p];
                    let local = local_ro[i];
                    let degen_self = render_transform_has_degenerate_scale(&nodes[i]);
                    (parent_world * local, parent_degen | degen_self)
                })
                .collect_into_vec(bfs_writes);
            for (slot, &i) in bfs_writes.iter().zip(level.iter()) {
                world_matrices[i] = slot.0;
                degenerate_scales[i] = slot.1;
                computed[i] = true;
            }
        }
    }

    /// Incremental world matrices: only recomputes indices with `computed[i] == false`.
    pub(super) fn compute_world_matrices_incremental(
        &mut self,
        scene_id: i32,
        nodes: &[RenderTransform],
        node_parents: &[i32],
    ) -> Result<(), SceneError> {
        let world_matrices = &mut self.world_matrices;
        let computed = &mut self.computed;
        let local_matrices = &mut self.local_matrices;
        let local_dirty = &mut self.local_dirty;
        let degenerate_scales = &mut self.degenerate_scales;
        let visit_epoch = &mut self.visit_epoch;
        let walk_epoch = &mut self.walk_epoch;
        let n = nodes.len();
        let mut stack: Vec<usize> = Vec::with_capacity(64.min(n));

        if visit_epoch.len() < n {
            visit_epoch.resize(n, 0);
        }
        if degenerate_scales.len() < n {
            degenerate_scales.resize(n, false);
        }

        for transform_index in (0..n).rev() {
            if computed[transform_index] {
                continue;
            }

            stack.clear();
            *walk_epoch = (*walk_epoch).wrapping_add(1);
            let epoch = *walk_epoch;

            let mut maybe_uppermost_matrix: Option<Mat4> = None;
            let mut id = transform_index;
            let mut cycle_detected = false;

            {
                profiling::scope!("world::upward_walk");
                loop {
                    if id >= n {
                        break;
                    }
                    if computed[id] {
                        maybe_uppermost_matrix = Some(world_matrices[id]);
                        break;
                    }
                    if visit_epoch[id] == epoch {
                        cycle_detected = true;
                        logger::trace!(
                            "parent cycle at scene {} transform {} -- local-only fallback",
                            scene_id,
                            id
                        );
                        break;
                    }
                    visit_epoch[id] = epoch;
                    stack.push(id);
                    let p = node_parents.get(id).copied().unwrap_or(-1);
                    if p < 0 || (p as usize) >= n || p == id as i32 {
                        break;
                    }
                    id = p as usize;
                }
            }

            if cycle_detected {
                for &cid in &stack {
                    if !computed[cid] {
                        let local = get_local_matrix(nodes, local_matrices, local_dirty, cid);
                        world_matrices[cid] = local;
                        degenerate_scales[cid] = render_transform_has_degenerate_scale(&nodes[cid]);
                        computed[cid] = true;
                    }
                }
                continue;
            }

            let (mut parent_matrix, mut parent_degenerate) = if let Some(m) = maybe_uppermost_matrix
            {
                (m, degenerate_scales.get(id).copied().unwrap_or(false))
            } else {
                let Some(top) = stack.pop() else {
                    continue;
                };
                let local = get_local_matrix(nodes, local_matrices, local_dirty, top);
                let degenerate = render_transform_has_degenerate_scale(&nodes[top]);
                world_matrices[top] = local;
                degenerate_scales[top] = degenerate;
                computed[top] = true;
                (local, degenerate)
            };

            while let Some(child_id) = stack.pop() {
                let local = get_local_matrix(nodes, local_matrices, local_dirty, child_id);
                parent_matrix *= local;
                parent_degenerate |= render_transform_has_degenerate_scale(&nodes[child_id]);
                world_matrices[child_id] = parent_matrix;
                degenerate_scales[child_id] = parent_degenerate;
                computed[child_id] = true;
            }
        }

        Ok(())
    }
}

/// Ensures cache vectors match `node_count`, invalidates if resized.
pub(super) fn ensure_cache_shapes(
    cache: &mut WorldTransformCache,
    node_count: usize,
    force_invalidate: bool,
) {
    if cache.world_matrices.len() != node_count || cache.degenerate_scales.len() != node_count {
        cache.world_matrices.resize(node_count, Mat4::IDENTITY);
        cache.computed.resize(node_count, false);
        cache.local_matrices.resize(node_count, Mat4::IDENTITY);
        cache.local_dirty.resize(node_count, true);
        cache.degenerate_scales.resize(node_count, false);
        cache.visit_epoch.resize(node_count, 0);
        cache.children_dirty = true;
        for c in &mut cache.computed {
            *c = false;
        }
    } else if force_invalidate {
        for c in &mut cache.computed {
            *c = false;
        }
    }
}

/// Runs incremental solve if anything is dirty or sizes changed.
///
/// Routes to the level-synchronous parallel bulk rebuild when the cache is fully dirty (typical
/// after a structural change or a force-invalidate) and the node count crosses
/// [`WORLD_BULK_REBUILD_PARALLEL_MIN`]. Smaller spaces and incremental updates keep using the
/// existing serial upward-walk algorithm where the per-node bookkeeping cost still dominates the
/// rayon dispatch overhead.
pub fn compute_world_matrices_for_space(
    scene_id: i32,
    nodes: &[RenderTransform],
    node_parents: &[i32],
    cache: &mut WorldTransformCache,
) -> Result<(), SceneError> {
    profiling::scope!("scene::compute_world_matrices");
    let n = nodes.len();
    if n == 0 {
        *cache = WorldTransformCache::default();
        return Ok(());
    }

    ensure_cache_shapes(cache, n, false);

    if cache.children_dirty {
        rebuild_children(node_parents, n, &mut cache.children);
        cache.children_dirty = false;
    }

    if n >= WORLD_BULK_REBUILD_PARALLEL_MIN && cache_is_fully_dirty(&cache.computed) {
        cache.compute_world_matrices_bulk_rebuild(scene_id, nodes, node_parents);
        return Ok(());
    }

    cache.compute_world_matrices_incremental(scene_id, nodes, node_parents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    /// Identity local pose used as the default node TRS in test fixtures.
    fn identity_xform() -> RenderTransform {
        RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    /// Translation-only local pose, convenient for asserting world-matrix products.
    fn translation_xform(x: f32, y: f32, z: f32) -> RenderTransform {
        RenderTransform {
            position: Vec3::new(x, y, z),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn fixup_transform_id_remaps_last_to_removed() {
        assert_eq!(fixup_transform_id(7, 3, 7), 3);
    }

    #[test]
    fn fixup_transform_id_returns_minus_one_when_old_equals_removed() {
        assert_eq!(fixup_transform_id(3, 3, 7), -1);
    }

    #[test]
    fn fixup_transform_id_passes_through_unrelated_indices() {
        assert_eq!(fixup_transform_id(2, 3, 7), 2);
        assert_eq!(fixup_transform_id(-1, 3, 7), -1);
    }

    #[test]
    fn rebuild_children_builds_parent_to_child_adjacency() {
        let parents = [-1, 0, 0, 1];
        let mut children = Vec::new();
        rebuild_children(&parents, 4, &mut children);
        assert_eq!(children.len(), 4);
        assert_eq!(children[0], vec![1, 2]);
        assert_eq!(children[1], vec![3]);
        assert!(children[2].is_empty());
        assert!(children[3].is_empty());
    }

    #[test]
    fn rebuild_children_ignores_self_loops_and_out_of_bounds_parents() {
        let parents = [1, 1, 5];
        let mut children = Vec::new();
        rebuild_children(&parents, 3, &mut children);
        assert_eq!(children[1], vec![0]);
        assert!(
            children[0].is_empty() && children[2].is_empty(),
            "self-loop on 1 and out-of-bounds parent 5 must be skipped"
        );
    }

    #[test]
    fn rebuild_children_clears_existing_children_before_rebuild() {
        let mut children = vec![vec![99usize]; 2];
        rebuild_children(&[-1, 0], 2, &mut children);
        assert_eq!(children[0], vec![1]);
        assert!(
            children[1].is_empty(),
            "stale child entries must be cleared"
        );
    }

    #[test]
    fn mark_descendants_uncomputed_propagates_through_subtree() {
        let children = vec![vec![1, 2], vec![3], vec![], vec![]];
        let mut computed = vec![false, true, true, true];
        mark_descendants_uncomputed(&children, &mut computed);
        assert_eq!(computed, vec![false, false, false, false]);
    }

    #[test]
    fn mark_descendants_uncomputed_no_op_when_all_computed() {
        let children = vec![vec![1], vec![]];
        let mut computed = vec![true, true];
        mark_descendants_uncomputed(&children, &mut computed);
        assert_eq!(computed, vec![true, true]);
    }

    #[test]
    fn mark_descendants_uncomputed_handles_empty_input() {
        let children: Vec<Vec<usize>> = Vec::new();
        let mut computed: Vec<bool> = Vec::new();
        mark_descendants_uncomputed(&children, &mut computed);
        assert!(computed.is_empty());
    }

    #[test]
    fn ensure_cache_shapes_resizes_and_clears_computed_on_grow() {
        let mut cache = WorldTransformCache::default();
        ensure_cache_shapes(&mut cache, 3, false);
        assert_eq!(cache.world_matrices.len(), 3);
        assert_eq!(cache.degenerate_scales.len(), 3);
        assert_eq!(cache.computed, vec![false, false, false]);
        assert!(
            cache.children_dirty,
            "growth must mark children adjacency dirty"
        );

        for c in &mut cache.computed {
            *c = true;
        }
        ensure_cache_shapes(&mut cache, 5, false);
        assert_eq!(cache.world_matrices.len(), 5);
        assert_eq!(cache.degenerate_scales.len(), 5);
        assert!(
            cache.computed.iter().all(|c| !*c),
            "resize must invalidate all computed flags"
        );
    }

    #[test]
    fn ensure_cache_shapes_force_invalidate_clears_computed_without_resize() {
        let mut cache = WorldTransformCache::default();
        ensure_cache_shapes(&mut cache, 2, false);
        for c in &mut cache.computed {
            *c = true;
        }
        ensure_cache_shapes(&mut cache, 2, true);
        assert!(cache.computed.iter().all(|c| !*c));
    }

    #[test]
    fn compute_world_matrices_for_space_empty_resets_cache() {
        let mut cache = WorldTransformCache::default();
        ensure_cache_shapes(&mut cache, 2, false);
        cache.computed[0] = true;
        compute_world_matrices_for_space(0, &[], &[], &mut cache).expect("ok");
        assert!(cache.world_matrices.is_empty());
        assert!(cache.computed.is_empty());
        assert!(cache.degenerate_scales.is_empty());
    }

    #[test]
    fn compute_world_matrices_for_space_single_root_uses_local_matrix() {
        let nodes = vec![translation_xform(4.0, 0.0, 0.0)];
        let parents = vec![-1];
        let mut cache = WorldTransformCache::default();
        compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");
        assert!(cache.computed[0]);
        let col3 = cache.world_matrices[0].col(3);
        assert!((col3.x - 4.0).abs() < 1e-5);
    }

    #[test]
    fn compute_world_matrices_for_space_two_level_chain_multiplies_in_order() {
        let nodes = vec![
            translation_xform(1.0, 0.0, 0.0),
            translation_xform(2.0, 0.0, 0.0),
        ];
        let parents = vec![-1, 0];
        let mut cache = WorldTransformCache::default();
        compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");
        let child_world = cache.world_matrices[1];
        let expected =
            render_transform_to_matrix(&nodes[0]) * render_transform_to_matrix(&nodes[1]);
        assert!(child_world.abs_diff_eq(expected, 1e-5));
    }

    /// Degenerate object scale on a parent marks every child in that transform chain.
    #[test]
    fn compute_world_matrices_for_space_propagates_degenerate_scale_to_children() {
        let mut collapsed_parent = identity_xform();
        collapsed_parent.scale = Vec3::new(0.0, 1.0, 1.0);
        let nodes = vec![collapsed_parent, identity_xform()];
        let parents = vec![-1, 0];
        let mut cache = WorldTransformCache::default();

        compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");

        assert_eq!(cache.degenerate_scales, vec![true, true]);
    }

    /// Negative nonzero object scale keeps the transform renderable for mirrored draw paths.
    #[test]
    fn compute_world_matrices_for_space_keeps_negative_nonzero_scale_renderable() {
        let mut mirrored = identity_xform();
        mirrored.scale = Vec3::new(-1.0, 1.0, 1.0);
        let nodes = vec![mirrored];
        let parents = vec![-1];
        let mut cache = WorldTransformCache::default();

        compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");

        assert_eq!(cache.degenerate_scales, vec![false]);
    }

    #[test]
    fn compute_world_matrices_for_space_cycle_falls_back_to_local_only() {
        let nodes = vec![identity_xform(), translation_xform(5.0, 0.0, 0.0)];
        let parents = vec![1, 0];
        let mut cache = WorldTransformCache::default();
        compute_world_matrices_for_space(42, &nodes, &parents, &mut cache).expect("cycle path");
        assert!(cache.computed.iter().all(|c| *c));
        let local1 = render_transform_to_matrix(&nodes[1]);
        assert!(
            cache.world_matrices[1].abs_diff_eq(local1, 1e-5),
            "cycle fallback must store local matrix unchanged"
        );
    }

    #[test]
    fn parallel_bulk_rebuild_matches_serial_on_large_chain() {
        // Constructed chain: each node parents the previous one. Above
        // WORLD_BULK_REBUILD_PARALLEL_MIN, the bulk-rebuild path triggers; the result must
        // be bit-identical to the existing serial incremental algorithm.
        let n = WORLD_BULK_REBUILD_PARALLEL_MIN + 7;
        let mut nodes = Vec::with_capacity(n);
        let mut parents = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(translation_xform(0.5, 0.0, 0.0));
            parents.push(if i == 0 { -1 } else { (i - 1) as i32 });
        }

        let mut parallel = WorldTransformCache::default();
        compute_world_matrices_for_space(0, &nodes, &parents, &mut parallel)
            .expect("parallel bulk");

        // Force the serial path: sub-threshold node count below WORLD_BULK_REBUILD_PARALLEL_MIN
        // still uses incremental, so we run it here and compare deeper subset
        // by directly invoking the incremental method via a fresh cache.
        let mut serial = WorldTransformCache::default();
        ensure_cache_shapes(&mut serial, n, false);
        if serial.children_dirty {
            rebuild_children(&parents, n, &mut serial.children);
            serial.children_dirty = false;
        }
        serial
            .compute_world_matrices_incremental(0, &nodes, &parents)
            .expect("serial");

        for i in 0..n {
            assert!(
                parallel.world_matrices[i].abs_diff_eq(serial.world_matrices[i], 1e-5),
                "world matrix mismatch at index {i}"
            );
            assert_eq!(parallel.degenerate_scales[i], serial.degenerate_scales[i]);
            assert!(parallel.computed[i] && serial.computed[i]);
        }
    }

    #[test]
    fn parallel_bulk_rebuild_handles_wide_tree() {
        // Multiple roots and a wide layer at depth 1: exercises level fan-out.
        let root_count = 16;
        let children_per_root = WORLD_BULK_REBUILD_PARALLEL_MIN / root_count + 8;
        let n = root_count + root_count * children_per_root;
        let mut nodes = Vec::with_capacity(n);
        let mut parents = Vec::with_capacity(n);
        for _ in 0..root_count {
            nodes.push(translation_xform(1.0, 0.0, 0.0));
            parents.push(-1);
        }
        for r in 0..root_count {
            for _ in 0..children_per_root {
                nodes.push(translation_xform(0.0, 1.0, 0.0));
                parents.push(r as i32);
            }
        }

        let mut parallel = WorldTransformCache::default();
        compute_world_matrices_for_space(0, &nodes, &parents, &mut parallel)
            .expect("parallel bulk");

        let mut serial = WorldTransformCache::default();
        ensure_cache_shapes(&mut serial, n, false);
        if serial.children_dirty {
            rebuild_children(&parents, n, &mut serial.children);
            serial.children_dirty = false;
        }
        serial
            .compute_world_matrices_incremental(0, &nodes, &parents)
            .expect("serial");

        for i in 0..n {
            assert!(
                parallel.world_matrices[i].abs_diff_eq(serial.world_matrices[i], 1e-5),
                "world matrix mismatch at index {i}"
            );
        }
    }

    #[test]
    fn compute_world_matrices_for_space_incremental_recomputes_only_dirty() {
        let nodes = vec![
            translation_xform(1.0, 0.0, 0.0),
            translation_xform(2.0, 0.0, 0.0),
        ];
        let parents = vec![-1, 0];
        let mut cache = WorldTransformCache::default();
        compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("first solve");
        let parent_world_before = cache.world_matrices[0];

        cache.computed[1] = false;
        cache.local_dirty[1] = true;
        compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("incremental");

        assert_eq!(
            cache.world_matrices[0], parent_world_before,
            "parent world matrix must not be re-derived when only the child is dirty"
        );
        assert!(cache.computed[1]);
    }
}
