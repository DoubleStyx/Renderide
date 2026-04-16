//! Incremental world-matrix propagation and child index for transform hierarchies.

use glam::Mat4;

use crate::shared::RenderTransform;

use super::error::SceneError;
use super::math::render_transform_to_matrix;

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
    /// Epoch per node for O(1) cycle detection during upward walks.
    pub(super) visit_epoch: Vec<u32>,
    /// Incremented before each upward walk.
    pub(super) walk_epoch: u32,
    /// Parent → children lists; rebuilt when structure changes.
    pub(super) children: Vec<Vec<usize>>,
    /// `children` must be rebuilt before descendant marking.
    pub(super) children_dirty: bool,
}

impl Default for WorldTransformCache {
    fn default() -> Self {
        Self {
            world_matrices: Vec::new(),
            computed: Vec::new(),
            local_matrices: Vec::new(),
            local_dirty: Vec::new(),
            visit_epoch: Vec::new(),
            walk_epoch: 0,
            children: Vec::new(),
            children_dirty: true,
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

/// Rebuilds parent → children adjacency.
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
        let child_list = children.get(i).map(Vec::as_slice).unwrap_or(&[]);
        stack.extend_from_slice(child_list);
        while let Some(child) = stack.pop() {
            computed[child] = false;
            let child_list = children.get(child).map(Vec::as_slice).unwrap_or(&[]);
            stack.extend_from_slice(child_list);
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

impl WorldTransformCache {
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
        let visit_epoch = &mut self.visit_epoch;
        let walk_epoch = &mut self.walk_epoch;
        let n = nodes.len();
        let mut stack: Vec<usize> = Vec::with_capacity(64.min(n));

        if visit_epoch.len() < n {
            visit_epoch.resize(n, 0);
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
                        "parent cycle at scene {} transform {} — local-only fallback",
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

            if cycle_detected {
                for &cid in &stack {
                    if !computed[cid] {
                        let local = get_local_matrix(nodes, local_matrices, local_dirty, cid);
                        world_matrices[cid] = local;
                        computed[cid] = true;
                    }
                }
                continue;
            }

            let mut parent_matrix = match maybe_uppermost_matrix {
                Some(m) => m,
                None => {
                    let top = match stack.pop() {
                        Some(t) => t,
                        None => continue,
                    };
                    let local = get_local_matrix(nodes, local_matrices, local_dirty, top);
                    world_matrices[top] = local;
                    computed[top] = true;
                    local
                }
            };

            while let Some(child_id) = stack.pop() {
                let local = get_local_matrix(nodes, local_matrices, local_dirty, child_id);
                parent_matrix *= local;
                world_matrices[child_id] = parent_matrix;
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
    if cache.world_matrices.len() != node_count {
        cache.world_matrices.resize(node_count, Mat4::IDENTITY);
        cache.computed.resize(node_count, false);
        cache.local_matrices.resize(node_count, Mat4::IDENTITY);
        cache.local_dirty.resize(node_count, true);
        cache.visit_epoch.resize(node_count, 0);
        cache.children_dirty = true;
        for c in cache.computed.iter_mut() {
            *c = false;
        }
    } else if force_invalidate {
        for c in cache.computed.iter_mut() {
            *c = false;
        }
    }
}

/// Runs incremental solve if anything is dirty or sizes changed.
pub fn compute_world_matrices_for_space(
    scene_id: i32,
    nodes: &[RenderTransform],
    node_parents: &[i32],
    cache: &mut WorldTransformCache,
) -> Result<(), SceneError> {
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

    cache.compute_world_matrices_incremental(scene_id, nodes, node_parents)
}
