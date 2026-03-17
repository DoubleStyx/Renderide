//! World matrix computation and transform hierarchy utilities.

use nalgebra::Matrix4;

use crate::scene::{math::render_transform_to_matrix, Scene};

use super::error::SceneError;

/// Per-scene cache for world matrices and computed flags.
pub(super) struct SceneCache {
    /// World-space matrices for each transform.
    pub world_matrices: Vec<Matrix4<f32>>,
    /// Whether each transform's world matrix has been computed this frame.
    pub computed: Vec<bool>,
}

/// Fixes transform ID references after swap_remove: removed ID becomes -1,
/// last index (swapped into removed slot) becomes removed ID.
pub(super) fn fixup_transform_id(old: i32, removed_id: i32, last_index: usize) -> i32 {
    if old == removed_id {
        -1
    } else if old == last_index as i32 {
        removed_id
    } else {
        old
    }
}

/// Marks descendants of uncomputed transforms as uncomputed.
/// Walks each node's parent chain to find the uppermost uncomputed ancestor;
/// if found, marks all nodes in that chain as uncomputed.
pub(super) fn mark_descendants_uncomputed(node_parents: &[i32], computed: &mut [bool]) {
    let n = computed.len();
    if n == 0 {
        return;
    }
    let mut checked = vec![false; n];
    for transform_index in (0..n).rev() {
        if checked[transform_index] {
            continue;
        }
        let mut maybe_last_non_computed: Option<usize> = None;
        let mut id = transform_index;
        let mut steps = 0;
        while id < n && steps < n {
            steps += 1;
            if !computed[id] {
                maybe_last_non_computed = Some(id);
            }
            if checked[id] {
                break;
            }
            let p = node_parents.get(id).copied().unwrap_or(-1);
            if p < 0 || (p as usize) >= n || p == id as i32 {
                break;
            }
            id = p as usize;
        }
        if let Some(last_non_computed) = maybe_last_non_computed {
            let mut id = transform_index;
            let mut steps = 0;
            while id != last_non_computed && id < n && steps < n {
                steps += 1;
                computed[id] = false;
                checked[id] = true;
                let p = node_parents.get(id).copied().unwrap_or(-1);
                if p < 0 || (p as usize) >= n || p == id as i32 {
                    break;
                }
                id = p as usize;
            }
        } else {
            let mut id = transform_index;
            let mut steps = 0;
            while id < n && steps < n {
                steps += 1;
                checked[id] = true;
                let p = node_parents.get(id).copied().unwrap_or(-1);
                if p < 0 || (p as usize) >= n || p == id as i32 {
                    break;
                }
                id = p as usize;
            }
        }
        checked[transform_index] = true;
    }
}

/// Incremental world matrix computation: only recomputes nodes with `computed[i] == false`.
/// Walks up from each uncomputed node to find the first computed ancestor, then multiplies down.
pub(super) fn compute_world_matrices_incremental(
    scene: &Scene,
    world_matrices: &mut [Matrix4<f32>],
    computed: &mut [bool],
) -> Result<(), SceneError> {
    let n = scene.nodes.len();
    let node_parents = &scene.node_parents;
    let nodes = &scene.nodes;
    let mut stack = Vec::with_capacity(64.min(n));

    for transform_index in (0..n).rev() {
        if computed[transform_index] {
            continue;
        }

        let mut maybe_uppermost_matrix: Option<Matrix4<f32>> = None;
        let mut id = transform_index;
        let mut steps = 0;
        while id < n && steps < n {
            steps += 1;
            if computed[id] {
                maybe_uppermost_matrix = Some(world_matrices[id]);
                break;
            }
            stack.push(id);
            let p = node_parents.get(id).copied().unwrap_or(-1);
            if p < 0 || (p as usize) >= n || p == id as i32 {
                break;
            }
            id = p as usize;
        }

        let mut parent_matrix = match maybe_uppermost_matrix {
            Some(m) => m,
            None => {
                let top = match stack.pop() {
                    Some(t) => t,
                    None => continue,
                };
                let local = render_transform_to_matrix(&nodes[top]);
                let uppermost = Matrix4::<f32>::identity() * local;
                world_matrices[top] = uppermost;
                computed[top] = true;
                uppermost
            }
        };

        while let Some(child_id) = stack.pop() {
            let local = render_transform_to_matrix(&nodes[child_id]);
            parent_matrix = parent_matrix * local;
            world_matrices[child_id] = parent_matrix;
            computed[child_id] = true;
        }
    }

    Ok(())
}

/// Full iterative DFS world matrix computation with cycle detection.
/// Used by tests; root-level nodes use identity as parent.
#[cfg(test)]
pub(crate) fn compute_world_matrices_from_scene(scene: &Scene) -> Vec<Matrix4<f32>> {
    let n = scene.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut world = vec![Matrix4::identity(); n];
    let mut visited = vec![false; n];
    let mut in_stack = vec![false; n];

    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        stack.push(start);
        in_stack[start] = true;
        while let Some(&i) = stack.last() {
            if visited[i] {
                in_stack[i] = false;
                stack.pop();
                continue;
            }
            let p = scene.node_parents.get(i).copied().unwrap_or(-1);
            let p_usize = if p >= 0 && (p as usize) < n && p != i as i32 {
                p as usize
            } else {
                let local = render_transform_to_matrix(&scene.nodes[i]);
                world[i] = Matrix4::<f32>::identity() * local;
                visited[i] = true;
                in_stack[i] = false;
                stack.pop();
                continue;
            };

            if in_stack[p_usize] {
                logger::trace!(
                    "Cycle detected in scene {} at transform {} (parent {}); treating as root",
                    scene.id,
                    i,
                    p
                );
                let local = render_transform_to_matrix(&scene.nodes[i]);
                world[i] = Matrix4::<f32>::identity() * local;
                visited[i] = true;
                in_stack[i] = false;
                stack.pop();
                continue;
            }

            if !visited[p_usize] {
                stack.push(p_usize);
                in_stack[p_usize] = true;
                continue;
            }

            let local = render_transform_to_matrix(&scene.nodes[i]);
            world[i] = world[p_usize] * local;
            visited[i] = true;
            in_stack[i] = false;
            stack.pop();
        }
    }

    world
}
