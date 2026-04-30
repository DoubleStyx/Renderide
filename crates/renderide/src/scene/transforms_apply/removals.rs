//! Dense transform removals and parent-pointer deltas for [`super::apply_transforms_update_extracted`].
//!
//! Removals run first in buffer order (host swap-with-last semantics) and emit
//! [`TransformRemovalEvent`]s the per-space orchestrator forwards to every dependent
//! renderable subsystem. Parent updates run after the post-removal regrowth.

use crate::scene::render_space::RenderSpaceState;
use crate::scene::world::WorldTransformCache;
use crate::shared::TransformParentUpdate;

use super::NodeDirtyMask;

/// One successful transform removal: dense index removed and last valid index before `swap_remove`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformRemovalEvent {
    /// Removed dense transform index (`i32`, same as host removal buffer entry).
    pub removed_index: i32,
    /// Last valid index in `nodes` before the slot was removed (swapped-into source).
    pub last_index_before_swap: usize,
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

        space.nodes.swap_remove(idx);
        space.node_parents.swap_remove(idx);
        if idx < cache.world_matrices.len() {
            cache.world_matrices.swap_remove(idx);
            cache.computed.swap_remove(idx);
            cache.local_matrices.swap_remove(idx);
            cache.local_dirty.swap_remove(idx);
            if idx < cache.degenerate_scales.len() {
                cache.degenerate_scales.swap_remove(idx);
            }
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

/// Applies parent pointer deltas from a pre-extracted slice.
pub(super) fn apply_transform_parent_updates_extracted(
    space: &mut RenderSpaceState,
    cache: &mut WorldTransformCache,
    parents: &[TransformParentUpdate],
    changed: &mut NodeDirtyMask,
    invalidate_world: &mut bool,
) {
    profiling::scope!("scene::apply_parent_updates");
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
