//! Batch-window routing for [`super::build_plan`].
//!
//! A "batch window" is a maximal run of consecutive draws sharing a `MaterialDrawBatchKey`. Within
//! a window the grouping policy is determined by material properties (intersection-pass, grab-pass,
//! transparency class) plus device capability (`supports_base_instance`).

use std::ops::Range;

use crate::world_mesh::draw_prep::WorldMeshDrawItem;
use crate::world_mesh::phase_classification::classify_world_mesh_batch;

use super::{DrawGroup, WorldMeshPhase};

/// Same-batch-key draw window and its subpass routing metadata.
#[derive(Clone, Debug)]
pub(super) struct BatchWindow {
    /// Draw index range covered by this window.
    pub(super) range: Range<usize>,
    /// Primary mesh render phase for the window.
    pub(super) phase: WorldMeshPhase,
    /// Whether every draw must remain a singleton group.
    pub(super) singleton: bool,
}

/// Returns the next same-batch-key window starting at `start`.
pub(super) fn next_batch_window(
    draws: &[WorldMeshDrawItem],
    start: usize,
    supports_base_instance: bool,
) -> BatchWindow {
    let key = &draws[start].batch_key;
    let mut end = start + 1;
    while end < draws.len() && &draws[end].batch_key == key {
        end += 1;
    }

    let classification = classify_world_mesh_batch(key);
    let order_dependent = !key.transparent_class.allows_relaxed_batching();

    BatchWindow {
        range: start..end,
        phase: classification.phase,
        singleton: !supports_base_instance
            || draws[start].skinned
            || (classification.post_skybox && order_dependent)
            || (key.alpha_blended && order_dependent)
            || classification.grab_pass,
    }
}

/// Appends `members` to `slab_layout` and returns a [`DrawGroup`] covering the new slab range.
#[inline]
pub(super) fn build_group(
    slab_layout: &mut Vec<usize>,
    representative_draw_idx: usize,
    members: &[usize],
) -> DrawGroup {
    let first_instance = slab_layout.len() as u32;
    slab_layout.extend_from_slice(members);
    let count = members.len() as u32;
    DrawGroup {
        representative_draw_idx,
        instance_range: first_instance..first_instance + count,
        material_packet_idx: 0,
    }
}
