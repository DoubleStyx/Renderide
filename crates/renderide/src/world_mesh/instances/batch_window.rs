//! Batch-window routing for [`super::build_plan`].
//!
//! A "batch window" is a maximal run of consecutive draws sharing a `MaterialDrawBatchKey`. Within
//! a window the grouping policy is determined by material properties (intersection-pass, grab-pass,
//! transparency class) plus device capability (`supports_base_instance`).

use std::ops::Range;

use crate::materials::{UNITY_RENDER_QUEUE_ALPHA_TEST, render_queue_is_transparent};
use crate::world_mesh::MaterialDrawBatchKey;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

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

    let intersect = key.embedded_requires_intersection_pass;
    let grab_pass = key.embedded_uses_scene_color_snapshot;
    let post_skybox = !intersect && !grab_pass && regular_window_records_after_skybox(key);
    let phase = phase_for_window(key, intersect, grab_pass, post_skybox);
    let order_dependent = !key.transparent_class.allows_relaxed_batching();
    debug_assert!(
        !(intersect && grab_pass),
        "intersection and grab-pass subpasses are mutually exclusive"
    );

    BatchWindow {
        range: start..end,
        phase,
        singleton: !supports_base_instance
            || draws[start].skinned
            || (post_skybox && order_dependent)
            || (key.alpha_blended && order_dependent)
            || grab_pass,
    }
}

/// Returns whether a regular forward draw must render after the skybox/background draw.
fn regular_window_records_after_skybox(key: &MaterialDrawBatchKey) -> bool {
    key.alpha_blended
        || render_queue_is_transparent(key.render_queue)
        || key.render_state.depth_write == Some(false)
}

/// Selects the primary phase for one same-batch-key window.
fn phase_for_window(
    key: &MaterialDrawBatchKey,
    intersect: bool,
    grab_pass: bool,
    post_skybox: bool,
) -> WorldMeshPhase {
    if intersect {
        WorldMeshPhase::Intersection
    } else if grab_pass {
        WorldMeshPhase::TransparentGrab
    } else if post_skybox {
        WorldMeshPhase::Transparent
    } else if key.render_queue >= UNITY_RENDER_QUEUE_ALPHA_TEST {
        WorldMeshPhase::ForwardAlphaTest
    } else {
        WorldMeshPhase::ForwardOpaque
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
