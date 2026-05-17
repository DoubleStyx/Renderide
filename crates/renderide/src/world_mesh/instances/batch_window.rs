//! Batch-window routing for [`super::build_plan`].
//!
//! A "batch window" is a maximal run of consecutive draws sharing a `MaterialDrawBatchKey`. Within
//! a window the grouping policy is determined by material properties (intersection-pass, grab-pass,
//! transparency class) plus device capability (`supports_base_instance`).

use std::ops::Range;

use crate::materials::render_queue_is_transparent;
use crate::world_mesh::MaterialDrawBatchKey;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::DrawGroup;

/// Same-batch-key draw window and its subpass routing metadata.
#[derive(Clone, Debug)]
pub(super) struct BatchWindow {
    /// Draw index range covered by this window.
    pub(super) range: Range<usize>,
    /// Whether the window belongs to the intersection subpass.
    pub(super) intersect: bool,
    /// Whether the window belongs to the regular post-skybox subpass.
    pub(super) post_skybox: bool,
    /// Whether the window belongs to the grab-pass transparent subpass.
    pub(super) grab_pass: bool,
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
    let order_dependent = !key.transparent_class.allows_relaxed_batching();
    debug_assert!(
        !(intersect && grab_pass),
        "intersection and grab-pass subpasses are mutually exclusive"
    );

    BatchWindow {
        range: start..end,
        intersect,
        post_skybox,
        grab_pass,
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

/// Selects the subpass group vector for a batch window.
pub(super) fn subpass_groups<'a>(
    regular_groups: &'a mut Vec<DrawGroup>,
    post_skybox_groups: &'a mut Vec<DrawGroup>,
    intersect_groups: &'a mut Vec<DrawGroup>,
    transparent_groups: &'a mut Vec<DrawGroup>,
    window: &BatchWindow,
) -> &'a mut Vec<DrawGroup> {
    if window.intersect {
        intersect_groups
    } else if window.grab_pass {
        transparent_groups
    } else if window.post_skybox {
        post_skybox_groups
    } else {
        regular_groups
    }
}

/// Appends `members` to `slab_layout` and pushes a [`DrawGroup`] covering the new slab range.
#[inline]
pub(super) fn emit_group(
    slab_layout: &mut Vec<usize>,
    target: &mut Vec<DrawGroup>,
    representative_draw_idx: usize,
    members: &[usize],
) {
    let first_instance = slab_layout.len() as u32;
    slab_layout.extend_from_slice(members);
    let count = members.len() as u32;
    target.push(DrawGroup {
        representative_draw_idx,
        instance_range: first_instance..first_instance + count,
        material_packet_idx: 0,
    });
}
