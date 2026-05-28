//! Ordering helpers for interleaving transparent post groups and grab-pass groups.

use crate::world_mesh::{InstancePlan, MeshPassKind, WorldMeshPhase};

/// Returns the phases that make up the transparent sequence mesh pass.
pub(super) fn transparent_sequence_phase_pair() -> (WorldMeshPhase, WorldMeshPhase) {
    match MeshPassKind::TransparentSequence.phases() {
        [transparent, grab] => (*transparent, *grab),
        _ => (WorldMeshPhase::Transparent, WorldMeshPhase::TransparentGrab),
    }
}

/// Returns whether the next sorted transparent sequence entry is a post group.
fn next_sequence_entry_is_post(plan: &InstancePlan, post_idx: usize, grab_idx: usize) -> bool {
    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    let Some(post) = plan.phase(transparent_phase).get(post_idx) else {
        return false;
    };
    let Some(grab) = plan.phase(grab_phase).get(grab_idx) else {
        return true;
    };
    post.representative_draw_idx <= grab.representative_draw_idx
}

/// Advances a pending transparent-post run when the next sorted item is a post group.
pub(super) fn advance_pending_post_run(
    plan: &InstancePlan,
    post_idx: &mut usize,
    grab_idx: usize,
    pending_post_start: &mut Option<usize>,
) -> bool {
    if !next_sequence_entry_is_post(plan, *post_idx, grab_idx) {
        return false;
    }
    if pending_post_start.is_none() {
        *pending_post_start = Some(*post_idx);
    }
    *post_idx += 1;
    true
}
