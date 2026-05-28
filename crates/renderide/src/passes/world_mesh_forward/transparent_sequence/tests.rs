//! Unit tests for transparent sequence ordering, snapshots, and MSAA resolves.

use crate::materials::SceneColorSnapshotMode;
use crate::world_mesh::{DrawGroup, InstancePlan, WorldMeshPhase};

use super::order::{advance_pending_post_run, transparent_sequence_phase_pair};
use super::transparent_sequence_pass_needed;

/// Operation emitted by the test-only transparent sequence simulator.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TransparentSequenceTestOp {
    /// A contiguous range of non-grab transparent groups was drawn.
    DrawPostRange(usize, usize),
    /// MSAA color was resolved before copying a grab snapshot.
    ResolveBeforeGrab(usize),
    /// A per-object grab snapshot was copied.
    SnapshotForGrab(usize),
    /// A reusable named-background grab snapshot was copied.
    SnapshotForNamedGrab(usize),
    /// A named-background grab reused the first named snapshot.
    ReuseNamedGrabSnapshot(usize),
    /// A grab group was drawn.
    DrawGrab(usize),
    /// A grab group was skipped after its snapshot copy failed.
    SkipGrabMissingSnapshot(usize),
    /// The final MSAA scene color was resolved.
    FinalResolve,
}

/// Collects simulated transparent sequence operations with successful snapshot copies.
fn collect_transparent_sequence_test_ops(
    plan: &InstancePlan,
    sample_count: u32,
) -> Vec<TransparentSequenceTestOp> {
    collect_transparent_sequence_test_ops_with_snapshot_result(plan, sample_count, true)
}

/// Collects simulated transparent sequence operations with a fixed snapshot-copy result.
fn collect_transparent_sequence_test_ops_with_snapshot_result(
    plan: &InstancePlan,
    sample_count: u32,
    snapshot_copy_succeeds: bool,
) -> Vec<TransparentSequenceTestOp> {
    collect_transparent_sequence_test_ops_with_modes(
        plan,
        sample_count,
        snapshot_copy_succeeds,
        &[],
    )
}

/// Collects simulated transparent sequence operations with explicit grab snapshot modes.
fn collect_transparent_sequence_test_ops_with_modes(
    plan: &InstancePlan,
    sample_count: u32,
    snapshot_copy_succeeds: bool,
    grab_modes: &[SceneColorSnapshotMode],
) -> Vec<TransparentSequenceTestOp> {
    let mut ops = Vec::new();
    let mut post_idx = 0usize;
    let mut grab_idx = 0usize;
    let mut pending_post_start = None;
    let mut scene_color_resolved_current = false;
    let mut named_background_snapshot_ready = false;

    let (transparent_phase, grab_phase) = transparent_sequence_phase_pair();
    let transparent_groups = plan.phase(transparent_phase);
    let grab_groups = plan.phase(grab_phase);

    while post_idx < transparent_groups.len() || grab_idx < grab_groups.len() {
        if advance_pending_post_run(plan, &mut post_idx, grab_idx, &mut pending_post_start) {
            continue;
        }

        if let Some(start) = pending_post_start.take() {
            ops.push(TransparentSequenceTestOp::DrawPostRange(start, post_idx));
            if sample_count > 1 {
                scene_color_resolved_current = false;
            }
        }
        let mode = grab_modes
            .get(grab_idx)
            .copied()
            .unwrap_or(SceneColorSnapshotMode::PerObjectGrab);
        let needs_snapshot_copy = match mode {
            SceneColorSnapshotMode::NamedBackgroundGrab => !named_background_snapshot_ready,
            SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => true,
        };
        if needs_snapshot_copy {
            if sample_count > 1 {
                ops.push(TransparentSequenceTestOp::ResolveBeforeGrab(grab_idx));
                scene_color_resolved_current = true;
            }
            match mode {
                SceneColorSnapshotMode::NamedBackgroundGrab => {
                    ops.push(TransparentSequenceTestOp::SnapshotForNamedGrab(grab_idx));
                }
                SceneColorSnapshotMode::PerObjectGrab | SceneColorSnapshotMode::None => {
                    ops.push(TransparentSequenceTestOp::SnapshotForGrab(grab_idx));
                }
            }
            if snapshot_copy_succeeds {
                if mode == SceneColorSnapshotMode::NamedBackgroundGrab {
                    named_background_snapshot_ready = true;
                }
            } else {
                ops.push(TransparentSequenceTestOp::SkipGrabMissingSnapshot(grab_idx));
                grab_idx += 1;
                continue;
            }
        } else {
            ops.push(TransparentSequenceTestOp::ReuseNamedGrabSnapshot(grab_idx));
        }
        ops.push(TransparentSequenceTestOp::DrawGrab(grab_idx));
        if sample_count > 1 {
            scene_color_resolved_current = false;
        }
        grab_idx += 1;
    }

    if let Some(start) = pending_post_start {
        ops.push(TransparentSequenceTestOp::DrawPostRange(start, post_idx));
        if sample_count > 1 {
            scene_color_resolved_current = false;
        }
    }
    if sample_count > 1 && !scene_color_resolved_current {
        ops.push(TransparentSequenceTestOp::FinalResolve);
    }
    ops
}

/// Builds a single draw group with a representative draw index.
fn group(representative_draw_idx: usize) -> DrawGroup {
    DrawGroup {
        representative_draw_idx,
        instance_range: 0..1,
        material_packet_idx: 0,
    }
}

/// Builds an instance plan with transparent and transparent-grab phase groups.
fn plan_with_transparent_groups(non_grab: Vec<DrawGroup>, grab: Vec<DrawGroup>) -> InstancePlan {
    let mut plan = InstancePlan::default();
    plan.phase_mut(WorldMeshPhase::Transparent).extend(non_grab);
    plan.phase_mut(WorldMeshPhase::TransparentGrab).extend(grab);
    plan
}

/// Empty MSAA transparent tails still need the final color resolve.
#[test]
fn msaa_empty_tail_still_records_final_resolve() {
    let plan = plan_with_transparent_groups(Vec::new(), Vec::new());

    assert_eq!(
        collect_transparent_sequence_test_ops(&plan, 4),
        vec![TransparentSequenceTestOp::FinalResolve]
    );
}

/// Pass admission includes opaque-only MSAA color resolves.
#[test]
fn pass_needed_includes_msaa_resolve_only_frames() {
    let empty = plan_with_transparent_groups(Vec::new(), Vec::new());
    assert!(!transparent_sequence_pass_needed(true, &empty, false, 1));
    assert!(!transparent_sequence_pass_needed(true, &empty, true, 1));
    assert!(transparent_sequence_pass_needed(true, &empty, true, 4));
    assert!(!transparent_sequence_pass_needed(false, &empty, true, 4));

    let transparent = plan_with_transparent_groups(vec![group(2)], Vec::new());
    assert!(transparent_sequence_pass_needed(
        true,
        &transparent,
        false,
        1
    ));
}

/// Non-grab transparent groups draw as sorted contiguous runs.
#[test]
fn non_grab_transparent_groups_stay_in_sorted_runs() {
    let plan = plan_with_transparent_groups(vec![group(2), group(4), group(8)], Vec::new());

    assert_eq!(
        collect_transparent_sequence_test_ops(&plan, 1),
        vec![TransparentSequenceTestOp::DrawPostRange(0, 3)]
    );
}

/// Grab groups copy snapshots immediately before drawing.
#[test]
fn grab_groups_trigger_snapshot_immediately_before_draw() {
    let plan = plan_with_transparent_groups(vec![group(1), group(9)], vec![group(5)]);

    assert_eq!(
        collect_transparent_sequence_test_ops(&plan, 1),
        vec![
            TransparentSequenceTestOp::DrawPostRange(0, 1),
            TransparentSequenceTestOp::SnapshotForGrab(0),
            TransparentSequenceTestOp::DrawGrab(0),
            TransparentSequenceTestOp::DrawPostRange(1, 2),
        ]
    );
}

/// Per-object grab groups take separate snapshots.
#[test]
fn multiple_grab_groups_take_multiple_snapshots() {
    let plan = plan_with_transparent_groups(Vec::new(), vec![group(3), group(7)]);

    assert_eq!(
        collect_transparent_sequence_test_ops(&plan, 1),
        vec![
            TransparentSequenceTestOp::SnapshotForGrab(0),
            TransparentSequenceTestOp::DrawGrab(0),
            TransparentSequenceTestOp::SnapshotForGrab(1),
            TransparentSequenceTestOp::DrawGrab(1),
        ]
    );
}

/// Named-background grab groups reuse the first named snapshot.
#[test]
fn named_grab_groups_reuse_the_first_named_snapshot() {
    let plan = plan_with_transparent_groups(Vec::new(), vec![group(3), group(7)]);

    assert_eq!(
        collect_transparent_sequence_test_ops_with_modes(
            &plan,
            1,
            true,
            &[
                SceneColorSnapshotMode::NamedBackgroundGrab,
                SceneColorSnapshotMode::NamedBackgroundGrab,
            ],
        ),
        vec![
            TransparentSequenceTestOp::SnapshotForNamedGrab(0),
            TransparentSequenceTestOp::DrawGrab(0),
            TransparentSequenceTestOp::ReuseNamedGrabSnapshot(1),
            TransparentSequenceTestOp::DrawGrab(1),
        ]
    );
}

/// Per-object grabs do not invalidate a previously copied named-background snapshot.
#[test]
fn per_object_grabs_still_copy_between_named_grab_reuse() {
    let plan = plan_with_transparent_groups(Vec::new(), vec![group(3), group(5), group(7)]);

    assert_eq!(
        collect_transparent_sequence_test_ops_with_modes(
            &plan,
            1,
            true,
            &[
                SceneColorSnapshotMode::NamedBackgroundGrab,
                SceneColorSnapshotMode::PerObjectGrab,
                SceneColorSnapshotMode::NamedBackgroundGrab,
            ],
        ),
        vec![
            TransparentSequenceTestOp::SnapshotForNamedGrab(0),
            TransparentSequenceTestOp::DrawGrab(0),
            TransparentSequenceTestOp::SnapshotForGrab(1),
            TransparentSequenceTestOp::DrawGrab(1),
            TransparentSequenceTestOp::ReuseNamedGrabSnapshot(2),
            TransparentSequenceTestOp::DrawGrab(2),
        ]
    );
}

/// Interleaved post and grab groups preserve sorted order while reusing named snapshots.
#[test]
fn interleaved_named_grabs_reuse_background_after_per_object_copy() {
    let plan = plan_with_transparent_groups(
        vec![group(4), group(9)],
        vec![group(2), group(6), group(11)],
    );

    assert_eq!(
        collect_transparent_sequence_test_ops_with_modes(
            &plan,
            1,
            true,
            &[
                SceneColorSnapshotMode::NamedBackgroundGrab,
                SceneColorSnapshotMode::PerObjectGrab,
                SceneColorSnapshotMode::NamedBackgroundGrab,
            ],
        ),
        vec![
            TransparentSequenceTestOp::SnapshotForNamedGrab(0),
            TransparentSequenceTestOp::DrawGrab(0),
            TransparentSequenceTestOp::DrawPostRange(0, 1),
            TransparentSequenceTestOp::SnapshotForGrab(1),
            TransparentSequenceTestOp::DrawGrab(1),
            TransparentSequenceTestOp::DrawPostRange(1, 2),
            TransparentSequenceTestOp::ReuseNamedGrabSnapshot(2),
            TransparentSequenceTestOp::DrawGrab(2),
        ]
    );
}

/// MSAA resolves before every copied grab snapshot and after the tail.
#[test]
fn msaa_resolves_before_each_grab_and_after_tail() {
    let plan = plan_with_transparent_groups(vec![group(1)], vec![group(3), group(7)]);

    assert_eq!(
        collect_transparent_sequence_test_ops(&plan, 4),
        vec![
            TransparentSequenceTestOp::DrawPostRange(0, 1),
            TransparentSequenceTestOp::ResolveBeforeGrab(0),
            TransparentSequenceTestOp::SnapshotForGrab(0),
            TransparentSequenceTestOp::DrawGrab(0),
            TransparentSequenceTestOp::ResolveBeforeGrab(1),
            TransparentSequenceTestOp::SnapshotForGrab(1),
            TransparentSequenceTestOp::DrawGrab(1),
            TransparentSequenceTestOp::FinalResolve,
        ]
    );
}

/// Snapshot-copy failure skips the affected grab draw.
#[test]
fn failed_snapshot_copy_skips_grab_draw() {
    let plan = plan_with_transparent_groups(Vec::new(), vec![group(3)]);

    assert_eq!(
        collect_transparent_sequence_test_ops_with_snapshot_result(&plan, 1, false),
        vec![
            TransparentSequenceTestOp::SnapshotForGrab(0),
            TransparentSequenceTestOp::SkipGrabMissingSnapshot(0),
        ]
    );
}

/// Post groups before a failed grab still count as recorded transparent-tail work.
#[test]
fn post_groups_before_failed_grab_still_count_as_recorded_tail() {
    let plan = plan_with_transparent_groups(vec![group(1)], vec![group(3)]);

    assert_eq!(
        collect_transparent_sequence_test_ops_with_snapshot_result(&plan, 4, false),
        vec![
            TransparentSequenceTestOp::DrawPostRange(0, 1),
            TransparentSequenceTestOp::ResolveBeforeGrab(0),
            TransparentSequenceTestOp::SnapshotForGrab(0),
            TransparentSequenceTestOp::SkipGrabMissingSnapshot(0),
        ]
    );
}
