//! Pose row validation, commit, and post-commit dirty-flag propagation.
//!
//! Row collection runs in two phases so the in-bounds filtering can fan out across rayon workers
//! above [`POSE_UPDATE_PARALLEL_MIN_ROWS`]: the parallel pass produces a [`PoseRow`] vector in
//! input order, and the serial commit pass repairs and writes each row into
//! [`RenderSpaceState::nodes`] under the lock that already protects the per-space apply.

use crate::scene::pose::repair_render_transform;
use crate::scene::render_space::RenderSpaceState;
use crate::scene::world::WorldTransformCache;
use crate::shared::{RenderTransform, TransformPoseUpdate};

use super::NodeDirtyMask;

/// Minimum pose-update count before [`collect_pose_rows`] fans out collection across rayon
/// workers. Below this threshold the scalar loop is faster than rayon dispatch overhead.
const POSE_UPDATE_PARALLEL_MIN_ROWS: usize = 128;

/// In-bounds pose row ready for serial repair and commit into [`RenderSpaceState::nodes`].
struct PoseRow {
    /// Dense transform index into [`RenderSpaceState::nodes`].
    transform_index: usize,
    /// Host pose from the row.
    pose: RenderTransform,
}

/// Index of the first sentinel `transform_id < 0` row, or `poses.len()` if no terminator is present.
#[inline]
fn pose_terminator_index(poses: &[TransformPoseUpdate]) -> usize {
    poses
        .iter()
        .position(|pu| pu.transform_id < 0)
        .unwrap_or(poses.len())
}

/// Walks the active prefix of `poses` once and produces one [`PoseRow`] per in-bounds entry.
fn collect_pose_rows(poses: &[TransformPoseUpdate], node_count: usize) -> Vec<PoseRow> {
    profiling::scope!("scene::collect_pose_rows");
    let active_len = pose_terminator_index(poses);
    let active = &poses[..active_len];
    let row_for = |pu: &TransformPoseUpdate| -> Option<PoseRow> {
        let idx = pu.transform_id as usize;
        if idx >= node_count {
            return None;
        }
        Some(PoseRow {
            transform_index: idx,
            pose: pu.pose,
        })
    };

    if active.len() >= POSE_UPDATE_PARALLEL_MIN_ROWS {
        use rayon::prelude::*;
        active.par_iter().filter_map(row_for).collect()
    } else {
        active.iter().filter_map(row_for).collect()
    }
}

/// Applies pose rows from a pre-extracted slice, repairing invalid components before commit.
///
/// Steady-state Resonite scenes route pose updates for every transform in an avatar skeleton each
/// frame even when most bones did not animate. We bitwise-compare each row to the existing scene
/// pose and skip the repair + write + dirty mark when they match. The mark drives downstream
/// world-matrix recomputation, so skipping it here also skips per-node work in the cache flush
/// and the prepared-renderables expansion later in the frame.
pub(super) fn apply_transform_pose_updates_extracted(
    space: &mut RenderSpaceState,
    poses: &[TransformPoseUpdate],
    _frame_index: i32,
    _sid: i32,
    changed: &mut NodeDirtyMask,
) {
    profiling::scope!("scene::apply_pose_updates");
    if poses.is_empty() {
        return;
    }
    let rows = collect_pose_rows(poses, space.nodes.len());
    for row in rows {
        let fallback = space.nodes[row.transform_index];
        // Bitwise-equal poses repair to themselves, so the cheaper compare lets us skip the dirty
        // mark that drives downstream world-matrix recompute and prepared-renderables refresh.
        if pose_matches(&row.pose, &fallback) {
            continue;
        }
        let repaired = repair_render_transform(&row.pose, &fallback);
        // After repair the value may still equal the existing scene pose (e.g. a clamp produced
        // the same number as the cached fallback). Treat that case as unchanged too.
        if pose_matches(&repaired, &fallback) {
            continue;
        }
        space.nodes[row.transform_index] = repaired;
        changed.mark(row.transform_index);
    }
}

/// Field-wise bitwise equality for [`RenderTransform`]. Defers to `glam`'s `PartialEq` on `Vec3`
/// and `Quat` so identical bit patterns compare equal.
#[inline]
fn pose_matches(a: &RenderTransform, b: &RenderTransform) -> bool {
    a.position == b.position && a.scale == b.scale && a.rotation == b.rotation
}

/// Marks per-node dirty flags after local transform edits.
pub(super) fn propagate_transform_change_dirty_flags(
    cache: &mut WorldTransformCache,
    changed: &NodeDirtyMask,
) {
    if !changed.any() {
        return;
    }
    for &i in changed.indices() {
        if i < cache.computed.len() {
            cache.computed[i] = false;
        }
        if i < cache.local_dirty.len() {
            cache.local_dirty[i] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;

    fn pose_at(x: f32) -> RenderTransform {
        RenderTransform {
            position: Vec3::new(x, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    /// [`pose_terminator_index`] returns the index of the first sentinel `transform_id < 0`.
    #[test]
    fn pose_terminator_index_finds_first_sentinel() {
        let pose = pose_at(0.0);
        let rows = vec![
            TransformPoseUpdate {
                transform_id: 0,
                pose,
            },
            TransformPoseUpdate {
                transform_id: 1,
                pose,
            },
            TransformPoseUpdate {
                transform_id: -1,
                pose,
            },
            TransformPoseUpdate {
                transform_id: 2,
                pose,
            },
        ];
        assert_eq!(pose_terminator_index(&rows), 2);
    }

    /// [`pose_terminator_index`] returns `len` when no sentinel is present.
    #[test]
    fn pose_terminator_index_no_sentinel_returns_len() {
        let pose = pose_at(0.0);
        let rows = vec![TransformPoseUpdate {
            transform_id: 0,
            pose,
        }];
        assert_eq!(pose_terminator_index(&rows), rows.len());
    }

    /// [`collect_pose_rows`] preserves input order, drops out-of-range transform indices, and
    /// keeps raw pose payloads for the serial repair pass.
    #[test]
    fn collect_pose_rows_preserves_order_and_raw_payload() {
        let valid = pose_at(2.0);
        let mut bad = pose_at(0.0);
        bad.position.x = f32::NAN;
        let rows = vec![
            TransformPoseUpdate {
                transform_id: 0,
                pose: valid,
            },
            TransformPoseUpdate {
                transform_id: 7,
                pose: valid,
            },
            TransformPoseUpdate {
                transform_id: 1,
                pose: bad,
            },
            TransformPoseUpdate {
                transform_id: -1,
                pose: valid,
            },
        ];
        let out = collect_pose_rows(&rows, 3);
        assert_eq!(
            out.len(),
            2,
            "out-of-range and sentinel rows must be dropped"
        );
        assert_eq!(out[0].transform_index, 0);
        assert_eq!(out[0].pose.position, valid.position);
        assert_eq!(out[1].transform_index, 1);
        assert!(out[1].pose.position.x.is_nan());
        assert_eq!(out[1].pose.position.y, bad.position.y);
        assert_eq!(out[1].pose.position.z, bad.position.z);
        assert_eq!(out[1].pose.scale, bad.scale);
        assert_eq!(out[1].pose.rotation, bad.rotation);
    }

    /// [`collect_pose_rows`] above [`POSE_UPDATE_PARALLEL_MIN_ROWS`] still preserves input order.
    #[test]
    fn collect_pose_rows_parallel_path_preserves_order() {
        let pose = pose_at(1.0);
        let n = POSE_UPDATE_PARALLEL_MIN_ROWS + 16;
        let mut rows = Vec::with_capacity(n + 1);
        for i in 0..n {
            rows.push(TransformPoseUpdate {
                transform_id: i as i32,
                pose,
            });
        }
        rows.push(TransformPoseUpdate {
            transform_id: -1,
            pose,
        });
        let out = collect_pose_rows(&rows, n);
        assert_eq!(out.len(), n);
        for (i, row) in out.iter().enumerate() {
            assert_eq!(row.transform_index, i);
            assert_eq!(row.pose.position, pose.position);
        }
    }

    /// Invalid host pose components are repaired component-wise and valid components still commit.
    #[test]
    fn invalid_pose_update_repairs_components_and_commits() {
        let mut existing = pose_at(42.0);
        existing.rotation = Quat::from_xyzw(0.0, 0.25, 0.0, 0.75);
        let mut bad = pose_at(1.0);
        bad.position.x = f32::NAN;
        bad.position.y = crate::scene::pose::POSE_VALIDATION_THRESHOLD;
        bad.scale.y = 2.0;
        bad.scale.z = f32::INFINITY;
        bad.rotation.w = f32::NAN;

        let mut space = RenderSpaceState::default();
        space.nodes.push(existing);
        let mut changed = NodeDirtyMask::new(space.nodes.len());

        apply_transform_pose_updates_extracted(
            &mut space,
            &[TransformPoseUpdate {
                transform_id: 0,
                pose: bad,
            }],
            9,
            2,
            &mut changed,
        );

        assert_eq!(space.nodes[0].position.x, existing.position.x);
        assert_eq!(
            space.nodes[0].position.y,
            crate::scene::pose::POSE_REPAIR_CLAMP_LIMIT
        );
        assert_eq!(space.nodes[0].position.z, bad.position.z);
        assert_eq!(space.nodes[0].scale.x, bad.scale.x);
        assert_eq!(space.nodes[0].scale.y, bad.scale.y);
        assert_eq!(
            space.nodes[0].scale.z,
            crate::scene::pose::POSE_REPAIR_CLAMP_LIMIT
        );
        assert_eq!(space.nodes[0].rotation.x, bad.rotation.x);
        assert_eq!(space.nodes[0].rotation.y, bad.rotation.y);
        assert_eq!(space.nodes[0].rotation.z, bad.rotation.z);
        assert_eq!(space.nodes[0].rotation.w, existing.rotation.w);
        assert!(changed.any());
    }

    /// A pose row that exactly matches the existing scene pose is a no-op: the dirty mask must
    /// stay clean so downstream world-matrix recompute and prepared-renderables refresh do not
    /// fire on bones that did not actually move this tick.
    #[test]
    fn pose_matching_existing_scene_pose_leaves_dirty_mask_clean() {
        let pose = pose_at(7.5);
        let mut space = RenderSpaceState::default();
        space.nodes.push(pose);
        let mut changed = NodeDirtyMask::new(space.nodes.len());

        apply_transform_pose_updates_extracted(
            &mut space,
            &[TransformPoseUpdate {
                transform_id: 0,
                pose,
            }],
            11,
            3,
            &mut changed,
        );

        assert_eq!(space.nodes[0].position, pose.position);
        assert_eq!(space.nodes[0].scale, pose.scale);
        assert_eq!(space.nodes[0].rotation, pose.rotation);
        assert!(!changed.any(), "matching pose must not mark the node dirty");
    }

    /// A genuine pose change must still mark the node dirty.
    #[test]
    fn pose_with_distinct_position_marks_node_dirty() {
        let mut space = RenderSpaceState::default();
        space.nodes.push(pose_at(1.0));
        let mut changed = NodeDirtyMask::new(space.nodes.len());

        apply_transform_pose_updates_extracted(
            &mut space,
            &[TransformPoseUpdate {
                transform_id: 0,
                pose: pose_at(2.0),
            }],
            13,
            5,
            &mut changed,
        );

        assert_eq!(space.nodes[0].position.x, 2.0);
        assert!(changed.any());
    }
}
