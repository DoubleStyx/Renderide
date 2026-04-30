//! Pose row validation, commit, and post-commit dirty-flag propagation.
//!
//! Validation runs in two phases so per-row [`PoseValidation::is_valid`] checks can fan out
//! across rayon workers above [`POSE_UPDATE_PARALLEL_MIN_ROWS`]: the parallel pass produces a
//! [`ValidatedPoseRow`] vector in input order, and the serial commit pass writes each row
//! into [`RenderSpaceState::nodes`] under the lock that already protects the per-space apply.

use crate::scene::pose::{PoseValidation, render_transform_identity};
use crate::scene::render_space::RenderSpaceState;
use crate::scene::world::WorldTransformCache;
use crate::shared::{RenderTransform, TransformPoseUpdate};

use super::NodeDirtyMask;

/// Minimum pose-update count before [`validate_pose_rows`] fans out validation across rayon
/// workers. Below this threshold the scalar loop is faster than rayon dispatch overhead.
const POSE_UPDATE_PARALLEL_MIN_ROWS: usize = 1024;

/// Validated pose row ready for serial commit into [`RenderSpaceState::nodes`].
struct ValidatedPoseRow {
    /// Dense transform index into [`RenderSpaceState::nodes`].
    transform_index: usize,
    /// Pose to commit (already substituted with [`render_transform_identity`] when the host row was rejected).
    pose: RenderTransform,
    /// `true` when the host row failed [`PoseValidation::is_valid`] (caller logs the rejection).
    rejected: bool,
    /// Original [`TransformPoseUpdate::transform_id`] for the rejection log line.
    raw_transform_id: i32,
}

/// Index of the first sentinel `transform_id < 0` row, or `poses.len()` if no terminator is present.
#[inline]
fn pose_terminator_index(poses: &[TransformPoseUpdate]) -> usize {
    poses
        .iter()
        .position(|pu| pu.transform_id < 0)
        .unwrap_or(poses.len())
}

/// Walks the active prefix of `poses` once and produces one [`ValidatedPoseRow`] per in-bounds entry.
fn validate_pose_rows(poses: &[TransformPoseUpdate], node_count: usize) -> Vec<ValidatedPoseRow> {
    profiling::scope!("scene::validate_pose_rows");
    let active_len = pose_terminator_index(poses);
    let active = &poses[..active_len];
    let row_for = |pu: &TransformPoseUpdate| -> Option<ValidatedPoseRow> {
        let idx = pu.transform_id as usize;
        if idx >= node_count {
            return None;
        }
        let valid = PoseValidation { pose: &pu.pose }.is_valid();
        Some(ValidatedPoseRow {
            transform_index: idx,
            pose: if valid {
                pu.pose
            } else {
                render_transform_identity()
            },
            rejected: !valid,
            raw_transform_id: pu.transform_id,
        })
    };

    if active.len() >= POSE_UPDATE_PARALLEL_MIN_ROWS {
        use rayon::prelude::*;
        active.par_iter().filter_map(row_for).collect()
    } else {
        active.iter().filter_map(row_for).collect()
    }
}

/// Applies pose rows from a pre-extracted slice, validating each against [`PoseValidation`].
pub(super) fn apply_transform_pose_updates_extracted(
    space: &mut RenderSpaceState,
    poses: &[TransformPoseUpdate],
    frame_index: i32,
    sid: i32,
    changed: &mut NodeDirtyMask,
) {
    profiling::scope!("scene::apply_pose_updates");
    if poses.is_empty() {
        return;
    }
    let validated = validate_pose_rows(poses, space.nodes.len());
    for row in validated {
        if row.rejected {
            logger::error!(
                "invalid pose scene={sid} transform={} frame={frame_index}: identity",
                row.raw_transform_id
            );
        }
        space.nodes[row.transform_index] = row.pose;
        changed.mark(row.transform_index);
    }
}

/// Marks per-node dirty flags after local transform edits.
pub(super) fn propagate_transform_change_dirty_flags(
    cache: &mut WorldTransformCache,
    changed: &NodeDirtyMask,
) {
    if !changed.any() {
        return;
    }
    let n = changed
        .flags()
        .len()
        .min(cache.computed.len().max(cache.local_dirty.len()));
    for (i, &dirty) in changed.flags()[..n].iter().enumerate() {
        if !dirty {
            continue;
        }
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

    /// [`validate_pose_rows`] preserves input order, drops out-of-range transform indices, and
    /// substitutes [`render_transform_identity`] for invalid poses.
    #[test]
    fn validate_pose_rows_preserves_order_and_substitutes_invalid() {
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
        let out = validate_pose_rows(&rows, 3);
        assert_eq!(
            out.len(),
            2,
            "out-of-range and sentinel rows must be dropped"
        );
        assert_eq!(out[0].transform_index, 0);
        assert!(!out[0].rejected);
        assert_eq!(out[1].transform_index, 1);
        assert!(out[1].rejected);
        let identity = render_transform_identity();
        assert_eq!(out[1].pose.position, identity.position);
        assert_eq!(out[1].pose.scale, identity.scale);
        assert_eq!(out[1].pose.rotation, identity.rotation);
    }

    /// [`validate_pose_rows`] above [`POSE_UPDATE_PARALLEL_MIN_ROWS`] still preserves input order.
    #[test]
    fn validate_pose_rows_parallel_path_preserves_order() {
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
        let out = validate_pose_rows(&rows, n);
        assert_eq!(out.len(), n);
        for (i, row) in out.iter().enumerate() {
            assert_eq!(row.transform_index, i);
            assert!(!row.rejected);
        }
    }
}
