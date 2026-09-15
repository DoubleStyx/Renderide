//! Parallel-admission policy for retained render-world maintenance.

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission};

/// Transform-root dirty records assigned to one expansion worker.
const DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS: usize = 1;
/// Retained node-index entries required before one root expansion scans in parallel.
const DIRTY_ROOT_NODE_SCAN_PARALLEL_MIN_NODES: usize = 128;
/// Retained node-index entries assigned to one root-expansion scan task.
const DIRTY_ROOT_NODE_SCAN_PARALLEL_CHUNK_NODES: usize = 64;
/// Render spaces assigned to one mesh-asset dirty expansion worker.
const MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES: usize = 1;
/// Dirty render spaces assigned to one retained-cache refresh worker.
const DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES: usize = 1;
/// Prepared-snapshot copy tasks assigned to one rebuild worker.
const SNAPSHOT_REBUILD_PARALLEL_CHUNK_TASKS: usize = 1;
/// Estimated dirty renderer/template work required before retained cache refresh uses Rayon.
pub(super) const DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS: usize = 64;
/// Retained draw templates targeted for one prepared-snapshot rebuild task.
pub(super) const SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES: usize = 256;
/// Retained draw-template count required before snapshot rebuild fan-out is considered.
pub(super) const SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS: usize =
    SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES * 2;

/// Returns the admission decision for transform-root dirty expansion.
pub(super) fn transform_root_expansion_admission(
    policy: FrameParallelPolicy,
    root_count: usize,
) -> ParallelAdmission {
    policy.admit_independent_items(
        FrameCpuWorkload::independent_items(root_count),
        DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS,
    )
}

/// Returns the admission decision for a large retained node-index scan.
pub(super) fn transform_root_node_scan_admission(
    policy: FrameParallelPolicy,
    node_count: usize,
) -> ParallelAdmission {
    if node_count < DIRTY_ROOT_NODE_SCAN_PARALLEL_MIN_NODES {
        return ParallelAdmission::Serial;
    }
    policy.admit_independent_items(
        FrameCpuWorkload::independent_items(node_count),
        DIRTY_ROOT_NODE_SCAN_PARALLEL_CHUNK_NODES,
    )
}

/// Returns the admission decision for mesh-asset dirty expansion.
pub(super) fn mesh_asset_expansion_admission(
    policy: FrameParallelPolicy,
    space_count: usize,
) -> ParallelAdmission {
    policy.admit_independent_items(
        FrameCpuWorkload::independent_items(space_count),
        MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES,
    )
}

/// Returns the admission decision for dirty retained-cache refresh.
pub(super) fn dirty_refresh_admission(
    policy: FrameParallelPolicy,
    space_count: usize,
    estimated_work_units: usize,
) -> ParallelAdmission {
    if estimated_work_units < DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS {
        return ParallelAdmission::Serial;
    }
    policy.admit_independent_items(
        FrameCpuWorkload::new(0, estimated_work_units, space_count),
        DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES,
    )
}

/// Returns the admission decision for retained prepared-snapshot rebuild.
pub(super) fn snapshot_rebuild_admission(
    policy: FrameParallelPolicy,
    task_count: usize,
    retained_draw_count: usize,
) -> ParallelAdmission {
    if retained_draw_count < SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS {
        return ParallelAdmission::Serial;
    }
    policy.admit_independent_items(
        FrameCpuWorkload::new(0, retained_draw_count, task_count),
        SNAPSHOT_REBUILD_PARALLEL_CHUNK_TASKS,
    )
}
