use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use super::planner::{
    CPU_FROXEL_PREFIX_CHUNK_SIZE, CPU_FROXEL_PREFIX_CHUNKS_PER_TASK,
    should_parallelize_cpu_froxel_prefix,
};
use super::types::CpuFroxelPrefixChunk;

/// Converts per-froxel counts into compact `[offset, count]` rows.
pub(super) fn prefix_counts_to_ranges(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
    if should_parallelize_cpu_froxel_prefix(counts.len()) {
        return prefix_counts_to_ranges_parallel(counts);
    }
    prefix_counts_to_ranges_serial(counts)
}

/// Serial prefix-sum implementation for small froxel-count arrays.
pub(super) fn prefix_counts_to_ranges_serial(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut offset = 0u64;
    for &count in counts {
        let range_offset = u32::try_from(offset).ok()?;
        ranges.push([range_offset, count]);
        offset = offset.checked_add(u64::from(count))?;
        if offset > u64::from(u32::MAX) {
            return None;
        }
    }
    let total_indices = usize::try_from(offset).ok()?;
    Some((ranges, total_indices))
}

/// Parallel prefix-sum implementation for large froxel-count arrays.
pub(super) fn prefix_counts_to_ranges_parallel(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
    let mut chunks = counts
        .par_chunks(CPU_FROXEL_PREFIX_CHUNK_SIZE)
        .with_min_len(CPU_FROXEL_PREFIX_CHUNKS_PER_TASK)
        .map(|counts| {
            let mut ranges = Vec::with_capacity(counts.len());
            let mut offset = 0u64;
            for &count in counts {
                let range_offset = u32::try_from(offset).ok()?;
                ranges.push([range_offset, count]);
                offset = offset.checked_add(u64::from(count))?;
            }
            Some(CpuFroxelPrefixChunk {
                ranges,
                total_count: offset,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut base_offset = 0u64;
    for chunk in &mut chunks {
        for range in &mut chunk.ranges {
            range[0] = u32::try_from(base_offset.checked_add(u64::from(range[0]))?).ok()?;
        }
        base_offset = base_offset.checked_add(chunk.total_count)?;
        if base_offset > u64::from(u32::MAX) {
            return None;
        }
    }

    let total_indices = usize::try_from(base_offset).ok()?;
    let mut ranges = Vec::with_capacity(counts.len());
    for chunk in chunks {
        ranges.extend(chunk.ranges);
    }
    Some((ranges, total_indices))
}

/// Appends one light index to one froxel's compact index range.
pub(super) fn write_membership(
    cluster_id: usize,
    light_idx: u32,
    ranges: &[[u32; 2]],
    cursors: &mut [u32],
    indices: &mut [u32],
) {
    let Some(range) = ranges.get(cluster_id) else {
        return;
    };
    let Some(cursor) = cursors.get_mut(cluster_id) else {
        return;
    };
    if *cursor >= range[1] {
        return;
    }
    let index_offset = u64::from(range[0]).checked_add(u64::from(*cursor));
    let Some(index) = index_offset.and_then(|offset| usize::try_from(offset).ok()) else {
        return;
    };
    let Some(dst) = indices.get_mut(index) else {
        return;
    };
    *dst = light_idx;
    *cursor += 1;
}

/// Atomically writes one light index at a chunk-local base offset.
pub(super) fn write_membership_atomic(
    cluster_id: usize,
    light_idx: u32,
    base: u32,
    cursors: &mut [u32],
    indices: &[AtomicU32],
) {
    let Some(cursor) = cursors.get_mut(cluster_id) else {
        return;
    };
    let index_offset = u64::from(base).checked_add(u64::from(*cursor));
    let Some(index) = index_offset.and_then(|offset| usize::try_from(offset).ok()) else {
        return;
    };
    let Some(dst) = indices.get(index) else {
        return;
    };
    dst.store(light_idx, Ordering::Relaxed);
    *cursor = cursor.saturating_add(1);
}
