use std::sync::atomic::AtomicU32;

use rayon::prelude::*;

use crate::gpu::GpuLight;
use crate::world_mesh::cluster::ClusterFrameParams;

use super::bounds::{build_eye_froxel_spheres, eye_froxel_spheres};
use super::geometry::assign_eye_lights_slice;
use super::planner::{
    CPU_FROXEL_LIGHT_CHUNK_SIZE, CPU_FROXEL_PARALLEL_CHUNK_TASKS, CPU_FROXEL_PREFIX_CHUNK_SIZE,
    should_parallelize_cpu_froxel_offsets, should_parallelize_cpu_froxel_prefix,
    total_cluster_count,
};
use super::prefix::{prefix_counts_to_ranges, write_membership_atomic};
use super::types::{
    CpuClusterAssignments, CpuFroxelChunkCounts, CpuFroxelParallelInputs, CpuFroxelStats,
    FroxelLayout,
};

pub(super) fn build_parallel(
    lights: &[GpuLight],
    eye_params: &[ClusterFrameParams],
    layouts: &[FroxelLayout],
    clusters_per_eye: u32,
) -> Option<CpuClusterAssignments> {
    profiling::scope!("clustered_light::cpu_froxel_parallel");
    let expected_clusters = usize::try_from(clusters_per_eye).ok()?;
    let total_clusters = total_cluster_count(clusters_per_eye, eye_params.len())?;
    let froxel_spheres_by_eye = build_eye_froxel_spheres(lights, eye_params, layouts)?;
    let inputs = CpuFroxelParallelInputs {
        lights,
        eye_params,
        layouts,
        froxel_spheres_by_eye: &froxel_spheres_by_eye,
        expected_clusters,
        total_clusters,
    };
    let chunks = count_parallel_light_chunks(&inputs)?;
    let (counts, stats) = merge_parallel_chunk_counts(&chunks);
    let (ranges, total_indices) = prefix_counts_to_ranges(&counts)?;
    let chunk_offsets = build_chunk_offsets(&chunks, &ranges);
    let indices = write_parallel_light_chunks(&inputs, chunks, &chunk_offsets, total_indices);

    Some(CpuClusterAssignments {
        ranges,
        indices,
        stats,
    })
}

fn count_parallel_light_chunks(
    inputs: &CpuFroxelParallelInputs<'_>,
) -> Option<CpuFroxelChunkCounts> {
    let chunk_count = inputs.lights.len().div_ceil(CPU_FROXEL_LIGHT_CHUNK_SIZE);
    let count_len = chunk_count.checked_mul(inputs.total_clusters)?;
    let mut chunks = CpuFroxelChunkCounts {
        counts: vec![0u32; count_len],
        stats: vec![CpuFroxelStats::default(); chunk_count],
        chunk_count,
        total_clusters: inputs.total_clusters,
    };
    if chunk_count == 0 || inputs.total_clusters == 0 {
        return Some(chunks);
    }

    let count_rows = chunks
        .counts
        .par_chunks_mut(inputs.total_clusters)
        .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS);
    let stats_rows = chunks
        .stats
        .par_iter_mut()
        .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS);
    count_rows
        .zip(stats_rows)
        .enumerate()
        .for_each(|(chunk_idx, (counts, stats))| {
            profiling::scope!("clustered_light::cpu_froxel_count_worker");
            let (light_start, light_end) = light_chunk_bounds(inputs.lights.len(), chunk_idx);
            let light_slice = &inputs.lights[light_start..light_end];
            for (eye_idx, (params, &layout)) in inputs
                .eye_params
                .iter()
                .zip(inputs.layouts.iter())
                .enumerate()
            {
                let froxel_spheres = eye_froxel_spheres(inputs.froxel_spheres_by_eye, eye_idx);
                let cluster_base = eye_idx * inputs.expected_clusters;
                let mut emit_count = |cluster_id: usize, _light_idx: u32| {
                    let Some(count) = counts.get_mut(cluster_id) else {
                        return;
                    };
                    *count = count.saturating_add(1);
                    stats.assigned_memberships = stats.assigned_memberships.saturating_add(1);
                };
                stats.culled_lights = stats.culled_lights.saturating_add(assign_eye_lights_slice(
                    light_slice,
                    light_start,
                    *params,
                    layout,
                    froxel_spheres,
                    cluster_base,
                    &mut emit_count,
                ));
            }
        });
    Some(chunks)
}

fn merge_parallel_chunk_counts(chunks: &CpuFroxelChunkCounts) -> (Vec<u32>, CpuFroxelStats) {
    profiling::scope!("clustered_light::merge_parallel_chunk_counts");
    let counts = if should_parallelize_cpu_froxel_prefix(chunks.total_clusters) {
        (0..chunks.total_clusters)
            .into_par_iter()
            .with_min_len(CPU_FROXEL_PREFIX_CHUNK_SIZE)
            .map(|cluster_id| merged_cluster_count(chunks, cluster_id))
            .collect()
    } else {
        merge_chunk_counts_serial(chunks)
    };
    let stats = merge_chunk_stats(chunks);
    (counts, stats)
}

fn merge_chunk_counts_serial(chunks: &CpuFroxelChunkCounts) -> Vec<u32> {
    if chunks.total_clusters == 0 {
        return Vec::new();
    }
    let mut counts = vec![0u32; chunks.total_clusters];
    for chunk_counts in chunks.counts.chunks(chunks.total_clusters) {
        for (total, &count) in counts.iter_mut().zip(chunk_counts.iter()) {
            *total = total.saturating_add(count);
        }
    }
    counts
}

fn merged_cluster_count(chunks: &CpuFroxelChunkCounts, cluster_id: usize) -> u32 {
    let mut total = 0u32;
    for chunk_idx in 0..chunks.chunk_count {
        total = total.saturating_add(chunk_cluster_count(chunks, chunk_idx, cluster_id));
    }
    total
}

fn merge_chunk_stats(chunks: &CpuFroxelChunkCounts) -> CpuFroxelStats {
    if chunks.chunk_count >= CPU_FROXEL_PARALLEL_CHUNK_TASKS * 2 && rayon::current_num_threads() > 1
    {
        return chunks
            .stats
            .par_iter()
            .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS)
            .copied()
            .reduce(CpuFroxelStats::default, merge_froxel_stats);
    }

    let mut stats = CpuFroxelStats::default();
    for &chunk_stats in &chunks.stats {
        stats = merge_froxel_stats(stats, chunk_stats);
    }
    stats
}

struct CpuFroxelChunkOffsets {
    offsets: Vec<u32>,
    chunk_count: usize,
    total_clusters: usize,
}

impl CpuFroxelChunkOffsets {
    fn offset(&self, chunk_idx: usize, cluster_id: usize) -> Option<u32> {
        if chunk_idx >= self.chunk_count || cluster_id >= self.total_clusters {
            return None;
        }
        self.offsets
            .get(cluster_id * self.chunk_count + chunk_idx)
            .copied()
    }
}

fn build_chunk_offsets(
    chunks: &CpuFroxelChunkCounts,
    ranges: &[[u32; 2]],
) -> CpuFroxelChunkOffsets {
    profiling::scope!("clustered_light::build_chunk_offsets");
    let mut offsets = vec![0u32; chunks.counts.len()];
    if chunks.chunk_count == 0 || chunks.total_clusters == 0 {
        return CpuFroxelChunkOffsets {
            offsets,
            chunk_count: chunks.chunk_count,
            total_clusters: chunks.total_clusters,
        };
    }

    if should_parallelize_cpu_froxel_offsets(chunks.total_clusters, chunks.chunk_count) {
        offsets
            .par_chunks_mut(chunks.chunk_count)
            .with_min_len(CPU_FROXEL_PREFIX_CHUNK_SIZE)
            .enumerate()
            .for_each(|(cluster_id, cluster_offsets)| {
                fill_cluster_offsets(chunks, ranges, cluster_id, cluster_offsets);
            });
    } else {
        for (cluster_id, cluster_offsets) in offsets.chunks_mut(chunks.chunk_count).enumerate() {
            fill_cluster_offsets(chunks, ranges, cluster_id, cluster_offsets);
        }
    }

    CpuFroxelChunkOffsets {
        offsets,
        chunk_count: chunks.chunk_count,
        total_clusters: chunks.total_clusters,
    }
}

fn fill_cluster_offsets(
    chunks: &CpuFroxelChunkCounts,
    ranges: &[[u32; 2]],
    cluster_id: usize,
    cluster_offsets: &mut [u32],
) {
    let mut next = ranges[cluster_id][0];
    for (chunk_idx, offset) in cluster_offsets.iter_mut().enumerate() {
        *offset = next;
        next = next.saturating_add(chunk_cluster_count(chunks, chunk_idx, cluster_id));
    }
}

#[inline]
fn chunk_cluster_count(chunks: &CpuFroxelChunkCounts, chunk_idx: usize, cluster_id: usize) -> u32 {
    chunks.counts[chunk_idx * chunks.total_clusters + cluster_id]
}

/// Combines two CPU froxel diagnostic records with saturating counters.
fn merge_froxel_stats(left: CpuFroxelStats, right: CpuFroxelStats) -> CpuFroxelStats {
    CpuFroxelStats {
        assigned_memberships: left
            .assigned_memberships
            .saturating_add(right.assigned_memberships),
        overflowed_memberships: left
            .overflowed_memberships
            .saturating_add(right.overflowed_memberships),
        culled_lights: left.culled_lights.saturating_add(right.culled_lights),
    }
}

fn write_parallel_light_chunks(
    inputs: &CpuFroxelParallelInputs<'_>,
    chunk_counts: CpuFroxelChunkCounts,
    chunk_offsets: &CpuFroxelChunkOffsets,
    total_indices: usize,
) -> Vec<u32> {
    let indices_atomic = (0..total_indices)
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>();
    let mut cursors = chunk_counts.counts;
    if inputs.total_clusters == 0 {
        return indices_atomic
            .into_iter()
            .map(AtomicU32::into_inner)
            .collect();
    }

    cursors
        .par_chunks_mut(inputs.total_clusters)
        .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS)
        .enumerate()
        .for_each(|(chunk_idx, cursors)| {
            profiling::scope!("clustered_light::cpu_froxel_write_worker");
            cursors.fill(0);
            let (light_start, light_end) = light_chunk_bounds(inputs.lights.len(), chunk_idx);
            let light_slice = &inputs.lights[light_start..light_end];
            for (eye_idx, (params, &layout)) in inputs
                .eye_params
                .iter()
                .zip(inputs.layouts.iter())
                .enumerate()
            {
                let froxel_spheres = eye_froxel_spheres(inputs.froxel_spheres_by_eye, eye_idx);
                let cluster_base = eye_idx * inputs.expected_clusters;
                let mut emit_index = |cluster_id: usize, light_idx: u32| {
                    let Some(base) = chunk_offsets.offset(chunk_idx, cluster_id) else {
                        return;
                    };
                    write_membership_atomic(cluster_id, light_idx, base, cursors, &indices_atomic);
                };
                assign_eye_lights_slice(
                    light_slice,
                    light_start,
                    *params,
                    layout,
                    froxel_spheres,
                    cluster_base,
                    &mut emit_index,
                );
            }
        });
    indices_atomic
        .into_iter()
        .map(AtomicU32::into_inner)
        .collect()
}

fn light_chunk_bounds(lights_len: usize, chunk_idx: usize) -> (usize, usize) {
    let start = chunk_idx * CPU_FROXEL_LIGHT_CHUNK_SIZE;
    let end = lights_len.min(start + CPU_FROXEL_LIGHT_CHUNK_SIZE);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::build_chunk_offsets;
    use crate::passes::clustered_light::froxel_cpu::types::{CpuFroxelChunkCounts, CpuFroxelStats};

    fn test_counts() -> CpuFroxelChunkCounts {
        CpuFroxelChunkCounts {
            counts: vec![1, 0, 3, 2, 4, 1],
            stats: vec![CpuFroxelStats::default(); 2],
            chunk_count: 2,
            total_clusters: 3,
        }
    }

    #[test]
    fn flat_chunk_offsets_match_cluster_prefix_order() {
        let ranges = [[0, 3], [3, 4], [7, 4]];
        let offsets = build_chunk_offsets(&test_counts(), &ranges);

        assert_eq!(offsets.offset(0, 0), Some(0));
        assert_eq!(offsets.offset(1, 0), Some(1));
        assert_eq!(offsets.offset(0, 1), Some(3));
        assert_eq!(offsets.offset(1, 1), Some(3));
        assert_eq!(offsets.offset(0, 2), Some(7));
        assert_eq!(offsets.offset(1, 2), Some(10));
    }
}
