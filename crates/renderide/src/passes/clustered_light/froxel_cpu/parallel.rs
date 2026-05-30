use std::num::Saturating;
use std::sync::atomic::AtomicU32;

use rayon::prelude::*;

use crate::gpu::GpuLight;
use crate::world_mesh::cluster::ClusterFrameParams;

use super::bounds::{build_eye_froxel_spheres, eye_froxel_spheres};
use super::geometry::assign_eye_lights_slice;
use super::planner::{
    CPU_FROXEL_LIGHT_CHUNK_SIZE, CPU_FROXEL_PARALLEL_CHUNK_TASKS, CPU_FROXEL_PREFIX_CHUNK_SIZE,
    should_parallelize_cpu_froxel_prefix, total_cluster_count,
};
use super::prefix::{prefix_counts_to_ranges, write_membership_atomic};
use super::types::{
    CpuClusterAssignments, CpuFroxelCountChunk, CpuFroxelParallelInputs, CpuFroxelStats,
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
    let chunks = count_parallel_light_chunks(&inputs);
    let (counts, stats) = merge_parallel_chunk_counts(&chunks, total_clusters);
    let (ranges, total_indices) = prefix_counts_to_ranges(&counts)?;
    let chunk_offsets = build_chunk_offsets(&chunks, &ranges, total_clusters);
    let indices = write_parallel_light_chunks(&inputs, &chunk_offsets, total_indices);

    Some(CpuClusterAssignments {
        ranges,
        indices,
        stats,
    })
}

fn count_parallel_light_chunks(inputs: &CpuFroxelParallelInputs<'_>) -> Vec<CpuFroxelCountChunk> {
    let chunk_count = inputs.lights.len().div_ceil(CPU_FROXEL_LIGHT_CHUNK_SIZE);
    let mut chunks = (0..chunk_count)
        .map(|_| CpuFroxelCountChunk {
            counts: vec![0u32; inputs.total_clusters],
            stats: CpuFroxelStats::default(),
        })
        .collect::<Vec<_>>();

    chunks
        .par_iter_mut()
        .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
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
                    let Some(count) = chunk.counts.get_mut(cluster_id) else {
                        return;
                    };
                    *count = count.saturating_add(1);
                    chunk.stats.assigned_memberships =
                        chunk.stats.assigned_memberships.saturating_add(1);
                };
                chunk.stats.culled_lights =
                    chunk
                        .stats
                        .culled_lights
                        .saturating_add(assign_eye_lights_slice(
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
    chunks
}

fn merge_parallel_chunk_counts(
    chunks: &[CpuFroxelCountChunk],
    total_clusters: usize,
) -> (Vec<u32>, CpuFroxelStats) {
    profiling::scope!("clustered_light::merge_parallel_chunk_counts");
    let counts = if should_parallelize_cpu_froxel_prefix(total_clusters) {
        (0..total_clusters)
            .into_par_iter()
            .with_min_len(CPU_FROXEL_PREFIX_CHUNK_SIZE)
            .map(|cluster_id| {
                chunks
                    .iter()
                    .map(|chunk| Saturating(chunk.counts[cluster_id]))
                    .sum::<Saturating<u32>>()
                    .0
            })
            .collect()
    } else {
        let mut counts = vec![0u32; total_clusters];
        for chunk in chunks {
            for (total, &count) in counts.iter_mut().zip(chunk.counts.iter()) {
                *total = total.saturating_add(count);
            }
        }
        counts
    };
    let stats = chunks
        .par_iter()
        .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS)
        .map(|chunk| chunk.stats)
        .reduce(CpuFroxelStats::default, merge_froxel_stats);
    (counts, stats)
}

fn build_chunk_offsets(
    chunks: &[CpuFroxelCountChunk],
    ranges: &[[u32; 2]],
    total_clusters: usize,
) -> Vec<Vec<u32>> {
    profiling::scope!("clustered_light::build_chunk_offsets");
    let chunk_count = chunks.len();
    // The amount of memory allocations/deallocations necessary
    // to properly parallelize computations and then transpose the Vec<Vec<_>>
    // is not worth the amount of time spent iterating here
    let mut chunk_offsets = (0..chunk_count)
        .map(|_| vec![0u32; total_clusters])
        .collect::<Vec<_>>();
    for cluster_id in 0..total_clusters {
        chunks
            .iter()
            .map(|chunk| chunk.counts[cluster_id])
            .enumerate()
            .fold(ranges[cluster_id][0], |offset, (chunk_idx, chunk)| {
                chunk_offsets[chunk_idx][cluster_id] = offset;
                offset.saturating_add(chunk)
            });
    }
    chunk_offsets
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
    chunk_offsets: &[Vec<u32>],
    total_indices: usize,
) -> Vec<u32> {
    let indices_atomic = (0..total_indices)
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>();
    chunk_offsets
        .par_iter()
        .with_min_len(CPU_FROXEL_PARALLEL_CHUNK_TASKS)
        .enumerate()
        .for_each(|(chunk_idx, offsets)| {
            profiling::scope!("clustered_light::cpu_froxel_write_worker");
            let (light_start, light_end) = light_chunk_bounds(inputs.lights.len(), chunk_idx);
            let light_slice = &inputs.lights[light_start..light_end];
            let mut cursors = vec![0u32; inputs.total_clusters];
            for (eye_idx, (params, &layout)) in inputs
                .eye_params
                .iter()
                .zip(inputs.layouts.iter())
                .enumerate()
            {
                let froxel_spheres = eye_froxel_spheres(inputs.froxel_spheres_by_eye, eye_idx);
                let cluster_base = eye_idx * inputs.expected_clusters;
                let mut emit_index = |cluster_id: usize, light_idx: u32| {
                    write_membership_atomic(
                        cluster_id,
                        light_idx,
                        offsets,
                        &mut cursors,
                        &indices_atomic,
                    );
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
