use crate::cpu_parallelism::{
    LIGHT_WORK_CHUNK_LIGHTS, LIGHT_WORK_PARALLEL_MIN_LIGHTS, admit_light_work_items,
    current_reference_worker_count, record_parallel_admission, reference_worker_count,
};
use crate::gpu::GpuLight;
use crate::world_mesh::cluster::ClusterFrameParams;

use super::bounds::{build_eye_froxel_spheres, eye_froxel_spheres};
use super::geometry::assign_eye_lights;
use super::parallel::build_parallel;
use super::prefix::{prefix_counts_to_ranges, write_membership};
use super::types::{CpuClusterAssignments, CpuFroxelStats, FroxelLayout};

/// Light count at which `Auto` mode starts considering CPU froxel assignment.
pub(in crate::passes::clustered_light) const AUTO_CPU_FROXEL_LIGHT_THRESHOLD: u32 = 64;
/// Lights assigned to one CPU froxel worker chunk.
pub(super) const CPU_FROXEL_LIGHT_CHUNK_SIZE: usize = LIGHT_WORK_CHUNK_LIGHTS;
/// Light count at which CPU froxel assignment fans out across worker chunks.
pub(super) const CPU_FROXEL_PARALLEL_MIN_LIGHTS: usize = LIGHT_WORK_PARALLEL_MIN_LIGHTS;
/// CPU froxel light chunks assigned to one worker task.
pub(super) const CPU_FROXEL_PARALLEL_CHUNK_TASKS: usize = 1;
/// Cluster-count stride for local prefix-sum and offset chunks.
pub(super) const CPU_FROXEL_PREFIX_CHUNK_SIZE: usize = 512;
/// Prefix chunks assigned to one Rayon worker leaf.
pub(super) const CPU_FROXEL_PREFIX_CHUNKS_PER_TASK: usize = 1;
/// Froxel count at which count merge, offset, and prefix work uses Rayon.
pub(super) const CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS: usize = CPU_FROXEL_PREFIX_CHUNK_SIZE * 2;

pub(in crate::passes::clustered_light) struct FroxelLightPlanner;

impl FroxelLightPlanner {
    /// Builds dynamic-range cluster assignments for every eye in `eye_params`.
    pub fn build(
        lights: &[GpuLight],
        eye_params: &[ClusterFrameParams],
        clusters_per_eye: u32,
    ) -> Option<CpuClusterAssignments> {
        profiling::scope!("clustered_light::cpu_froxel_build");
        if eye_params.is_empty() {
            return Some(CpuClusterAssignments::default());
        }
        let layouts = validated_eye_layouts(eye_params, clusters_per_eye)?;
        let admission = admit_light_work_items(lights.len(), current_reference_worker_count());
        record_parallel_admission("cpu_froxel_lights", lights.len(), lights.len(), admission);
        if should_parallelize_cpu_froxel_lights(lights.len()) {
            build_parallel(lights, eye_params, &layouts, clusters_per_eye)
        } else {
            build_serial(lights, eye_params, &layouts, clusters_per_eye)
        }
    }
}

/// Returns whether CPU froxel assignment should split light ranges over Rayon.
#[inline]
pub(super) fn should_parallelize_cpu_froxel_lights(light_count: usize) -> bool {
    should_parallelize_cpu_froxel_lights_with_workers(light_count, current_reference_worker_count())
}

/// Returns whether CPU froxel assignment should split light ranges for a known worker count.
#[inline]
pub(super) const fn should_parallelize_cpu_froxel_lights_with_workers(
    light_count: usize,
    worker_count: usize,
) -> bool {
    light_count >= CPU_FROXEL_PARALLEL_MIN_LIGHTS
        && admit_light_work_items(light_count, worker_count).is_parallel()
}

/// Returns whether CPU froxel prefix and merge helpers should use Rayon.
#[inline]
pub(super) fn should_parallelize_cpu_froxel_prefix(cluster_count: usize) -> bool {
    should_parallelize_cpu_froxel_prefix_with_workers(
        cluster_count,
        current_reference_worker_count(),
    )
}

/// Returns whether CPU froxel prefix and merge helpers should use Rayon for a known worker count.
#[inline]
pub(super) const fn should_parallelize_cpu_froxel_prefix_with_workers(
    cluster_count: usize,
    worker_count: usize,
) -> bool {
    reference_worker_count(worker_count) > 1
        && cluster_count >= CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS
}

/// Returns whether CPU froxel chunk-offset building should use Rayon.
#[inline]
pub(super) fn should_parallelize_cpu_froxel_offsets(
    cluster_count: usize,
    chunk_count: usize,
) -> bool {
    should_parallelize_cpu_froxel_offsets_with_workers(
        cluster_count,
        chunk_count,
        current_reference_worker_count(),
    )
}

/// Returns whether CPU froxel chunk-offset building should use Rayon for a known worker count.
#[inline]
pub(super) const fn should_parallelize_cpu_froxel_offsets_with_workers(
    cluster_count: usize,
    chunk_count: usize,
    worker_count: usize,
) -> bool {
    chunk_count >= 2
        && should_parallelize_cpu_froxel_prefix_with_workers(cluster_count, worker_count)
}

pub(super) fn validated_eye_layouts(
    eye_params: &[ClusterFrameParams],
    clusters_per_eye: u32,
) -> Option<Vec<FroxelLayout>> {
    let expected = usize::try_from(clusters_per_eye).ok()?;
    let mut layouts = Vec::with_capacity(eye_params.len());
    for params in eye_params {
        let layout = FroxelLayout::from_cluster_params(params);
        if layout.cluster_count()? != expected {
            return None;
        }
        layouts.push(layout);
    }
    Some(layouts)
}

pub(super) fn total_cluster_count(clusters_per_eye: u32, eye_count: usize) -> Option<usize> {
    usize::try_from(clusters_per_eye)
        .ok()?
        .checked_mul(eye_count)
}

pub(super) fn build_serial(
    lights: &[GpuLight],
    eye_params: &[ClusterFrameParams],
    layouts: &[FroxelLayout],
    clusters_per_eye: u32,
) -> Option<CpuClusterAssignments> {
    let expected_clusters = usize::try_from(clusters_per_eye).ok()?;
    let total_clusters = total_cluster_count(clusters_per_eye, eye_params.len())?;
    let froxel_spheres_by_eye = build_eye_froxel_spheres(lights, eye_params, layouts)?;
    let mut counts = vec![0u32; total_clusters];
    let mut stats = CpuFroxelStats::default();

    for (eye_idx, (params, &layout)) in eye_params.iter().zip(layouts.iter()).enumerate() {
        let froxel_spheres = eye_froxel_spheres(&froxel_spheres_by_eye, eye_idx);
        let cluster_base = eye_idx.checked_mul(expected_clusters)?;
        let mut emit_count = |cluster_id: usize, _light_idx: u32| {
            let Some(count) = counts.get_mut(cluster_id) else {
                return;
            };
            *count = count.saturating_add(1);
            stats.assigned_memberships = stats.assigned_memberships.saturating_add(1);
        };
        stats.culled_lights = stats.culled_lights.saturating_add(assign_eye_lights(
            lights,
            *params,
            layout,
            froxel_spheres,
            cluster_base,
            &mut emit_count,
        ));
    }

    let (ranges, total_indices) = prefix_counts_to_ranges(&counts)?;
    let mut indices = vec![0u32; total_indices];
    let mut cursors = vec![0u32; total_clusters];

    for (eye_idx, (params, &layout)) in eye_params.iter().zip(layouts.iter()).enumerate() {
        let froxel_spheres = eye_froxel_spheres(&froxel_spheres_by_eye, eye_idx);
        let cluster_base = eye_idx.checked_mul(expected_clusters)?;
        let mut emit_index = |cluster_id: usize, light_idx: u32| {
            write_membership(cluster_id, light_idx, &ranges, &mut cursors, &mut indices);
        };
        assign_eye_lights(
            lights,
            *params,
            layout,
            froxel_spheres,
            cluster_base,
            &mut emit_index,
        );
    }

    Some(CpuClusterAssignments {
        ranges,
        indices,
        stats,
    })
}
