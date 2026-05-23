//! CPU light-centric froxel assignment for clustered forward lighting.
//!
//! This path mirrors the clustered-light storage contract (`cluster_light_counts` range rows plus
//! compact `cluster_light_indices`) so dense-light frames can use a light-centric alternative to
//! the O(froxels x lights) GPU scan.

use std::sync::atomic::{AtomicU32, Ordering};

use glam::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::gpu::GpuLight;
use crate::world_mesh::cluster::{
    CLUSTER_COUNT_Z, ClusterFrameParams, TILE_SIZE, sanitize_cluster_clip_planes,
};

/// Light count at which `Auto` mode starts considering CPU froxel assignment.
pub(super) const AUTO_CPU_FROXEL_LIGHT_THRESHOLD: u32 = 64;
/// Lights assigned to one CPU froxel worker chunk.
const CPU_FROXEL_LIGHT_CHUNK_SIZE: usize = 32;
/// Light count at which CPU froxel assignment fans out across worker chunks.
const CPU_FROXEL_PARALLEL_MIN_LIGHTS: usize = CPU_FROXEL_LIGHT_CHUNK_SIZE * 2;
/// CPU froxel light chunks assigned to one worker task.
const CPU_FROXEL_PARALLEL_CHUNK_TASKS: usize = 1;
/// Cluster-count stride for local prefix-sum chunks.
const CPU_FROXEL_PREFIX_CHUNK_SIZE: usize = 256;
/// Prefix chunks assigned to one Rayon worker leaf.
const CPU_FROXEL_PREFIX_CHUNKS_PER_TASK: usize = 1;
/// Froxel count at which count merge, offset, and prefix work uses Rayon.
const CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS: usize = CPU_FROXEL_PREFIX_CHUNK_SIZE * 2;

/// Point light tag in [`GpuLight::light_type`].
const LIGHT_TYPE_POINT: u32 = 0;
/// Directional light tag in [`GpuLight::light_type`].
const LIGHT_TYPE_DIRECTIONAL: u32 = 1;
/// Spot light tag in [`GpuLight::light_type`].
const LIGHT_TYPE_SPOT: u32 = 2;
/// Cluster AABB padding used by the clustered-light compute shader.
const CLUSTER_BOUNDARY_EPSILON: f32 = 0.00001;
/// Largest half-angle cosine used by spotlight culling, equivalent to a 0.5 degree half-angle.
const SPOT_CULL_MIN_COS_HALF: f32 = 0.999_961_9;
/// Half-angle cosine below which spotlights use range-sphere culling to avoid wide-cone misses.
const SPOT_CULL_WIDE_COS_HALF: f32 = 0.5;
/// Small distance pad for cone/sphere boundary comparisons.
const SPOT_CULL_DISTANCE_EPSILON: f32 = 0.00001;

/// Cluster-grid layout for one eye.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FroxelLayout {
    /// Cluster count in screen X.
    pub cluster_count_x: u32,
    /// Cluster count in screen Y.
    pub cluster_count_y: u32,
    /// Cluster count in depth.
    pub cluster_count_z: u32,
    /// Viewport width in physical pixels.
    pub viewport_width: u32,
    /// Viewport height in physical pixels.
    pub viewport_height: u32,
}

impl FroxelLayout {
    /// Builds a layout from the frame's clustered camera params.
    fn from_cluster_params(params: &ClusterFrameParams) -> Self {
        Self {
            cluster_count_x: params.cluster_count_x.max(1),
            cluster_count_y: params.cluster_count_y.max(1),
            cluster_count_z: CLUSTER_COUNT_Z.max(1),
            viewport_width: params.viewport_width.max(1),
            viewport_height: params.viewport_height.max(1),
        }
    }

    /// Number of froxels in this eye.
    fn cluster_count(self) -> Option<usize> {
        let xy = self.cluster_count_x.checked_mul(self.cluster_count_y)?;
        xy.checked_mul(self.cluster_count_z).map(|v| v as usize)
    }
}

/// Per-frame CPU-produced cluster storage matching the existing WGSL buffers.
#[derive(Clone, Debug, Default)]
pub(super) struct CpuClusterAssignments {
    /// Per-froxel `[offset, count]` rows addressing [`Self::indices`].
    pub ranges: Vec<[u32; 2]>,
    /// Compact light indices for every froxel membership.
    pub indices: Vec<u32>,
    /// Assignment diagnostics for profiling and tests.
    pub stats: CpuFroxelStats,
}

/// CPU froxel assignment diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuFroxelStats {
    /// Number of light/froxel memberships emitted into compact storage.
    pub assigned_memberships: u64,
    /// Number of light/froxel memberships dropped because compact storage could not represent them.
    pub overflowed_memberships: u64,
    /// Number of lights rejected before assignment because their conservative bounds miss the view.
    pub culled_lights: u32,
}

struct CpuFroxelCountChunk {
    counts: Vec<u32>,
    stats: CpuFroxelStats,
}

/// Local prefix-sum result for one cluster-count chunk.
struct CpuFroxelPrefixChunk {
    /// Range rows with offsets relative to the start of this chunk.
    ranges: Vec<[u32; 2]>,
    /// Sum of every count in this chunk.
    total_count: u64,
}

struct CpuFroxelParallelInputs<'a> {
    lights: &'a [GpuLight],
    eye_params: &'a [ClusterFrameParams],
    layouts: &'a [FroxelLayout],
    froxel_spheres_by_eye: &'a [Vec<FroxelSphere>],
    expected_clusters: usize,
    total_clusters: usize,
}

/// Stateless CPU froxel assignment entry point.
pub(super) struct FroxelLightPlanner;

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
        if should_parallelize_cpu_froxel_lights(lights.len()) {
            build_parallel(lights, eye_params, &layouts, clusters_per_eye)
        } else {
            build_serial(lights, eye_params, &layouts, clusters_per_eye)
        }
    }
}

/// Returns whether CPU froxel assignment should split light ranges over Rayon.
#[inline]
fn should_parallelize_cpu_froxel_lights(light_count: usize) -> bool {
    light_count >= CPU_FROXEL_PARALLEL_MIN_LIGHTS
}

/// Returns whether CPU froxel prefix and merge helpers should use Rayon.
#[inline]
fn should_parallelize_cpu_froxel_prefix(cluster_count: usize) -> bool {
    cluster_count >= CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS
}

fn validated_eye_layouts(
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

fn total_cluster_count(clusters_per_eye: u32, eye_count: usize) -> Option<usize> {
    usize::try_from(clusters_per_eye)
        .ok()?
        .checked_mul(eye_count)
}

fn build_serial(
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

fn build_parallel(
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
    let chunk_offsets = build_parallel_chunk_offsets(&chunks, &ranges, total_clusters);
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
    let counts = if should_parallelize_cpu_froxel_prefix(total_clusters) {
        (0..total_clusters)
            .into_par_iter()
            .with_min_len(CPU_FROXEL_PREFIX_CHUNK_SIZE)
            .map(|cluster_id| {
                chunks.iter().fold(0u32, |total, chunk| {
                    total.saturating_add(chunk.counts[cluster_id])
                })
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

fn build_parallel_chunk_offsets(
    chunks: &[CpuFroxelCountChunk],
    ranges: &[[u32; 2]],
    total_clusters: usize,
) -> Vec<Vec<u32>> {
    let chunk_count = chunks.len();
    if should_parallelize_cpu_froxel_prefix(total_clusters) && chunk_count >= 2 {
        let per_cluster_offsets = (0..total_clusters)
            .into_par_iter()
            .with_min_len(CPU_FROXEL_PREFIX_CHUNK_SIZE)
            .map(|cluster_id| {
                let mut next = ranges[cluster_id][0];
                chunks
                    .iter()
                    .map(|chunk| {
                        let offset = next;
                        next = next.saturating_add(chunk.counts[cluster_id]);
                        offset
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut chunk_offsets = (0..chunk_count)
            .map(|_| vec![0u32; total_clusters])
            .collect::<Vec<_>>();
        for (cluster_id, offsets) in per_cluster_offsets.into_iter().enumerate() {
            for (chunk_idx, offset) in offsets.into_iter().enumerate() {
                chunk_offsets[chunk_idx][cluster_id] = offset;
            }
        }
        return chunk_offsets;
    }

    let mut chunk_offsets = (0..chunk_count)
        .map(|_| vec![0u32; total_clusters])
        .collect::<Vec<_>>();
    for cluster_id in 0..total_clusters {
        let mut next = ranges[cluster_id][0];
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            chunk_offsets[chunk_idx][cluster_id] = next;
            next = next.saturating_add(chunk.counts[cluster_id]);
        }
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

/// Assigns every light to one eye's froxel grid.
fn assign_eye_lights(
    lights: &[GpuLight],
    params: ClusterFrameParams,
    layout: FroxelLayout,
    froxel_spheres: &[FroxelSphere],
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) -> u32 {
    assign_eye_lights_slice(
        lights,
        0,
        params,
        layout,
        froxel_spheres,
        cluster_base,
        emit,
    )
}

fn assign_eye_lights_slice(
    lights: &[GpuLight],
    light_index_base: usize,
    params: ClusterFrameParams,
    layout: FroxelLayout,
    froxel_spheres: &[FroxelSphere],
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) -> u32 {
    let view = params.world_to_view;
    let view_scale = params.world_to_view_scale_max();
    let mut culled_lights = 0u32;
    for (local_light_idx, light) in lights.iter().enumerate() {
        let Some(light_idx) = light_index_base
            .checked_add(local_light_idx)
            .and_then(|idx| u32::try_from(idx).ok())
        else {
            culled_lights = culled_lights.saturating_add(1);
            continue;
        };
        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            assign_directional(light_idx, layout, cluster_base, emit);
            continue;
        }
        let Some(bounds) =
            light_froxel_bounds(light, view, params.proj, view_scale, layout, params)
        else {
            culled_lights = culled_lights.saturating_add(1);
            continue;
        };
        assign_bounded_light(
            light_idx,
            bounds,
            layout,
            froxel_spheres,
            cluster_base,
            emit,
        );
    }
    culled_lights
}

/// Inclusive froxel bounds touched by a light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FroxelBounds {
    /// First X froxel.
    x0: u32,
    /// Last X froxel.
    x1: u32,
    /// First Y froxel.
    y0: u32,
    /// Last Y froxel.
    y1: u32,
    /// First Z froxel.
    z0: u32,
    /// Last Z froxel.
    z1: u32,
}

/// View-space AABB for one froxel.
#[derive(Clone, Copy, Debug)]
struct FroxelAabb {
    /// Minimum view-space corner.
    min: Vec3,
    /// Maximum view-space corner.
    max: Vec3,
}

impl FroxelAabb {
    /// Returns a sphere that fully contains this AABB.
    fn bounding_sphere(self) -> FroxelSphere {
        let center = (self.min + self.max) * 0.5;
        FroxelSphere {
            center,
            radius: (self.max - center).length(),
        }
    }
}

/// Bounding sphere that fully contains one froxel.
#[derive(Clone, Copy, Debug)]
struct FroxelSphere {
    /// Sphere center in view space.
    center: Vec3,
    /// Sphere radius in view-space units.
    radius: f32,
}

/// Conservative assignment data for one point or spot light.
#[derive(Clone, Copy, Debug)]
struct BoundedLight {
    /// Inclusive froxel range touched by the light's broad range sphere.
    bounds: FroxelBounds,
    /// Spotlight-specific fine culling data.
    spot: Option<SpotCull>,
}

/// View-space spotlight cone used for per-froxel fine culling.
#[derive(Clone, Copy, Debug)]
struct SpotCull {
    /// Cone apex in view space.
    apex: Vec3,
    /// Normalized cone axis in view space.
    axis: Vec3,
    /// Cosine of the spotlight half-angle.
    cos_half: f32,
    /// Light range in view-space units.
    range: f32,
}

impl SpotCull {
    /// Returns whether this spotlight can affect a froxel bounding sphere.
    fn intersects_froxel_sphere(self, sphere: FroxelSphere) -> bool {
        if !sphere_sphere_intersect(self.apex, self.range, sphere.center, sphere.radius) {
            return false;
        }

        let clamped_cos_half = self.cos_half.clamp(0.0, SPOT_CULL_MIN_COS_HALF);
        if clamped_cos_half <= SPOT_CULL_WIDE_COS_HALF {
            return true;
        }

        let sin_half = (1.0 - clamped_cos_half * clamped_cos_half).max(0.0).sqrt();
        let offset = sphere.center - self.apex;
        let axis_dist = offset.dot(self.axis);
        if axis_dist < -sphere.radius || axis_dist > self.range + sphere.radius {
            return false;
        }

        let lateral_len = (offset.length_squared() - axis_dist * axis_dist)
            .max(0.0)
            .sqrt();
        let closest_cone_distance = clamped_cos_half * lateral_len - axis_dist * sin_half;
        closest_cone_distance <= sphere.radius + SPOT_CULL_DISTANCE_EPSILON
    }
}

/// Computes conservative froxel bounds for point and spot lights.
fn light_froxel_bounds(
    light: &GpuLight,
    view: Mat4,
    proj: Mat4,
    view_scale: f32,
    layout: FroxelLayout,
    params: ClusterFrameParams,
) -> Option<BoundedLight> {
    let center = transform_point(view, Vec3::from_array(light.position));
    let radius = (light.range * view_scale).max(0.0);
    if radius <= 0.0 || !radius.is_finite() {
        return None;
    }

    let spot = if light.light_type == LIGHT_TYPE_SPOT {
        let axis = transform_vector(view, Vec3::from_array(light.direction))
            .try_normalize()
            .unwrap_or(Vec3::Z);
        Some(SpotCull {
            apex: center,
            axis,
            cos_half: light.spot_cos_half_angle,
            range: radius,
        })
    } else if light.light_type != LIGHT_TYPE_POINT {
        return None;
    } else {
        None
    };

    let (near, far) = params.sanitized_clip_planes();
    let raw_nearest_depth = -(center.z + radius);
    let raw_farthest_depth = -(center.z - radius);
    if raw_farthest_depth < near || raw_nearest_depth > far {
        return None;
    }

    let nearest_depth = raw_nearest_depth.clamp(near, far);
    let farthest_depth = raw_farthest_depth.clamp(near, far);
    let z0 = cluster_z_from_depth(nearest_depth, near, far, layout.cluster_count_z);
    let z1 = cluster_z_from_depth(farthest_depth, near, far, layout.cluster_count_z);
    let (x0, x1, y0, y1) = projected_sphere_xy_bounds(center, radius, proj, near, far, layout)?;

    Some(BoundedLight {
        bounds: FroxelBounds {
            x0,
            x1,
            y0,
            y1,
            z0: z0.min(z1),
            z1: z0.max(z1),
        },
        spot,
    })
}

/// Returns whether two spheres overlap.
fn sphere_sphere_intersect(a_center: Vec3, a_radius: f32, b_center: Vec3, b_radius: f32) -> bool {
    let radius = a_radius + b_radius;
    a_center.distance_squared(b_center) <= radius * radius
}

/// Builds froxel bounding spheres for every eye.
fn build_eye_froxel_spheres(
    lights: &[GpuLight],
    eye_params: &[ClusterFrameParams],
    layouts: &[FroxelLayout],
) -> Option<Vec<Vec<FroxelSphere>>> {
    if !lights
        .iter()
        .any(|light| light.light_type == LIGHT_TYPE_SPOT)
    {
        return Some(Vec::new());
    }

    let mut all_spheres = Vec::with_capacity(eye_params.len());
    for (params, &layout) in eye_params.iter().zip(layouts.iter()) {
        all_spheres.push(froxel_bounding_spheres(*params, layout)?);
    }
    Some(all_spheres)
}

/// Returns one eye's froxel spheres, or an empty slice when no spotlights need them.
fn eye_froxel_spheres(
    froxel_spheres_by_eye: &[Vec<FroxelSphere>],
    eye_idx: usize,
) -> &[FroxelSphere] {
    froxel_spheres_by_eye
        .get(eye_idx)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// Builds bounding spheres for every froxel in one eye.
fn froxel_bounding_spheres(
    params: ClusterFrameParams,
    layout: FroxelLayout,
) -> Option<Vec<FroxelSphere>> {
    let inv_proj = params.proj.inverse();
    if !inv_proj.is_finite() {
        return None;
    }

    let cluster_count = layout.cluster_count()?;
    let mut spheres = Vec::with_capacity(cluster_count);
    for z in 0..layout.cluster_count_z {
        for y in 0..layout.cluster_count_y {
            for x in 0..layout.cluster_count_x {
                spheres.push(cluster_aabb(params, inv_proj, layout, x, y, z)?.bounding_sphere());
            }
        }
    }
    Some(spheres)
}

/// Computes the view-space AABB for one froxel.
fn cluster_aabb(
    params: ClusterFrameParams,
    inv_proj: Mat4,
    layout: FroxelLayout,
    cluster_x: u32,
    cluster_y: u32,
    cluster_z: u32,
) -> Option<FroxelAabb> {
    let (near_depth, far_depth) = cluster_z_depth_bounds(
        cluster_z,
        layout.cluster_count_z,
        params.near_clip,
        params.far_clip,
    );
    let tile_near = -near_depth;
    let tile_far = -far_depth;

    let w = layout.viewport_width as f32;
    let h = layout.viewport_height as f32;
    let px_min = cluster_x.saturating_mul(TILE_SIZE) as f32;
    let px_max = cluster_x
        .saturating_add(1)
        .saturating_mul(TILE_SIZE)
        .min(layout.viewport_width) as f32;
    let py_min = cluster_y.saturating_mul(TILE_SIZE) as f32;
    let py_max = cluster_y
        .saturating_add(1)
        .saturating_mul(TILE_SIZE)
        .min(layout.viewport_height) as f32;
    let ndc_left = 2.0 * px_min / w - 1.0;
    let ndc_right = 2.0 * px_max / w - 1.0;
    let ndc_top = 1.0 - 2.0 * py_min / h;
    let ndc_bottom = 1.0 - 2.0 * py_max / h;

    let ndc_bl = Vec2::new(ndc_left, ndc_bottom);
    let ndc_br = Vec2::new(ndc_right, ndc_bottom);
    let ndc_tl = Vec2::new(ndc_left, ndc_top);
    let ndc_tr = Vec2::new(ndc_right, ndc_top);

    let points = [
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_bl, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_br, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tl, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tr, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_bl, tile_far)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_br, tile_far)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tl, tile_far)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tr, tile_far)?,
    ];
    let (mut min_v, mut max_v) = min_max_points(&points);

    if cluster_z == 0 {
        let camera_clip = params.proj * Vec4::new(0.0, 0.0, 0.0, 1.0);
        if camera_clip.w.abs() > 1e-8 {
            let zero_points = [
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_bl, 0.0)?,
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_br, 0.0)?,
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tl, 0.0)?,
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tr, 0.0)?,
            ];
            let (zero_min, zero_max) = min_max_points(&zero_points);
            min_v = min_v.min(zero_min);
            max_v = max_v.max(zero_max);
        } else {
            min_v = min_v.min(Vec3::ZERO);
            max_v = max_v.max(Vec3::ZERO);
        }
    }

    let extent = max_v - min_v;
    let max_extent = extent.x.max(extent.y).max(extent.z).max(1.0);
    let pad = max_extent * CLUSTER_BOUNDARY_EPSILON;
    Some(FroxelAabb {
        min: min_v - Vec3::splat(pad),
        max: max_v + Vec3::splat(pad),
    })
}

/// Returns the component-wise min and max for a fixed set of view-space points.
fn min_max_points<const N: usize>(points: &[Vec3; N]) -> (Vec3, Vec3) {
    let mut min_v = Vec3::splat(f32::INFINITY);
    let mut max_v = Vec3::splat(f32::NEG_INFINITY);
    for &point in points {
        min_v = min_v.min(point);
        max_v = max_v.max(point);
    }
    (min_v, max_v)
}

/// Computes logarithmic clustered-depth bounds for one Z slice.
fn cluster_z_depth_bounds(
    cluster_z: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
) -> (f32, f32) {
    let z_count = cluster_count_z.max(1);
    let z = cluster_z.min(z_count - 1);
    let (near_safe, far_safe) = sanitize_cluster_clip_planes(near_clip, far_clip);
    let ratio = (far_safe / near_safe).max(1.0 + f32::EPSILON);
    let zf = z as f32;
    let num_z = z_count as f32;
    (
        near_safe * ratio.powf(zf / num_z),
        near_safe * ratio.powf((zf + 1.0) / num_z),
    )
}

/// Reconstructs a view-space point from NDC X/Y and view-space Z.
fn view_point_at_ndc_xy_and_z(
    proj: Mat4,
    inv_proj: Mat4,
    ndc_xy: Vec2,
    view_z: f32,
) -> Option<Vec3> {
    ndc_z_from_view_z(proj, view_z)
        .and_then(|ndc_z| ndc_to_view(inv_proj, Vec3::new(ndc_xy.x, ndc_xy.y, ndc_z)))
}

/// Reconstructs a view-space point from NDC coordinates.
fn ndc_to_view(inv_proj: Mat4, ndc: Vec3) -> Option<Vec3> {
    let clip = inv_proj * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
    let point = if clip.w.abs() <= 1e-8 {
        clip.truncate()
    } else {
        clip.truncate() / clip.w
    };
    point.is_finite().then_some(point)
}

/// Projects view-space Z to NDC Z using the frame projection.
fn ndc_z_from_view_z(proj: Mat4, view_z: f32) -> Option<f32> {
    let clip = proj * Vec4::new(0.0, 0.0, view_z, 1.0);
    if clip.w.abs() <= 1e-8 || !clip.w.is_finite() {
        return None;
    }
    let ndc_z = clip.z / clip.w;
    ndc_z.is_finite().then_some(ndc_z)
}

/// Transforms a world-space point by `matrix`.
fn transform_point(matrix: Mat4, point: Vec3) -> Vec3 {
    (matrix * point.extend(1.0)).truncate()
}

/// Transforms a world-space vector by `matrix`.
fn transform_vector(matrix: Mat4, vector: Vec3) -> Vec3 {
    (matrix * vector.extend(0.0)).truncate()
}

/// Maps positive depth to a logarithmic clustered Z slice.
fn cluster_z_from_depth(depth: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let z_count = cluster_count_z.max(1);
    let (near_safe, far_safe) = sanitize_cluster_clip_planes(near_clip, far_clip);
    let ratio = (far_safe / near_safe).max(1.0 + f32::EPSILON);
    let z = (depth.clamp(near_safe, far_safe) / near_safe).log(ratio) * z_count as f32;
    z.clamp(0.0, z_count.saturating_sub(1) as f32) as u32
}

/// Computes conservative screen-space froxel bounds for a view-space sphere.
fn projected_sphere_xy_bounds(
    center: Vec3,
    radius: f32,
    proj: Mat4,
    near: f32,
    far: f32,
    layout: FroxelLayout,
) -> Option<(u32, u32, u32, u32)> {
    let near_z = (center.z + radius).min(-near).max(-far);
    let far_z = (center.z - radius).min(-near).max(-far);
    let mut ndc_min = Vec2::splat(f32::INFINITY);
    let mut ndc_max = Vec2::splat(f32::NEG_INFINITY);
    for z in [near_z, far_z] {
        for x_sign in [-1.0, 1.0] {
            for y_sign in [-1.0, 1.0] {
                let p = Vec3::new(center.x + radius * x_sign, center.y + radius * y_sign, z);
                let ndc = project_view_point(proj, p)?;
                ndc_min = ndc_min.min(ndc);
                ndc_max = ndc_max.max(ndc);
            }
        }
    }
    let x0 = ndc_x_to_cluster(ndc_min.x, layout);
    let x1 = ndc_x_to_cluster(ndc_max.x, layout);
    let y0 = ndc_y_to_cluster(ndc_max.y, layout);
    let y1 = ndc_y_to_cluster(ndc_min.y, layout);
    Some((x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1)))
}

/// Projects a view-space point into normalized device coordinates.
fn project_view_point(proj: Mat4, point: Vec3) -> Option<Vec2> {
    let clip = proj * Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w.abs() <= 1e-8 || !clip.w.is_finite() {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    (ndc.x.is_finite() && ndc.y.is_finite()).then(|| ndc.truncate())
}

/// Converts NDC X to a froxel coordinate.
fn ndc_x_to_cluster(ndc_x: f32, layout: FroxelLayout) -> u32 {
    let px = ((ndc_x.clamp(-1.0, 1.0) + 1.0) * 0.5 * layout.viewport_width as f32).floor();
    (px as u32 / TILE_SIZE).min(layout.cluster_count_x - 1)
}

/// Converts NDC Y to a froxel coordinate with top-left screen origin.
fn ndc_y_to_cluster(ndc_y: f32, layout: FroxelLayout) -> u32 {
    let py = ((1.0 - ndc_y.clamp(-1.0, 1.0)) * 0.5 * layout.viewport_height as f32).floor();
    (py as u32 / TILE_SIZE).min(layout.cluster_count_y - 1)
}

/// Assigns a directional light to every froxel.
fn assign_directional(
    light_idx: u32,
    layout: FroxelLayout,
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) {
    let Some(cluster_count) = layout.cluster_count() else {
        return;
    };
    for cluster_local in 0..cluster_count {
        emit(cluster_base + cluster_local, light_idx);
    }
}

/// Assigns a bounded local light to its touched froxel range.
fn assign_bounded_light(
    light_idx: u32,
    light: BoundedLight,
    layout: FroxelLayout,
    froxel_spheres: &[FroxelSphere],
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) {
    for z in light.bounds.z0..=light.bounds.z1 {
        for y in light.bounds.y0..=light.bounds.y1 {
            for x in light.bounds.x0..=light.bounds.x1 {
                let local = x + layout.cluster_count_x * (y + layout.cluster_count_y * z);
                let local_usize = local as usize;
                if let Some(spot) = light.spot {
                    let Some(&froxel_sphere) = froxel_spheres.get(local_usize) else {
                        continue;
                    };
                    if !spot.intersects_froxel_sphere(froxel_sphere) {
                        continue;
                    }
                }
                emit(cluster_base + local_usize, light_idx);
            }
        }
    }
}

/// Converts per-froxel counts into compact `[offset, count]` rows.
fn prefix_counts_to_ranges(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
    if should_parallelize_cpu_froxel_prefix(counts.len()) {
        return prefix_counts_to_ranges_parallel(counts);
    }
    prefix_counts_to_ranges_serial(counts)
}

/// Serial prefix-sum implementation for small froxel-count arrays.
fn prefix_counts_to_ranges_serial(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
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
fn prefix_counts_to_ranges_parallel(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
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
fn write_membership(
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

fn write_membership_atomic(
    cluster_id: usize,
    light_idx: u32,
    offsets: &[u32],
    cursors: &mut [u32],
    indices: &[AtomicU32],
) {
    let Some(&base) = offsets.get(cluster_id) else {
        return;
    };
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

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use super::*;

    /// Builds a compact 2x2x16 test layout.
    fn test_params() -> ClusterFrameParams {
        ClusterFrameParams {
            near_clip: 0.1,
            far_clip: 100.0,
            world_to_view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            cluster_count_x: 2,
            cluster_count_y: 2,
            viewport_width: 64,
            viewport_height: 64,
            projection_flags: 0,
        }
    }

    /// Builds a larger 8x8x16 layout that crosses the parallel prefix threshold.
    fn large_test_params() -> ClusterFrameParams {
        ClusterFrameParams {
            cluster_count_x: 8,
            cluster_count_y: 8,
            viewport_width: 256,
            viewport_height: 256,
            ..test_params()
        }
    }

    /// Builds a point light at `position`.
    fn point_light(position: Vec3, range: f32) -> GpuLight {
        GpuLight {
            position: position.to_array(),
            range,
            light_type: LIGHT_TYPE_POINT,
            ..Default::default()
        }
    }

    /// Builds a spot light at `position`.
    fn spot_light(
        position: Vec3,
        direction: Vec3,
        range: f32,
        full_angle_degrees: f32,
    ) -> GpuLight {
        let half_angle = (full_angle_degrees * 0.5).to_radians();
        GpuLight {
            position: position.to_array(),
            direction: direction.to_array(),
            range,
            light_type: LIGHT_TYPE_SPOT,
            spot_cos_half_angle: half_angle.cos().clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Builds a directional light.
    fn directional_light() -> GpuLight {
        GpuLight {
            light_type: LIGHT_TYPE_DIRECTIONAL,
            ..Default::default()
        }
    }

    /// Builds a spotlight cull primitive using a half-angle in degrees.
    fn spot_cull(half_angle_degrees: f32, range: f32) -> SpotCull {
        SpotCull {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            cos_half: half_angle_degrees.to_radians().cos().clamp(0.0, 1.0),
            range,
        }
    }

    /// Returns the compact light-index slice for one cluster.
    fn cluster_indices(assignments: &CpuClusterAssignments, cluster_id: usize) -> &[u32] {
        let [offset, count] = assignments.ranges[cluster_id];
        let start = offset as usize;
        let end = start + count as usize;
        &assignments.indices[start..end]
    }

    #[test]
    fn cpu_froxel_light_parallel_gate_starts_at_two_chunks() {
        assert_eq!(
            CPU_FROXEL_PARALLEL_MIN_LIGHTS,
            CPU_FROXEL_LIGHT_CHUNK_SIZE * 2
        );
        assert!(!should_parallelize_cpu_froxel_lights(
            CPU_FROXEL_PARALLEL_MIN_LIGHTS - 1
        ));
        assert!(should_parallelize_cpu_froxel_lights(
            CPU_FROXEL_PARALLEL_MIN_LIGHTS
        ));
    }

    #[test]
    fn cpu_froxel_prefix_parallel_gate_starts_at_prefix_chunk() {
        assert_eq!(
            CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
            CPU_FROXEL_PREFIX_CHUNK_SIZE * 2
        );
        assert!(!should_parallelize_cpu_froxel_prefix(
            CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS - 1
        ));
        assert!(should_parallelize_cpu_froxel_prefix(
            CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS
        ));
    }

    #[test]
    fn empty_lights_write_zero_ranges_without_indices() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");
        assert_eq!(assignments.ranges.len(), 64);
        assert!(assignments.ranges.iter().all(|range| range[1] == 0));
        assert!(assignments.indices.is_empty());
    }

    #[test]
    fn directional_light_hits_every_froxel() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[directional_light()],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        assert!(assignments.ranges.iter().all(|range| range[1] == 1));
        assert_eq!(cluster_indices(&assignments, 0), &[0]);
    }

    #[test]
    fn local_light_touches_subset_of_froxels() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[point_light(Vec3::new(0.0, 0.0, -5.0), 0.25)],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        let touched = assignments
            .ranges
            .iter()
            .filter(|range| range[1] > 0)
            .count();
        assert!(touched > 0);
        assert!(touched < assignments.ranges.len());
    }

    #[test]
    fn spotlight_cull_keeps_edge_touching_froxel() {
        let spot = spot_cull(30.0, 10.0);
        let axis_dist = 5.0f32;
        let radius = 0.25f32;
        let cone_edge = axis_dist * 30.0f32.to_radians().tan();
        let sphere = FroxelSphere {
            center: Vec3::new(cone_edge + radius * 0.5, 0.0, axis_dist),
            radius,
        };

        assert!(spot.intersects_froxel_sphere(sphere));
    }

    #[test]
    fn spotlight_cull_keeps_froxel_crossing_apex_plane() {
        let sphere = FroxelSphere {
            center: Vec3::new(0.0, 0.0, -0.05),
            radius: 0.1,
        };

        assert!(spot_cull(20.0, 10.0).intersects_froxel_sphere(sphere));
    }

    #[test]
    fn spotlight_cull_keeps_froxel_crossing_range_end() {
        let sphere = FroxelSphere {
            center: Vec3::new(0.0, 0.0, 10.04),
            radius: 0.05,
        };

        assert!(spot_cull(20.0, 10.0).intersects_froxel_sphere(sphere));
    }

    #[test]
    fn spotlight_cull_clamps_tiny_angles_conservatively() {
        let min_half_angle = 0.5f32.to_radians();
        let sphere = FroxelSphere {
            center: Vec3::new(5.0 * min_half_angle.tan() * 0.5, 0.0, 5.0),
            radius: 0.001,
        };
        let spot = SpotCull {
            apex: Vec3::ZERO,
            axis: Vec3::Z,
            cos_half: 1.0,
            range: 10.0,
        };

        assert!(spot.intersects_froxel_sphere(sphere));
    }

    #[test]
    fn spotlight_cull_uses_range_for_wide_cones() {
        let sphere = FroxelSphere {
            center: Vec3::new(5.0, 0.0, 1.0),
            radius: 0.1,
        };

        assert!(spot_cull(75.0, 10.0).intersects_froxel_sphere(sphere));
    }

    #[test]
    fn compact_indices_store_all_lights() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[directional_light(), directional_light()],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        assert_eq!(assignments.ranges[0][1], 2);
        assert_eq!(cluster_indices(&assignments, 0), &[0, 1]);
    }

    #[test]
    fn compact_indices_do_not_truncate_dense_clusters() {
        let params = test_params();
        let lights = (0..70).map(|_| directional_light()).collect::<Vec<_>>();
        let assignments = FroxelLightPlanner::build(
            &lights,
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        assert!(assignments.ranges.iter().all(|range| range[1] == 70));
        assert_eq!(cluster_indices(&assignments, 0).len(), 70);
        assert_eq!(assignments.stats.overflowed_memberships, 0);
    }

    #[test]
    fn parallel_froxel_build_matches_serial_build() {
        let params = [test_params(), test_params()];
        let clusters_per_eye =
            params[0].cluster_count_x * params[0].cluster_count_y * CLUSTER_COUNT_Z;
        let layouts = validated_eye_layouts(&params, clusters_per_eye).expect("layouts");
        let lights = (0..CPU_FROXEL_PARALLEL_MIN_LIGHTS + 13)
            .map(|idx| {
                if idx % 7 == 0 {
                    spot_light(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, 3.0, 60.0)
                } else if idx % 5 == 0 {
                    point_light(Vec3::new((idx % 3) as f32 - 1.0, 0.0, -5.0), 0.5)
                } else {
                    directional_light()
                }
            })
            .collect::<Vec<_>>();

        let serial = build_serial(&lights, &params, &layouts, clusters_per_eye).expect("serial");
        let parallel =
            build_parallel(&lights, &params, &layouts, clusters_per_eye).expect("parallel");

        assert_eq!(parallel.ranges, serial.ranges);
        assert_eq!(parallel.indices, serial.indices);
        assert_eq!(parallel.stats, serial.stats);
    }

    #[test]
    fn parallel_prefix_counts_match_serial_prefix() {
        let counts = (0..CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS + 257)
            .map(|idx| (idx % 7) as u32)
            .collect::<Vec<_>>();

        let serial = prefix_counts_to_ranges_serial(&counts).expect("serial");
        let parallel = prefix_counts_to_ranges_parallel(&counts).expect("parallel");

        assert_eq!(parallel, serial);
    }

    #[test]
    fn parallel_froxel_build_matches_serial_for_large_cluster_grid() {
        let params = [large_test_params(), large_test_params()];
        let clusters_per_eye =
            params[0].cluster_count_x * params[0].cluster_count_y * CLUSTER_COUNT_Z;
        let layouts = validated_eye_layouts(&params, clusters_per_eye).expect("layouts");
        let lights = (0..CPU_FROXEL_PARALLEL_MIN_LIGHTS + 13)
            .map(|idx| {
                if idx % 7 == 0 {
                    spot_light(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, 3.0, 60.0)
                } else if idx % 5 == 0 {
                    point_light(Vec3::new((idx % 3) as f32 - 1.0, 0.0, -5.0), 0.5)
                } else {
                    directional_light()
                }
            })
            .collect::<Vec<_>>();

        let serial = build_serial(&lights, &params, &layouts, clusters_per_eye).expect("serial");
        let parallel =
            build_parallel(&lights, &params, &layouts, clusters_per_eye).expect("parallel");

        assert_eq!(parallel.ranges, serial.ranges);
        assert_eq!(parallel.indices, serial.indices);
        assert_eq!(parallel.stats, serial.stats);
    }
}
