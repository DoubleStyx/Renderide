use glam::{Mat4, Vec3};

use crate::gpu::GpuLight;
use crate::world_mesh::cluster::{CLUSTER_COUNT_Z, ClusterFrameParams};

use super::bounds::{
    FROXEL_SPHERE_PARALLEL_MIN_FROXELS, FroxelSphere, LIGHT_TYPE_DIRECTIONAL, LIGHT_TYPE_POINT,
    LIGHT_TYPE_SPOT, SpotCull, should_parallelize_froxel_spheres_with_workers,
};
use super::parallel::build_parallel;
use super::planner::{
    CPU_FROXEL_LIGHT_CHUNK_SIZE, CPU_FROXEL_PARALLEL_MIN_LIGHTS, CPU_FROXEL_PREFIX_CHUNK_SIZE,
    CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS, FroxelLightPlanner, build_serial,
    should_parallelize_cpu_froxel_lights_with_workers,
    should_parallelize_cpu_froxel_offsets_with_workers,
    should_parallelize_cpu_froxel_prefix_with_workers, validated_eye_layouts,
};
use super::prefix::{prefix_counts_to_ranges_parallel, prefix_counts_to_ranges_serial};
use super::types::CpuClusterAssignments;

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
fn spot_light(position: Vec3, direction: Vec3, range: f32, full_angle_degrees: f32) -> GpuLight {
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
    assert!(!should_parallelize_cpu_froxel_lights_with_workers(
        CPU_FROXEL_PARALLEL_MIN_LIGHTS - 1,
        4
    ));
    assert!(should_parallelize_cpu_froxel_lights_with_workers(
        CPU_FROXEL_PARALLEL_MIN_LIGHTS,
        4
    ));
    assert!(!should_parallelize_cpu_froxel_lights_with_workers(
        CPU_FROXEL_PARALLEL_MIN_LIGHTS,
        1
    ));
}

#[test]
fn cpu_froxel_prefix_parallel_gate_requires_workers_and_two_chunks() {
    assert_eq!(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
        CPU_FROXEL_PREFIX_CHUNK_SIZE * 2
    );
    assert!(!should_parallelize_cpu_froxel_prefix_with_workers(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS - 1,
        4
    ));
    assert!(should_parallelize_cpu_froxel_prefix_with_workers(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
        4
    ));
    assert!(!should_parallelize_cpu_froxel_prefix_with_workers(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
        1
    ));
}

#[test]
fn cpu_froxel_offset_parallel_gate_requires_multiple_light_chunks() {
    assert!(!should_parallelize_cpu_froxel_offsets_with_workers(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
        1,
        4
    ));
    assert!(should_parallelize_cpu_froxel_offsets_with_workers(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
        2,
        4
    ));
    assert!(!should_parallelize_cpu_froxel_offsets_with_workers(
        CPU_FROXEL_PREFIX_PARALLEL_MIN_CLUSTERS,
        2,
        1
    ));
}

#[test]
fn froxel_sphere_parallel_gate_requires_workers_and_two_chunks() {
    assert!(!should_parallelize_froxel_spheres_with_workers(
        FROXEL_SPHERE_PARALLEL_MIN_FROXELS - 1,
        4
    ));
    assert!(should_parallelize_froxel_spheres_with_workers(
        FROXEL_SPHERE_PARALLEL_MIN_FROXELS,
        4
    ));
    assert!(!should_parallelize_froxel_spheres_with_workers(
        FROXEL_SPHERE_PARALLEL_MIN_FROXELS,
        1
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
    let clusters_per_eye = params[0].cluster_count_x * params[0].cluster_count_y * CLUSTER_COUNT_Z;
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
    let parallel = build_parallel(&lights, &params, &layouts, clusters_per_eye).expect("parallel");

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
    let clusters_per_eye = params[0].cluster_count_x * params[0].cluster_count_y * CLUSTER_COUNT_Z;
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
    let parallel = build_parallel(&lights, &params, &layouts, clusters_per_eye).expect("parallel");

    assert_eq!(parallel.ranges, serial.ranges);
    assert_eq!(parallel.indices, serial.indices);
    assert_eq!(parallel.stats, serial.stats);
}
