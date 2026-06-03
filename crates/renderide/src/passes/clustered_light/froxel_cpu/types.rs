use crate::gpu::GpuLight;
use crate::world_mesh::cluster::{CLUSTER_COUNT_Z, ClusterFrameParams};

use super::bounds::EyeFroxelSpheres;

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
    pub(super) fn from_cluster_params(params: &ClusterFrameParams) -> Self {
        Self {
            cluster_count_x: params.cluster_count_x.max(1),
            cluster_count_y: params.cluster_count_y.max(1),
            cluster_count_z: CLUSTER_COUNT_Z.max(1),
            viewport_width: params.viewport_width.max(1),
            viewport_height: params.viewport_height.max(1),
        }
    }

    /// Number of froxels in this eye.
    pub(super) fn cluster_count(self) -> Option<usize> {
        let xy = self.cluster_count_x.checked_mul(self.cluster_count_y)?;
        xy.checked_mul(self.cluster_count_z).map(|v| v as usize)
    }
}

/// Per-frame CPU-produced cluster storage matching the existing WGSL buffers.
#[derive(Clone, Debug, Default)]
pub(in crate::passes::clustered_light) struct CpuClusterAssignments {
    /// Per-froxel `[offset, count]` rows addressing [`Self::indices`].
    pub ranges: Vec<[u32; 2]>,
    /// Compact light indices for every froxel membership.
    pub indices: Vec<u32>,
    /// Assignment diagnostics for profiling and tests.
    pub stats: CpuFroxelStats,
}

/// CPU froxel assignment diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::passes::clustered_light) struct CpuFroxelStats {
    /// Number of light/froxel memberships emitted into compact storage.
    pub assigned_memberships: u64,
    /// Number of light/froxel memberships dropped because compact storage could not represent them.
    pub overflowed_memberships: u64,
    /// Number of lights rejected before assignment because their conservative bounds miss the view.
    pub culled_lights: u32,
}

/// Flat per-light-chunk froxel membership counts for the parallel CPU path.
pub(super) struct CpuFroxelChunkCounts {
    /// Chunk-major count rows addressed by `chunk_idx * total_clusters + cluster_id`.
    pub(super) counts: Vec<u32>,
    /// Per-light-chunk assignment diagnostics.
    pub(super) stats: Vec<CpuFroxelStats>,
    /// Number of light chunks in this frame's parallel build.
    pub(super) chunk_count: usize,
    /// Number of froxels across all eyes in this frame.
    pub(super) total_clusters: usize,
}

/// Local prefix-sum result for one cluster-count chunk.
pub(super) struct CpuFroxelPrefixChunk {
    /// Range rows with offsets relative to the start of this chunk.
    pub(super) ranges: Vec<[u32; 2]>,
    /// Sum of every count in this chunk.
    pub(super) total_count: u64,
}

/// Shared read-only inputs used by the parallel CPU froxel passes.
pub(super) struct CpuFroxelParallelInputs<'a> {
    /// Lights submitted for the current clustered-light build.
    pub(super) lights: &'a [GpuLight],
    /// Per-eye cluster frame parameters.
    pub(super) eye_params: &'a [ClusterFrameParams],
    /// Validated per-eye froxel layouts.
    pub(super) layouts: &'a [FroxelLayout],
    /// Flat spotlight culling sphere cache.
    pub(super) froxel_spheres_by_eye: &'a EyeFroxelSpheres,
    /// Froxel count expected for each eye.
    pub(super) expected_clusters: usize,
    /// Froxel count across every eye.
    pub(super) total_clusters: usize,
}
