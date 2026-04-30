//! Action variants and CPU-froxel upload returned from
//! [`super::ClusteredLightPass::prepare_record_action`].
//!
//! The clustered-light pass picks one of four record actions per view tick: skip, complete
//! (CPU froxel finished the work), clear-counts (zero-light shortcut), or run the GPU scan.

use crate::backend::GpuLight;
use crate::camera::ViewId;
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::world_mesh::cluster_frame::ClusterFrameParams;

use super::froxel_cpu::FroxelLightPlanner;

/// Prepared work selected by clustered-light recording.
pub(super) enum ClusteredLightRecordAction {
    /// Nothing to record for this view.
    Skip,
    /// CPU assignment already uploaded the cluster buffers.
    Done,
    /// Clear cluster counts because the frame has no lights.
    ClearZero(ClusteredLightClearData),
    /// Run the existing GPU scan compute path.
    GpuScan(ClusteredLightGpuScanData),
}

/// Data needed to clear empty per-cluster light counts.
pub(super) struct ClusteredLightClearData {
    /// Shared cluster-count buffer.
    pub cluster_light_counts: wgpu::Buffer,
    /// Number of clusters produced per eye.
    pub clusters_per_eye: u32,
    /// Number of eyes represented by this view.
    pub eye_count: usize,
}

/// Inputs for CPU froxel assignment and upload.
pub(super) struct CpuFroxelRecordData<'a> {
    /// Deferred upload sink for compatible cluster buffers.
    pub upload_batch: &'a FrameUploadBatch,
    /// CPU-side light rows packed during frame preparation.
    pub lights: &'a [GpuLight],
    /// Shared cluster-count buffer.
    pub cluster_light_counts: &'a wgpu::Buffer,
    /// Shared packed cluster-index buffer.
    pub cluster_light_indices: &'a wgpu::Buffer,
    /// Per-eye cluster frame params.
    pub eye_params: &'a [ClusterFrameParams],
    /// Number of clusters produced per eye.
    pub clusters_per_eye: u32,
    /// Graph view id.
    pub view_id: ViewId,
    /// Scene light count.
    pub light_count: u32,
}

/// Data needed to run the GPU clustered-light scan.
pub(super) struct ClusteredLightGpuScanData {
    /// Graph view id.
    pub view_id: ViewId,
    /// Shared cluster-buffer cache version.
    pub cluster_ver: u64,
    /// Shared cluster-count buffer.
    pub cluster_light_counts: wgpu::Buffer,
    /// Shared packed cluster-index buffer.
    pub cluster_light_indices: wgpu::Buffer,
    /// Per-view cluster params uniform buffer.
    pub params_buffer: wgpu::Buffer,
    /// Frame light storage buffer.
    pub lights_buffer: wgpu::Buffer,
    /// Per-eye cluster frame params.
    pub eye_params: Vec<ClusterFrameParams>,
    /// Number of clusters produced per eye.
    pub clusters_per_eye: u32,
    /// Scene light count.
    pub light_count: u32,
    /// Target viewport size in pixels.
    pub viewport: (u32, u32),
}

/// Attempts CPU light-centric froxel assignment and uploads compatible cluster buffers.
///
/// Returns `true` when assignment succeeded and the cluster buffers were uploaded; `false`
/// when assignment failed or was skipped, prompting the caller to fall back to GPU scan.
pub(super) fn try_record_cpu_froxel(data: CpuFroxelRecordData<'_>) -> bool {
    let Some(assignments) =
        FroxelLightPlanner::build(data.lights, data.eye_params, data.clusters_per_eye)
    else {
        logger::warn!(
            "ClusteredLight: CPU froxel assignment failed for {:?}; falling back to GPU scan",
            data.view_id
        );
        return false;
    };

    profiling::scope!("clustered_light::cpu_froxel_upload");
    data.upload_batch.write_buffer(
        data.cluster_light_counts,
        0,
        bytemuck::cast_slice(&assignments.counts),
    );
    if !assignments.indices.is_empty() {
        data.upload_batch.write_buffer(
            data.cluster_light_indices,
            0,
            bytemuck::cast_slice(&assignments.indices),
        );
    }
    logger::trace!(
        "ClusteredLight: CPU froxel assignment view={view_id:?} lights={} eyes={} memberships={} overflowed={}",
        data.light_count,
        data.eye_params.len(),
        assignments.stats.assigned_memberships,
        assignments.stats.overflowed_memberships,
        view_id = data.view_id
    );
    true
}
