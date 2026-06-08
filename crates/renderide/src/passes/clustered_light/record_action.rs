//! Action variants returned from [`super::ClusteredLightPass::prepare_record_action`].
//!
//! The clustered-light pass picks one of three record actions per view tick: skip, clear ranges
//! for the zero-light shortcut, or run the GPU scan.

use crate::camera::ViewId;
use crate::world_mesh::cluster::ClusterFrameParams;

/// Prepared work selected by clustered-light recording.
pub(super) enum ClusteredLightRecordAction {
    /// Nothing to record for this view.
    Skip,
    /// Clear cluster ranges because the frame has no lights.
    ClearZero(ClusteredLightClearData),
    /// Run the existing GPU scan compute path.
    GpuScan(ClusteredLightGpuScanData),
}

/// Data needed to clear empty per-cluster light ranges.
pub(super) struct ClusteredLightClearData {
    /// Shared cluster-range buffer.
    pub cluster_light_counts: wgpu::Buffer,
    /// Number of clusters produced per eye.
    pub clusters_per_eye: u32,
    /// Number of eyes represented by this view.
    pub eye_count: usize,
}

/// Data needed to run the GPU clustered-light scan.
pub(super) struct ClusteredLightGpuScanData {
    /// Graph view id.
    pub view_id: ViewId,
    /// Shared cluster-buffer cache version.
    pub cluster_ver: u64,
    /// Shared cluster-range buffer.
    pub cluster_light_counts: wgpu::Buffer,
    /// Shared compact cluster-index buffer.
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
