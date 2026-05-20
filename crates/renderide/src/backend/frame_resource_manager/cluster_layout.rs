//! Shared-cluster pre-record layouts and uniform buffer allocation helpers.

use crate::backend::cluster_gpu::{CLUSTER_COUNT_Z, TILE_SIZE};
use crate::camera::ViewId;
use crate::gpu::CLUSTER_PARAMS_UNIFORM_SIZE;
use crate::graph_inputs::PreRecordViewResourceLayout;

use super::super::frame_gpu::PerViewSceneSnapshotSyncParams;

/// Unique shared-cluster pre-record layout after removing view-local snapshot fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ClusterPreRecordLayout {
    /// Viewport width in physical pixels.
    pub(super) width: u32,
    /// Viewport height in physical pixels.
    pub(super) height: u32,
    /// Whether cluster buffers need two-eye storage.
    pub(super) stereo: bool,
    /// Minimum compact light-index words required for this cluster layout.
    pub(super) index_capacity_words: u64,
}

/// Converts a view resource layout into the view-local scene snapshot sync request.
pub(super) fn per_view_snapshot_sync_params(
    layout: PreRecordViewResourceLayout,
) -> PerViewSceneSnapshotSyncParams {
    PerViewSceneSnapshotSyncParams {
        viewport: (layout.width, layout.height),
        depth_format: layout.depth_format,
        color_format: layout.color_format,
        multiview: layout.stereo,
        needs_depth_snapshot: layout.needs_depth_snapshot,
        needs_color_snapshot: layout.needs_color_snapshot,
    }
}

/// Returns stable unique cluster layouts while preserving first-seen view order.
pub(super) fn unique_cluster_pre_record_layouts(
    view_layouts: &[PreRecordViewResourceLayout],
    light_count_for_view: impl Fn(ViewId) -> u32,
) -> Vec<ClusterPreRecordLayout> {
    let mut out: Vec<ClusterPreRecordLayout> = Vec::new();
    for layout in view_layouts {
        let Some(index_capacity_words) =
            cluster_index_capacity_for_layout(*layout, light_count_for_view(layout.view_id))
        else {
            logger::warn!(
                "skipping impossible cluster capacity for viewport {}x{} stereo={}",
                layout.width,
                layout.height,
                layout.stereo
            );
            continue;
        };
        if let Some(existing) = out.iter_mut().find(|existing| {
            existing.width == layout.width
                && existing.height == layout.height
                && existing.stereo == layout.stereo
        }) {
            existing.index_capacity_words = existing.index_capacity_words.max(index_capacity_words);
        } else {
            out.push(ClusterPreRecordLayout {
                width: layout.width,
                height: layout.height,
                stereo: layout.stereo,
                index_capacity_words,
            });
        }
    }
    out
}

/// Returns compact index-buffer capacity for a layout if the calculation fits in `u64`.
pub(super) fn cluster_index_capacity_for_layout(
    layout: PreRecordViewResourceLayout,
    light_count: u32,
) -> Option<u64> {
    let cluster_x = layout.width.max(1).div_ceil(TILE_SIZE);
    let cluster_y = layout.height.max(1).div_ceil(TILE_SIZE);
    let eye_count = if layout.stereo { 2_u64 } else { 1_u64 };
    let capacity = u64::from(cluster_x)
        .checked_mul(u64::from(cluster_y))?
        .checked_mul(u64::from(CLUSTER_COUNT_Z))?
        .checked_mul(eye_count)?
        .checked_mul(u64::from(light_count.max(1)))?;
    u32::try_from(capacity).is_ok().then_some(capacity)
}

/// Allocates the per-view `ClusterParams` uniform buffer. Sized for one slot (mono) or two
/// slots (stereo). Used by `ClusteredLightPass` to write camera matrices per-view without
/// racing against other views' writes in the shared graph upload sink.
pub(super) fn make_cluster_params_buffer(device: &wgpu::Device, stereo: bool) -> wgpu::Buffer {
    let eye_multiplier = if stereo { 2 } else { 1 };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_view_cluster_params_uniform"),
        size: CLUSTER_PARAMS_UNIFORM_SIZE * eye_multiplier,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(Buffer, "backend::per_view_cluster_params_uniform");
    buffer
}
