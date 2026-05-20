//! Scene-depth snapshot recording for the graph-managed world-mesh forward pass.
//!
//! When MSAA is active, depth is resolved first (delegating to [`super::depth_resolve`]). The
//! resolved single-sample depth is then copied into the sampled scene-depth snapshot used by
//! intersection materials.

use crate::gpu::MsaaDepthResolveResources;
use crate::graph_inputs::{GraphPassFrame, MsaaViews};
use crate::profiling::GpuProfilerHandle;
use crate::world_mesh::WorldMeshHelperNeeds;

use super::PreparedWorldMeshForwardFrame;
use super::depth_resolve::encode_msaa_depth_resolve_for_frame;

/// Resolves MSAA depth when needed, then copies the single-sample frame depth into the sampled
/// scene-depth snapshot used by intersection materials.
pub(crate) fn encode_world_mesh_forward_depth_snapshot(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    msaa_views: Option<&MsaaViews>,
    msaa_depth_resolve: Option<&MsaaDepthResolveResources>,
    profiler: Option<&GpuProfilerHandle>,
) -> bool {
    profiling::scope!("world_mesh_forward::encode_depth_snapshot");
    if !depth_snapshot_recording_needed(prepared.helper_needs) {
        return false;
    }

    if frame.view.sample_count > 1
        && let (Some(msaa_views), Some(res)) = (msaa_views, msaa_depth_resolve)
    {
        encode_msaa_depth_resolve_for_frame(device, encoder, frame, msaa_views, res, profiler);
    }

    if !frame.shared.frame_resources.has_frame_gpu() {
        return false;
    }
    let copy_query =
        profiler.map(|p| p.begin_query("world_mesh_forward::scene_depth_snapshot_copy", encoder));
    let copied = frame
        .shared
        .frame_resources
        .copy_scene_depth_snapshot_for_view(
            frame.view.view_id,
            encoder,
            frame.view.depth_texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    if let (Some(profiler), Some(query)) = (profiler, copy_query) {
        profiler.end_query(encoder, query);
    }
    copied
}

/// Returns whether the scene-depth snapshot copy should be recorded for this view.
fn depth_snapshot_recording_needed(helper_needs: WorldMeshHelperNeeds) -> bool {
    helper_needs.depth_snapshot
}

#[cfg(test)]
mod tests {
    use super::depth_snapshot_recording_needed;
    use crate::world_mesh::WorldMeshHelperNeeds;

    #[test]
    fn depth_snapshot_recording_follows_helper_needs() {
        assert!(!depth_snapshot_recording_needed(WorldMeshHelperNeeds {
            depth_snapshot: false,
            color_snapshot: true,
        }));
        assert!(depth_snapshot_recording_needed(WorldMeshHelperNeeds {
            depth_snapshot: true,
            color_snapshot: false,
        }));
    }
}
