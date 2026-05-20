//! Scene-color snapshot helper for the world-mesh transparent sequence.

use crate::graph_inputs::GraphPassFrame;
use crate::render_graph::context::GraphResolvedResources;
use crate::world_mesh::WorldMeshHelperNeeds;

use super::PreparedWorldMeshForwardFrame;
use crate::profiling::GpuProfilerHandle;

use super::WorldMeshForwardGraphResources;

/// Copies the resolved HDR scene color into the sampled scene-color snapshot used by the next
/// grab-pass material group.
pub(crate) fn encode_world_mesh_forward_color_snapshot(
    graph_resources: &GraphResolvedResources,
    encoder: &mut wgpu::CommandEncoder,
    frame: &GraphPassFrame<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
    profiler: Option<&GpuProfilerHandle>,
) -> bool {
    profiling::scope!("world_mesh_forward::encode_color_snapshot");
    if !color_snapshot_recording_needed(prepared.helper_needs) {
        logger::warn!(
            "world mesh color snapshot copy: helper needs did not request a color snapshot"
        );
        return false;
    }
    if !frame.shared.frame_resources.has_frame_gpu() {
        logger::warn!("world mesh color snapshot copy: frame GPU resources are unavailable");
        return false;
    }
    let Some(source_color) = graph_resources.transient_texture(resources.scene_color_hdr) else {
        logger::warn!("world mesh color snapshot copy: resolved scene color source is unavailable");
        return false;
    };
    let copy_query =
        profiler.map(|p| p.begin_query("world_mesh_forward::scene_color_snapshot_copy", encoder));
    let copied = frame
        .shared
        .frame_resources
        .copy_scene_color_snapshot_for_view(
            frame.view.view_id,
            encoder,
            &source_color.texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    if let (Some(profiler), Some(query)) = (profiler, copy_query) {
        profiler.end_query(encoder, query);
    }
    copied
}

/// Returns whether the scene-color snapshot copy should be recorded for this view.
fn color_snapshot_recording_needed(helper_needs: WorldMeshHelperNeeds) -> bool {
    helper_needs.color_snapshot
}

#[cfg(test)]
mod tests {
    use crate::world_mesh::WorldMeshHelperNeeds;

    use super::color_snapshot_recording_needed;

    #[test]
    fn color_snapshot_recording_follows_helper_needs() {
        assert!(!color_snapshot_recording_needed(WorldMeshHelperNeeds {
            depth_snapshot: true,
            color_snapshot: false,
        }));
        assert!(color_snapshot_recording_needed(WorldMeshHelperNeeds {
            depth_snapshot: false,
            color_snapshot: true,
        }));
    }
}
