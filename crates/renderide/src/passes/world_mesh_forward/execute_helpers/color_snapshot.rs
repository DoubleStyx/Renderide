//! Scene-color snapshot helper for the graph-managed world-mesh forward pass.

use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::frame_params::FrameRenderParams;
use crate::world_mesh::WorldMeshHelperNeeds;

use super::super::PreparedWorldMeshForwardFrame;

use super::super::WorldMeshForwardGraphResources;

/// Copies the resolved HDR scene color into the sampled scene-color snapshot used by grab-pass
/// transparent materials.
pub(crate) fn encode_world_mesh_forward_color_snapshot(
    graph_resources: Option<&GraphResolvedResources>,
    encoder: &mut wgpu::CommandEncoder,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> bool {
    if !color_snapshot_recording_needed(prepared.helper_needs) {
        return false;
    }
    if frame.shared.frame_resources.frame_gpu().is_none() {
        return false;
    }
    let Some(source_color) =
        graph_resources.and_then(|graph| graph.transient_texture(resources.scene_color_hdr))
    else {
        return false;
    };
    frame
        .shared
        .frame_resources
        .copy_scene_color_snapshot_for_view(
            frame.view.view_id,
            encoder,
            &source_color.texture,
            frame.view.viewport_px,
            prepared.pipeline.use_multiview,
        );
    true
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
