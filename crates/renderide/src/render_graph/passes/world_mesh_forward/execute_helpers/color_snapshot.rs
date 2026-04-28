//! Scene-color snapshot helper for the graph-managed world-mesh forward pass.

use crate::render_graph::context::GraphResolvedResources;
use crate::render_graph::frame_params::{FrameRenderParams, PreparedWorldMeshForwardFrame};

use super::super::WorldMeshForwardGraphResources;

/// Copies the resolved HDR scene color into the sampled scene-color snapshot used by grab-pass
/// transparent materials.
pub(crate) fn encode_world_mesh_forward_color_snapshot(
    graph_resources: Option<&GraphResolvedResources>,
    encoder: &mut wgpu::CommandEncoder,
    frame: &mut FrameRenderParams<'_>,
    prepared: &PreparedWorldMeshForwardFrame,
    resources: WorldMeshForwardGraphResources,
) -> bool {
    if prepared.plan.transparent_groups.is_empty() {
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
