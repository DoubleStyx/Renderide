//! Material draw-packet resolution entry point for backend world-mesh frame planning.

use crate::frame_upload_batch::GraphUploadSink;
use crate::graph_inputs::OffscreenWriteTarget;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::{
    MaterialBatchBoundary, MaterialBatchPacket, MaterialDrawResolver, WorldMeshForwardPipelineState,
};

/// Resolves per-batch pipeline sets and `@group(1)` bind groups for the sorted draw list.
///
/// This wrapper keeps the forward-pass helper boundary stable while the concrete abstraction
/// lives with draw prep. Backend frame planning uses the material batch pipeline key so prepared
/// packets cannot drift on grab-pass MSAA, front-face, blend, render-state, or shader permutation.
pub(super) fn precompute_material_resolve_batches(
    encode: &WorldMeshForwardEncodeRefs<'_>,
    uploads: GraphUploadSink<'_>,
    draws: &[WorldMeshDrawItem],
    pipeline: &WorldMeshForwardPipelineState,
    offscreen_write_target: OffscreenWriteTarget,
    boundaries_scratch: &mut Vec<MaterialBatchBoundary>,
) -> Vec<MaterialBatchPacket> {
    MaterialDrawResolver::new(
        encode,
        uploads,
        pipeline.pass_desc,
        pipeline.shader_perm,
        offscreen_write_target,
        pipeline.front_face_flip,
    )
    .resolve_batches(draws, boundaries_scratch)
}
