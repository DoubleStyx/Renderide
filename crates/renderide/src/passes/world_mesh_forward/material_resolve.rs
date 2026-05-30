//! Material draw-packet resolution entry point for backend world-mesh frame planning.

use crate::graph_inputs::OffscreenWriteTarget;
use crate::materials::MaterialPipelineDesc;
use crate::materials::ShaderPermutation;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::{MaterialBatchBoundary, MaterialBatchPacket, MaterialDrawResolver};

/// Resolves per-batch pipeline sets and `@group(1)` bind groups for the sorted draw list.
///
/// This wrapper keeps the forward-pass helper boundary stable while the concrete abstraction
/// lives with draw prep. Backend frame planning uses the material batch pipeline key so prepared
/// packets cannot drift on grab-pass MSAA, front-face, blend, render-state, or shader permutation.
pub(super) fn precompute_material_resolve_batches(
    encode: &WorldMeshForwardEncodeRefs<'_>,
    uploads: GraphUploadSink<'_>,
    draws: &[WorldMeshDrawItem],
    shader_perm: ShaderPermutation,
    pass_desc: &MaterialPipelineDesc,
    offscreen_write_target: OffscreenWriteTarget,
    boundaries_scratch: &mut Vec<MaterialBatchBoundary>,
) -> Vec<MaterialBatchPacket> {
    MaterialDrawResolver::new(
        encode,
        uploads,
        *pass_desc,
        shader_perm,
        offscreen_write_target,
    )
    .resolve_batches(draws, boundaries_scratch)
}
