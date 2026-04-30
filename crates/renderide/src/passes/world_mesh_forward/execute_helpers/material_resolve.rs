//! Material draw-packet resolution entry point for the world-mesh forward prepare pass.

use crate::backend::WorldMeshForwardEncodeRefs;
use crate::materials::MaterialPipelineDesc;
use crate::materials::ShaderPermutation;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

use super::super::{MaterialBatchPacket, MaterialDrawResolver};

/// Resolves per-batch pipeline sets and `@group(1)` bind groups for the sorted draw list.
///
/// This wrapper keeps the forward-pass helper boundary stable while the concrete abstraction
/// lives with draw prep. Both graph pre-warm and record-time resolution now use
/// [`super::super::PipelineVariantKey`], which prevents the two
/// paths from drifting on grab-pass MSAA, front-face, blend, render-state, or shader permutation.
pub(super) fn precompute_material_resolve_batches(
    encode: &WorldMeshForwardEncodeRefs<'_>,
    queue: &wgpu::Queue,
    draws: &[WorldMeshDrawItem],
    shader_perm: ShaderPermutation,
    pass_desc: &MaterialPipelineDesc,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> Vec<MaterialBatchPacket> {
    MaterialDrawResolver::new(
        encode,
        queue,
        *pass_desc,
        shader_perm,
        offscreen_write_render_texture_asset_id,
    )
    .resolve_batches(draws)
}
