//! Shared draw-candidate evaluation for world-mesh collection.

use super::*;

/// View-local material-slot draw candidate shared by scene-walk and prepared collection.
pub(super) struct DrawCandidate {
    /// Render space containing the source renderer.
    pub(super) space_id: RenderSpaceId,
    /// Scene node id used for transform and filter decisions.
    pub(super) node_id: i32,
    /// Mesh asset id referenced by the source renderer.
    pub(super) mesh_asset_id: i32,
    /// Material slot index within the source renderer.
    pub(super) slot_index: usize,
    /// First index in the mesh index buffer.
    pub(super) first_index: u32,
    /// Number of indices emitted by the draw.
    pub(super) index_count: u32,
    /// Overlay layer flag copied into cull and draw metadata.
    pub(super) is_overlay: bool,
    /// Renderer sorting order copied into transparent ordering.
    pub(super) sorting_order: i32,
    /// Whether this draw uses skinned vertex streams.
    pub(super) skinned: bool,
    /// Whether skinning writes world-space positions.
    pub(super) world_space_deformed: bool,
    /// Material asset after render-context override resolution.
    pub(super) material_asset_id: i32,
    /// Property block associated with material slot zero.
    pub(super) property_block_id: Option<i32>,
}

/// Builds a draw item from a cull-surviving material-slot candidate without allocating.
pub(super) fn evaluate_draw_candidate(
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    candidate: DrawCandidate,
    front_face: RasterFrontFace,
    rigid_world_matrix: Option<Mat4>,
    alpha_distance_sq: f32,
) -> Option<WorldMeshDrawItem> {
    if candidate.index_count == 0 || candidate.material_asset_id < 0 {
        return None;
    }
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id: candidate.material_asset_id,
        mesh_property_block_slot0: candidate.property_block_id,
    };
    let batch_key = batch_key_for_slot_cached(
        candidate.material_asset_id,
        candidate.property_block_id,
        candidate.skinned,
        front_face,
        cache,
        MaterialResolveCtx {
            dict: ctx.material_dict,
            router: ctx.material_router,
            pipeline_property_ids: ctx.pipeline_property_ids,
            shader_perm: ctx.shader_perm,
        },
    );
    let camera_distance_sq = if batch_key.alpha_blended {
        alpha_distance_sq
    } else {
        0.0
    };
    Some(WorldMeshDrawItem {
        space_id: candidate.space_id,
        node_id: candidate.node_id,
        mesh_asset_id: candidate.mesh_asset_id,
        slot_index: candidate.slot_index,
        first_index: candidate.first_index,
        index_count: candidate.index_count,
        is_overlay: candidate.is_overlay,
        sorting_order: candidate.sorting_order,
        skinned: candidate.skinned,
        world_space_deformed: candidate.world_space_deformed,
        collect_order: 0,
        camera_distance_sq,
        lookup_ids,
        batch_key,
        rigid_world_matrix,
    })
}
