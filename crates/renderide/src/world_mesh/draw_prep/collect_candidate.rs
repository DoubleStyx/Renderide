//! Shared draw-candidate evaluation for world-mesh collection.

use super::*;
use crate::scene::MeshRendererInstanceId;

/// View-local material-slot draw candidate shared by scene-walk and prepared collection.
pub(super) struct DrawCandidate {
    /// Render space containing the source renderer.
    pub(super) space_id: RenderSpaceId,
    /// Scene node id used for transform and filter decisions.
    pub(super) node_id: i32,
    /// Dense renderer index inside the static or skinned renderer table selected by [`Self::skinned`].
    pub(super) renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    pub(super) instance_id: MeshRendererInstanceId,
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
    /// Whether blendshape scatter writes cache-backed positions for this draw.
    pub(super) blendshape_deformed: bool,
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
    let batch_key_hash = super::super::types::compute_batch_key_hash(&batch_key);
    Some(WorldMeshDrawItem {
        space_id: candidate.space_id,
        node_id: candidate.node_id,
        renderable_index: candidate.renderable_index,
        instance_id: candidate.instance_id,
        mesh_asset_id: candidate.mesh_asset_id,
        slot_index: candidate.slot_index,
        first_index: candidate.first_index,
        index_count: candidate.index_count,
        is_overlay: candidate.is_overlay,
        sorting_order: candidate.sorting_order,
        skinned: candidate.skinned,
        world_space_deformed: candidate.world_space_deformed,
        blendshape_deformed: candidate.blendshape_deformed,
        collect_order: 0,
        camera_distance_sq,
        lookup_ids,
        batch_key,
        batch_key_hash,
        rigid_world_matrix,
    })
}

#[cfg(test)]
mod tests {
    //! CPU-only draw-candidate identity tests.

    use glam::{Mat4, Vec3};

    use super::*;
    use crate::assets::material::{MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry};
    use crate::gpu_pools::MeshPool;
    use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderingContext;

    #[test]
    fn evaluate_draw_candidate_preserves_renderer_identity_separate_from_node_id() {
        let scene = SceneCoordinator::new();
        let mesh_pool = MeshPool::default_pool();
        let store = MaterialPropertyStore::new();
        let material_dict = MaterialDictionary::new(&store);
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let registry = PropertyIdRegistry::new();
        let property_ids = MaterialPipelinePropertyIds::new(&registry);
        let cache = FrameMaterialBatchCache::new();
        let ctx = DrawCollectionContext {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            material_router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: ShaderPermutation::default(),
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            transform_filter: None,
            material_cache: None,
            prepared: None,
        };
        let candidate = DrawCandidate {
            space_id: RenderSpaceId(3),
            node_id: 9,
            renderable_index: 42,
            instance_id: MeshRendererInstanceId(99),
            mesh_asset_id: 7,
            slot_index: 0,
            first_index: 0,
            index_count: 3,
            is_overlay: false,
            sorting_order: 0,
            skinned: true,
            world_space_deformed: true,
            blendshape_deformed: true,
            material_asset_id: 11,
            property_block_id: None,
        };

        let item = evaluate_draw_candidate(
            &ctx,
            &cache,
            candidate,
            RasterFrontFace::Clockwise,
            None,
            0.0,
        )
        .expect("draw item");

        assert_eq!(item.node_id, 9);
        assert_eq!(item.renderable_index, 42);
        assert_eq!(item.instance_id, MeshRendererInstanceId(99));
    }
}
