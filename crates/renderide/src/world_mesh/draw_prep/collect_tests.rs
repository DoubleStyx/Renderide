//! Tests for world-mesh draw collection helpers.

use glam::{Mat4, Quat, Vec3};

use super::*;
use crate::gpu_pools::MeshPool;
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{RenderTransform, RenderingContext};

/// Builds a unit-scale transform for draw-prep tests.
fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Minimal prepared draw used to exercise transform-scale filtering before mesh lookup.
fn prepared_draw(space_id: RenderSpaceId) -> FramePreparedDraw {
    FramePreparedDraw {
        space_id,
        renderable_index: 0,
        instance_id: MeshRendererInstanceId(11),
        node_id: 0,
        mesh_asset_id: 7,
        is_overlay: false,
        sorting_order: 0,
        skinned: false,
        world_space_deformed: false,
        blendshape_deformed: false,
        slot_index: 0,
        first_index: 0,
        index_count: 3,
        material_asset_id: 9,
        property_block_id: None,
    }
}

/// Prepared collection can collapse material-slot runs from the same source renderer.
#[test]
fn prepared_draws_share_renderer_groups_material_slots_only() {
    let space_id = RenderSpaceId(27);
    let first_slot = prepared_draw(space_id);
    let mut second_slot = prepared_draw(space_id);
    second_slot.slot_index = 1;
    second_slot.first_index = 3;
    second_slot.material_asset_id = 10;
    let mut next_renderer = second_slot.clone();
    next_renderer.renderable_index = 1;
    next_renderer.instance_id = MeshRendererInstanceId(12);

    assert!(prepared_draws_share_renderer(&first_slot, &second_slot));
    assert!(!prepared_draws_share_renderer(&second_slot, &next_renderer));
}

/// Unit-scale renderer nodes remain eligible for draw collection.
#[test]
fn draw_transform_scale_filter_allows_unit_scale() {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(28);
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);

    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
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

    assert!(!transform_chain_has_degenerate_scale(&ctx, space_id, 0));
}
