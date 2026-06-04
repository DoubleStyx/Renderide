//! Tests for world-mesh draw collection helpers.

use glam::{Mat4, Quat, Vec3};
use hashbrown::HashSet;

use super::world_matrix::front_face_for_draw_matrices;
use super::*;
use crate::cpu_parallelism::{ParallelAdmission, RENDER_COMMAND_CHUNK_DRAWS};
use crate::gpu_pools::MeshPool;
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry};
use crate::materials::{
    MaterialPipelinePropertyIds, MaterialRouter, RasterFrontFace, RasterPipelineKind,
};
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{RenderTransform, RenderingContext};
use crate::world_mesh::CameraTransformDrawFilter;

/// Builds a unit-scale transform for draw-prep tests.
fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Builds an identity transform with the requested scale.
fn scaled_transform(scale: Vec3) -> RenderTransform {
    RenderTransform {
        scale,
        ..identity_transform()
    }
}

/// Evaluates the draw transform-scale filter for one root node.
fn transform_scale_filter_result(scale: Vec3) -> bool {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(28);
    scene.test_seed_space_identity_worlds(space_id, vec![scaled_transform(scale)], vec![-1]);

    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let ctx = DrawCollectionInputs {
        scene_assets: DrawCollectionSceneAssets {
            scene: &scene,
            mesh_pool: &mesh_pool,
        },
        materials: DrawCollectionMaterialInputs {
            dict: &material_dict,
            router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: ShaderPermutation::default(),
        },
        view: DrawCollectionViewInputs {
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            mesh_lod_bias: 2.0,
            transform_filter: None,
            render_space_filter: None,
            reflection_probes: None,
        },
        caches: DrawCollectionFrameCaches {
            material_cache: None,
            prepared: None,
        },
    };

    transform_chain_has_degenerate_scale(&ctx, space_id, 0)
}

/// Evaluates the Hidden-layer view policy for one optional camera transform filter.
fn hidden_visibility_for_filter(filter: Option<&CameraTransformDrawFilter>) -> bool {
    let scene = SceneCoordinator::new();
    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let ctx = DrawCollectionInputs {
        scene_assets: DrawCollectionSceneAssets {
            scene: &scene,
            mesh_pool: &mesh_pool,
        },
        materials: DrawCollectionMaterialInputs {
            dict: &material_dict,
            router: &router,
            pipeline_property_ids: &property_ids,
            shader_perm: ShaderPermutation::default(),
        },
        view: DrawCollectionViewInputs {
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            mesh_lod_bias: 2.0,
            transform_filter: filter,
            render_space_filter: None,
            reflection_probes: None,
        },
        caches: DrawCollectionFrameCaches {
            material_cache: None,
            prepared: None,
        },
    };
    hidden_layers_visible_in_view(&ctx)
}

/// Minimal prepared draw used to exercise transform-scale filtering before mesh lookup.
fn prepared_draw(space_id: RenderSpaceId) -> FramePreparedDraw {
    FramePreparedDraw {
        space_id,
        renderable_index: 0,
        instance_id: MeshRendererInstanceId(11),
        renderer_ordinal: 0,
        node_id: 0,
        mesh_asset_id: 7,
        is_overlay: false,
        is_hidden: false,
        sorting_order: 0,
        skinned: false,
        world_space_deformed: false,
        blendshape_deformed: false,
        tangent_blendshape_deform_active: false,
        slot_index: 0,
        first_index: 0,
        index_count: 3,
        material_asset_id: 9,
        property_block_id: None,
        cull_geometry: None,
        rigid_world_matrix_override: None,
        particle_draw: crate::particles::ParticleDrawParams::default(),
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
    let mut hidden_slot = second_slot.clone();
    hidden_slot.is_hidden = true;

    assert!(prepared_draws_share_renderer(&first_slot, &second_slot));
    assert!(!prepared_draws_share_renderer(&second_slot, &hidden_slot));
    assert!(!prepared_draws_share_renderer(&second_slot, &next_renderer));
}

#[test]
fn world_space_deformed_front_face_uses_deform_root_parity() {
    let mirrored = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));

    assert_eq!(
        front_face_for_draw_matrices(true, None, Some(mirrored)),
        RasterFrontFace::CounterClockwise
    );
    assert_eq!(
        front_face_for_draw_matrices(false, None, Some(mirrored)),
        RasterFrontFace::Clockwise
    );
}

/// Unit-scale renderer nodes remain eligible for draw collection.
#[test]
fn draw_transform_scale_filter_allows_unit_scale() {
    assert!(!transform_scale_filter_result(Vec3::ONE));
}

/// Planar zero-scale renderer nodes remain eligible for draw collection.
#[test]
fn draw_transform_scale_filter_allows_planar_zero_scale() {
    assert!(!transform_scale_filter_result(Vec3::new(1.0, 0.0, 1.0)));
}

/// Line-scale renderer nodes are not eligible for triangle draw collection.
#[test]
fn draw_transform_scale_filter_rejects_line_scale() {
    assert!(transform_scale_filter_result(Vec3::new(1.0, 0.0, 0.0)));
}

/// Point-scale renderer nodes are not eligible for triangle draw collection.
#[test]
fn draw_transform_scale_filter_rejects_point_scale() {
    assert!(transform_scale_filter_result(Vec3::ZERO));
}

#[test]
fn hidden_layer_visibility_requires_non_empty_selective_filter() {
    let exclude_only = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::from_iter([1]),
    };
    let empty_selective = CameraTransformDrawFilter {
        only: Some(HashSet::new()),
        exclude: HashSet::new(),
    };
    let selective = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };

    assert!(!hidden_visibility_for_filter(None));
    assert!(!hidden_visibility_for_filter(Some(&exclude_only)));
    assert!(!hidden_visibility_for_filter(Some(&empty_selective)));
    assert!(hidden_visibility_for_filter(Some(&selective)));
}

#[test]
fn prepared_collect_parallelism_requires_draw_heavy_work_and_multiple_tasks() {
    let threshold = RENDER_COMMAND_CHUNK_DRAWS * 2;

    assert_eq!(
        prepared_collect_admission(2, threshold - 1, 2),
        ParallelAdmission::Serial
    );
    assert_eq!(
        prepared_collect_admission(1, threshold, 2),
        ParallelAdmission::Serial
    );
    assert!(prepared_collect_admission(2, threshold, 2).is_parallel());
    assert_eq!(
        prepared_collect_admission(2, threshold, 1),
        ParallelAdmission::Serial
    );
}
