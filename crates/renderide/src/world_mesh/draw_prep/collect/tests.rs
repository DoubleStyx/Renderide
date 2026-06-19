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
use crate::shared::{LayerType, RenderTransform, RenderingContext, ShadowCastMode};
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

struct TestDrawContextResources<'a> {
    scene: &'a SceneCoordinator,
    mesh_pool: &'a MeshPool,
    material_dict: &'a MaterialDictionary<'a>,
    router: &'a MaterialRouter,
    property_ids: &'a MaterialPipelinePropertyIds,
}

/// Builds a draw-collection context for CPU-only draw-prep tests.
fn test_draw_context<'a>(
    resources: TestDrawContextResources<'a>,
    transform_filter: Option<&'a CameraTransformDrawFilter>,
    render_space_scope: ViewRenderSpaceScope,
    layer_policy: ViewLayerPolicy,
) -> DrawCollectionInputs<'a> {
    DrawCollectionInputs {
        scene_assets: DrawCollectionSceneAssets {
            scene: resources.scene,
            mesh_pool: resources.mesh_pool,
        },
        materials: DrawCollectionMaterialInputs {
            dict: resources.material_dict,
            router: resources.router,
            pipeline_property_ids: resources.property_ids,
            shader_perm: ShaderPermutation::default(),
        },
        view: DrawCollectionViewInputs {
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
            view_origin_world: Vec3::ZERO,
            culling: None,
            mesh_lod_bias: 2.0,
            transform_filter,
            transform_filter_space: render_space_scope.single_space(),
            render_space_scope,
            layer_policy,
            reflection_probes: None,
        },
        caches: DrawCollectionFrameCaches {
            material_cache: None,
            prepared: None,
        },
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
    let ctx = test_draw_context(
        TestDrawContextResources {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            router: &router,
            property_ids: &property_ids,
        },
        None,
        ViewRenderSpaceScope::AllActive,
        ViewLayerPolicy::MainView,
    );

    transform_chain_has_degenerate_scale(&ctx, space_id, 0)
}

/// Evaluates the special-layer view policy for one optional camera transform filter.
fn special_layer_visibility_for_filter(
    filter: Option<&CameraTransformDrawFilter>,
    layer_policy: ViewLayerPolicy,
    special_layer: Option<LayerType>,
) -> bool {
    let scene = SceneCoordinator::new();
    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let ctx = test_draw_context(
        TestDrawContextResources {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            router: &router,
            property_ids: &property_ids,
        },
        filter,
        ViewRenderSpaceScope::AllActive,
        layer_policy,
    );
    special_layer_visible_in_view(&ctx, special_layer)
}

/// Evaluates whether a private render space is visible under one draw context.
fn private_space_visibility_for_filter(
    filter: Option<&CameraTransformDrawFilter>,
    layer_policy: ViewLayerPolicy,
) -> bool {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(29);
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    scene.test_set_space_private(space_id, true);
    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let ctx = test_draw_context(
        TestDrawContextResources {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            router: &router,
            property_ids: &property_ids,
        },
        filter,
        ViewRenderSpaceScope::single(space_id),
        layer_policy,
    );
    render_space_visible_in_view(&ctx, space_id)
}

/// Evaluates private-space visibility for an all-active camera view.
fn private_space_visibility_for_all_active(layer_policy: ViewLayerPolicy) -> bool {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(30);
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    scene.test_set_space_private(space_id, true);
    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let ctx = test_draw_context(
        TestDrawContextResources {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            router: &router,
            property_ids: &property_ids,
        },
        None,
        ViewRenderSpaceScope::AllActive,
        layer_policy,
    );
    render_space_visible_in_view(&ctx, space_id)
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
        shadow_cast_mode: ShadowCastMode::On,
        skinned: false,
        world_space_deformed: false,
        blendshape_deformed: false,
        tangent_blendshape_deform_active: false,
        slot_index: 0,
        material_stack_order: None,
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
fn camera_layer_policy_hides_hidden_and_overlay_without_selective_roots() {
    assert!(!special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::camera(false),
        Some(LayerType::Hidden)
    ));
    assert!(!special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::camera(false),
        Some(LayerType::Overlay)
    ));
}

#[test]
fn camera_layer_policy_shows_hidden_and_overlay_for_selective_roots() {
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

    assert!(!special_layer_visibility_for_filter(
        Some(&exclude_only),
        ViewLayerPolicy::camera(false),
        Some(LayerType::Hidden)
    ));
    assert!(!special_layer_visibility_for_filter(
        Some(&empty_selective),
        ViewLayerPolicy::camera(false),
        Some(LayerType::Hidden)
    ));
    assert!(special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::camera(false),
        Some(LayerType::Hidden)
    ));
    assert!(special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::camera(false),
        Some(LayerType::Overlay)
    ));
}

#[test]
fn main_view_layer_policy_excludes_special_layers_unless_selective() {
    let selective = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };

    assert!(!special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::MainView,
        Some(LayerType::Hidden)
    ));
    assert!(!special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::MainView,
        Some(LayerType::Overlay)
    ));
    assert!(special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::MainView,
        Some(LayerType::Hidden)
    ));
    assert!(special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::MainView,
        Some(LayerType::Overlay)
    ));
}

#[test]
fn desktop_overlay_layer_policy_includes_only_overlay_layer() {
    assert!(special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::DesktopOverlay,
        Some(LayerType::Overlay)
    ));
    assert!(!special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::DesktopOverlay,
        Some(LayerType::Hidden)
    ));
    assert!(!special_layer_visibility_for_filter(
        None,
        ViewLayerPolicy::DesktopOverlay,
        None
    ));
}

#[test]
fn desktop_overlay_layer_policy_ignores_selective_camera_roots() {
    let selective = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };

    assert!(special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::DesktopOverlay,
        Some(LayerType::Overlay)
    ));
    assert!(!special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::DesktopOverlay,
        Some(LayerType::Hidden)
    ));
    assert!(!special_layer_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::DesktopOverlay,
        None
    ));
}

#[test]
fn selected_camera_overlay_renders_as_non_overlay() {
    let scene = SceneCoordinator::new();
    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let filter = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([1])),
        exclude: HashSet::new(),
    };
    let ctx = test_draw_context(
        TestDrawContextResources {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            router: &router,
            property_ids: &property_ids,
        },
        Some(&filter),
        ViewRenderSpaceScope::AllActive,
        ViewLayerPolicy::camera(false),
    );

    assert!(!effective_overlay_in_view(&ctx, true));
}

#[test]
fn camera_layer_policy_filters_private_spaces_without_private_ui() {
    let selective = CameraTransformDrawFilter {
        only: Some(HashSet::from_iter([0])),
        exclude: HashSet::new(),
    };

    assert!(!private_space_visibility_for_filter(
        None,
        ViewLayerPolicy::camera(false)
    ));
    assert!(private_space_visibility_for_filter(
        None,
        ViewLayerPolicy::camera(true)
    ));
    assert!(private_space_visibility_for_filter(
        Some(&selective),
        ViewLayerPolicy::camera(false)
    ));
}

#[test]
fn all_active_camera_scope_respects_render_private_ui_for_private_spaces() {
    assert!(!private_space_visibility_for_all_active(
        ViewLayerPolicy::camera(false)
    ));
    assert!(private_space_visibility_for_all_active(
        ViewLayerPolicy::camera(true)
    ));
}

#[test]
fn transform_filter_masks_are_source_space_bound() {
    let mut scene = SceneCoordinator::new();
    let source = RenderSpaceId(31);
    let other = RenderSpaceId(32);
    scene.test_seed_space_identity_worlds(
        source,
        vec![identity_transform(), identity_transform()],
        vec![-1, 0],
    );
    scene.test_seed_space_identity_worlds(
        other,
        vec![identity_transform(), identity_transform()],
        vec![-1, 0],
    );
    let mesh_pool = MeshPool::default_pool();
    let store = MaterialPropertyStore::new();
    let material_dict = MaterialDictionary::new(&store);
    let router = MaterialRouter::new(RasterPipelineKind::Null);
    let registry = PropertyIdRegistry::new();
    let property_ids = MaterialPipelinePropertyIds::new(&registry);
    let exclude_only = CameraTransformDrawFilter {
        only: None,
        exclude: HashSet::from_iter([1]),
    };
    let mut ctx = test_draw_context(
        TestDrawContextResources {
            scene: &scene,
            mesh_pool: &mesh_pool,
            material_dict: &material_dict,
            router: &router,
            property_ids: &property_ids,
        },
        Some(&exclude_only),
        ViewRenderSpaceScope::AllActive,
        ViewLayerPolicy::camera(false),
    );
    ctx.view.transform_filter_space = Some(source);

    let masks = build_per_space_filter_masks(&[source, other], &ctx);

    assert!(masks.contains_key(&source));
    assert!(!masks.contains_key(&other));
    assert!(transform_filter_for_space(&ctx, source).is_some());
    assert!(transform_filter_for_space(&ctx, other).is_none());
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
