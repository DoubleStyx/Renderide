//! Tests for the parent module.

use super::cluster_layout::{
    ClusterPreRecordLayout, cluster_index_capacity_for_layout, per_view_snapshot_sync_params,
    unique_cluster_pre_record_layouts,
};
use super::*;

use glam::{Mat4, Quat, Vec3};

use crate::camera::{HostCameraFrame, ViewId};
use crate::graph_inputs::PreRecordViewResourceLayout;
use crate::mesh_deform::{SkinCacheKey, SkinCacheRendererKind};
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{
    LightData, LightType, LightsBufferRendererState, RenderTransform, RenderingContext, ShadowType,
};
use crate::world_mesh::WorldMeshCullProjParams;

/// Returns an identity host transform for scene fixtures.
fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Builds a pre-record layout for pure frame-resource planning tests.
fn pre_record_layout(
    width: u32,
    height: u32,
    stereo: bool,
    needs_depth_snapshot: bool,
    needs_color_snapshot: bool,
) -> PreRecordViewResourceLayout {
    PreRecordViewResourceLayout {
        view_id: ViewId::Main,
        width,
        height,
        stereo,
        sample_count: 1,
        depth_format: wgpu::TextureFormat::Depth32Float,
        color_format: wgpu::TextureFormat::Rgba16Float,
        needs_depth_snapshot,
        needs_color_snapshot,
    }
}

#[test]
fn new_manager_has_no_per_view_draw() {
    let mgr = FrameResourceManager::new();
    let secondary = ViewId::secondary_camera(RenderSpaceId(42), 0);
    assert!(mgr.per_view_per_draw(ViewId::Main).is_none());
    assert!(mgr.per_view_per_draw(secondary).is_none());
}

#[test]
fn new_manager_has_no_per_view_frame() {
    let mgr = FrameResourceManager::new();
    let secondary = ViewId::secondary_camera(RenderSpaceId(42), 0);
    assert!(mgr.per_view_frame(ViewId::Main).is_none());
    assert!(mgr.per_view_frame(secondary).is_none());
}

#[test]
fn mesh_deform_submission_replaces_visible_filter_and_clears_dispatch_flag() {
    let first = SkinCacheKey::new(
        RenderSpaceId(1),
        SkinCacheRendererKind::Skinned,
        MeshRendererInstanceId(10),
    );
    let second = SkinCacheKey::new(
        RenderSpaceId(2),
        SkinCacheRendererKind::Skinned,
        MeshRendererInstanceId(20),
    );
    let mut mgr = FrameResourceManager::new();

    mgr.begin_mesh_deform_submission(hashbrown::HashSet::from_iter([first]));
    mgr.set_mesh_deform_dispatched_this_submission();
    assert!(mgr.mesh_deform_dispatched_this_submission());

    mgr.begin_mesh_deform_submission(hashbrown::HashSet::from_iter([second]));

    assert!(!mgr.mesh_deform_dispatched_this_submission());
    let visible = mgr
        .visible_mesh_deform_keys_snapshot()
        .expect("visible deform filter");
    assert_eq!(visible.len(), 1);
    assert!(visible.contains(&second));
    assert!(!visible.contains(&first));
}

/// Shared pre-record work deduplicates only the cluster allocation shape, not snapshot needs.
#[test]
fn cluster_pre_record_layouts_ignore_snapshot_fields() {
    let dashboard = pre_record_layout(512, 256, false, false, true);
    let mut dashboard_depth = pre_record_layout(512, 256, false, true, false);
    dashboard_depth.view_id = ViewId::secondary_camera(RenderSpaceId(7), 0);
    let main = pre_record_layout(1920, 1080, false, false, false);
    let light_count = 3;
    let dashboard_depth_light_count = 7;

    let layouts =
        unique_cluster_pre_record_layouts(&[dashboard, dashboard_depth, main], |view_id| {
            if view_id == dashboard_depth.view_id {
                dashboard_depth_light_count
            } else {
                light_count
            }
        });

    assert_eq!(
        layouts,
        vec![
            ClusterPreRecordLayout {
                width: 512,
                height: 256,
                stereo: false,
                index_capacity_words: cluster_index_capacity_for_layout(
                    dashboard_depth,
                    dashboard_depth_light_count,
                )
                .unwrap(),
            },
            ClusterPreRecordLayout {
                width: 1920,
                height: 1080,
                stereo: false,
                index_capacity_words: cluster_index_capacity_for_layout(main, light_count).unwrap(),
            },
        ]
    );
}

/// Snapshot sync requests stay per-view, so an unrelated view cannot become the grab winner.
#[test]
fn per_view_snapshot_sync_params_preserve_grab_need_per_view() {
    let dashboard = pre_record_layout(512, 256, false, false, true);
    let main = pre_record_layout(1920, 1080, false, false, false);

    let dashboard_sync = per_view_snapshot_sync_params(dashboard);
    let main_sync = per_view_snapshot_sync_params(main);

    assert_eq!(dashboard_sync.viewport, (512, 256));
    assert!(dashboard_sync.needs_color_snapshot);
    assert!(!dashboard_sync.needs_depth_snapshot);
    assert_eq!(main_sync.viewport, (1920, 1080));
    assert!(!main_sync.needs_color_snapshot);
    assert!(!main_sync.needs_depth_snapshot);
}

#[test]
fn retire_nonexistent_is_noop() {
    let mut mgr = FrameResourceManager::new();
    let secondary = ViewId::secondary_camera(RenderSpaceId(99), 0);
    mgr.retire_per_view_per_draw(ViewId::Main);
    mgr.retire_per_view_per_draw(secondary);
    mgr.retire_per_view_frame(ViewId::Main);
    mgr.retire_per_view_frame(secondary);
    mgr.retire_view(secondary);
}

fn make_light_data_with_intensity(color_x: f32, intensity: f32) -> LightData {
    make_light_data_at_with_intensity(Vec3::ZERO, 10.0, color_x, intensity)
}

fn make_light_data_at_with_intensity(
    point: Vec3,
    range: f32,
    color_x: f32,
    intensity: f32,
) -> LightData {
    LightData {
        point,
        orientation: Quat::IDENTITY,
        color: Vec3::new(color_x, 0.0, 0.0),
        intensity,
        range,
        angle: 45.0,
    }
}

fn make_light_data(color_x: f32) -> LightData {
    make_light_data_with_intensity(color_x, 1.0)
}

fn make_light_data_at(point: Vec3, range: f32, color_x: f32) -> LightData {
    make_light_data_at_with_intensity(point, range, color_x, 1.0)
}

fn make_state(global_unique_id: i32) -> LightsBufferRendererState {
    LightsBufferRendererState {
        renderable_index: 0,
        global_unique_id,
        shadow_strength: 0.0,
        shadow_near_plane: 0.0,
        shadow_map_resolution: 0,
        shadow_bias: 0.0,
        shadow_normal_bias: 0.0,
        cookie_texture_asset_id: -1,
        light_type: LightType::Point,
        shadow_type: ShadowType::None,
        _padding: [0; 2],
    }
}

fn make_indexed_state(
    renderable_index: i32,
    global_unique_id: i32,
    light_type: LightType,
) -> LightsBufferRendererState {
    LightsBufferRendererState {
        renderable_index,
        light_type,
        ..make_state(global_unique_id)
    }
}

fn make_shadowed_state(global_unique_id: i32) -> LightsBufferRendererState {
    LightsBufferRendererState {
        shadow_strength: 0.75,
        shadow_near_plane: 0.25,
        shadow_bias: 0.01,
        shadow_normal_bias: 0.02,
        shadow_type: ShadowType::Soft,
        ..make_state(global_unique_id)
    }
}

fn seed_space_with_light(
    scene: &mut SceneCoordinator,
    space_id: RenderSpaceId,
    global_unique_id: i32,
    color_x: f32,
) {
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    let cache = scene.light_cache_mut();
    cache.store_full(global_unique_id, vec![make_light_data(color_x)]);
    cache.apply_update(space_id.0, &[], &[0], &[make_state(global_unique_id)]);
}

fn identity_light_cull_desc() -> FrameLightCullDesc {
    FrameLightCullDesc {
        host_camera: HostCameraFrame::default(),
        proj: WorldMeshCullProjParams {
            world_proj: Mat4::IDENTITY,
            overlay_proj: Mat4::IDENTITY,
            vr_stereo: None,
        },
    }
}

fn seed_space_with_signed_light(
    scene: &mut SceneCoordinator,
    space_id: RenderSpaceId,
    global_unique_id: i32,
    color_x: f32,
    intensity: f32,
) {
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    let cache = scene.light_cache_mut();
    cache.store_full(
        global_unique_id,
        vec![make_light_data_with_intensity(color_x, intensity)],
    );
    cache.apply_update(space_id.0, &[], &[0], &[make_state(global_unique_id)]);
}

#[test]
fn prepare_lights_from_scene_keeps_and_detects_negative_lights() {
    let mut scene = SceneCoordinator::new();
    seed_space_with_signed_light(&mut scene, RenderSpaceId(1), 100, 1.0, -2.0);
    seed_space_with_signed_light(&mut scene, RenderSpaceId(2), 200, -0.5, -2.0);

    let mut mgr = FrameResourceManager::new();
    mgr.prepare_lights_from_scene(&scene);

    assert_eq!(mgr.frame_lights().len(), 2);
    assert!(mgr.signed_scene_color_required());
}

/// Lights from inactive render spaces must not leak into the frame's GPU light buffer.
///
/// Regression: `prepare_lights_from_scene` used to iterate every tracked render space, so
/// after a world switch (host marks the old space `is_active = false` but keeps it resident)
/// its lights persisted into the new world's shading. Every other per-space pipeline
/// (renderables, deform, secondary cameras, material-batch cache) filters by `is_active`;
/// lights must follow the same rule.
#[test]
fn prepare_lights_from_scene_skips_inactive_spaces() {
    let mut scene = SceneCoordinator::new();
    let space_a = RenderSpaceId(1);
    let space_b = RenderSpaceId(2);
    seed_space_with_light(&mut scene, space_a, 100, 1.0);
    seed_space_with_light(&mut scene, space_b, 200, 0.5);

    // Both spaces active: both lights contribute.
    let mut mgr = FrameResourceManager::new();
    mgr.prepare_lights_from_scene(&scene);
    assert_eq!(mgr.frame_lights().len(), 2);

    // Focus space A only.
    scene.test_set_space_active(space_b, false);
    mgr.reset_light_prep_for_tick();
    mgr.prepare_lights_from_scene(&scene);
    let packed = mgr.frame_lights();
    assert_eq!(packed.len(), 1);
    assert!((packed[0].color[0] - 1.0).abs() < 1e-5);

    // Switch focus to space B; A's light must not carry over.
    scene.test_set_space_active(space_a, false);
    scene.test_set_space_active(space_b, true);
    mgr.reset_light_prep_for_tick();
    mgr.prepare_lights_from_scene(&scene);
    let packed = mgr.frame_lights();
    assert_eq!(packed.len(), 1);
    assert!((packed[0].color[0] - 0.5).abs() < 1e-5);
}

#[test]
fn prepare_lights_for_views_culls_light_volumes_before_packing() {
    let mut scene = SceneCoordinator::new();
    let space = RenderSpaceId(6);
    scene.test_seed_space_identity_worlds(
        space,
        vec![
            identity_transform(),
            identity_transform(),
            identity_transform(),
        ],
        vec![-1, -1, -1],
    );
    let cache = scene.light_cache_mut();
    cache.store_full(100, vec![make_light_data_at(Vec3::ZERO, 0.25, 1.0)]);
    cache.store_full(
        200,
        vec![make_light_data_at(Vec3::new(4.0, 0.0, 0.0), 0.25, 0.5)],
    );
    cache.store_full(
        300,
        vec![make_light_data_at(Vec3::new(100.0, 0.0, 0.0), 10.0, 0.25)],
    );
    cache.apply_update(
        space.0,
        &[],
        &[0, 1, 2],
        &[
            make_indexed_state(0, 100, LightType::Point),
            make_indexed_state(1, 200, LightType::Point),
            make_indexed_state(2, 300, LightType::Directional),
        ],
    );

    let main = ViewId::Main;
    let overlay = ViewId::MainOverlay;
    let desc = FrameLightViewDesc {
        view_id: main,
        render_context: RenderingContext::UserView,
        render_space_filter: Some(space),
        head_output_transform: Mat4::IDENTITY,
        render_shadows: true,
        cull: Some(identity_light_cull_desc()),
    };
    let mut overlay_desc = desc;
    overlay_desc.view_id = overlay;
    let mut mgr = FrameResourceManager::new();
    mgr.prepare_lights_for_views(&scene, [desc, overlay_desc], None);

    let stats = mgr.light_visibility_stats();
    assert_eq!(stats.space_count, 2);
    assert_eq!(stats.lights_before_cull, 6);
    assert_eq!(stats.indexed_lights, 4);
    assert_eq!(stats.fallback_lights, 2);
    assert_eq!(stats.rejected_lights, 2);
    assert_eq!(stats.lights_after_cull, 4);
    assert_eq!(stats.packed_lights, 4);
    assert_eq!(stats.linear_queries, 2);
    assert_eq!(stats.bvh_queries, 0);
    assert_eq!(stats.light_aabb_tests, 4);
    mgr.reset_light_prep_for_tick();
    assert_eq!(mgr.light_visibility_stats(), Default::default());

    for view_id in [main, overlay] {
        let packed = mgr.frame_lights_for_view(view_id);
        assert_eq!(packed.len(), 2);
        assert!(packed.iter().any(|light| light.light_type == 1));
        assert!(
            packed
                .iter()
                .any(|light| light.light_type == 0 && (light.color[0] - 1.0).abs() < 1e-5)
        );
        assert!(
            packed
                .iter()
                .all(|light| (light.color[0] - 0.5).abs() >= 1e-5)
        );
    }
}

#[test]
fn prepare_lights_for_views_keeps_secondary_light_positions_view_local() {
    let mut scene = SceneCoordinator::new();
    let space = RenderSpaceId(7);
    let local = RenderTransform {
        position: Vec3::new(1.0, 0.0, 0.0),
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    };
    scene.test_seed_space_identity_worlds(space, vec![local], vec![-1]);
    scene.test_set_space_overlay(space, true);
    scene.test_set_space_root_transform(
        space,
        RenderTransform {
            position: Vec3::new(2.0, 0.0, 0.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        },
    );
    let cache = scene.light_cache_mut();
    cache.store_full(100, vec![make_light_data(1.0)]);
    cache.apply_update(space.0, &[], &[0], &[make_state(100)]);

    let first = ViewId::secondary_camera(space, 0);
    let second = ViewId::secondary_camera(space, 1);
    let mut mgr = FrameResourceManager::new();
    mgr.prepare_lights_for_views(
        &scene,
        [
            FrameLightViewDesc {
                view_id: first,
                render_context: RenderingContext::UserView,
                render_space_filter: Some(space),
                head_output_transform: Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
                render_shadows: true,
                cull: None,
            },
            FrameLightViewDesc {
                view_id: second,
                render_context: RenderingContext::UserView,
                render_space_filter: Some(space),
                head_output_transform: Mat4::from_translation(Vec3::new(30.0, 0.0, 0.0)),
                render_shadows: true,
                cull: None,
            },
        ],
        None,
    );

    let first_lights = mgr.frame_lights_for_view(first);
    let second_lights = mgr.frame_lights_for_view(second);
    let stats = mgr.light_visibility_stats();
    assert_eq!(stats.space_count, 2);
    assert_eq!(stats.cull_disabled_spaces, 2);
    assert_eq!(stats.lights_before_cull, 2);
    assert_eq!(stats.lights_after_cull, 2);
    assert_eq!(stats.packed_lights, 2);
    assert_eq!(stats.indexed_lights, 0);
    assert_eq!(stats.rejected_lights, 0);
    assert_eq!(first_lights.len(), 1);
    assert_eq!(second_lights.len(), 1);
    assert!((first_lights[0].position[0] - 9.0).abs() < 1e-4);
    assert!((second_lights[0].position[0] - 29.0).abs() < 1e-4);
}

#[test]
fn prepare_lights_for_views_can_disable_shadow_metadata_per_view() {
    let mut scene = SceneCoordinator::new();
    let space = RenderSpaceId(8);
    scene.test_seed_space_identity_worlds(space, vec![identity_transform()], vec![-1]);
    let cache = scene.light_cache_mut();
    cache.store_full(100, vec![make_light_data(1.0)]);
    cache.apply_update(space.0, &[], &[0], &[make_shadowed_state(100)]);

    let shadows_on = ViewId::secondary_camera(space, 0);
    let shadows_off = ViewId::secondary_camera(space, 1);
    let mut mgr = FrameResourceManager::new();
    mgr.prepare_lights_for_views(
        &scene,
        [
            FrameLightViewDesc {
                view_id: shadows_on,
                render_context: RenderingContext::UserView,
                render_space_filter: Some(space),
                head_output_transform: Mat4::IDENTITY,
                render_shadows: true,
                cull: None,
            },
            FrameLightViewDesc {
                view_id: shadows_off,
                render_context: RenderingContext::UserView,
                render_space_filter: Some(space),
                head_output_transform: Mat4::IDENTITY,
                render_shadows: false,
                cull: None,
            },
        ],
        None,
    );

    let on = mgr.frame_lights_for_view(shadows_on);
    let off = mgr.frame_lights_for_view(shadows_off);
    assert_eq!(on.len(), 1);
    assert_eq!(off.len(), 1);
    assert_eq!(on[0].color, off[0].color);
    assert_eq!(on[0].intensity, off[0].intensity);
    assert_eq!(on[0].shadow_type, 2);
    assert_eq!(on[0].shadow_strength, 0.75);
    assert_eq!(off[0].shadow_type, 0);
    assert_eq!(off[0].shadow_strength, 0.0);
    assert_eq!(off[0].shadow_near_plane, 0.0);
    assert_eq!(off[0].shadow_bias, 0.0);
    assert_eq!(off[0].shadow_normal_bias, 0.0);
}
