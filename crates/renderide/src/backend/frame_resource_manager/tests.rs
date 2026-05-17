//! Tests for the parent module.

use super::cluster_layout::{
    ClusterPreRecordLayout, cluster_index_capacity_for_layout, per_view_snapshot_sync_params,
    unique_cluster_pre_record_layouts,
};
use super::*;

use glam::{Mat4, Quat, Vec3};

use crate::camera::ViewId;
use crate::mesh_deform::{SkinCacheKey, SkinCacheRendererKind};
use crate::render_graph::frame_params::PreRecordViewResourceLayout;
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{
    LightData, LightType, LightsBufferRendererState, RenderTransform, RenderingContext, ShadowType,
};

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
    LightData {
        point: Vec3::ZERO,
        orientation: Quat::IDENTITY,
        color: Vec3::new(color_x, 0.0, 0.0),
        intensity,
        range: 10.0,
        angle: 45.0,
    }
}

fn make_light_data(color_x: f32) -> LightData {
    make_light_data_with_intensity(color_x, 1.0)
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

fn seed_space_with_light(
    scene: &mut SceneCoordinator,
    space_id: RenderSpaceId,
    global_unique_id: i32,
    color_x: f32,
) {
    scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
    let cache = scene.light_cache_mut();
    cache.store_full(global_unique_id, vec![make_light_data(color_x)]);
    cache.apply_update(space_id.0, &[], &[0], &[make_state(global_unique_id)]);
}

fn seed_space_with_signed_light(
    scene: &mut SceneCoordinator,
    space_id: RenderSpaceId,
    global_unique_id: i32,
    color_x: f32,
    intensity: f32,
) {
    scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
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
    assert!((packed[0].color[0] - 0.214_041_14).abs() < 1e-5);
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
            },
            FrameLightViewDesc {
                view_id: second,
                render_context: RenderingContext::UserView,
                render_space_filter: Some(space),
                head_output_transform: Mat4::from_translation(Vec3::new(30.0, 0.0, 0.0)),
            },
        ],
    );

    let first_lights = mgr.frame_lights_for_view(first);
    let second_lights = mgr.frame_lights_for_view(second);
    assert_eq!(first_lights.len(), 1);
    assert_eq!(second_lights.len(), 1);
    assert!((first_lights[0].position[0] - 9.0).abs() < 1e-4);
    assert!((second_lights[0].position[0] - 29.0).abs() < 1e-4);
}
