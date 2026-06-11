//! Data-only tests for [`RendererRuntime::collect_prepared_views`]. No GPU is created.

use std::path::PathBuf;
use std::sync::Arc;

use super::*;
use crate::camera::StereoViewMatrices;
use crate::config::{RendererSettings, RendererSettingsHandle};
use crate::connection::ConnectionParams;
use crate::gpu::OutputDepthMode;
use crate::materials::ShaderPermutation;

fn build_runtime() -> RendererRuntime {
    let settings: RendererSettingsHandle =
        Arc::new(std::sync::RwLock::new(RendererSettings::default()));
    RendererRuntime::new(
        Option::<ConnectionParams>::None,
        settings,
        PathBuf::from("test_config.toml"),
    )
}

const TEST_EXTENT: (u32, u32) = (1920, 1080);

fn test_eye(position: glam::Vec3) -> EyeView {
    let view = glam::Mat4::from_translation(-position);
    let proj = glam::Mat4::IDENTITY;
    EyeView::new(view, proj, proj * view, position)
}

fn collect_default_desktop_views(runtime: &RendererRuntime) -> ViewFamilyPlan<'_> {
    runtime.collect_prepared_views_without_secondaries(
        PrimaryViewRequest::DesktopMain,
        TEST_EXTENT,
        RenderPathProfile::desktop_main(),
        RenderPathProfile::desktop_main(),
    )
}

#[test]
fn secondary_cameras_use_single_sample_profile_policy() {
    assert_eq!(
        RenderPathProfile::secondary_camera(ViewPostProcessing::primary_view())
            .sample_count_policy(),
        crate::render_graph::compiled::RenderPathSampleCountPolicy::SingleSample
    );
}

#[test]
fn secondary_cameras_use_camera_render_context() {
    assert_eq!(secondary_camera_render_context(), RenderingContext::Camera);
}

#[test]
fn camera_portals_use_mode_specific_render_contexts() {
    assert_eq!(
        camera_portal_render_context(CameraPortalMode::Mirror),
        RenderingContext::Mirror
    );
    assert_eq!(
        camera_portal_render_context(CameraPortalMode::Portal),
        RenderingContext::Portal
    );
}

#[test]
fn secondary_camera_write_target_uses_double_buffer_policy() {
    let double_buffered = 1u16 << 2;
    let post_processing = 1u16 << 6;

    assert_eq!(
        secondary_camera_write_target(9, 0).render_texture_self_sampling(),
        Some(RenderTextureSelfSampling::Suppress)
    );
    assert_eq!(
        secondary_camera_write_target(9, double_buffered).render_texture_self_sampling(),
        Some(RenderTextureSelfSampling::AllowPreviousContents)
    );
    assert_eq!(
        secondary_camera_write_target(9, double_buffered | post_processing)
            .render_texture_self_sampling(),
        Some(RenderTextureSelfSampling::Suppress)
    );
}

#[test]
fn secondary_camera_flags_drive_layer_and_shadow_policy() {
    let render_private_ui = 1u16 << 3;
    let render_shadows = 1u16 << 5;

    assert_eq!(
        secondary_camera_layer_policy(0),
        ViewLayerPolicy::camera(false)
    );
    assert_eq!(
        secondary_camera_layer_policy(render_private_ui),
        ViewLayerPolicy::camera(true)
    );
    assert!(!secondary_camera_shadows_enabled(0));
    assert!(secondary_camera_shadows_enabled(render_shadows));
}

#[test]
fn secondary_camera_render_space_scope_tracks_selective_roots() {
    let space = RenderSpaceId(42);

    assert_eq!(
        camera_render_space_scope(false, space),
        ViewRenderSpaceScope::AllActive
    );
    assert_eq!(
        camera_render_space_scope(true, space),
        ViewRenderSpaceScope::single(space)
    );
}

#[test]
fn empty_scene_desktop_mode_yields_only_main_view() {
    let runtime = build_runtime();
    let views = collect_default_desktop_views(&runtime);
    let plans = views.plans();
    assert_eq!(plans.len(), 1);
    assert!(matches!(plans[0].target, FrameViewPlanTarget::Swapchain));
    assert_eq!(plans[0].view_id, ViewId::Main);
    assert!(plans[0].draw_filter.is_none());
    assert!(views.requirements().any_post_processing);
}

#[test]
fn empty_scene_vr_secondaries_only_yields_empty_vec() {
    let runtime = build_runtime();
    let views = runtime.collect_prepared_views_without_secondaries(
        PrimaryViewRequest::None,
        TEST_EXTENT,
        RenderPathProfile::desktop_main(),
        RenderPathProfile::xr_hmd(),
    );
    assert!(
        views.is_empty(),
        "no HMD, no secondaries, and main view excluded -- nothing to render"
    );
    assert_eq!(views.frame_global().view_id, ViewId::Main);
    assert!(views.frame_global().post_processing.is_enabled());
}

#[test]
fn empty_scene_desktop_secondaries_only_uses_desktop_frame_global_fallback() {
    let runtime = build_runtime();
    let views = runtime.collect_prepared_views_without_secondaries(
        PrimaryViewRequest::None,
        TEST_EXTENT,
        RenderPathProfile::desktop_main(),
        RenderPathProfile::headless_main(),
    );

    assert!(views.is_empty());
    assert_eq!(views.frame_global().view_id, ViewId::Main);
    assert!(!views.frame_global().post_processing.is_enabled());
    assert_eq!(
        views.frame_global().render_context,
        runtime.scene.active_main_render_context()
    );
}

#[test]
fn desktop_main_view_overrides_frame_global_fallback() {
    let runtime = build_runtime();
    let views = runtime.collect_prepared_views_without_secondaries(
        PrimaryViewRequest::DesktopMain,
        TEST_EXTENT,
        RenderPathProfile::headless_main(),
        RenderPathProfile::desktop_main(),
    );

    assert_eq!(views.plans().len(), 1);
    assert_eq!(views.frame_global().view_id, ViewId::Main);
    assert!(!views.frame_global().post_processing.is_enabled());
}

#[test]
fn main_view_carries_runtime_host_camera() {
    let mut runtime = build_runtime();
    runtime.host_camera.frame_index = 42;
    runtime.host_camera.desktop_fov_degrees = 75.0;
    let views = collect_default_desktop_views(&runtime);
    let main = &views.plans()[0];
    assert_eq!(main.host_camera.frame_index, 42);
    assert_eq!(main.host_camera.desktop_fov_degrees, 75.0);
}

/// Pins the contract from the April 2026 cull regression: the main desktop `FrameViewPlan`
/// must carry the main target extent supplied to `collect_prepared_views`. A zero or stale
/// extent produces a degenerate `build_world_mesh_cull_proj_params` frustum and flickering
/// scene-object culling.
#[test]
fn main_view_viewport_matches_supplied_target_extent() {
    let runtime = build_runtime();
    let views = runtime.collect_prepared_views_without_secondaries(
        PrimaryViewRequest::DesktopMain,
        (1280, 720),
        RenderPathProfile::desktop_main(),
        RenderPathProfile::desktop_main(),
    );
    let main = views
        .plans()
        .iter()
        .find(|v| v.view_id == ViewId::Main)
        .expect("desktop primary request yields a main view");
    assert_eq!(main.viewport_px, (1280, 720));
}

#[test]
fn main_view_uses_default_shader_permutation_and_depth_mode() {
    let runtime = build_runtime();
    let view = runtime.build_main_desktop_view(TEST_EXTENT);
    assert_eq!(view.shader_permutation(), ShaderPermutation(0));
    assert_eq!(view.output_depth_mode(), OutputDepthMode::DesktopSingle);
    assert_eq!(view.clear.mode, CameraClearMode::Skybox);
    assert_eq!(view.post_processing(), ViewPostProcessing::primary_view());
    assert_eq!(
        view.profile.id(),
        crate::render_graph::compiled::RenderPathProfileId::DesktopMain
    );
}

#[test]
fn main_view_uses_headless_profile_for_headless_output() {
    let runtime = build_runtime();
    let view =
        runtime.build_main_view_with_profile(TEST_EXTENT, RenderPathProfile::headless_main(), None);

    assert_eq!(view.post_processing(), ViewPostProcessing::disabled());
    assert_eq!(
        view.profile.id(),
        crate::render_graph::compiled::RenderPathProfileId::HeadlessMain
    );
}

/// Secondary view identity follows camera identity even when cameras share a render target.
#[test]
fn secondary_camera_view_ids_do_not_alias_shared_render_targets() {
    let first = secondary_camera_view_id(RenderSpaceId(9), 12, 0);
    let second = secondary_camera_view_id(RenderSpaceId(9), 13, 1);
    let fallback = secondary_camera_view_id(RenderSpaceId(9), -1, 2);

    assert_ne!(first, second);
    assert_ne!(first, fallback);
    assert_eq!(
        fallback,
        ViewId::SecondaryCamera(crate::camera::SecondaryCameraId::new(RenderSpaceId(9), 2))
    );
}

#[test]
fn camera_portal_stereo_render_rects_split_target_halves() {
    let (left, right) = camera_portal_stereo_render_rects((1024, 512)).expect("even split rects");
    assert_eq!(left.origin_px, (0, 0));
    assert_eq!(left.extent_px, (512, 512));
    assert_eq!(right.origin_px, (512, 0));
    assert_eq!(right.extent_px, (512, 512));

    let (left, right) = camera_portal_stereo_render_rects((5, 3)).expect("odd split rects");
    assert_eq!(left.origin_px, (0, 0));
    assert_eq!(left.extent_px, (2, 3));
    assert_eq!(right.origin_px, (2, 0));
    assert_eq!(right.extent_px, (3, 3));

    assert!(camera_portal_stereo_render_rects((1, 512)).is_none());
    assert!(camera_portal_stereo_render_rects((1024, 0)).is_none());
}

#[test]
fn camera_portal_source_view_plans_use_stereo_eye_sources_and_half_rects() {
    let mut runtime = build_runtime();
    let left_position = glam::Vec3::new(-0.03, 1.7, 0.25);
    let right_position = glam::Vec3::new(0.03, 1.7, 0.25);
    runtime.host_camera.vr_active = true;
    runtime.host_camera.stereo = Some(StereoViewMatrices::new(
        test_eye(left_position),
        test_eye(right_position),
    ));

    let plans = runtime
        .camera_portal_source_view_plans(TEST_EXTENT, (1024, 512))
        .expect("stereo plans");
    let collected: Vec<_> = plans.iter().collect();

    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0].eye_index, 0);
    assert_eq!(collected[0].source.world_position, left_position);
    assert_eq!(collected[0].render_rect.origin_px, (0, 0));
    assert_eq!(collected[0].render_rect.extent_px, (512, 512));
    assert_eq!(collected[1].eye_index, 1);
    assert_eq!(collected[1].source.world_position, right_position);
    assert_eq!(collected[1].render_rect.origin_px, (512, 0));
    assert_eq!(collected[1].render_rect.extent_px, (512, 512));
}

#[test]
fn camera_portal_source_view_plans_fall_back_to_mono_without_active_stereo() {
    let runtime = build_runtime();

    let plans = runtime
        .camera_portal_source_view_plans(TEST_EXTENT, (512, 512))
        .expect("mono plan");
    let collected: Vec<_> = plans.iter().collect();

    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].eye_index, 0);
    assert_eq!(collected[0].render_rect.origin_px, (0, 0));
    assert_eq!(collected[0].render_rect.extent_px, (512, 512));
}

#[test]
fn camera_portal_view_ids_do_not_alias_stereo_halves() {
    let left = camera_portal_eye_view_id(RenderSpaceId(1), 4, 0, 0);
    let right = camera_portal_eye_view_id(RenderSpaceId(1), 4, 0, 1);

    assert_ne!(left, right);
    let ViewId::CameraPortal(left_id) = left else {
        panic!("left portal id");
    };
    let ViewId::CameraPortal(right_id) = right else {
        panic!("right portal id");
    };
    assert_eq!(left_id.eye_index, 0);
    assert_eq!(right_id.eye_index, 1);
}

#[test]
fn secondary_view_tasks_sort_by_depth_then_space_then_camera() {
    let mut tasks = vec![
        (RenderSpaceId(2), 0.0, 4),
        (RenderSpaceId(1), 0.0, 3),
        (RenderSpaceId(1), -5.0, 9),
        (RenderSpaceId(1), 0.0, 1),
    ];

    sort_secondary_view_tasks(&mut tasks);

    assert_eq!(
        tasks,
        vec![
            (RenderSpaceId(1), -5.0, 9),
            (RenderSpaceId(1), 0.0, 1),
            (RenderSpaceId(1), 0.0, 3),
            (RenderSpaceId(2), 0.0, 4),
        ]
    );
}

#[test]
fn prepared_view_helpers_honor_explicit_camera_view_origin() {
    let runtime = build_runtime();
    let mut view = runtime.build_main_desktop_view(TEST_EXTENT);
    view.host_camera.head_output_transform =
        glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(view.view_origin_world(), glam::Vec3::new(1.0, 2.0, 3.0));
    view.host_camera.explicit_view = Some(EyeView::new(
        glam::Mat4::IDENTITY,
        glam::Mat4::IDENTITY,
        glam::Mat4::IDENTITY,
        glam::Vec3::new(7.0, 8.0, 9.0),
    ));
    assert_eq!(view.view_origin_world(), glam::Vec3::new(7.0, 8.0, 9.0));
}

#[test]
fn prepared_view_helpers_prefer_eye_world_position_over_head_output() {
    let runtime = build_runtime();
    let mut view = runtime.build_main_desktop_view(TEST_EXTENT);
    view.host_camera.head_output_transform =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.0));
    view.host_camera.eye_world_position = Some(glam::Vec3::new(4.0, 5.0, 6.0));
    assert_eq!(
        view.view_origin_world(),
        glam::Vec3::new(4.0, 5.0, 6.0),
        "eye_world_position must override the head-output (render-space root) translation \
         so PBS view-direction math sees the eye, not the floor anchor"
    );
    view.host_camera.explicit_view = Some(EyeView::new(
        glam::Mat4::IDENTITY,
        glam::Mat4::IDENTITY,
        glam::Mat4::IDENTITY,
        glam::Vec3::new(7.0, 8.0, 9.0),
    ));
    assert_eq!(
        view.view_origin_world(),
        glam::Vec3::new(7.0, 8.0, 9.0),
        "explicit camera view still wins over eye_world_position"
    );
}
