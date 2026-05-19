use super::*;
use crate::camera::{CameraClipPlanes, EyeView};
use crate::materials::{RasterPipelineKind, RasterPrimitiveTopology};
use crate::shared::ShadowCastMode;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
use crate::world_mesh::{DrawGroup, InstancePlan, WorldMeshDrawItem};

fn shadow_light() -> GpuLight {
    GpuLight {
        shadow_type: 1,
        shadow_strength: 1.0,
        intensity: 1.0,
        range: 10.0,
        ..GpuLight::default()
    }
}

fn perspective_test_camera(viewport: Viewport) -> HostCameraFrame {
    let clip = CameraClipPlanes::new(0.1, 100.0);
    let projection = reverse_z_perspective(viewport.aspect(), 70.0_f32.to_radians(), 0.1, 100.0);
    HostCameraFrame {
        clip,
        desktop_fov_degrees: 70.0,
        projection_kind: CameraProjectionKind::Perspective,
        explicit_view: Some(EyeView::new(
            Mat4::IDENTITY,
            projection,
            projection,
            Vec3::ZERO,
        )),
        eye_world_position: Some(Vec3::ZERO),
        ..Default::default()
    }
}

fn assert_shadow_clip_contains(view_proj: Mat4, point: Vec3) {
    let clip = view_proj * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    assert!(
        (-1.0001..=1.0001).contains(&ndc.x),
        "x outside shadow clip: {ndc:?}",
    );
    assert!(
        (-1.0001..=1.0001).contains(&ndc.y),
        "y outside shadow clip: {ndc:?}",
    );
    assert!(
        (-0.0001..=1.0001).contains(&ndc.z),
        "z outside shadow clip: {ndc:?}",
    );
}

fn shadow_draw(node_id: i32) -> WorldMeshDrawItem {
    let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 7,
        node_id,
        slot_index: 0,
        collect_order: node_id as usize,
        alpha_blended: false,
    });
    item.shadow_cast_mode = ShadowCastMode::On;
    item.batch_key.primitive_topology = RasterPrimitiveTopology::TriangleList;
    item
}

fn draw_group(draw_idx: usize) -> DrawGroup {
    DrawGroup {
        representative_draw_idx: draw_idx,
        instance_range: draw_idx as u32..draw_idx as u32 + 1,
        material_packet_idx: 0,
    }
}

fn shadow_view(view_proj: Mat4) -> PlannedShadowView {
    PlannedShadowView {
        gpu: GpuShadowView::default(),
        view_proj,
    }
}

fn identity_view() -> PlannedShadowView {
    shadow_view(Mat4::IDENTITY)
}

#[test]
fn light_casts_shadows_requires_shadow_type_strength_intensity_and_range() {
    assert!(light_casts_shadows(&shadow_light()));

    let mut no_type = shadow_light();
    no_type.shadow_type = 0;
    assert!(!light_casts_shadows(&no_type));

    let mut no_strength = shadow_light();
    no_strength.shadow_strength = 0.0;
    assert!(!light_casts_shadows(&no_strength));

    let mut no_intensity = shadow_light();
    no_intensity.intensity = 0.0;
    assert!(!light_casts_shadows(&no_intensity));

    let mut invalid_range = shadow_light();
    invalid_range.range = f32::NAN;
    assert!(!light_casts_shadows(&invalid_range));
}

#[test]
fn shadow_depth_bias_is_scaled_to_shadow_texels() {
    let bias = sanitized_depth_bias(0.05, 1024);
    assert!((bias - (0.5 / 1024.0)).abs() < 1e-8);

    let clamped = sanitized_depth_bias(10.0, 1024);
    assert!((clamped - (MAX_SHADOW_DEPTH_BIAS_TEXELS / 1024.0)).abs() < 1e-8);

    let fallback = sanitized_depth_bias(0.0, 1024);
    assert!((fallback - (DEFAULT_SHADOW_DEPTH_BIAS_TEXELS / 1024.0)).abs() < 1e-8);
}

#[test]
fn shadow_slope_bias_is_scaled_to_shadow_texels() {
    let bias = sanitized_slope_bias(2048);
    assert!((bias - (SHADOW_SLOPE_BIAS_TEXELS / 2048.0)).abs() < 1e-8);
}

#[test]
fn directional_cascade_fit_contains_near_lower_frustum_point() {
    let scene = SceneCoordinator::new();
    let viewport = Viewport::new(1280, 720);
    let camera = perspective_test_camera(viewport);
    let fit = directional_cascade_fit(
        &scene,
        &camera,
        viewport,
        Vec3::new(-0.35, -1.0, -0.25).normalize(),
        0.1,
        4.0,
        1024,
    )
    .expect("directional cascade should fit a valid perspective camera");

    assert_shadow_clip_contains(fit.view_proj, Vec3::new(0.0, -0.12, -0.2));
}

#[test]
fn directional_cascade_fit_contains_all_slice_corners() {
    let scene = SceneCoordinator::new();
    let viewport = Viewport::new(1280, 720);
    let camera = perspective_test_camera(viewport);
    let projection = camera_projection(&camera, viewport);
    let corners = cascade_frustum_corners_world(
        camera.projection_kind,
        projection,
        camera_world_to_view(&scene, &camera),
        0.1,
        4.0,
    )
    .expect("frustum corners should be reconstructable");
    let fit = directional_shadow_matrix_from_corners(
        &corners,
        Vec3::new(-0.35, -1.0, -0.25).normalize(),
        1024,
    )
    .expect("directional cascade should fit valid corners");

    for corner in corners {
        assert_shadow_clip_contains(fit.view_proj, corner);
    }
}

#[test]
fn compact_shadow_phase_drops_slab_slots_for_culled_groups() {
    let mut visible = shadow_draw(0);
    visible.world_aabb = Some((Vec3::new(-0.25, -0.25, 0.1), Vec3::new(0.25, 0.25, 0.5)));
    let mut culled = shadow_draw(1);
    culled.world_aabb = Some((Vec3::new(4.0, 4.0, 4.0), Vec3::new(5.0, 5.0, 5.0)));
    let draws = vec![visible, culled];
    let instance_plan = InstancePlan {
        slab_layout: vec![0, 1],
        regular_groups: vec![draw_group(0), draw_group(1)],
        post_skybox_groups: Vec::new(),
        intersect_groups: Vec::new(),
        transparent_groups: Vec::new(),
    };
    let phases = build_shadow_phases_for_views(
        &draws,
        &instance_plan,
        &shadow_pipeline_state(),
        &[identity_view()],
    );

    let compact = compact_shadow_phases_from_source_layout(phases, &instance_plan.slab_layout);

    assert_eq!(compact.len(), 1);
    assert_eq!(compact[0].slab_slot_base, 0);
    assert_eq!(compact[0].slab_layout, vec![0]);
    assert_eq!(compact_shadow_phases_total_slots(&compact), 1);
    assert_eq!(compact[0].phase.depth_runs[0].group.instance_range, 0..1);
}

#[test]
fn compact_shadow_phases_get_distinct_slab_bases_per_view() {
    let draws = vec![shadow_draw(0), shadow_draw(1)];
    let instance_plan = InstancePlan {
        slab_layout: vec![0, 1],
        regular_groups: vec![draw_group(0), draw_group(1)],
        post_skybox_groups: Vec::new(),
        intersect_groups: Vec::new(),
        transparent_groups: Vec::new(),
    };
    let phases = build_shadow_phases_for_views(
        &draws,
        &instance_plan,
        &shadow_pipeline_state(),
        &[identity_view(), identity_view()],
    );

    let compact = compact_shadow_phases_from_source_layout(phases, &instance_plan.slab_layout);

    assert_eq!(compact.len(), 2);
    assert_eq!(compact[0].slab_slot_base, 0);
    assert_eq!(compact[0].slab_layout, vec![0, 1]);
    assert_eq!(compact[1].slab_slot_base, 2);
    assert_eq!(compact[1].slab_layout, vec![0, 1]);
    assert_eq!(compact_shadow_phases_total_slots(&compact), 4);
}

#[test]
fn compact_shadow_phase_remaps_material_and_generic_runs() {
    let mut material = shadow_draw(0);
    material.batch_key.pipeline = RasterPipelineKind::EmbeddedStem("pbsmetallic_default".into());
    let generic = shadow_draw(1);
    let draws = vec![material, generic];
    let instance_plan = InstancePlan {
        slab_layout: vec![0, 1],
        regular_groups: vec![draw_group(0), draw_group(1)],
        post_skybox_groups: Vec::new(),
        intersect_groups: Vec::new(),
        transparent_groups: Vec::new(),
    };
    let phases = build_shadow_phases_for_views(
        &draws,
        &instance_plan,
        &shadow_pipeline_state(),
        &[identity_view()],
    );

    let compact = compact_shadow_phases_from_source_layout(phases, &instance_plan.slab_layout);

    assert_eq!(compact[0].slab_layout, vec![0, 1]);
    assert_eq!(compact[0].phase.material_runs.len(), 1);
    assert_eq!(
        compact[0].phase.material_runs[0]
            .group
            .representative_draw_idx,
        0
    );
    assert_eq!(compact[0].phase.material_runs[0].group.instance_range, 0..1);
    assert_eq!(compact[0].phase.depth_runs.len(), 1);
    assert_eq!(
        compact[0].phase.depth_runs[0].group.representative_draw_idx,
        1
    );
    assert_eq!(compact[0].phase.depth_runs[0].group.instance_range, 1..2);
}

#[test]
fn compact_shadow_phase_splits_disabled_member_inside_group() {
    let caster = shadow_draw(0);
    let mut disabled = shadow_draw(1);
    disabled.shadow_cast_mode = ShadowCastMode::Off;
    let draws = vec![caster, disabled];
    let instance_plan = InstancePlan {
        slab_layout: vec![0, 1],
        regular_groups: vec![DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..2,
            material_packet_idx: 0,
        }],
        post_skybox_groups: Vec::new(),
        intersect_groups: Vec::new(),
        transparent_groups: Vec::new(),
    };
    let phases = build_shadow_phases_for_views(
        &draws,
        &instance_plan,
        &shadow_pipeline_state(),
        &[identity_view()],
    );

    let compact = compact_shadow_phases_from_source_layout(phases, &instance_plan.slab_layout);

    assert_eq!(compact[0].slab_layout, vec![0]);
    assert_eq!(compact[0].phase.depth_runs.len(), 1);
    assert_eq!(compact[0].phase.depth_runs[0].group.instance_range, 0..1);
}

#[test]
fn shadow_phase_cache_reuses_identical_plan_and_view() {
    let draws = vec![shadow_draw(0), shadow_draw(1)];
    let instance_plan = InstancePlan {
        slab_layout: vec![0, 1],
        regular_groups: vec![DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..2,
            material_packet_idx: 0,
        }],
        post_skybox_groups: Vec::new(),
        intersect_groups: Vec::new(),
        transparent_groups: Vec::new(),
    };
    let prepared =
        build_shadow_caster_prepared_plan(&draws, &instance_plan, &shadow_pipeline_state());
    let mut cache = WorldMeshShadowPhaseCache::default();
    let views = [identity_view()];

    let first = cache.build_compact_phases(&prepared, &instance_plan.slab_layout, &views);
    let second = cache.build_compact_phases(&prepared, &instance_plan.slab_layout, &views);

    assert_eq!(first.cache_stats.hits, 0);
    assert_eq!(first.cache_stats.misses, 1);
    assert_eq!(second.cache_stats.hits, 1);
    assert_eq!(second.cache_stats.misses, 0);
    assert_eq!(second.phases[0].slab_layout, vec![0, 1]);
    assert_eq!(
        second.phases[0].phase.depth_runs[0].group.instance_range,
        0..2
    );
}
