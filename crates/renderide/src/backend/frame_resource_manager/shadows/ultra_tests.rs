use std::sync::Arc;

use crate::backend::HostShadowQuality;
use crate::backend::frame_resource_manager::per_view_state::PreparedViewLights;
use crate::camera::ViewId;
use crate::gpu::GpuLight;
use crate::materials::RasterPipelineKind;
use crate::shared::{
    LightType, QualityConfig, ShadowCascadeMode, ShadowCastMode, ShadowResolutionMode,
};
use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshDrawItem, WorldMeshDrawPlan};

use super::{POINT_FACE_COUNT, light_type_u32};

fn shadowed_light(light_type: LightType) -> GpuLight {
    GpuLight {
        position: [3.0, 4.0, 5.0],
        light_type: light_type_u32(light_type),
        shadow_type: 1,
        shadow_strength: 1.0,
        shadow_near_plane: 0.05,
        shadow_bias: 0.25,
        range: 8.0,
        spot_cos_half_angle: 0.5,
        ..GpuLight::default()
    }
}

fn pbs_draw(node_id: i32, shadow_cast_mode: ShadowCastMode) -> WorldMeshDrawItem {
    let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id,
        slot_index: 0,
        collect_order: node_id.max(0) as usize,
        alpha_blended: false,
    });
    item.shadow_cast_mode = shadow_cast_mode;
    item.batch_key.pipeline = RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));
    item
}

fn prefetched_plan(items: Vec<WorldMeshDrawItem>) -> WorldMeshDrawPlan {
    WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
        WorldMeshDrawCollection {
            draws_pre_cull: items.len(),
            items,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        },
        None,
    )))
}

#[test]
fn ultra_shadow_planning_caps_quality_resolution_by_light_type() {
    let mut manager = super::FrameResourceManager::new();
    let lights = &mut manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights;
    lights.push(shadowed_light(LightType::Directional));
    lights.push(shadowed_light(LightType::Spot));
    lights.push(shadowed_light(LightType::Point));
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);
    let quality = HostShadowQuality::from_quality_config(&QualityConfig {
        shadow_cascades: ShadowCascadeMode::FourCascades,
        shadow_resolution: ShadowResolutionMode::Ultra,
        ..Default::default()
    });

    manager.prepare_shadow_frame_for_views(quality, [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.requested_resolution, 4096);
    assert_eq!(plan.render_views.len(), 11);
    for view in &plan.render_views[..4] {
        assert_eq!(view.kind, crate::gpu::SHADOW_VIEW_KIND_DIRECTIONAL);
        assert_eq!(view.resolution, 4096);
    }
    assert_eq!(plan.render_views[4].kind, crate::gpu::SHADOW_VIEW_KIND_SPOT);
    assert_eq!(plan.render_views[4].resolution, 2048);
    for view in &plan.render_views[5..] {
        assert_eq!(view.kind, crate::gpu::SHADOW_VIEW_KIND_POINT);
        assert_eq!(view.resolution, 1024);
    }
    assert_eq!(plan.metadata[0].params[1], 1.0 / 4096.0);
    assert_eq!(plan.metadata[4].params[1], 1.0 / 2048.0);
    assert_eq!(plan.metadata[5].params[1], 1.0 / 1024.0);
}

#[test]
fn custom_shadow_resolution_override_bypasses_quality_light_type_cap() {
    let mut manager = super::FrameResourceManager::new();
    let mut point = shadowed_light(LightType::Point);
    point.shadow_map_resolution = 2048;
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(point);
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);
    let quality = HostShadowQuality::from_quality_config(&QualityConfig {
        shadow_resolution: ShadowResolutionMode::Ultra,
        ..Default::default()
    });

    manager.prepare_shadow_frame_for_views(quality, [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.requested_resolution, 2048);
    assert_eq!(plan.render_views.len(), POINT_FACE_COUNT as usize);
    for view in &plan.render_views {
        assert_eq!(view.resolution, 2048);
    }
}

#[test]
fn applying_actual_atlas_resolution_updates_shadow_metadata() {
    let mut manager = super::FrameResourceManager::new();
    let lights = &mut manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights;
    let mut custom_spot = shadowed_light(LightType::Spot);
    custom_spot.shadow_map_resolution = 512;
    custom_spot.shadow_normal_bias = 1.0;
    let mut ultra_spot = shadowed_light(LightType::Spot);
    ultra_spot.shadow_normal_bias = 1.0;
    lights.push(custom_spot);
    lights.push(ultra_spot);
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);
    let quality = HostShadowQuality::from_quality_config(&QualityConfig {
        shadow_resolution: ShadowResolutionMode::Ultra,
        ..Default::default()
    });

    manager.prepare_shadow_frame_for_views(quality, [(ViewId::Main, &draw_plan)]);
    let old_bias = manager.shadow_frame_plan().metadata[1].light_params[2];
    manager.apply_shadow_atlas_resolution(1024);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.requested_resolution, 1024);
    assert_eq!(plan.render_views[0].resolution, 512);
    assert_eq!(plan.render_views[1].resolution, 1024);
    assert_eq!(plan.metadata[0].params[1], 1.0 / 512.0);
    assert_eq!(plan.metadata[1].params[1], 1.0 / 1024.0);
    assert_eq!(plan.metadata[0].atlas_rect, [0.0, 0.0, 0.5, 0.5]);
    assert_eq!(plan.metadata[1].atlas_rect, [0.0, 0.0, 1.0, 1.0]);
    assert!((plan.metadata[1].light_params[2] - old_bias * 2.0).abs() <= f32::EPSILON);
}
