use std::sync::Arc;

use hashbrown::HashMap;

use crate::backend::HostShadowQuality;
use crate::backend::frame_resource_manager::per_view_state::PreparedViewLights;
use crate::camera::ViewId;
use crate::gpu::{GpuLight, GpuLimits};
use crate::materials::RasterPipelineKind;
use crate::shared::{LightType, ShadowCastMode};
use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
use crate::world_mesh::{
    PrefetchedWorldMeshViewDraws, WorldMeshDrawItem, WorldMeshDrawPlan, WorldMeshPhase,
};
use glam::Vec3;

use super::{
    POINT_FACE_COUNT, light_type_u32, point_shadow_projection, shadow_view_count_for_light,
};

fn limits_with_shadow_format_usages<const N: usize>(
    features: [(wgpu::TextureFormat, wgpu::TextureUsages); N],
) -> GpuLimits {
    let mut format_features = HashMap::new();
    for (format, allowed_usages) in features {
        format_features.insert(
            format,
            wgpu::TextureFormatFeatures {
                allowed_usages,
                flags: wgpu::TextureFormatFeatureFlags::empty(),
            },
        );
    }
    GpuLimits::synthetic_for_tests(
        wgpu::Limits {
            max_texture_dimension_2d: 1024,
            max_texture_array_layers: 8,
            ..Default::default()
        },
        wgpu::Features::empty(),
        format_features,
    )
}

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
fn point_lights_plan_six_shadow_views() {
    assert_eq!(
        shadow_view_count_for_light(
            light_type_u32(LightType::Point),
            HostShadowQuality::default()
        ),
        POINT_FACE_COUNT
    );
}

#[test]
fn directional_lights_use_host_cascade_count() {
    let quality = HostShadowQuality {
        cascade_count: 2,
        ..HostShadowQuality::default()
    };
    assert_eq!(
        shadow_view_count_for_light(light_type_u32(LightType::Directional), quality),
        2
    );
}

#[test]
fn clear_assignment_removes_shadow_view_link() {
    let mut light = GpuLight {
        shadow_view_start: 4,
        shadow_view_count: 2,
        shadow_flags: 7,
        ..GpuLight::default()
    };
    super::clear_light_shadow_assignment(&mut light);
    assert_eq!(light.shadow_view_start, 0);
    assert_eq!(light.shadow_view_count, 0);
    assert_eq!(light.shadow_flags, 0);
}

#[test]
fn point_shadow_faces_have_distinct_projection_matrices() {
    let light = GpuLight {
        range: 8.0,
        shadow_near_plane: 0.05,
        ..GpuLight::default()
    };
    let position = Vec3::new(1.0, 2.0, 3.0);
    let mut seen = Vec::new();
    for face in 0..POINT_FACE_COUNT {
        let matrix = point_shadow_projection(&light, position, face).to_cols_array();
        assert!(
            !seen.iter().any(|existing| existing == &matrix),
            "point shadow face {face} reused a previous projection"
        );
        seen.push(matrix);
    }
}

#[test]
fn point_shadow_faces_share_caster_set_slab_range() {
    let mut manager = super::FrameResourceManager::new();
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(GpuLight {
            position: [3.0, 4.0, 5.0],
            light_type: light_type_u32(LightType::Point),
            shadow_type: 1,
            shadow_strength: 1.0,
            shadow_near_plane: 0.05,
            shadow_bias: 0.25,
            range: 8.0,
            ..GpuLight::default()
        });

    let mut first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
        material_asset_id: 1,
        property_block: None,
        skinned: false,
        sorting_order: 0,
        mesh_asset_id: 1,
        node_id: 1,
        slot_index: 0,
        collect_order: 0,
        alpha_blended: false,
    });
    first.batch_key.pipeline = RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));
    let mut second = first.clone();
    second.node_id = 2;
    second.collect_order = 1;

    let draw_plan = WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
        WorldMeshDrawCollection {
            items: vec![first, second],
            draws_pre_cull: 2,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        },
        None,
    )));
    manager
        .prepare_shadow_frame_for_views(HostShadowQuality::default(), [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.render_views.len(), POINT_FACE_COUNT as usize);
    assert_eq!(plan.caster_sets.len(), 1);
    assert_eq!(plan.requested_draw_slots, 2);

    let caster_set = &plan.caster_sets[0];
    assert_eq!(caster_set.slab_slot_offset, 0);
    assert_eq!(caster_set.draws.len(), 2);
    assert_eq!(caster_set.instance_plan.slab_layout.len(), 2);

    for view in &plan.render_views {
        assert_eq!(view.kind, crate::gpu::SHADOW_VIEW_KIND_POINT);
        assert_eq!(view.light_position, Vec3::new(3.0, 4.0, 5.0));
        assert_eq!(view.light_range, 8.0);
        assert_eq!(view.shadow_bias, 0.25);
        assert_eq!(view.caster_set_index, 0);
        assert_eq!(view.groups(WorldMeshPhase::ForwardOpaque).len(), 1);
        assert_eq!(
            view.groups(WorldMeshPhase::ForwardOpaque)[0].instance_range,
            0..2
        );
    }
}

#[test]
fn shadow_planning_excludes_shadow_cast_mode_off_draws() {
    let mut manager = super::FrameResourceManager::new();
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(shadowed_light(LightType::Spot));
    let draw_plan = prefetched_plan(vec![
        pbs_draw(1, ShadowCastMode::Off),
        pbs_draw(2, ShadowCastMode::On),
        pbs_draw(3, ShadowCastMode::ShadowOnly),
    ]);

    manager
        .prepare_shadow_frame_for_views(HostShadowQuality::default(), [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.render_views.len(), 1);
    assert_eq!(plan.caster_sets.len(), 1);
    assert_eq!(plan.render_views[0].caster_set_index, 0);
    let nodes = plan.caster_sets[0]
        .draws
        .iter()
        .map(|item| item.node_id)
        .collect::<Vec<_>>();
    assert_eq!(nodes, vec![2, 3]);
    assert_eq!(plan.requested_draw_slots, 2);
}

#[test]
fn shadow_caster_plan_merges_forward_material_state_changes() {
    let first = pbs_draw(1, ShadowCastMode::On);
    let mut second = pbs_draw(2, ShadowCastMode::On);
    second.batch_key.material_asset_id = 44;
    second.batch_key.shader_asset_id = 55;
    second.batch_key.property_block_slot0 = Some(66);

    let plan = super::build_shadow_caster_plan(&[first, second], true);

    assert_eq!(plan.slab_layout, vec![0, 1]);
    assert_eq!(plan.phase_len(WorldMeshPhase::ForwardOpaque), 1);
    assert_eq!(
        plan.phase(WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..2
    );
}

#[test]
fn shadow_caster_plan_keeps_deformed_draws_singleton() {
    let mut first = pbs_draw(1, ShadowCastMode::On);
    first.world_space_deformed = true;
    let mut second = pbs_draw(2, ShadowCastMode::On);
    second.blendshape_deformed = true;

    let plan = super::build_shadow_caster_plan(&[first, second], true);

    assert_eq!(plan.slab_layout, vec![0, 1]);
    assert_eq!(plan.phase_len(WorldMeshPhase::ForwardOpaque), 2);
    assert_eq!(
        plan.phase(WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..1
    );
    assert_eq!(
        plan.phase(WorldMeshPhase::ForwardOpaque)[1].instance_range,
        1..2
    );
}

#[test]
fn shadow_caster_plan_keeps_downlevel_draws_singleton() {
    let first = pbs_draw(1, ShadowCastMode::On);
    let second = pbs_draw(2, ShadowCastMode::On);

    let plan = super::build_shadow_caster_plan(&[first, second], false);

    assert_eq!(plan.slab_layout, vec![0, 1]);
    assert_eq!(plan.phase_len(WorldMeshPhase::ForwardOpaque), 2);
    assert!(
        plan.phase(WorldMeshPhase::ForwardOpaque)
            .iter()
            .all(|group| group.instance_range.end - group.instance_range.start == 1)
    );
}

#[test]
fn shadow_planning_uses_per_light_resolution_and_metadata_bias() {
    let mut manager = super::FrameResourceManager::new();
    let lights = &mut manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights;
    let mut low_resolution = shadowed_light(LightType::Spot);
    low_resolution.shadow_map_resolution = 512;
    low_resolution.shadow_normal_bias = 2.0;
    lights.push(low_resolution);
    lights.push(shadowed_light(LightType::Spot));
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

    manager
        .prepare_shadow_frame_for_views(HostShadowQuality::default(), [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.render_views.len(), 2);
    assert_eq!(
        plan.requested_resolution,
        HostShadowQuality::default().tile_resolution
    );
    assert_eq!(plan.render_views[0].resolution, 512);
    assert_eq!(
        plan.render_views[1].resolution,
        HostShadowQuality::default().tile_resolution
    );
    assert_eq!(plan.metadata[0].params[1], 1.0 / 512.0);
    assert_eq!(plan.metadata[0].atlas_rect, [0.0, 0.0, 0.25, 0.25]);
    assert_eq!(plan.metadata[1].atlas_rect, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(plan.metadata[0].light_params[3], 0.25);
    assert!(plan.metadata[0].light_params[2] > 0.0);
}

#[test]
fn shadow_planning_requests_only_custom_resolution_when_all_lights_override() {
    let mut manager = super::FrameResourceManager::new();
    let mut light = shadowed_light(LightType::Spot);
    light.shadow_map_resolution = 512;
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(light);
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

    manager
        .prepare_shadow_frame_for_views(HostShadowQuality::default(), [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert_eq!(plan.requested_resolution, 512);
    assert_eq!(plan.render_views[0].resolution, 512);
    assert_eq!(plan.metadata[0].atlas_rect, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn shadow_planning_disables_when_depth_atlas_format_is_not_renderable() {
    let mut manager = super::FrameResourceManager::new();
    manager.limits = Some(Arc::new(limits_with_shadow_format_usages([
        (
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        (
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (
            wgpu::TextureFormat::Depth16Unorm,
            wgpu::TextureUsages::empty(),
        ),
    ])));
    let mut light = shadowed_light(LightType::Spot);
    light.shadow_view_start = 7;
    light.shadow_view_count = 3;
    light.shadow_flags = 5;
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(light);
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

    manager
        .prepare_shadow_frame_for_views(HostShadowQuality::default(), [(ViewId::Main, &draw_plan)]);

    let plan = manager.shadow_frame_plan();
    assert!(plan.render_views.is_empty());
    assert!(plan.metadata.is_empty());
    assert_eq!(plan.requested_draw_slots, 0);
    assert_eq!(manager.shadow_resource_request(), None);

    let light = &manager.per_view_lights.get(ViewId::Main).unwrap().lights[0];
    assert_eq!(light.shadow_view_start, 0);
    assert_eq!(light.shadow_view_count, 0);
    assert_eq!(light.shadow_flags, 0);
}
