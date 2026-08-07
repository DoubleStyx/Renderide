use std::sync::Arc;

use hashbrown::HashMap;

use crate::backend::HostShadowQuality;
use crate::backend::frame_resource_manager::per_view_state::PreparedViewLights;
use crate::camera::ViewId;
use crate::gpu::{GpuLight, GpuLimits};
use crate::gpu_pools::MeshPool;
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
    WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws::new(
        WorldMeshDrawCollection {
            draws_pre_cull: items.len(),
            items: items.into(),
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

    let draw_plan = WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws::new(
        WorldMeshDrawCollection {
            items: vec![first, second].into(),
            draws_pre_cull: 2,
            draws_culled: 0,
            draws_hi_z_culled: 0,
            visibility: Default::default(),
            arrangement: Default::default(),
        },
        None,
    )));
    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );

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

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );

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

    assert_eq!(plan.instance_plan.slab_layout, vec![0, 1]);
    assert_eq!(plan.instance_plan.phase_len(WorldMeshPhase::ForwardOpaque), 1);
    assert_eq!(
        plan.instance_plan.phase(WorldMeshPhase::ForwardOpaque)[0].instance_range,
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

    assert_eq!(plan.instance_plan.slab_layout, vec![0, 1]);
    assert_eq!(plan.instance_plan.phase_len(WorldMeshPhase::ForwardOpaque), 2);
    assert_eq!(
        plan.instance_plan.phase(WorldMeshPhase::ForwardOpaque)[0].instance_range,
        0..1
    );
    assert_eq!(
        plan.instance_plan.phase(WorldMeshPhase::ForwardOpaque)[1].instance_range,
        1..2
    );
}

#[test]
fn shadow_caster_plan_keeps_downlevel_draws_singleton() {
    let first = pbs_draw(1, ShadowCastMode::On);
    let second = pbs_draw(2, ShadowCastMode::On);

    let plan = super::build_shadow_caster_plan(&[first, second], false);

    assert_eq!(plan.instance_plan.slab_layout, vec![0, 1]);
    assert_eq!(plan.instance_plan.phase_len(WorldMeshPhase::ForwardOpaque), 2);
    assert!(
        plan.instance_plan.phase(WorldMeshPhase::ForwardOpaque)
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

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );

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

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );

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

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );

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

#[test]
fn unchanged_static_shadow_input_reuses_caster_and_visibility_packets() {
    let mut manager = super::FrameResourceManager::new();
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(shadowed_light(LightType::Spot));
    let draw_plan = prefetched_plan(vec![
        pbs_draw(1, ShadowCastMode::On),
        pbs_draw(2, ShadowCastMode::On),
    ]);

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );
    let first = manager.shadow_frame_plan();
    let first_draws = Arc::clone(&first.caster_sets[0].draws);
    let first_slab_ptr = first.caster_sets[0].instance_plan.slab_layout.as_ptr();
    let first_visibility = Arc::clone(first.render_views[0].visible_groups_arc());
    assert_eq!(first.cache_stats.caster_plan_misses, 1);
    assert_eq!(first.cache_stats.visibility_misses, 1);

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );
    let second = manager.shadow_frame_plan();

    assert_eq!(second.cache_stats.caster_plan_hits, 1);
    assert_eq!(second.cache_stats.visibility_hits, 1);
    assert_eq!(second.cache_stats.avoided_caster_draw_scans, 2);
    assert!(Arc::ptr_eq(&first_draws, &second.caster_sets[0].draws));
    assert_eq!(
        first_slab_ptr,
        second.caster_sets[0].instance_plan.slab_layout.as_ptr(),
        "the cached O(draws) slab/group allocation should move forward, not be cloned"
    );
    assert!(Arc::ptr_eq(
        &first_visibility,
        second.render_views[0].visible_groups_arc()
    ));
}

#[test]
fn exact_shadow_view_signature_invalidates_visibility_without_regrouping_casters() {
    let mut manager = super::FrameResourceManager::new();
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(shadowed_light(LightType::Spot));
    let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );
    manager
        .per_view_lights
        .get_mut(ViewId::Main)
        .unwrap()
        .lights[0]
        .range = 19.0;
    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &draw_plan)],
    );
    let plan = manager.shadow_frame_plan();

    assert_eq!(plan.cache_stats.caster_plan_hits, 1);
    assert_eq!(plan.cache_stats.visibility_hits, 0);
    assert_eq!(plan.cache_stats.visibility_misses, 1);
}

#[test]
fn equal_draw_values_with_new_arc_identity_do_not_hit_shadow_planning_cache() {
    let mut manager = super::FrameResourceManager::new();
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(shadowed_light(LightType::Spot));
    let first_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);
    let replacement_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &first_plan)],
    );
    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        None,
        [(ViewId::Main, &replacement_plan)],
    );
    let plan = manager.shadow_frame_plan();

    assert_eq!(plan.cache_stats.caster_plan_hits, 0);
    assert_eq!(plan.cache_stats.caster_plan_misses, 1);
    assert_eq!(plan.cache_stats.visibility_hits, 0);
    assert_eq!(plan.cache_stats.visibility_misses, 1);
}

#[test]
fn stable_mesh_generation_reuses_visible_caster_content_hash() {
    let mut manager = super::FrameResourceManager::new();
    let mesh_pool = MeshPool::default_pool();
    manager
        .per_view_lights
        .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
        .lights
        .push(shadowed_light(LightType::Spot));
    let draw_plan = prefetched_plan(vec![
        pbs_draw(1, ShadowCastMode::On),
        pbs_draw(2, ShadowCastMode::On),
    ]);

    for _ in 0..2 {
        manager.prepare_shadow_frame_for_views(
            HostShadowQuality::default(),
            Some(&mesh_pool),
            [(ViewId::Main, &draw_plan)],
        );
        let resolution = manager.shadow_frame_plan().requested_resolution;
        manager.finalize_shadow_frame_after_atlas_sync(resolution, false);
    }
    manager.prepare_shadow_frame_for_views(
        HostShadowQuality::default(),
        Some(&mesh_pool),
        [(ViewId::Main, &draw_plan)],
    );
    let plan = manager.shadow_frame_plan();

    assert_eq!(plan.cache_stats.content_hash_hits, 1);
    assert_eq!(plan.cache_stats.avoided_content_hash_draws, 2);
    assert!(
        plan.rendering_layer_indices.is_empty(),
        "stable static depth contents should reuse their persistent atlas layer"
    );
}

#[test]
fn shadow_budget_keeps_the_nearest_punctual_lights_and_directional_first() {
    // A 4-user session measured 13 shadow layers driving 81% of all draw submissions, so the
    // punctual budget is the direct lever on shadow cost. It only stays usable if truncating it
    // drops the lights nobody looks at, not whichever ones happen to be first in the list.
    use glam::Vec3;
    let far = {
        let mut light = shadowed_light(LightType::Point);
        light.position = [200.0, 0.0, 0.0];
        light
    };
    let near = {
        let mut light = shadowed_light(LightType::Point);
        light.position = [1.0, 0.0, 0.0];
        light
    };
    let sun = shadowed_light(LightType::Directional);

    // Declaration order deliberately puts the least useful light first.
    let lights = [far, near, sun];
    // Viewer at the origin looking down -Z, so the "near" light really is the near one.
    use crate::backend::ShadowCameraFit;
    use glam::Mat4;
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(1.0, 1.6, 0.1, 100.0);
    let fit = ShadowCameraFit::from_world_to_clip(proj * view, 0.1, 100.0).expect("finite fit");

    let order = super::shadow_light_priority_order(&lights, Some(&fit));

    assert_eq!(
        order.first().copied(),
        Some(2),
        "directional lights are not punctual-budgeted and must lead"
    );
    assert_eq!(
        order[1], 1,
        "the near point light must outrank the distant one despite being declared later"
    );
}

#[test]
fn shadow_light_order_is_declaration_order_without_a_camera_fit() {
    let lights = [
        shadowed_light(LightType::Point),
        shadowed_light(LightType::Spot),
    ];

    assert_eq!(
        super::shadow_light_priority_order(&lights, None),
        vec![0, 1],
        "with no viewer to rank against, ordering must stay stable"
    );
}

#[test]
fn dynamic_casters_sort_last_and_no_group_straddles_the_boundary() {
    // The whole static/dynamic shadow split rests on this: if a group could contain both a skinned
    // and a rigid caster, "redraw only the dynamic half" would silently drop static geometry.
    let mut skinned = pbs_draw(1, ShadowCastMode::On);
    skinned.skinned = true;
    let mut deformed = pbs_draw(2, ShadowCastMode::On);
    deformed.blendshape_deformed = true;
    // Declared with the dynamic casters first so ordering cannot come out right by accident.
    let draws = vec![
        skinned,
        pbs_draw(3, ShadowCastMode::On),
        deformed,
        pbs_draw(4, ShadowCastMode::On),
    ];

    let plan = super::build_shadow_caster_plan(&draws, true);
    let groups = plan.instance_plan.phase(WorldMeshPhase::ForwardOpaque);

    assert_eq!(
        plan.first_dynamic_instance, 2,
        "the two rigid casters must occupy the static prefix"
    );
    for group in groups {
        let start = group.instance_range.start;
        let end = group.instance_range.end;
        let straddles = start < plan.first_dynamic_instance && end > plan.first_dynamic_instance;
        assert!(!straddles, "group {start}..{end} straddles the boundary");
    }
    for &slot in &plan.instance_plan.slab_layout[plan.first_dynamic_instance as usize..] {
        assert!(
            super::shadow_draw_is_dynamic(&draws[slot]),
            "static caster landed in the dynamic suffix"
        );
    }
}

#[test]
fn caster_set_without_dynamic_draws_reports_an_empty_dynamic_suffix() {
    let draws = vec![pbs_draw(1, ShadowCastMode::On), pbs_draw(2, ShadowCastMode::On)];

    let plan = super::build_shadow_caster_plan(&draws, true);

    assert_eq!(
        plan.first_dynamic_instance,
        plan.instance_plan.slab_layout.len() as u32,
        "a fully static set must leave nothing in the dynamic suffix"
    );
}

mod render_scope {
    use super::super::{
        ShadowCasterContentHash, ShadowRenderScope, ShadowRetainedDepth, shadow_render_scope,
    };

    fn content(hash: u64, has_dynamic: bool) -> ShadowCasterContentHash {
        ShadowCasterContentHash {
            hash,
            reusable: true,
            has_dynamic,
        }
    }

    fn retained(live: Option<u64>, store: Option<u64>, available: bool) -> ShadowRetainedDepth {
        ShadowRetainedDepth {
            live_caster_sig: live,
            store_caster_sig: store,
            store_available: available,
        }
    }

    #[test]
    fn a_settled_static_layer_records_nothing() {
        assert_eq!(
            shadow_render_scope(content(7, false), retained(Some(7), None, true)),
            ShadowRenderScope::Reuse
        );
    }

    #[test]
    fn a_moved_static_layer_redraws_in_full() {
        assert_eq!(
            shadow_render_scope(content(7, false), retained(Some(6), None, true)),
            ShadowRenderScope::Full
        );
    }

    #[test]
    fn one_avatar_no_longer_forces_the_whole_layer_to_redraw() {
        // This is the entire point of the split. Before, has_dynamic meant a full redraw of every
        // static caster in the layer, 13 times a frame with 4 users in the room.
        assert_eq!(
            shadow_render_scope(content(7, true), retained(None, Some(7), true)),
            ShadowRenderScope::DynamicOverStatic {
                refresh_static: false
            }
        );
    }

    #[test]
    fn a_dynamic_layer_with_stale_static_depth_refreshes_the_store_first() {
        assert_eq!(
            shadow_render_scope(content(7, true), retained(None, Some(6), true)),
            ShadowRenderScope::DynamicOverStatic {
                refresh_static: true
            }
        );
        assert_eq!(
            shadow_render_scope(content(7, true), retained(None, None, true)),
            ShadowRenderScope::DynamicOverStatic {
                refresh_static: true
            }
        );
    }

    #[test]
    fn without_a_store_dynamic_layers_fall_back_to_the_old_full_redraw() {
        assert_eq!(
            shadow_render_scope(content(7, true), retained(None, Some(7), false)),
            ShadowRenderScope::Full
        );
    }

    #[test]
    fn unreusable_content_always_redraws_everything() {
        // Mutated mesh residency: stale geometry would render under an unchanged hash.
        let not_reusable = ShadowCasterContentHash {
            hash: 7,
            reusable: false,
            has_dynamic: false,
        };

        assert_eq!(
            shadow_render_scope(not_reusable, retained(Some(7), Some(7), true)),
            ShadowRenderScope::Full
        );
    }
}
