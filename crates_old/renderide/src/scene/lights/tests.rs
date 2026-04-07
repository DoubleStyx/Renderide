use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use nalgebra::{Quaternion, Vector3};

use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState, ShadowType};

use super::{LightCache, ResolvedLight, light_casts_shadows};

fn make_light_data(pos: (f32, f32, f32), color: (f32, f32, f32)) -> LightData {
    LightData {
        point: Vector3::new(pos.0, pos.1, pos.2),
        orientation: Quaternion::identity(),
        color: Vector3::new(color.0, color.1, color.2),
        intensity: 1.0,
        range: 10.0,
        angle: 45.0,
    }
}

fn make_state(
    renderable_index: i32,
    global_unique_id: i32,
    light_type: LightType,
) -> LightsBufferRendererState {
    LightsBufferRendererState {
        renderable_index,
        global_unique_id,
        shadow_strength: 0.0,
        shadow_near_plane: 0.0,
        shadow_map_resolution: 0,
        shadow_bias: 0.0,
        shadow_normal_bias: 0.0,
        cookie_texture_asset_id: -1,
        light_type,
        shadow_type: ShadowType::none,
        _padding: [0; 2],
    }
}

#[test]
fn test_light_cache_store_full_and_apply_additions() {
    let mut cache = LightCache::new();
    let space_id = 0;
    // One buffer (global_unique_id=100) with 2 lights
    let light_data = vec![
        make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
    ];
    cache.store_full(100, light_data);

    let additions: Vec<i32> = vec![0];
    let states = vec![make_state(0, 100, LightType::point)];
    cache.apply_update(space_id, &[], &additions, &states);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].data.point.x, 1.0);
    assert_eq!(lights[0].state.global_unique_id, 100);
    assert_eq!(lights[1].data.point.y, 2.0);
    assert_eq!(lights[1].state.light_type, LightType::point);
}

#[test]
fn test_light_cache_removals() {
    let mut cache = LightCache::new();
    let space_id = 0;
    // Three buffers, one light each
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
    cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);

    let additions: Vec<i32> = vec![0, 1, 2];
    let states = vec![
        make_state(0, 100, LightType::point),
        make_state(1, 101, LightType::point),
        make_state(2, 102, LightType::point),
    ];
    cache.apply_update(space_id, &[], &additions, &states);
    assert_eq!(
        cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights")
            .len(),
        3
    );

    // Remove buffer renderable_index 1 (global_unique_id 101); dirty states may be empty.
    cache.apply_update(space_id, &[1], &[], &[]);
    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].state.global_unique_id, 100);
    assert_eq!(lights[1].state.global_unique_id, 102);
}

#[test]
fn test_light_cache_resolve_world_space() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);

    let additions: Vec<i32> = vec![0];
    let states = vec![make_state(0, 100, LightType::point)];
    cache.apply_update(space_id, &[], &additions, &states);

    let world_matrix = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let resolved = cache.resolve_lights(
        space_id,
        |tid| {
            if tid == 0 { Some(world_matrix) } else { None }
        },
    );

    assert_eq!(resolved.len(), 1);
    assert!((resolved[0].world_position.x - 11.0).abs() < 1e-5);
    assert!((resolved[0].world_position.y - 0.0).abs() < 1e-5);
    assert!((resolved[0].world_position.z - 0.0).abs() < 1e-5);
}

/// Host local forward is +Z; world propagation must match `R * local_z` (not −Z).
#[test]
fn resolve_lights_propagation_uses_local_pos_z() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]);
    cache.apply_update(
        space_id,
        &[],
        &[0],
        &[make_state(0, 100, LightType::directional)],
    );

    let world_rot_y = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let resolved = cache.resolve_lights(
        space_id,
        |tid| {
            if tid == 0 { Some(world_rot_y) } else { None }
        },
    );
    assert_eq!(resolved.len(), 1);
    let expected = world_rot_y
        .transform_vector3(Vec3::new(0.0, 0.0, 1.0))
        .normalize();
    let d = resolved[0].world_direction;
    assert!(
        (d - expected).length() < 1e-4,
        "expected {:?}, got {:?}",
        expected,
        d
    );
}

#[test]
fn test_resolve_lights_with_fallback_from_buffers() {
    let mut cache = LightCache::new();
    let space_id = 0;
    let light_data = vec![
        make_light_data((5.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        make_light_data((0.0, 3.0, 0.0), (0.0, 1.0, 0.0)),
    ];
    cache.store_full(space_id, light_data);
    // No apply_update: spaces is empty, but buffers has data.

    let resolved = cache.resolve_lights_with_fallback(space_id, |_| None);

    assert_eq!(resolved.len(), 2);
    assert!((resolved[0].world_position.x - 5.0).abs() < 1e-5);
    assert!((resolved[0].color.x - 1.0).abs() < 1e-5);
    assert_eq!(resolved[0].light_type, LightType::point);
    assert_eq!(resolved[0].global_unique_id, -1);
    assert!((resolved[1].world_position.y - 3.0).abs() < 1e-5);
    assert!((resolved[1].color.y - 1.0).abs() < 1e-5);
}

#[test]
fn test_resolve_lights_with_fallback_single_buffer_when_no_update() {
    let mut cache = LightCache::new();
    let space_id = 3;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    // No apply_update: spaces is empty. Single buffer with key 100 (not space_id).
    // Fallback uses the single buffer when buffers.len() == 1.

    let resolved = cache.resolve_lights_with_fallback(space_id, |_| None);

    assert_eq!(resolved.len(), 1);
    assert!((resolved[0].world_position.x - 1.0).abs() < 1e-5);
    assert_eq!(resolved[0].global_unique_id, -1);
}

fn make_light_state(
    renderable_index: i32,
    light_type: LightType,
    intensity: f32,
    range: f32,
    color: (f32, f32, f32),
) -> LightState {
    LightState {
        renderable_index,
        intensity,
        range,
        spot_angle: 45.0,
        color: nalgebra::Vector4::new(color.0, color.1, color.2, 1.0),
        shadow_strength: 0.0,
        shadow_near_plane: 0.0,
        shadow_map_resolution_override: 0,
        shadow_bias: 0.0,
        shadow_normal_bias: 0.0,
        cookie_texture_asset_id: -1,
        r#type: light_type,
        shadow_type: ShadowType::none,
        _padding: [0; 2],
    }
}

#[test]
fn test_apply_regular_lights_update() {
    let mut cache = LightCache::new();
    let space_id = 1;
    let states = vec![
        make_light_state(0, LightType::point, 2.0, 15.0, (1.0, 0.0, 0.0)),
        make_light_state(1, LightType::directional, 1.0, 0.0, (0.0, 1.0, 0.0)),
    ];
    let additions: Vec<i32> = vec![0, 1];
    cache.apply_regular_lights_update(space_id, &[], &additions, &states);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].data.intensity, 2.0);
    assert_eq!(lights[0].data.range, 15.0);
    assert_eq!(lights[0].state.light_type, LightType::point);
    assert_eq!(lights[0].transform_id, 0);
    assert_eq!(lights[1].state.light_type, LightType::directional);
    assert_eq!(lights[1].transform_id, 1);

    let world = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
    let resolved = cache.resolve_lights(space_id, |tid| if tid == 0 { Some(world) } else { None });
    assert_eq!(resolved.len(), 2);
    assert!((resolved[0].world_position.x - 5.0).abs() < 1e-5);
    assert!((resolved[0].intensity - 2.0).abs() < 1e-5);
}

#[test]
fn test_regular_lights_use_additions_and_persist_transforms() {
    let mut cache = LightCache::new();
    let space_id = 2;
    let states = vec![
        make_light_state(0, LightType::point, 1.0, 10.0, (1.0, 0.0, 0.0)),
        make_light_state(1, LightType::point, 1.0, 10.0, (0.0, 1.0, 0.0)),
        make_light_state(2, LightType::point, 1.0, 10.0, (0.0, 0.0, 1.0)),
    ];
    let additions: Vec<i32> = vec![100, 101, 102];
    cache.apply_regular_lights_update(space_id, &[], &additions, &states);

    let world_100 = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let world_101 = Mat4::from_translation(Vec3::new(0.0, 20.0, 0.0));
    let world_102 = Mat4::from_translation(Vec3::new(0.0, 0.0, 30.0));

    let resolved = cache.resolve_lights(space_id, |tid| match tid {
        100 => Some(world_100),
        101 => Some(world_101),
        102 => Some(world_102),
        _ => None,
    });

    assert_eq!(resolved.len(), 3);
    assert!((resolved[0].world_position.x - 10.0).abs() < 1e-5);
    assert!((resolved[0].world_position.y - 0.0).abs() < 1e-5);
    assert!((resolved[1].world_position.y - 20.0).abs() < 1e-5);
    assert!((resolved[2].world_position.z - 30.0).abs() < 1e-5);

    cache.apply_regular_lights_update(space_id, &[], &[], &states);

    let resolved2 = cache.resolve_lights(space_id, |tid| match tid {
        100 => Some(world_100),
        101 => Some(world_101),
        102 => Some(world_102),
        _ => None,
    });

    assert_eq!(resolved2.len(), 3);
    assert!((resolved2[0].world_position.x - 10.0).abs() < 1e-5);
    assert!((resolved2[1].world_position.y - 20.0).abs() < 1e-5);
    assert!((resolved2[2].world_position.z - 30.0).abs() < 1e-5);
}

#[test]
fn test_resolve_lights_buffer_light_range_scaled_by_parent() {
    let mut cache = LightCache::new();
    let space_id = 0;
    let light_data = vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))];
    cache.store_full(100, light_data);

    let additions: Vec<i32> = vec![0];
    let states = vec![make_state(0, 100, LightType::point)];
    cache.apply_update(space_id, &[], &additions, &states);

    let world_matrix =
        Mat4::from_scale_rotation_translation(Vec3::splat(2.0), Quat::IDENTITY, Vec3::ZERO);
    let resolved = cache.resolve_lights(
        space_id,
        |tid| if tid == 0 { Some(world_matrix) } else { None },
    );

    assert_eq!(resolved.len(), 1);
    assert!(
        (resolved[0].range - 20.0).abs() < 1e-5,
        "buffer light range should be scaled by parent (10 * 2 = 20), got {}",
        resolved[0].range
    );
}

#[test]
fn test_resolve_lights_with_fallback_prefers_spaces() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(
        100,
        vec![
            make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
        ],
    );
    let additions: Vec<i32> = vec![0];
    let states = vec![make_state(0, 100, LightType::point)];
    cache.apply_update(space_id, &[], &additions, &states);

    let world_matrix = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let resolved = cache.resolve_lights_with_fallback(space_id, |tid| {
        if tid == 0 { Some(world_matrix) } else { None }
    });

    assert_eq!(resolved.len(), 2);
    assert!((resolved[0].world_position.x - 11.0).abs() < 1e-5);
    assert_eq!(resolved[0].global_unique_id, 100);
}

/// Second frame carries only dirty regular lights; untouched slots must remain.
#[test]
fn test_incremental_regular_lights_patches_without_dropping_others() {
    let mut cache = LightCache::new();
    let space_id = 7;
    let states_full = vec![
        make_light_state(0, LightType::point, 1.0, 10.0, (1.0, 0.0, 0.0)),
        make_light_state(1, LightType::point, 1.0, 10.0, (0.0, 1.0, 0.0)),
        make_light_state(2, LightType::point, 1.0, 10.0, (0.0, 0.0, 1.0)),
    ];
    let additions: Vec<i32> = vec![10, 11, 12];
    cache.apply_regular_lights_update(space_id, &[], &additions, &states_full);

    let dirty_only = vec![make_light_state(
        1,
        LightType::point,
        99.0,
        10.0,
        (0.0, 1.0, 0.0),
    )];
    cache.apply_regular_lights_update(space_id, &[], &[], &dirty_only);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("space should have lights");
    assert_eq!(lights.len(), 3);
    let by_r: HashMap<i32, f32> = lights
        .iter()
        .map(|c| (c.state.renderable_index, c.data.intensity))
        .collect();
    assert!((by_r[&0] - 1.0).abs() < 1e-5);
    assert!((by_r[&1] - 99.0).abs() < 1e-5);
    assert!((by_r[&2] - 1.0).abs() < 1e-5);
}

/// Second frame updates one buffer only; the other buffer’s lights must remain.
#[test]
fn test_incremental_buffer_lights_patches_without_dropping_other_buffers() {
    let mut cache = LightCache::new();
    let space_id = 8;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(200, vec![make_light_data((0.0, 5.0, 0.0), (0.0, 1.0, 0.0))]);

    let additions: Vec<i32> = vec![0, 1];
    let states_both = vec![
        make_state(0, 100, LightType::point),
        make_state(1, 200, LightType::point),
    ];
    cache.apply_update(space_id, &[], &additions, &states_both);
    assert_eq!(
        cache
            .get_lights_for_space(space_id)
            .map(|s| s.len())
            .unwrap_or(0),
        2
    );

    let states_dirty_100_only = vec![make_state(0, 100, LightType::spot)];
    cache.apply_update(space_id, &[], &[], &states_dirty_100_only);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("space should have lights");
    assert_eq!(lights.len(), 2);
    let types: HashMap<i32, LightType> = lights
        .iter()
        .map(|c| (c.state.global_unique_id, c.state.light_type))
        .collect();
    assert_eq!(types[&100], LightType::spot);
    assert_eq!(types[&200], LightType::point);
    assert!(
        (lights
            .iter()
            .find(|c| c.state.global_unique_id == 200)
            .unwrap()
            .data
            .point
            .y
            - 5.0)
            .abs()
            < 1e-5
    );
}

/// Regular and buffer lights in one space: partial buffer update must not remove regular lights.
#[test]
fn test_mixed_regular_and_buffer_incremental() {
    let mut cache = LightCache::new();
    let space_id = 9;
    cache.store_full(50, vec![make_light_data((3.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);

    cache.apply_update(space_id, &[], &[0], &[make_state(0, 50, LightType::point)]);
    let reg = vec![make_light_state(
        0,
        LightType::point,
        2.0,
        4.0,
        (0.5, 0.5, 0.5),
    )];
    cache.apply_regular_lights_update(space_id, &[], &[99], &reg);

    assert_eq!(
        cache
            .get_lights_for_space(space_id)
            .map(|s| s.len())
            .unwrap_or(0),
        2
    );

    cache.apply_update(
        space_id,
        &[],
        &[],
        &[make_state(0, 50, LightType::directional)],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("space should have lights");
    assert_eq!(lights.len(), 2);
    let buffer = lights
        .iter()
        .find(|c| c.state.global_unique_id == 50)
        .unwrap();
    assert_eq!(buffer.state.light_type, LightType::directional);
    let regular = lights
        .iter()
        .find(|c| c.state.global_unique_id == -1)
        .unwrap();
    assert!((regular.data.intensity - 2.0).abs() < 1e-5);
    assert_eq!(regular.transform_id, 99);
}

#[test]
fn light_casts_shadows_matches_shadow_type_and_strength() {
    let r = ResolvedLight {
        world_position: Vec3::ZERO,
        world_direction: Vec3::Z,
        color: Vec3::ONE,
        intensity: 1.0,
        range: 10.0,
        spot_angle: 45.0,
        light_type: LightType::point,
        global_unique_id: 0,
        shadow_type: ShadowType::none,
        shadow_strength: 1.0,
        shadow_near_plane: 0.0,
        shadow_bias: 0.0,
        shadow_normal_bias: 0.0,
    };
    assert!(!light_casts_shadows(&r));
    let mut hard = r;
    hard.shadow_type = ShadowType::hard;
    assert!(light_casts_shadows(&hard));
    hard.shadow_strength = 0.0;
    assert!(!light_casts_shadows(&hard));
}

#[test]
fn resolve_lights_copies_shadow_fields_from_buffer_state() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]);
    let mut st = make_state(0, 100, LightType::spot);
    st.shadow_type = ShadowType::soft;
    st.shadow_strength = 0.75;
    st.shadow_near_plane = 0.05;
    st.shadow_bias = 0.02;
    st.shadow_normal_bias = 0.03;
    cache.apply_update(space_id, &[], &[0], &[st]);

    let resolved =
        cache.resolve_lights(
            space_id,
            |tid| {
                if tid == 0 { Some(Mat4::IDENTITY) } else { None }
            },
        );
    assert_eq!(resolved.len(), 1);
    let l = &resolved[0];
    assert_eq!(l.shadow_type, ShadowType::soft);
    assert!((l.shadow_strength - 0.75).abs() < 1e-5);
    assert!((l.shadow_near_plane - 0.05).abs() < 1e-5);
    assert!((l.shadow_bias - 0.02).abs() < 1e-5);
    assert!((l.shadow_normal_bias - 0.03).abs() < 1e-5);
}

#[test]
fn resolve_lights_with_fallback_uses_default_shadow_fields() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(
        space_id,
        vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))],
    );
    let resolved = cache.resolve_lights_with_fallback(space_id, |_| None);
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].shadow_type, ShadowType::none);
    assert_eq!(resolved[0].shadow_strength, 0.0);
    assert_eq!(resolved[0].shadow_near_plane, 0.0);
    assert_eq!(resolved[0].shadow_bias, 0.0);
    assert_eq!(resolved[0].shadow_normal_bias, 0.0);
}
