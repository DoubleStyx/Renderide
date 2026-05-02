//! Unit tests for [`LightCache`](super::LightCache).

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState, ShadowType};

use super::LightCache;

fn make_light_data(pos: (f32, f32, f32), color: (f32, f32, f32)) -> LightData {
    LightData {
        point: Vec3::new(pos.0, pos.1, pos.2),
        orientation: Quat::IDENTITY,
        color: Vec3::new(color.0, color.1, color.2),
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
        shadow_type: ShadowType::None,
        _padding: [0; 2],
    }
}

fn make_regular_state(renderable_index: i32, intensity: f32, range: f32) -> LightState {
    LightState {
        renderable_index,
        intensity,
        range,
        spot_angle: 45.0,
        color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
        shadow_strength: 0.0,
        shadow_near_plane: 0.0,
        shadow_map_resolution_override: 0,
        shadow_bias: 0.0,
        shadow_normal_bias: 0.0,
        cookie_texture_asset_id: -1,
        r#type: LightType::Point,
        shadow_type: ShadowType::None,
        _padding: [0; 2],
    }
}

#[test]
fn light_cache_store_full_and_apply_additions() {
    let mut cache = LightCache::new();
    let space_id = 0;
    let light_data = vec![
        make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
    ];
    cache.store_full(100, light_data);

    let additions: Vec<i32> = vec![0];
    let states = vec![make_state(0, 100, LightType::Point)];
    cache.apply_update(space_id, &[], &additions, &states);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].data.point.x, 1.0);
    assert_eq!(lights[0].state.global_unique_id, 100);
    assert_eq!(lights[1].data.point.y, 2.0);
    assert_eq!(lights[1].state.light_type, LightType::Point);
}

#[test]
fn store_full_refreshes_existing_buffer_contributions() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);

    cache.store_full(100, vec![make_light_data((2.0, 0.0, 0.0), (0.0, 1.0, 0.0))]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 1);
    assert!((lights[0].data.point.x - 2.0).abs() < 1e-5);
    assert!((lights[0].data.color.y - 1.0).abs() < 1e-5);
    assert_eq!(lights[0].state.global_unique_id, 100);
}

#[test]
fn store_full_after_state_creates_buffer_contributions() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);
    assert_eq!(
        cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should exist")
            .len(),
        0
    );

    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 1);
    assert!((lights[0].data.color.y - 1.0).abs() < 1e-5);
    assert_eq!(lights[0].transform_id, 0);
}

#[test]
fn light_cache_version_changes_on_light_mutations() {
    let mut cache = LightCache::new();
    let version0 = cache.version();

    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    let version1 = cache.version();
    assert_ne!(version0, version1);

    cache.apply_update(0, &[], &[0], &[make_state(0, 100, LightType::Point)]);
    let version2 = cache.version();
    assert_ne!(version1, version2);

    cache.apply_update(0, &[0], &[], &[]);
    assert_ne!(version2, cache.version());
}

#[test]
fn apply_update_replacing_renderable_guid_removes_previous_buffer() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(200, vec![make_light_data((2.0, 0.0, 0.0), (0.0, 1.0, 0.0))]);

    cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);
    cache.apply_update(space_id, &[], &[], &[make_state(0, 200, LightType::Point)]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].state.global_unique_id, 200);
    assert!((lights[0].data.color.y - 1.0).abs() < 1e-5);
}

#[test]
fn light_cache_removals() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
    cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);

    let additions: Vec<i32> = vec![0, 1, 2];
    let states = vec![
        make_state(0, 100, LightType::Point),
        make_state(1, 101, LightType::Point),
        make_state(2, 102, LightType::Point),
    ];
    cache.apply_update(space_id, &[], &additions, &states);
    assert_eq!(
        cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights")
            .len(),
        3
    );

    cache.apply_update(space_id, &[1], &[], &[]);
    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    // Swap-remove of index 1 moves the last entry (guid=102) into slot 1. Space vec is
    // rebuilt from the dense list so output order is [guid=100, guid=102].
    assert_eq!(lights[0].state.global_unique_id, 100);
    assert_eq!(lights[1].state.global_unique_id, 102);
}

#[test]
fn light_cache_resolve_world_space() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);

    let additions: Vec<i32> = vec![0];
    let states = vec![make_state(0, 100, LightType::Point)];
    cache.apply_update(space_id, &[], &additions, &states);

    let world_matrix = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let resolved = cache.resolve_lights(space_id, |tid| (tid == 0).then_some(world_matrix));

    assert_eq!(resolved.len(), 1);
    assert!((resolved[0].world_position.x - 11.0).abs() < 1e-5);
    assert!((resolved[0].world_position.y - 0.0).abs() < 1e-5);
    assert!((resolved[0].world_position.z - 0.0).abs() < 1e-5);
}

#[test]
fn resolve_lights_with_fallback_does_not_synthesize_raw_buffers() {
    let mut cache = LightCache::new();
    let space_id = 0;
    let light_data = vec![
        make_light_data((5.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        make_light_data((0.0, 3.0, 0.0), (0.0, 1.0, 0.0)),
    ];
    cache.store_full(space_id, light_data);

    let resolved = cache.resolve_lights_with_fallback(space_id, |_| None);

    assert!(
        resolved.is_empty(),
        "raw LightData buffers require an active LightsBufferRendererState before rendering"
    );
}

#[test]
fn gpu_light_from_resolved_point() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);
    let resolved = cache.resolve_lights(space_id, |_| Some(Mat4::IDENTITY));
    assert_eq!(resolved.len(), 1);
    let gpu = crate::backend::GpuLight::from_resolved(&resolved[0]);
    assert_eq!(gpu.light_type, 0);
    assert!((gpu.position[0] - 1.0).abs() < 1e-5);
}

/// Regression: removing a middle buffer-renderer slot must swap-remove the last entry into
/// the freed index so the dense list stays aligned with the host's swap-remove reindexing.
#[test]
fn buffer_renderer_swap_remove_middle_preserves_last_entry() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
    cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);

    cache.apply_update(
        space_id,
        &[],
        &[10, 11, 12],
        &[
            make_state(0, 100, LightType::Point),
            make_state(1, 101, LightType::Point),
            make_state(2, 102, LightType::Point),
        ],
    );

    cache.apply_update(space_id, &[1], &[], &[]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].state.global_unique_id, 100);
    assert_eq!(lights[0].transform_id, 10);
    // Swapped-in entry keeps its original transform id and buffer guid.
    assert_eq!(lights[1].state.global_unique_id, 102);
    assert_eq!(lights[1].transform_id, 12);
}

/// Regression: after removing and re-adding, the dense list must have exactly one entry per
/// live renderable with its provided transform id -- no ghost carrying `transform_id = 0`.
#[test]
fn buffer_renderer_remove_then_add_has_no_ghost() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
    cache.apply_update(
        space_id,
        &[],
        &[10, 11],
        &[
            make_state(0, 100, LightType::Point),
            make_state(1, 101, LightType::Point),
        ],
    );

    cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);
    cache.apply_update(
        space_id,
        &[0],
        &[12],
        &[make_state(1, 102, LightType::Point)],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].state.global_unique_id, 101);
    assert_eq!(lights[0].transform_id, 11);
    assert_eq!(lights[1].state.global_unique_id, 102);
    assert_eq!(lights[1].transform_id, 12);
}

/// Regression: a state-only update after a swap-remove must apply to the swapped entry --
/// the original `transform_id` must be preserved (no fallback to `transform_id = 0`, which
/// would render the light at the world origin).
#[test]
fn buffer_renderer_state_only_update_after_swap_uses_swapped_transform() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
    cache.apply_update(
        space_id,
        &[],
        &[10, 11],
        &[
            make_state(0, 100, LightType::Point),
            make_state(1, 101, LightType::Point),
        ],
    );

    cache.apply_update(space_id, &[0], &[], &[]);

    cache.apply_update(
        space_id,
        &[],
        &[],
        &[make_state(0, 101, LightType::Directional)],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].state.global_unique_id, 101);
    assert_eq!(lights[0].state.light_type, LightType::Directional);
    // Critical: transform id is preserved, not reset to 0 (which would put the light at origin).
    assert_eq!(lights[0].transform_id, 11);
}

/// Regression: the same swap-remove contract must hold for regular (Unity `Light`)
/// renderables -- the reported bug was about a Light component added via the component picker
/// ending up with `renderable_index = -1` / at the world origin after add/remove cycles.
#[test]
fn regular_light_swap_remove_middle_preserves_last_entry() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.apply_regular_lights_update(
        space_id,
        &[],
        &[10, 11, 12],
        &[
            make_regular_state(0, 1.0, 10.0),
            make_regular_state(1, 2.0, 20.0),
            make_regular_state(2, 3.0, 30.0),
        ],
    );

    cache.apply_regular_lights_update(space_id, &[1], &[], &[]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert!((lights[0].data.intensity - 1.0).abs() < 1e-5);
    assert_eq!(lights[0].transform_id, 10);
    // Former index 2 swapped into index 1; its transform id and state follow it.
    assert!((lights[1].data.intensity - 3.0).abs() < 1e-5);
    assert_eq!(lights[1].transform_id, 12);
}

#[test]
fn regular_light_remove_then_add_has_no_ghost() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.apply_regular_lights_update(
        space_id,
        &[],
        &[10, 11],
        &[
            make_regular_state(0, 1.0, 10.0),
            make_regular_state(1, 2.0, 20.0),
        ],
    );

    cache.apply_regular_lights_update(space_id, &[0], &[12], &[make_regular_state(1, 3.0, 30.0)]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    // Former index 1 (intensity 2.0, transform 11) swapped into index 0.
    assert!((lights[0].data.intensity - 2.0).abs() < 1e-5);
    assert_eq!(lights[0].transform_id, 11);
    // New renderable at index 1 with transform id 12.
    assert!((lights[1].data.intensity - 3.0).abs() < 1e-5);
    assert_eq!(lights[1].transform_id, 12);
}

/// Regression: reproduces the "light at origin" symptom -- a state update for a swapped
/// regular light must address the swapped entry and preserve its non-zero transform id.
#[test]
fn regular_light_state_only_update_after_swap_uses_swapped_transform() {
    let mut cache = LightCache::new();
    let space_id = 0;
    cache.apply_regular_lights_update(
        space_id,
        &[],
        &[10, 11],
        &[
            make_regular_state(0, 1.0, 10.0),
            make_regular_state(1, 2.0, 20.0),
        ],
    );

    cache.apply_regular_lights_update(space_id, &[0], &[], &[]);

    cache.apply_regular_lights_update(space_id, &[], &[], &[make_regular_state(0, 5.0, 50.0)]);

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 1);
    assert!((lights[0].data.intensity - 5.0).abs() < 1e-5);
    // Critical: the swapped entry keeps its original transform id, not the `0` origin fallback.
    assert_eq!(lights[0].transform_id, 11);
}

/// Regression: when an unrelated transform elsewhere in the space is swap-removed and the
/// host moves the last transform into the freed slot, the light's stored `transform_id`
/// must be rolled forward so it still points at its slot's transform. Without the fixup,
/// `get_world_matrix` returns `None` and the light falls back to `Mat4::IDENTITY` -> visible
/// at world origin. Reproduces the user-reported bug directly.
#[test]
fn regular_light_transform_id_follows_swap_remove() {
    use crate::scene::transforms_apply::TransformRemovalEvent;

    let mut cache = LightCache::new();
    let space_id = 0;
    // Two lights: one at transform 5 (some unrelated slot), one at transform 42 (which
    // happens to be the last transform in the dense list on the host).
    cache.apply_regular_lights_update(
        space_id,
        &[],
        &[5, 42],
        &[
            make_regular_state(0, 1.0, 10.0),
            make_regular_state(1, 2.0, 20.0),
        ],
    );

    // Simulate: transform 10 was removed on the host (unrelated slot lost its last
    // renderable); the last transform (index 42) was swap-moved into slot 10.
    cache.fixup_for_transform_removals(
        space_id,
        &[TransformRemovalEvent {
            removed_index: 10,
            last_index_before_swap: 42,
        }],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    // Unrelated light keeps transform 5.
    assert_eq!(lights[0].transform_id, 5);
    assert!((lights[0].data.intensity - 1.0).abs() < 1e-5);
    // The light whose transform was at the pre-swap last index (42) now points at the
    // freed slot (10). Without the fixup, this would still read 42 -> out of range -> origin.
    assert_eq!(lights[1].transform_id, 10);
    assert!((lights[1].data.intensity - 2.0).abs() < 1e-5);
}

/// Regression: same swap-remove fixup for buffer-renderer lights.
#[test]
fn buffer_renderer_transform_id_follows_swap_remove() {
    use crate::scene::transforms_apply::TransformRemovalEvent;

    let mut cache = LightCache::new();
    let space_id = 0;
    cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
    cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
    cache.apply_update(
        space_id,
        &[],
        &[5, 42],
        &[
            make_state(0, 100, LightType::Point),
            make_state(1, 101, LightType::Point),
        ],
    );

    cache.fixup_for_transform_removals(
        space_id,
        &[TransformRemovalEvent {
            removed_index: 10,
            last_index_before_swap: 42,
        }],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    // Unrelated renderer (transform 5, buffer 100) is untouched.
    assert_eq!(lights[0].state.global_unique_id, 100);
    assert_eq!(lights[0].transform_id, 5);
    // Swapped-in renderer follows: transform 42 -> 10.
    assert_eq!(lights[1].state.global_unique_id, 101);
    assert_eq!(lights[1].transform_id, 10);
}

/// Regression: when a light's OWN transform is removed (its slot lost all renderables),
/// the fixup drops it defensively. In a real host stream the light would also be in the
/// same frame's removals array; this guards against that invariant regressing.
#[test]
fn regular_light_whose_own_transform_was_removed_is_dropped() {
    use crate::scene::transforms_apply::TransformRemovalEvent;

    let mut cache = LightCache::new();
    let space_id = 0;
    cache.apply_regular_lights_update(
        space_id,
        &[],
        &[5, 42],
        &[
            make_regular_state(0, 1.0, 10.0),
            make_regular_state(1, 2.0, 20.0),
        ],
    );

    // Remove transform 5 (which is the first light's transform). last_index_before_swap = 42
    // -> the second light's transform moves into slot 5.
    cache.fixup_for_transform_removals(
        space_id,
        &[TransformRemovalEvent {
            removed_index: 5,
            last_index_before_swap: 42,
        }],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].transform_id, 5);
    assert!((lights[0].data.intensity - 2.0).abs() < 1e-5);
}

/// Regression: an unrelated transform removal (not touching any light) must leave every
/// cached light's `transform_id` untouched.
#[test]
fn light_fixup_ignores_unrelated_removals() {
    use crate::scene::transforms_apply::TransformRemovalEvent;

    let mut cache = LightCache::new();
    let space_id = 0;
    cache.apply_regular_lights_update(
        space_id,
        &[],
        &[10, 11],
        &[
            make_regular_state(0, 1.0, 10.0),
            make_regular_state(1, 2.0, 20.0),
        ],
    );

    cache.fixup_for_transform_removals(
        space_id,
        &[TransformRemovalEvent {
            removed_index: 5,
            last_index_before_swap: 42,
        }],
    );

    let lights = cache
        .get_lights_for_space(space_id)
        .expect("test setup: space should have lights");
    assert_eq!(lights.len(), 2);
    assert_eq!(lights[0].transform_id, 10);
    assert_eq!(lights[1].transform_id, 11);
}
