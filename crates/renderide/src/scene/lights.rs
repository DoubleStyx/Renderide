//! Light cache and resolved light types for rendering.
//!
//! Stores light data per space, merges with scene transforms, and produces
//! world-space resolved lights for the render loop.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState};

/// Cached light entry combining pose data from submission with state from updates.
#[derive(Clone, Debug)]
pub struct CachedLight {
    /// Local-space pose and color from LightsBufferRendererSubmission.
    pub data: LightData,
    /// Renderable index, type, and shadow params from LightsBufferRendererUpdate.
    pub state: LightsBufferRendererState,
    /// Transform index for world matrix lookup. Set from additions (host sends transform indices).
    pub transform_id: usize,
}

impl CachedLight {
    /// Creates a new cached light with default state when only data is available.
    pub fn from_data(data: LightData) -> Self {
        Self {
            data,
            state: LightsBufferRendererState::default(),
            transform_id: 0,
        }
    }
}

/// Resolved light in world space, ready for the render loop.
#[derive(Clone, Debug)]
pub struct ResolvedLight {
    /// World-space position.
    pub world_position: Vec3,
    /// World-space direction (normalized; -Z for spot/directional).
    pub world_direction: Vec3,
    /// RGB color.
    pub color: Vec3,
    /// Light intensity.
    pub intensity: f32,
    /// Attenuation range (point/spot).
    pub range: f32,
    /// Spot angle in degrees (spot only).
    pub spot_angle: f32,
    /// Light type: point, directional, or spot.
    pub light_type: LightType,
    /// Global unique ID for consumed feedback.
    pub global_unique_id: i32,
}

/// Cache of lights per lights buffer (keyed by GlobalUniqueId) and per space.
///
/// A RenderSpace can have multiple LightsBufferRenderer components; each has its own
/// GlobalUniqueId and buffer of lights. We aggregate lights from all buffers in a space.
#[derive(Clone, Debug)]
pub struct LightCache {
    /// Full light data per buffer ID (GlobalUniqueId). Populated from LightsBufferRendererSubmission.
    buffers: HashMap<i32, Vec<LightData>>,
    /// Per-space light entries (data + state). Key: space_id.
    spaces: HashMap<i32, Vec<CachedLight>>,
    /// Maps (space_id, global_unique_id) to transform_id for buffers we've seen.
    buffer_transforms: HashMap<(i32, i32), usize>,
    /// Maps (space_id, renderable_index) to transform_id for regular lights (Light components).
    /// Persisted across frames; additions provide transform indices for new lights.
    regular_light_transforms: HashMap<(i32, i32), usize>,
}

impl LightCache {
    /// Creates a new empty light cache.
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            spaces: HashMap::new(),
            buffer_transforms: HashMap::new(),
            regular_light_transforms: HashMap::new(),
        }
    }

    /// Returns the number of light buffers stored (for diagnostics).
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Stores full light data from a submission. Overwrites any existing buffer.
    pub fn store_full(&mut self, lights_buffer_unique_id: i32, light_data: Vec<LightData>) {
        self.buffers.insert(lights_buffer_unique_id, light_data);
    }

    /// Applies incremental update (states, removals, additions) for a space.
    ///
    /// Aggregates lights from all buffers in the space using state.global_unique_id.
    /// Each buffer (LightsBufferRenderer) has many lights; we create one CachedLight per light.
    /// Additions are transform indices for new buffers; we persist transform_id per buffer.
    pub fn apply_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightsBufferRendererState],
    ) {
        // 1. Apply removals: remove states at these indices (buffer/slot indices)
        let mut removal_indices: Vec<usize> = removals
            .iter()
            .take_while(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();
        removal_indices.sort_by(|a, b| b.cmp(a));

        let remaining_states: Vec<LightsBufferRendererState> = states
            .iter()
            .enumerate()
            .filter(|(i, _)| !removal_indices.contains(i))
            .map(|(_, s)| *s)
            .collect();

        // Clear persisted transforms for removed buffers
        for &idx in &removal_indices {
            if let Some(state) = states.get(idx) {
                self.buffer_transforms
                    .remove(&(space_id, state.global_unique_id));
            }
        }

        // 2. Build (state, transform_id) for each remaining buffer
        let mut additions_iter = additions
            .iter()
            .take_while(|&&t| t >= 0)
            .map(|&t| t as usize);

        let mut buffer_infos: Vec<(LightsBufferRendererState, usize)> = Vec::new();
        for state in &remaining_states {
            if state.renderable_index < 0 {
                break;
            }
            let key = (space_id, state.global_unique_id);
            let transform_id = if let Some(&tid) = self.buffer_transforms.get(&key) {
                tid
            } else if let Some(tid) = additions_iter.next() {
                self.buffer_transforms.insert(key, tid);
                tid
            } else {
                0
            };
            buffer_infos.push((*state, transform_id));
        }

        // 3. Rebuild entries: for each buffer, emit one CachedLight per LightData
        let mut entries = Vec::new();
        for (state, transform_id) in buffer_infos {
            let buffer_data = self
                .buffers
                .get(&state.global_unique_id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            for data in buffer_data {
                entries.push(CachedLight {
                    data: *data,
                    state,
                    transform_id,
                });
            }
        }

        self.spaces.insert(space_id, entries);
    }

    /// Applies regular light updates (Light components) from `lights_update`.
    ///
    /// Each LightState describes one scene light. Position/direction come from the transform
    /// at the index provided in `additions` (host sends transform indices). We persist the
    /// mapping (space_id, renderable_index) -> transform_id across frames.
    pub fn apply_regular_lights_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightState],
    ) {
        let mut removal_indices: Vec<usize> = removals
            .iter()
            .take_while(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();
        removal_indices.sort_by(|a, b| b.cmp(a));

        for &idx in &removal_indices {
            if let Some(state) = states.get(idx)
                && state.renderable_index >= 0
            {
                self.regular_light_transforms
                    .remove(&(space_id, state.renderable_index));
            }
        }

        let remaining_states: Vec<LightState> = states
            .iter()
            .enumerate()
            .filter(|(i, _)| !removal_indices.contains(i))
            .map(|(_, s)| *s)
            .collect();

        let mut additions_iter = additions
            .iter()
            .take_while(|&&t| t >= 0)
            .map(|&t| t as usize);

        let mut entries = Vec::with_capacity(remaining_states.len());
        for state in &remaining_states {
            if state.renderable_index < 0 {
                break;
            }
            let key = (space_id, state.renderable_index);
            let transform_id = if let Some(&tid) = self.regular_light_transforms.get(&key) {
                tid
            } else if let Some(tid) = additions_iter.next() {
                self.regular_light_transforms.insert(key, tid);
                tid
            } else {
                0
            };

            let data = LightData {
                point: nalgebra::Vector3::new(0.0, 0.0, 0.0),
                orientation: nalgebra::Quaternion::identity(),
                color: nalgebra::Vector3::new(state.color.x, state.color.y, state.color.z),
                intensity: state.intensity,
                range: state.range,
                angle: state.spot_angle,
            };
            let state_converted = LightsBufferRendererState {
                renderable_index: state.renderable_index,
                global_unique_id: -1,
                shadow_strength: state.shadow_strength,
                shadow_near_plane: state.shadow_near_plane,
                shadow_map_resolution: state.shadow_map_resolution_override,
                shadow_bias: state.shadow_bias,
                shadow_normal_bias: state.shadow_normal_bias,
                cookie_texture_asset_id: state.cookie_texture_asset_id,
                light_type: state.r#type,
                shadow_type: state.shadow_type,
                _padding: [0; 2],
            };
            entries.push(CachedLight {
                data,
                state: state_converted,
                transform_id,
            });
        }

        let space = self.spaces.entry(space_id).or_default();
        space.retain(|c| c.state.global_unique_id >= 0);
        space.extend(entries);
    }

    /// Returns cached lights for a space. Call after apply_update.
    pub fn get_lights_for_space(&self, space_id: i32) -> Option<&[CachedLight]> {
        self.spaces.get(&space_id).map(|v| v.as_slice())
    }

    /// Returns mutable cached lights for a space.
    pub fn get_lights_for_space_mut(&mut self, space_id: i32) -> Option<&mut Vec<CachedLight>> {
        self.spaces.get_mut(&space_id)
    }

    /// Removes a space's lights (e.g. when space is removed).
    pub fn remove_space(&mut self, space_id: i32) {
        self.spaces.remove(&space_id);
        self.buffer_transforms
            .retain(|(sid, _), _| *sid != space_id);
        self.regular_light_transforms
            .retain(|(sid, _), _| *sid != space_id);
    }

    /// Resolves cached lights to world space using scene world matrices.
    pub fn resolve_lights(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        let Some(lights) = self.get_lights_for_space(space_id) else {
            return Vec::new();
        };

        const FORWARD: Vec3 = Vec3::new(0.0, 0.0, -1.0);

        let mut resolved = Vec::with_capacity(lights.len());
        for cached in lights {
            let world = get_world_matrix(cached.transform_id).unwrap_or(Mat4::IDENTITY);

            let point = cached.data.point;
            let p = Vec3::new(point.x, point.y, point.z);
            let world_pos = world.transform_point3(p);

            let ori = cached.data.orientation;
            let q = Quat::from_xyzw(ori.i, ori.j, ori.k, ori.w);
            let world_dir = (world.to_scale_rotation_translation().1 * q) * FORWARD;
            let world_dir = if world_dir.length_squared() > 1e-10 {
                world_dir.normalize()
            } else {
                Vec3::NEG_Z
            };

            let color = cached.data.color;
            let color = Vec3::new(color.x, color.y, color.z);

            resolved.push(ResolvedLight {
                world_position: world_pos,
                world_direction: world_dir,
                color,
                intensity: cached.data.intensity,
                range: cached.data.range,
                spot_angle: cached.data.angle,
                light_type: cached.state.light_type,
                global_unique_id: cached.state.global_unique_id,
            });
        }
        resolved
    }

    /// Resolves lights for a space, falling back to raw buffer data when `spaces` is empty.
    ///
    /// When the host sends `LightsBufferRendererSubmission` but not `LightsBufferRendererUpdate`
    /// with additions, `spaces` stays empty and `resolve_lights` returns nothing. This method
    /// synthesizes resolved lights from `buffers` using default state (point light, identity
    /// transform) so lights can render.
    ///
    /// Fallback logic: (1) If a buffer exists with key `space_id`, use it (legacy single-buffer).
    /// (2) If exactly one buffer exists and no spaces data, use it (host may omit update for
    /// single-buffer case). Multiple buffers without update cannot be mapped to a space.
    pub fn resolve_lights_with_fallback(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        let from_spaces = self.resolve_lights(space_id, get_world_matrix);
        if !from_spaces.is_empty() {
            return from_spaces;
        }

        let light_data = self.buffers.get(&space_id).or_else(|| {
            if self.buffers.len() == 1 {
                self.buffers.values().next()
            } else {
                None
            }
        });
        let Some(light_data) = light_data else {
            return Vec::new();
        };
        if light_data.is_empty() {
            return Vec::new();
        }

        const FORWARD: Vec3 = Vec3::new(0.0, 0.0, -1.0);

        let mut resolved = Vec::with_capacity(light_data.len());
        for data in light_data {
            let p = Vec3::new(data.point.x, data.point.y, data.point.z);
            let q = Quat::from_xyzw(
                data.orientation.i,
                data.orientation.j,
                data.orientation.k,
                data.orientation.w,
            );
            let world_dir = q * FORWARD;
            let world_dir = if world_dir.length_squared() > 1e-10 {
                world_dir.normalize()
            } else {
                Vec3::NEG_Z
            };

            resolved.push(ResolvedLight {
                world_position: p,
                world_direction: world_dir,
                color: Vec3::new(data.color.x, data.color.y, data.color.z),
                intensity: data.intensity,
                range: data.range,
                spot_angle: data.angle,
                light_type: LightType::point,
                global_unique_id: -1,
            });
        }
        resolved
    }
}

impl Default for LightCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::ShadowType;
    use nalgebra::{Quaternion, Vector3};

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

        let lights = cache.get_lights_for_space(space_id).unwrap();
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
        assert_eq!(cache.get_lights_for_space(space_id).unwrap().len(), 3);

        // Remove buffer at index 1 (global_unique_id 101)
        cache.apply_update(space_id, &[1], &[], &states);
        let lights = cache.get_lights_for_space(space_id).unwrap();
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
        let resolved =
            cache.resolve_lights(
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

        let lights = cache.get_lights_for_space(space_id).unwrap();
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].data.intensity, 2.0);
        assert_eq!(lights[0].data.range, 15.0);
        assert_eq!(lights[0].state.light_type, LightType::point);
        assert_eq!(lights[0].transform_id, 0);
        assert_eq!(lights[1].state.light_type, LightType::directional);
        assert_eq!(lights[1].transform_id, 1);

        let world = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let resolved =
            cache.resolve_lights(space_id, |tid| if tid == 0 { Some(world) } else { None });
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
}
