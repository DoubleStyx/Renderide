//! [`LightCache`]: merges incremental host light updates and resolves to world space.

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState, ShadowType};

use super::types::{CachedLight, ResolvedLight};

/// Local axis for light propagation before world transform (host forward = **+Z**).
const LOCAL_LIGHT_PROPAGATION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// CPU-side cache: buffer submissions, per-render-space flattened lights, regular vs buffer paths.
///
/// Populated from [`crate::shared::FrameSubmitData`] light batches and
/// [`crate::shared::LightsBufferRendererSubmission`]. GPU upload uses
/// [`Self::resolve_lights_with_fallback`] after world matrices are current.
#[derive(Clone, Debug)]
pub struct LightCache {
    buffers: HashMap<i32, Vec<LightData>>,
    spaces: HashMap<i32, Vec<CachedLight>>,
    buffer_contributions: HashMap<(i32, i32), Vec<CachedLight>>,
    buffer_by_renderable: HashMap<(i32, i32), i32>,
    regular_lights: HashMap<(i32, i32), CachedLight>,
    buffer_transforms: HashMap<(i32, i32), usize>,
    regular_light_transforms: HashMap<(i32, i32), usize>,
}

impl LightCache {
    /// Empty cache.
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            spaces: HashMap::new(),
            buffer_contributions: HashMap::new(),
            buffer_by_renderable: HashMap::new(),
            regular_lights: HashMap::new(),
            buffer_transforms: HashMap::new(),
            regular_light_transforms: HashMap::new(),
        }
    }

    /// Number of distinct light buffers stored from submissions (diagnostics).
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Stores full [`LightData`] rows from a host submission (overwrites prior buffer id).
    pub fn store_full(&mut self, lights_buffer_unique_id: i32, light_data: Vec<LightData>) {
        self.buffers.insert(lights_buffer_unique_id, light_data);
    }

    fn rebuild_space_vec(&mut self, space_id: i32) {
        let mut v = Vec::new();
        let mut guids: Vec<i32> = self
            .buffer_contributions
            .keys()
            .filter(|(sid, _)| *sid == space_id)
            .map(|(_, g)| *g)
            .collect();
        guids.sort_unstable();
        for g in guids {
            if let Some(chunk) = self.buffer_contributions.get(&(space_id, g)) {
                v.extend(chunk.iter().cloned());
            }
        }
        let mut r_indices: Vec<i32> = self
            .regular_lights
            .keys()
            .filter(|(sid, _)| *sid == space_id)
            .map(|(_, r)| *r)
            .collect();
        r_indices.sort_unstable();
        for r in r_indices {
            if let Some(light) = self.regular_lights.get(&(space_id, r)) {
                v.push(light.clone());
            }
        }
        self.spaces.insert(space_id, v);
    }

    /// Applies [`LightsBufferRendererUpdate`]: removals, additions (transform indices), states.
    pub fn apply_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightsBufferRendererState],
    ) {
        let removal_set: HashSet<i32> = removals.iter().take_while(|&&i| i >= 0).copied().collect();

        for &ridx in &removal_set {
            if let Some(guid) = self.buffer_by_renderable.remove(&(space_id, ridx)) {
                self.buffer_contributions.remove(&(space_id, guid));
                self.buffer_transforms.remove(&(space_id, guid));
            }
        }

        let mut additions_iter = additions
            .iter()
            .take_while(|&&t| t >= 0)
            .map(|&t| t as usize);

        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            if removal_set.contains(&state.renderable_index) {
                continue;
            }
            let guid = state.global_unique_id;
            self.buffer_by_renderable
                .insert((space_id, state.renderable_index), guid);

            let key_tf = (space_id, guid);
            let transform_id = if let Some(&tid) = self.buffer_transforms.get(&key_tf) {
                tid
            } else if let Some(tid) = additions_iter.next() {
                self.buffer_transforms.insert(key_tf, tid);
                tid
            } else {
                0
            };

            let buffer_data = self.buffers.get(&guid).map(|v| v.as_slice()).unwrap_or(&[]);
            let mut entries = Vec::with_capacity(buffer_data.len());
            for data in buffer_data {
                entries.push(CachedLight {
                    data: *data,
                    state: *state,
                    transform_id,
                });
            }
            self.buffer_contributions.insert((space_id, guid), entries);
        }

        self.rebuild_space_vec(space_id);
    }

    /// Applies regular [`LightState`] updates (Unity `Light` components).
    pub fn apply_regular_lights_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightState],
    ) {
        let removal_set: HashSet<i32> = removals.iter().take_while(|&&i| i >= 0).copied().collect();

        for &ridx in &removal_set {
            self.regular_lights.remove(&(space_id, ridx));
            self.regular_light_transforms.remove(&(space_id, ridx));
        }

        let mut additions_iter = additions
            .iter()
            .take_while(|&&t| t >= 0)
            .map(|&t| t as usize);

        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            if removal_set.contains(&state.renderable_index) {
                continue;
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
            self.regular_lights.insert(
                key,
                CachedLight {
                    data,
                    state: state_converted,
                    transform_id,
                },
            );
        }

        self.rebuild_space_vec(space_id);
    }

    /// Cached lights for `space_id` after the last apply.
    pub fn get_lights_for_space(&self, space_id: i32) -> Option<&[CachedLight]> {
        self.spaces.get(&space_id).map(|v| v.as_slice())
    }

    /// Drops all light entries tied to a removed render space.
    pub fn remove_space(&mut self, space_id: i32) {
        self.spaces.remove(&space_id);
        self.buffer_contributions
            .retain(|(sid, _), _| *sid != space_id);
        self.buffer_by_renderable
            .retain(|(sid, _), _| *sid != space_id);
        self.regular_lights.retain(|(sid, _), _| *sid != space_id);
        self.buffer_transforms
            .retain(|(sid, _), _| *sid != space_id);
        self.regular_light_transforms
            .retain(|(sid, _), _| *sid != space_id);
    }

    /// Resolves cached lights using space-local transform world matrices (caller composes root).
    pub fn resolve_lights(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        let Some(lights) = self.get_lights_for_space(space_id) else {
            return Vec::new();
        };

        let mut resolved = Vec::with_capacity(lights.len());
        for cached in lights {
            let world = get_world_matrix(cached.transform_id).unwrap_or(Mat4::IDENTITY);

            let point = cached.data.point;
            let p = Vec3::new(point.x, point.y, point.z);
            let world_pos = world.transform_point3(p);

            let ori = cached.data.orientation;
            let q = Quat::from_xyzw(ori.i, ori.j, ori.k, ori.w);
            let world_dir = (world.to_scale_rotation_translation().1 * q) * LOCAL_LIGHT_PROPAGATION;
            let world_dir = if world_dir.length_squared() > 1e-10 {
                world_dir.normalize()
            } else {
                LOCAL_LIGHT_PROPAGATION
            };

            let color = cached.data.color;
            let color = Vec3::new(color.x, color.y, color.z);

            let range = if cached.state.global_unique_id >= 0 {
                let (scale, _, _) = world.to_scale_rotation_translation();
                let uniform_scale = (scale.x + scale.y + scale.z) / 3.0;
                cached.data.range * uniform_scale
            } else {
                cached.data.range
            };

            resolved.push(ResolvedLight {
                world_position: world_pos,
                world_direction: world_dir,
                color,
                intensity: cached.data.intensity,
                range,
                spot_angle: cached.data.angle,
                light_type: cached.state.light_type,
                global_unique_id: cached.state.global_unique_id,
                shadow_type: cached.state.shadow_type,
                shadow_strength: cached.state.shadow_strength,
                shadow_near_plane: cached.state.shadow_near_plane,
                shadow_bias: cached.state.shadow_bias,
                shadow_normal_bias: cached.state.shadow_normal_bias,
            });
        }
        resolved
    }

    /// Like [`Self::resolve_lights`], but if the flattened list is empty, synthesizes from raw buffer data.
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

        let mut resolved = Vec::with_capacity(light_data.len());
        for data in light_data {
            let p = Vec3::new(data.point.x, data.point.y, data.point.z);
            let q = Quat::from_xyzw(
                data.orientation.i,
                data.orientation.j,
                data.orientation.k,
                data.orientation.w,
            );
            let world_dir = q * LOCAL_LIGHT_PROPAGATION;
            let world_dir = if world_dir.length_squared() > 1e-10 {
                world_dir.normalize()
            } else {
                LOCAL_LIGHT_PROPAGATION
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
                shadow_type: ShadowType::none,
                shadow_strength: 0.0,
                shadow_near_plane: 0.0,
                shadow_bias: 0.0,
                shadow_normal_bias: 0.0,
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
    use glam::{Mat4, Vec3};
    use nalgebra::{Quaternion, Vector3};

    use crate::shared::{LightData, LightType, LightsBufferRendererState, ShadowType};

    use super::LightCache;

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
    fn light_cache_store_full_and_apply_additions() {
        let mut cache = LightCache::new();
        let space_id = 0;
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
    fn light_cache_removals() {
        let mut cache = LightCache::new();
        let space_id = 0;
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

        cache.apply_update(space_id, &[1], &[], &[]);
        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[1].state.global_unique_id, 102);
    }

    #[test]
    fn light_cache_resolve_world_space() {
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
                    if tid == 0 {
                        Some(world_matrix)
                    } else {
                        None
                    }
                },
            );

        assert_eq!(resolved.len(), 1);
        assert!((resolved[0].world_position.x - 11.0).abs() < 1e-5);
        assert!((resolved[0].world_position.y - 0.0).abs() < 1e-5);
        assert!((resolved[0].world_position.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn resolve_lights_with_fallback_from_buffers() {
        let mut cache = LightCache::new();
        let space_id = 0;
        let light_data = vec![
            make_light_data((5.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            make_light_data((0.0, 3.0, 0.0), (0.0, 1.0, 0.0)),
        ];
        cache.store_full(space_id, light_data);

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
    fn gpu_light_from_resolved_point() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::point)]);
        let resolved = cache.resolve_lights(space_id, |_| Some(Mat4::IDENTITY));
        assert_eq!(resolved.len(), 1);
        let gpu = crate::backend::GpuLight::from_resolved(&resolved[0]);
        assert_eq!(gpu.light_type, 0);
        assert!((gpu.position[0] - 1.0).abs() < 1e-5);
    }
}
