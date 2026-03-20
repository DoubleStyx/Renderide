//! Light cache and resolved light types for rendering.
//!
//! Stores light data per space, merges with scene transforms, and produces
//! world-space resolved lights for the render loop.
//!
//! FrooxEngine sends **incremental** `LightRenderablesUpdate` and `LightsBufferRendererUpdate`
//! batches (dirty renderables only), matching `ChangesHandlingRenderableComponentManager` on the
//! host. This cache **merges** each batch into persistent per-slot storage (like Unity’s
//! `RenderableStateChangeManager`), then flattens into [`LightCache::spaces`] for resolve.

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState};

/// Local axis for light propagation before world transform: host **`transform.forward`** (+Z).
const LOCAL_LIGHT_PROPAGATION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

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
    /// World-space propagation direction (normalized): local **+Z** (host `transform.forward`)
    /// rotated by [`LightData::orientation`] and the light’s world matrix. PBR shaders use
    /// `normalize(-world_direction)` for directional **to-light** and the same axis for spot cones.
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
///
/// Buffer renderables and regular `Light` component renderables each have their own
/// renderable-index namespace on the host; they are stored in separate maps here.
#[derive(Clone, Debug)]
pub struct LightCache {
    /// Full light data per buffer ID (GlobalUniqueId). Populated from LightsBufferRendererSubmission.
    buffers: HashMap<i32, Vec<LightData>>,
    /// Flattened per-space list of cached lights, rebuilt after each merge. Key: `space_id`.
    spaces: HashMap<i32, Vec<CachedLight>>,
    /// Expanded buffer lights per `(space_id, global_unique_id)`. Updated only for buffers present
    /// in a given [`LightsBufferRendererUpdate`] batch; untouched buffers keep their last contribution.
    buffer_contributions: HashMap<(i32, i32), Vec<CachedLight>>,
    /// Maps `(space_id, buffer_renderable_index)` to that buffer’s `global_unique_id` for removals.
    buffer_by_renderable: HashMap<(i32, i32), i32>,
    /// Regular scene lights keyed by `(space_id, light_renderable_index)`.
    regular_lights: HashMap<(i32, i32), CachedLight>,
    /// Maps `(space_id, global_unique_id)` to transform_id for buffer parents.
    buffer_transforms: HashMap<(i32, i32), usize>,
    /// Maps `(space_id, renderable_index)` to transform_id for regular lights.
    regular_light_transforms: HashMap<(i32, i32), usize>,
}

impl LightCache {
    /// Creates a new empty light cache.
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

    /// Returns the number of light buffers stored (for diagnostics).
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Stores full light data from a submission. Overwrites any existing buffer.
    pub fn store_full(&mut self, lights_buffer_unique_id: i32, light_data: Vec<LightData>) {
        self.buffers.insert(lights_buffer_unique_id, light_data);
    }

    /// Rebuilds [`LightCache::spaces`] for `space_id` from buffer contributions and regular lights
    /// in deterministic order (sorted buffer `global_unique_id`, then sorted regular renderable index).
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

    /// Applies incremental update (states, removals, additions) for a space.
    ///
    /// Merges each [`LightsBufferRendererState`] in `states` into
    /// [`LightCache::buffer_contributions`]. Removals use **buffer renderable indices** (host
    /// `RenderableIndex` values), not positions in the `states` span. Buffers not listed in
    /// `states` keep their previous contribution. Each buffer contributes one [`CachedLight`] per
    /// [`LightData`] row from [`LightCache::buffers`]. Additions are transform indices for newly
    /// registered buffer components.
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

    /// Applies regular light updates (Light components) from `lights_update`.
    ///
    /// Each [`LightState`] in `states` patches the slot `(space_id, renderable_index)`; slots not
    /// listed keep their previous data. Removals use **light renderable indices** from the host.
    /// Additions supply transform indices for newly allocated light renderables.
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

    /// Resolves cached lights to world space using scene world matrices.
    ///
    /// For buffer lights (LightsBufferRenderer), range is scaled by the parent transform's
    /// average lossy scale to match Renderite.Unity's LightsBufferRenderer behavior.
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
    use std::collections::HashMap;

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
        let resolved =
            cache.resolve_lights(
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
        let resolved =
            cache.resolve_lights(
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
}
