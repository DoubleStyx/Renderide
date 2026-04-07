//! [`LightCache`]: merges incremental host light updates and resolves to world space.

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState, ShadowType};

use super::types::{CachedLight, ResolvedLight};

/// Local axis for light propagation before world transform: host **`transform.forward`** (+Z).
const LOCAL_LIGHT_PROPAGATION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

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
                shadow_type: cached.state.shadow_type,
                shadow_strength: cached.state.shadow_strength,
                shadow_near_plane: cached.state.shadow_near_plane,
                shadow_bias: cached.state.shadow_bias,
                shadow_normal_bias: cached.state.shadow_normal_bias,
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
