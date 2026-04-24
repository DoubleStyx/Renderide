//! [`LightCache`]: merges incremental host light updates and resolves to world space.

use hashbrown::HashMap;
use std::collections::HashSet;

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightsBufferRendererState};

use super::types::{CachedLight, ResolvedLight};

/// Local axis for light propagation before world transform (host forward = **+Z**).
const LOCAL_LIGHT_PROPAGATION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// CPU-side cache: buffer submissions, per-render-space flattened lights, regular vs buffer paths.
///
/// Populated from [`crate::shared::FrameSubmitData`] light batches and
/// [`crate::shared::LightsBufferRendererSubmission`]. GPU upload uses
/// [`Self::resolve_lights`] after world matrices are current.
#[derive(Clone, Debug)]
pub struct LightCache {
    version: u64,
    buffers: HashMap<i32, Vec<LightData>>,
    spaces: HashMap<i32, Vec<CachedLight>>,
    buffer_contributions: HashMap<(i32, i32), Vec<CachedLight>>,
    buffer_by_renderable: HashMap<(i32, i32), i32>,
    buffer_states: HashMap<(i32, i32), LightsBufferRendererState>,
    regular_lights: HashMap<(i32, i32), CachedLight>,
    buffer_transforms: HashMap<(i32, i32), usize>,
    regular_light_transforms: HashMap<(i32, i32), usize>,
    /// Reused by [`Self::rebuild_space_vec`] for sorted buffer GUID keys (avoids per-update allocations).
    rebuild_guids_scratch: Vec<i32>,
    /// Reused by [`Self::rebuild_space_vec`] for sorted regular renderable indices.
    rebuild_regular_indices_scratch: Vec<i32>,
}

impl LightCache {
    /// Empty cache.
    pub fn new() -> Self {
        Self {
            version: 0,
            buffers: HashMap::new(),
            spaces: HashMap::new(),
            buffer_contributions: HashMap::new(),
            buffer_by_renderable: HashMap::new(),
            buffer_states: HashMap::new(),
            regular_lights: HashMap::new(),
            buffer_transforms: HashMap::new(),
            regular_light_transforms: HashMap::new(),
            rebuild_guids_scratch: Vec::new(),
            rebuild_regular_indices_scratch: Vec::new(),
        }
    }

    /// Number of distinct light buffers stored from submissions (diagnostics).
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Monotonic generation for renderable light output.
    pub fn version(&self) -> u64 {
        self.version
    }

    fn mark_changed(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Stores full [`LightData`] rows from a host submission (overwrites prior buffer id).
    pub fn store_full(&mut self, lights_buffer_unique_id: i32, light_data: Vec<LightData>) {
        self.buffers.insert(lights_buffer_unique_id, light_data);
        self.refresh_buffer_contributions(lights_buffer_unique_id);
        self.mark_changed();
    }

    fn build_buffer_entries(
        buffer_data: &[LightData],
        state: LightsBufferRendererState,
        transform_id: usize,
    ) -> Vec<CachedLight> {
        let mut entries = Vec::with_capacity(buffer_data.len());
        for data in buffer_data {
            entries.push(CachedLight {
                data: *data,
                state,
                transform_id,
            });
        }
        entries
    }

    fn refresh_buffer_contributions(&mut self, lights_buffer_unique_id: i32) {
        profiling::scope!("lights::refresh_buffer_contributions");
        let Some(buffer_data) = self.buffers.get(&lights_buffer_unique_id).cloned() else {
            return;
        };

        let mut keys = Vec::new();
        keys.extend(
            self.buffer_states
                .keys()
                .filter(|(_, guid)| *guid == lights_buffer_unique_id)
                .copied(),
        );

        let mut dirty_spaces = Vec::new();
        for key in keys {
            let Some(&state) = self.buffer_states.get(&key) else {
                continue;
            };
            let transform_id = self.buffer_transforms.get(&key).copied().unwrap_or(0);
            let entries = Self::build_buffer_entries(&buffer_data, state, transform_id);
            self.buffer_contributions.insert(key, entries);
            dirty_spaces.push(key.0);
        }

        dirty_spaces.sort_unstable();
        dirty_spaces.dedup();
        for space_id in dirty_spaces {
            self.rebuild_space_vec(space_id);
        }
    }

    /// Rebuilds the flattened per-space light list after buffer or regular-light map changes.
    ///
    /// Reuses the [`Vec`] stored in [`Self::spaces`] and scratch buffers for sort keys so steady-state
    /// updates avoid allocating new vectors every time.
    fn rebuild_space_vec(&mut self, space_id: i32) {
        profiling::scope!("lights::rebuild_space_vec");
        let v = self.spaces.entry(space_id).or_default();
        v.clear();

        self.rebuild_guids_scratch.clear();
        self.rebuild_guids_scratch.extend(
            self.buffer_contributions
                .keys()
                .filter(|(sid, _)| *sid == space_id)
                .map(|(_, g)| *g),
        );
        self.rebuild_guids_scratch.sort_unstable();
        for g in self.rebuild_guids_scratch.iter().copied() {
            if let Some(chunk) = self.buffer_contributions.get(&(space_id, g)) {
                v.extend(chunk.iter().cloned());
            }
        }

        self.rebuild_regular_indices_scratch.clear();
        self.rebuild_regular_indices_scratch.extend(
            self.regular_lights
                .keys()
                .filter(|(sid, _)| *sid == space_id)
                .map(|(_, r)| *r),
        );
        self.rebuild_regular_indices_scratch.sort_unstable();
        for r in self.rebuild_regular_indices_scratch.iter().copied() {
            if let Some(light) = self.regular_lights.get(&(space_id, r)) {
                v.push(light.clone());
            }
        }
    }

    /// Applies [`LightsBufferRendererUpdate`]: removals, additions (transform indices), states.
    pub fn apply_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightsBufferRendererState],
    ) {
        profiling::scope!("lights::apply_update");
        let removal_set: HashSet<i32> = removals.iter().take_while(|&&i| i >= 0).copied().collect();

        for &ridx in &removal_set {
            if let Some(guid) = self.buffer_by_renderable.remove(&(space_id, ridx)) {
                self.buffer_contributions.remove(&(space_id, guid));
                self.buffer_transforms.remove(&(space_id, guid));
                self.buffer_states.remove(&(space_id, guid));
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
            if let Some(previous_guid) = self
                .buffer_by_renderable
                .insert((space_id, state.renderable_index), guid)
            {
                if previous_guid != guid {
                    self.buffer_contributions.remove(&(space_id, previous_guid));
                    self.buffer_transforms.remove(&(space_id, previous_guid));
                    self.buffer_states.remove(&(space_id, previous_guid));
                }
            }

            let key_tf = (space_id, guid);
            let transform_id = if let Some(&tid) = self.buffer_transforms.get(&key_tf) {
                tid
            } else if let Some(tid) = additions_iter.next() {
                self.buffer_transforms.insert(key_tf, tid);
                tid
            } else {
                0
            };
            self.buffer_states.insert(key_tf, *state);

            let buffer_data = self.buffers.get(&guid).map(|v| v.as_slice()).unwrap_or(&[]);
            let entries = Self::build_buffer_entries(buffer_data, *state, transform_id);
            self.buffer_contributions.insert((space_id, guid), entries);
        }

        self.rebuild_space_vec(space_id);
        self.mark_changed();
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
                point: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                color: Vec3::new(state.color.x, state.color.y, state.color.z),
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
        self.mark_changed();
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
        self.buffer_states.retain(|(sid, _), _| *sid != space_id);
        self.regular_lights.retain(|(sid, _), _| *sid != space_id);
        self.buffer_transforms
            .retain(|(sid, _), _| *sid != space_id);
        self.regular_light_transforms
            .retain(|(sid, _), _| *sid != space_id);
        self.mark_changed();
    }

    /// Resolves cached lights using space-local transform world matrices (caller composes root).
    pub fn resolve_lights(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        let mut out = Vec::new();
        self.resolve_lights_into(space_id, get_world_matrix, &mut out);
        out
    }

    /// Like [`Self::resolve_lights`], but appends into `out` (caller clears when replacing content).
    pub fn resolve_lights_into(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
        out: &mut Vec<ResolvedLight>,
    ) {
        profiling::scope!("lights::resolve_lights_into");
        let Some(lights) = self.get_lights_for_space(space_id) else {
            return;
        };

        out.reserve(lights.len());
        for cached in lights {
            let world = get_world_matrix(cached.transform_id).unwrap_or(Mat4::IDENTITY);

            let point = cached.data.point;
            let p = Vec3::new(point.x, point.y, point.z);
            let world_pos = world.transform_point3(p);

            let ori = cached.data.orientation;
            let q = ori;
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

            out.push(ResolvedLight {
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
    }

    /// Alias for [`Self::resolve_lights`] kept for callers that distinguish the "with fallback" name.
    ///
    /// Raw buffer submissions are not renderable by themselves; a matching renderer state is required.
    pub fn resolve_lights_with_fallback(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        self.resolve_lights(space_id, get_world_matrix)
    }

    /// Alias for [`Self::resolve_lights_into`] kept for callers that distinguish the "with fallback" name.
    pub fn resolve_lights_with_fallback_into(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
        out: &mut Vec<ResolvedLight>,
    ) {
        self.resolve_lights_into(space_id, get_world_matrix, out);
    }
}

impl Default for LightCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::shared::{LightData, LightType, LightsBufferRendererState, ShadowType};

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
}
