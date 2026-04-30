//! Regular Unity-`Light`-component apply path for [`super::LightCache`].
//!
//! Lives next to [`super::buffer_renderers`] so the two host-renderable paths can evolve
//! independently while sharing [`super::LightCache::rebuild_space_vec`] and the resolve
//! pipeline in the parent module.

use glam::{Quat, Vec3};

use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::scene::world::fixup_transform_id;
use crate::shared::{LightData, LightState, LightsBufferRendererState};

use super::super::types::CachedLight;
use super::{DEAD_TRANSFORM_ID, LightCache};

impl LightCache {
    /// Applies regular [`LightState`] updates (Unity `Light` components).
    ///
    /// Three-phase pipeline matching the host's `RenderableManager.HandleUpdate`: removals via
    /// [`Vec::swap_remove`] to mirror the host's reindexing, additions append placeholder
    /// [`CachedLight`]s carrying the transform id from the `additions` buffer, and states
    /// address entries by index.
    pub fn apply_regular_lights_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightState],
    ) {
        profiling::scope!("lights::apply_regular_lights_update");
        let v = self.regular_lights.entry(space_id).or_default();

        for &idx in removals.iter().take_while(|&&i| i >= 0) {
            let idx_usize = idx as usize;
            if idx_usize >= v.len() {
                logger::warn!(
                    "light_cache: regular-light removal index {idx} out of range (space_id={space_id}, len={})",
                    v.len()
                );
                continue;
            }
            v.swap_remove(idx_usize);
        }

        for &t in additions.iter().take_while(|&&t| t >= 0) {
            v.push(CachedLight {
                data: LightData::default(),
                state: LightsBufferRendererState::default(),
                transform_id: t as usize,
            });
        }

        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx_usize = state.renderable_index as usize;
            let Some(slot) = v.get_mut(idx_usize) else {
                logger::warn!(
                    "light_cache: regular-light state index {} out of range (space_id={space_id}, len={})",
                    state.renderable_index,
                    v.len()
                );
                continue;
            };
            slot.data = LightData {
                point: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                color: Vec3::new(state.color.x, state.color.y, state.color.z),
                intensity: state.intensity,
                range: state.range,
                angle: state.spot_angle,
            };
            slot.state = LightsBufferRendererState {
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
        }

        self.rebuild_space_vec(space_id);
        self.mark_changed();
    }

    /// Rolls each regular-light entry's `transform_id` forward through `removals`. Drops entries
    /// whose own transform was removed (fixup returns `-1`) and returns `true` if anything
    /// changed so the caller can rebuild and mark dirty exactly once across both light paths.
    pub(super) fn fixup_regular_lights_for_transform_removals(
        &mut self,
        space_id: i32,
        removals: &[TransformRemovalEvent],
    ) -> bool {
        let Some(v) = self.regular_lights.get_mut(&space_id) else {
            return false;
        };
        let mut dirty = false;
        for removal in removals {
            for light in v.iter_mut() {
                if light.transform_id == DEAD_TRANSFORM_ID {
                    continue;
                }
                let fixed = fixup_transform_id(
                    light.transform_id as i32,
                    removal.removed_index,
                    removal.last_index_before_swap,
                );
                if fixed < 0 {
                    light.transform_id = DEAD_TRANSFORM_ID;
                    dirty = true;
                } else if (fixed as usize) != light.transform_id {
                    light.transform_id = fixed as usize;
                    dirty = true;
                }
            }
        }
        let before = v.len();
        v.retain(|l| {
            if l.transform_id == DEAD_TRANSFORM_ID {
                logger::warn!(
                    "light_cache: regular light dropped during transform-removal fixup (space_id={space_id})"
                );
                false
            } else {
                true
            }
        });
        if v.len() != before {
            dirty = true;
        }
        dirty
    }
}
