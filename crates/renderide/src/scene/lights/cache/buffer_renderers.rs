//! Buffer-renderer apply path for [`super::LightCache`].
//!
//! `LightsBufferRendererSubmission` carries packed [`crate::shared::LightData`] rows keyed by
//! `global_unique_id` with one [`super::BufferRenderer`] per host renderable; per-space output
//! is fanned out by [`super::LightCache::rebuild_space_vec`].

use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::scene::world::fixup_transform_id;
use crate::shared::LightsBufferRendererState;

use super::{BufferRenderer, DEAD_TRANSFORM_ID, LightCache};

impl LightCache {
    /// Applies [`crate::shared::LightsBufferRendererUpdate`]: removals, additions, then states.
    ///
    /// Order is fixed (**removals → additions → states**) to mirror the host
    /// `RenderableManager.HandleUpdate`. Removal uses [`Vec::swap_remove`] so the renderer's
    /// dense list stays in lockstep with the host's swap-remove reindexing; additions append
    /// placeholder entries whose transform ids come from the `additions` buffer; state rows
    /// address those entries by index.
    pub fn apply_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightsBufferRendererState],
    ) {
        profiling::scope!("lights::apply_update");
        let v = self.buffer_renderers.entry(space_id).or_default();

        for &idx in removals.iter().take_while(|&&i| i >= 0) {
            let idx_usize = idx as usize;
            if idx_usize >= v.len() {
                logger::warn!(
                    "light_cache: buffer-renderer removal index {idx} out of range (space_id={space_id}, len={})",
                    v.len()
                );
                continue;
            }
            v.swap_remove(idx_usize);
        }

        for &t in additions.iter().take_while(|&&t| t >= 0) {
            v.push(BufferRenderer {
                transform_id: t as usize,
                state: LightsBufferRendererState::default(),
            });
        }

        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx_usize = state.renderable_index as usize;
            let Some(slot) = v.get_mut(idx_usize) else {
                logger::warn!(
                    "light_cache: buffer-renderer state index {} out of range (space_id={space_id}, len={})",
                    state.renderable_index,
                    v.len()
                );
                continue;
            };
            slot.state = *state;
        }

        self.rebuild_space_vec(space_id);
        self.mark_changed();
    }

    /// Rolls each buffer-renderer entry's `transform_id` forward through `removals`. Drops
    /// entries whose own transform was removed and returns `true` if anything changed.
    pub(super) fn fixup_buffer_renderers_for_transform_removals(
        &mut self,
        space_id: i32,
        removals: &[TransformRemovalEvent],
    ) -> bool {
        let Some(v) = self.buffer_renderers.get_mut(&space_id) else {
            return false;
        };
        let mut dirty = false;
        for removal in removals {
            for br in v.iter_mut() {
                if br.transform_id == DEAD_TRANSFORM_ID {
                    continue;
                }
                let fixed = fixup_transform_id(
                    br.transform_id as i32,
                    removal.removed_index,
                    removal.last_index_before_swap,
                );
                if fixed < 0 {
                    br.transform_id = DEAD_TRANSFORM_ID;
                    dirty = true;
                } else if (fixed as usize) != br.transform_id {
                    br.transform_id = fixed as usize;
                    dirty = true;
                }
            }
        }
        let before = v.len();
        v.retain(|br| {
            if br.transform_id == DEAD_TRANSFORM_ID {
                logger::warn!(
                    "light_cache: buffer renderer dropped during transform-removal fixup (space_id={space_id})"
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
