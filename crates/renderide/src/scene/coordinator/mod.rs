//! Owns all [`RenderSpaceState`](super::render_space::RenderSpaceState) instances and applies per-frame host data.

pub mod parallel_apply;
mod world_queries;

use hashbrown::HashMap;
use std::collections::HashSet;

use glam::Mat4;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::FrameSubmitData;
use crate::shared::RenderingContext;

use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::lights::{
    apply_light_renderables_update, apply_lights_buffer_renderers_update, LightCache, ResolvedLight,
};
use super::math::multiply_root;
use super::render_overrides::MeshRendererOverrideTarget;
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::{compute_world_matrices_for_space, ensure_cache_shapes, WorldTransformCache};

use parallel_apply::{
    apply_extracted_render_space_update, extract_render_space_update, light_updates_view,
    ExtractedRenderSpaceUpdate, PerSpaceApplyInputs,
};

/// Warns when more than one non-overlay render space is marked active (breaks main-camera assumptions).
fn warn_if_multiple_active_non_overlay_spaces(data: &FrameSubmitData) {
    let active_non_overlay = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .count();
    if active_non_overlay > 1 {
        logger::warn!(
            "FrameSubmitData: {active_non_overlay} active non-overlay render spaces (expected at most one for main camera parity)"
        );
    }
}

/// Best-effort reflection probe SH2 host task markers (Unity `RenderSpace.HandleUpdate` preamble).
fn mark_reflection_probe_sh2_task_failures(shm: &mut SharedMemoryAccessor, data: &FrameSubmitData) {
    for update in &data.render_spaces {
        if let Some(ref sh2) = update.reflection_probe_sh2_taks {
            super::reflection_probe_sh2::mark_reflection_probe_sh2_tasks_failed(shm, sh2);
        }
    }
}

#[cfg(test)]
mod tests;

/// Scene registry: one entry per host render space, Unity `RenderingManager` dictionary semantics.
pub struct SceneCoordinator {
    spaces: HashMap<RenderSpaceId, RenderSpaceState>,
    world_caches: HashMap<RenderSpaceId, WorldTransformCache>,
    world_dirty: HashSet<RenderSpaceId>,
    light_cache: LightCache,
    /// Reused in [`Self::flush_world_caches`] to avoid per-flush `Vec` allocation.
    world_dirty_flush_scratch: Vec<RenderSpaceId>,
    /// Reused in [`Self::remove_render_spaces_not_in_submit`].
    remove_spaces_scratch: Vec<RenderSpaceId>,
}

impl Default for SceneCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCoordinator {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            spaces: HashMap::new(),
            world_caches: HashMap::new(),
            world_dirty: HashSet::new(),
            light_cache: LightCache::new(),
            world_dirty_flush_scratch: Vec::new(),
            remove_spaces_scratch: Vec::new(),
        }
    }

    /// Host light cache (submissions + incremental updates); CPU-side only.
    pub fn light_cache(&self) -> &LightCache {
        &self.light_cache
    }

    /// Mutable light cache ([`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) store, tests).
    pub fn light_cache_mut(&mut self) -> &mut LightCache {
        &mut self.light_cache
    }

    /// Render space ids currently present (stable order not guaranteed).
    pub fn render_space_ids(&self) -> impl Iterator<Item = RenderSpaceId> + '_ {
        self.spaces.keys().copied()
    }

    /// Number of host render spaces currently tracked.
    pub fn render_space_count(&self) -> usize {
        self.spaces.len()
    }

    /// Total static and skinned mesh renderables across all spaces.
    pub fn total_mesh_renderable_count(&self) -> usize {
        self.spaces
            .values()
            .map(|s| s.static_mesh_renderers.len() + s.skinned_mesh_renderers.len())
            .sum()
    }

    /// Resolves active lights in world space for `id`.
    pub fn resolve_lights_world(&self, id: RenderSpaceId) -> Vec<ResolvedLight> {
        let sid = id.0;
        self.light_cache
            .resolve_lights(sid, |transform_idx| self.world_matrix(id, transform_idx))
    }

    /// Appends world-space lights for `id` into `out` (caller typically [`Vec::clear`]s once per frame).
    /// Same semantics as [`Self::resolve_lights_world`] without allocating a new [`Vec`].
    pub fn resolve_lights_world_into(&self, id: RenderSpaceId, out: &mut Vec<ResolvedLight>) {
        let sid = id.0;
        self.light_cache.resolve_lights_into(
            sid,
            |transform_idx| self.world_matrix(id, transform_idx),
            out,
        );
    }

    /// Read-only access for debugging / future systems.
    pub fn space(&self, id: RenderSpaceId) -> Option<&RenderSpaceState> {
        self.spaces.get(&id)
    }

    /// Main non-overlay render space, matching Unity's single active main-space expectation.
    pub fn active_main_space(&self) -> Option<&RenderSpaceState> {
        self.spaces
            .values()
            .filter(|s| s.is_active && !s.is_overlay)
            .min_by_key(|s| s.id.0)
    }

    /// Current head-output render context for the main view.
    pub fn active_main_render_context(&self) -> RenderingContext {
        self.active_main_space()
            .map(RenderSpaceState::main_render_context)
            .unwrap_or(RenderingContext::UserView)
    }

    /// Cached world matrix from the host transform hierarchy (parent chain only).
    ///
    /// This matches object/light/bone placement: [`RenderSpaceState::root_transform`] is **not**
    /// applied here—it drives the view basis via [`RenderSpaceState::view_transform`], not mesh
    /// model matrices.
    pub fn world_matrix(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_caches
            .get(&id)?
            .world_matrices
            .get(transform_index)
            .copied()
    }

    /// Alias for [`Self::world_matrix`].
    pub fn world_matrix_local(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_matrix(id, transform_index)
    }

    /// Hierarchy world matrix left-multiplied by [`RenderSpaceState::root_transform`].
    ///
    /// Use only when a host contract explicitly requires this composite. Default rendering uses
    /// [`Self::world_matrix`].
    pub fn world_matrix_including_space_root(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        let local = self.world_matrix(id, transform_index)?;
        Some(multiply_root(local, &space.root_transform))
    }

    /// Material override for the given renderer + slot in the given render context.
    pub fn overridden_material_asset_id(
        &self,
        space_id: RenderSpaceId,
        context: RenderingContext,
        skinned: bool,
        renderable_index: usize,
        slot_index: usize,
    ) -> Option<i32> {
        let space = self.spaces.get(&space_id)?;
        let target = if skinned {
            MeshRendererOverrideTarget::Skinned(renderable_index as i32)
        } else {
            MeshRendererOverrideTarget::Static(renderable_index as i32)
        };
        space.overridden_material_asset_id(context, target, slot_index)
    }

    /// Recomputes cached world matrices for every dirty space (no-op if caches clean).
    ///
    /// The per-space solve is data-independent (each [`WorldTransformCache`] is keyed by a
    /// distinct [`RenderSpaceId`]), so we drain dirty caches into a `Vec`, run the incremental
    /// solve in parallel via rayon, and reinsert successful results afterwards. On error the
    /// offending space is left marked dirty so the next flush retries; the first error observed
    /// is surfaced as the function result.
    pub fn flush_world_caches(&mut self) -> Result<(), SceneError> {
        profiling::scope!("scene::flush_world_caches");
        use rayon::prelude::*;

        self.world_dirty_flush_scratch.clear();
        self.world_dirty_flush_scratch
            .extend(self.world_dirty.iter().copied());

        // Drop caches for dirty spaces that no longer exist and drain caches for surviving
        // spaces into a work vec. This runs on the main thread because it mutates `self`.
        let mut work: Vec<(RenderSpaceId, WorldTransformCache)> =
            Vec::with_capacity(self.world_dirty_flush_scratch.len());
        for id in self.world_dirty_flush_scratch.iter().copied() {
            if !self.spaces.contains_key(&id) {
                self.world_caches.remove(&id);
                self.world_dirty.remove(&id);
                continue;
            }
            let cache = self.world_caches.remove(&id).unwrap_or_default();
            work.push((id, cache));
        }

        if work.is_empty() {
            return Ok(());
        }

        // `&self.spaces` is a shared borrow across rayon workers; `HashMap::get` is `Sync` for
        // `Sync` keys and values. Each task owns its own cache.
        let spaces = &self.spaces;
        let results: Vec<(RenderSpaceId, Result<WorldTransformCache, SceneError>)> = work
            .into_par_iter()
            .map(|(id, mut cache)| {
                let space = match spaces.get(&id) {
                    Some(s) => s,
                    // Space was removed between drain and dispatch — preserve cache as-is so the
                    // reinsert step below drops it via the `Ok` path (caller treats this as a no-op).
                    None => return (id, Ok(cache)),
                };
                let n = space.nodes.len();
                ensure_cache_shapes(&mut cache, n, false);
                let result = compute_world_matrices_for_space(
                    id.0,
                    &space.nodes,
                    &space.node_parents,
                    &mut cache,
                );
                (id, result.map(|()| cache))
            })
            .collect();

        let mut first_err: Option<SceneError> = None;
        for (id, result) in results {
            match result {
                Ok(cache) => {
                    self.world_caches.insert(id, cache);
                    self.world_dirty.remove(&id);
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    // Leave `world_dirty` set so the next flush retries this space.
                }
            }
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    }

    /// Applies [`FrameSubmitData`]: transforms, meshes, skinned meshes, lights (Unity order).
    ///
    /// Two-phase pipeline:
    ///
    /// 1. **Phase A (serial):** [`extract_render_space_update`] reads every shared-memory buffer
    ///    referenced by each [`RenderSpaceUpdate`] into owned vectors. Header fields
    ///    ([`RenderSpaceState::apply_update_header`]) are also applied here while we still hold a
    ///    serial borrow on the spaces map.
    /// 2. **Phase B (parallel above one space):** per-space mutation runs over the drained
    ///    `(RenderSpaceState, WorldTransformCache, ExtractedRenderSpaceUpdate)` tuples. Each
    ///    tuple owns disjoint state, so rayon workers cannot race.
    /// 3. **Phase C (serial):** light updates target the shared
    ///    [`crate::scene::lights::LightCache`] and run after the parallel apply.
    pub fn apply_frame_submit(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &FrameSubmitData,
    ) -> Result<(), SceneError> {
        profiling::scope!("scene::apply_frame_submit");
        warn_if_multiple_active_non_overlay_spaces(data);
        mark_reflection_probe_sh2_task_failures(shm, data);

        let mut seen = HashSet::new();

        // Phase A: serial pre-extract + ensure entries + apply header fields.
        let mut extracted_per_space: Vec<ExtractedRenderSpaceUpdate> =
            Vec::with_capacity(data.render_spaces.len());
        {
            profiling::scope!("scene::apply_frame_submit::extract");
            for update in &data.render_spaces {
                let id = RenderSpaceId(update.id);
                seen.insert(id);
                let space = self.spaces.entry(id).or_insert_with(|| RenderSpaceState {
                    id,
                    ..Default::default()
                });
                space.id = id;
                space.apply_update_header(update);
                self.world_caches.entry(id).or_default();

                let extracted = extract_render_space_update(shm, update, data.frame_index)?;
                extracted_per_space.push(extracted);
            }
        }

        // Phase B: per-space apply (parallel for >1 space, serial otherwise).
        self.apply_extracted_per_space(extracted_per_space)?;

        // Phase C: light updates (still serial: shared LightCache).
        {
            profiling::scope!("scene::apply_frame_submit::lights");
            for update in &data.render_spaces {
                let view = light_updates_view(update);
                if let Some(lu) = view.lights_update {
                    apply_light_renderables_update(&mut self.light_cache, shm, lu, view.space_id)?;
                }
                if let Some(lbu) = view.lights_buffer_renderers_update {
                    apply_lights_buffer_renderers_update(
                        &mut self.light_cache,
                        shm,
                        lbu,
                        view.space_id,
                    )?;
                }
            }
        }

        self.remove_render_spaces_not_in_submit(&seen);
        Ok(())
    }

    /// Drains per-space state, runs Phase B (parallel where it pays), and re-inserts the results.
    ///
    /// Drives the rayon fan-out used by [`Self::apply_frame_submit`]. For one or zero entries we
    /// stay serial to skip rayon dispatch overhead. Per-space dirty cache marks are merged into
    /// [`Self::world_dirty`] on the main thread before reinsert.
    fn apply_extracted_per_space(
        &mut self,
        extracted_per_space: Vec<ExtractedRenderSpaceUpdate>,
    ) -> Result<(), SceneError> {
        if extracted_per_space.is_empty() {
            return Ok(());
        }
        profiling::scope!("scene::apply_frame_submit::apply");

        // Drain spaces and caches into a Vec so they can move into worker closures.
        let mut work: Vec<(
            RenderSpaceId,
            RenderSpaceState,
            WorldTransformCache,
            ExtractedRenderSpaceUpdate,
            Vec<TransformRemovalEvent>,
        )> = Vec::with_capacity(extracted_per_space.len());
        for extracted in extracted_per_space {
            let id = extracted.space_id;
            let Some(space) = self.spaces.remove(&id) else {
                continue;
            };
            let cache = self.world_caches.remove(&id).unwrap_or_default();
            work.push((id, space, cache, extracted, Vec::new()));
        }

        if work.len() <= 1 {
            for (id, mut space, mut cache, extracted, mut removal_events) in work {
                let dirty = apply_extracted_render_space_update(
                    &extracted,
                    PerSpaceApplyInputs {
                        space: &mut space,
                        cache: &mut cache,
                        removal_events: &mut removal_events,
                    },
                );
                if dirty {
                    self.world_dirty.insert(id);
                }
                self.spaces.insert(id, space);
                self.world_caches.insert(id, cache);
            }
            return Ok(());
        }

        use rayon::prelude::*;
        let processed: Vec<(RenderSpaceId, RenderSpaceState, WorldTransformCache, bool)> = work
            .into_par_iter()
            .map(
                |(id, mut space, mut cache, extracted, mut removal_events)| {
                    let dirty = apply_extracted_render_space_update(
                        &extracted,
                        PerSpaceApplyInputs {
                            space: &mut space,
                            cache: &mut cache,
                            removal_events: &mut removal_events,
                        },
                    );
                    (id, space, cache, dirty)
                },
            )
            .collect();
        for (id, space, cache, dirty) in processed {
            if dirty {
                self.world_dirty.insert(id);
            }
            self.spaces.insert(id, space);
            self.world_caches.insert(id, cache);
        }
        Ok(())
    }

    /// Drops render spaces that were absent from this submit’s id set.
    fn remove_render_spaces_not_in_submit(&mut self, seen: &HashSet<RenderSpaceId>) {
        self.remove_spaces_scratch.clear();
        self.remove_spaces_scratch
            .extend(self.spaces.keys().copied().filter(|id| !seen.contains(id)));
        for id in self.remove_spaces_scratch.iter().copied() {
            self.light_cache.remove_space(id.0);
            self.spaces.remove(&id);
            self.world_caches.remove(&id);
            self.world_dirty.remove(&id);
        }
    }
}
