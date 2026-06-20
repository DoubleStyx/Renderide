//! Owns all [`RenderSpaceState`](super::render_space::RenderSpaceState) instances and applies per-frame host data.

mod apply;
mod dirty;
mod queries;
mod reports;

use hashbrown::HashMap;
use std::collections::HashSet;

use glam::Mat4;

use crate::cpu_parallelism::{
    ParallelAdmission, admit_renderable_update_items, current_reference_worker_count,
    record_parallel_admission,
};
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{BlitToDisplayState, FrameSubmitData, LayerType, RenderSH2, RenderingContext};

use super::DrainedReflectionProbeRenderChanges;
use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::lights::{
    LightCache, ResolvedLight, apply_light_renderables_update, apply_lights_buffer_renderers_update,
};
#[cfg(test)]
use super::math::multiply_root;
use super::overrides::MeshRendererOverrideTarget;
use super::read::{
    SceneLightRead, SceneMeshRendererRead, SceneReflectionProbeRead, SceneSpaceRead,
    SceneTransformRead, WorldMeshSceneRead,
};
use super::render_space::{RenderSpaceState, RenderSpaceView};
use super::transforms::TransformRemovalEvent;
use super::world::{WorldTransformCache, compute_world_matrices_for_space, ensure_cache_shapes};

use apply::{ExtractedRenderSpaceUpdate, extract_render_space_update, light_updates_view};
use dirty::{
    extracted_update_affects_reflection_probes, extracted_update_affects_render_world,
    note_render_world_dirty_for_extracted_update, render_world_header_changed,
};
pub use reports::{
    RenderWorldBoundsDirty, RenderWorldMaterialOverrideDirty, RenderWorldRendererDirty,
    RenderWorldRendererKind, RenderWorldTransformDirty, SceneApplyReport, SceneCacheFlushReport,
};

/// Dirty render spaces assigned to one world-cache flush worker.
const WORLD_CACHE_FLUSH_PARALLEL_CHUNK_SPACES: usize = 1;
/// Dirty render-space count required before world-cache flush fans out.
const WORLD_CACHE_FLUSH_PARALLEL_MIN_SPACES: usize = WORLD_CACHE_FLUSH_PARALLEL_CHUNK_SPACES * 2;

/// Returns the world-cache flush admission decision for a known worker count.
#[inline]
fn world_cache_flush_admission(
    space_count: usize,
    work_units: usize,
    worker_count: usize,
) -> ParallelAdmission {
    let work_admission = admit_renderable_update_items(work_units, worker_count);
    if space_count >= WORLD_CACHE_FLUSH_PARALLEL_MIN_SPACES && work_admission.is_parallel() {
        ParallelAdmission::Parallel {
            chunk_size: WORLD_CACHE_FLUSH_PARALLEL_CHUNK_SPACES,
        }
    } else {
        ParallelAdmission::Serial
    }
}

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

#[cfg(test)]
mod tests;

/// Scene registry: one entry per host render space.
pub struct SceneCoordinator {
    /// Backed by [`hashbrown::HashMap`] for O(1) per-id lookup on the per-frame
    /// `apply_frame_submit` lift/reinsert path. Iteration order is non-deterministic; callers
    /// that need a stable order go through [`Self::render_space_ids`] which sorts ids by host
    /// `RenderSpaceId` value at iteration time.
    spaces: HashMap<RenderSpaceId, RenderSpaceState>,
    world_caches: HashMap<RenderSpaceId, WorldTransformCache>,
    world_dirty: HashSet<RenderSpaceId>,
    light_cache: LightCache,
    world_flush_scratch: WorldCacheFlushScratch,
    apply_scratch: SceneApplyScratch,
}

#[derive(Default)]
struct WorldCacheFlushScratch {
    /// Reused in [`SceneCoordinator::flush_world_caches`] to avoid per-flush `Vec` allocation.
    dirty_spaces: Vec<RenderSpaceId>,
}

#[derive(Default)]
struct SceneApplyScratch {
    /// Reused in [`SceneCoordinator::remove_render_spaces_not_in_submit`].
    remove_spaces: Vec<RenderSpaceId>,
    /// Per-space transform swap-remove events emitted during Phase B of the current frame's
    /// apply. Consumed by Phase C so [`LightCache::fixup_for_transform_removals`] can roll
    /// cached `transform_id`s forward before the light update applies. Cleared at the top of
    /// every [`SceneCoordinator::apply_frame_submit`] so stale events never leak into later
    /// frames; the per-space [`Vec`] allocations are retained across frames to keep the
    /// steady-state path allocation-free.
    transform_removals_by_space: HashMap<RenderSpaceId, Vec<TransformRemovalEvent>>,
    /// Reused [`HashSet`] of render space ids seen in the current
    /// [`FrameSubmitData::render_spaces`]; cleared at the top of every
    /// [`SceneCoordinator::apply_frame_submit`] and consumed by
    /// [`SceneCoordinator::remove_render_spaces_not_in_submit`].
    seen: HashSet<RenderSpaceId>,
    /// Reused per-space [`ExtractedRenderSpaceUpdate`] buffer for Phase A of every
    /// [`SceneCoordinator::apply_frame_submit`]; drained into Phase B, then refilled next frame.
    extracted: Vec<ExtractedRenderSpaceUpdate>,
    /// Reused per-space work buffer for [`SceneCoordinator::apply_extracted_per_space`]'s Phase B
    /// drain.
    ///
    /// Holds one tuple per space whose state was lifted out of the coordinator spaces and world
    /// caches for the parallel apply. Drained in place after the loop so the allocation persists
    /// across frames.
    work: Vec<ApplyWorkSlot>,
}

/// One per-space work slot held in [`SceneApplyScratch::work`].
struct ApplyWorkSlot {
    /// Render space identity for reinsert and dirty-cache tracking.
    id: RenderSpaceId,
    /// Lifted per-space scene state.
    space: RenderSpaceState,
    /// Lifted per-space world transform cache.
    cache: WorldTransformCache,
    /// Pre-extracted host update payload to apply.
    extracted: ExtractedRenderSpaceUpdate,
    /// Estimated row work carried by [`Self::extracted`].
    work_units: usize,
    /// Reused transform-removal side buffer for this work item.
    removal_events: Vec<TransformRemovalEvent>,
    /// Whether applying this slot dirtied the world transform cache.
    world_dirty: bool,
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
            world_flush_scratch: WorldCacheFlushScratch::default(),
            apply_scratch: SceneApplyScratch::default(),
        }
    }

    /// Mutable light cache ([`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) store, tests).
    pub fn light_cache_mut(&mut self) -> &mut LightCache {
        &mut self.light_cache
    }

    /// Render space ids currently present, ordered by host id for deterministic traversal.
    ///
    /// The backing [`hashbrown::HashMap`] iterates in unspecified order, so this method copies the
    /// keys into a small owned [`Vec`] and sorts by [`RenderSpaceId`] before returning the
    /// iterator. The allocation is negligible at typical scene sizes (1-10 active render spaces);
    /// callers that have observed a sorted contract from the prior `BTreeMap` backing are
    /// preserved.
    pub fn render_space_ids(&self) -> impl Iterator<Item = RenderSpaceId> {
        let mut ids: Vec<RenderSpaceId> = self.spaces.keys().copied().collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids.into_iter()
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

    /// Appends render-context-aware world-space lights for `id` into `out`.
    ///
    /// This uses the same transform basis as view draw collection so clustered-light culling and
    /// forward shading see lights in the same world space as the meshes for that view.
    pub fn resolve_lights_for_render_context_into(
        &self,
        id: RenderSpaceId,
        context: RenderingContext,
        head_output_transform: Mat4,
        out: &mut Vec<ResolvedLight>,
    ) {
        let sid = id.0;
        self.light_cache.resolve_lights_into(
            sid,
            |transform_idx| {
                self.world_matrix_for_render_context(
                    id,
                    transform_idx,
                    context,
                    head_output_transform,
                )
            },
            out,
        );
    }

    /// Estimates cached light rows visible to a view's optional render-space filter.
    pub fn candidate_light_count_for_render_space_filter(
        &self,
        render_space_filter: Option<RenderSpaceId>,
    ) -> usize {
        if let Some(id) = render_space_filter {
            return self
                .spaces
                .get(&id)
                .filter(|space| space.is_active)
                .map_or(0, |_| self.light_cache.cached_light_count_for_space(id.0));
        }
        self.spaces
            .iter()
            .filter(|(_, space)| space.is_active)
            .map(|(id, _)| self.light_cache.cached_light_count_for_space(id.0))
            .sum()
    }

    /// Read-only access for debugging / future systems.
    pub fn space(&self, id: RenderSpaceId) -> Option<RenderSpaceView<'_>> {
        self.spaces.get(&id).map(RenderSpaceView::new)
    }

    /// Main non-overlay render space, matching the host's single active main-space expectation.
    pub fn active_main_space(&self) -> Option<RenderSpaceView<'_>> {
        self.spaces
            .values()
            .filter(|s| s.is_active && !s.is_overlay)
            .min_by_key(|s| s.id.0)
            .map(RenderSpaceView::new)
    }

    /// Drains host changed-probe render requests after the latest scene apply.
    pub fn take_reflection_probe_render_changes(&mut self) -> DrainedReflectionProbeRenderChanges {
        let mut out = DrainedReflectionProbeRenderChanges::default();
        for space in self.spaces.values_mut() {
            let mut drained = super::reflection_probe::drain_reflection_probe_render_changes(space);
            out.completed.append(&mut drained.completed);
            out.scene_captures.append(&mut drained.scene_captures);
        }
        out
    }

    /// Latest explicit [`BlitToDisplayState`] targeting `display_index` from any active render
    /// space, or [`None`] if no active blit covers that display.
    ///
    /// When multiple blits target the same display, traversal is stable: active render spaces are
    /// visited by ascending [`RenderSpaceId`] and dense renderables by ascending table index, with
    /// later matches winning. `is_overlay` spaces are included so per-user and mirror blits keep
    /// working in overlay worlds. This only returns explicit host `BlitToDisplay` rows. Entries
    /// whose state has not yet been initialized by a `states` row are skipped.
    pub fn active_blit_for_display(&self, display_index: i16) -> Option<BlitToDisplayState> {
        let mut latest: Option<BlitToDisplayState> = None;
        for id in self.render_space_ids() {
            let Some(space) = self.spaces.get(&id) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for entry in &space.blit_to_displays {
                if !entry.state_initialized {
                    continue;
                }
                if entry.state.display_index != display_index {
                    continue;
                }
                if entry.state.texture_id < 0 {
                    continue;
                }
                latest = Some(entry.state);
            }
        }
        latest
    }

    /// Current head-output render context for the main view.
    pub fn active_main_render_context(&self) -> RenderingContext {
        self.active_main_space()
            .map_or(RenderingContext::UserView, |space| {
                space.main_render_context()
            })
    }

    /// Cached world matrix from the host transform hierarchy (parent chain only).
    ///
    /// This matches object/light/bone placement: [`RenderSpaceState::root_transform`] is **not**
    /// applied here--it drives the view basis via [`RenderSpaceState::view_transform`], not mesh
    /// model matrices.
    pub fn world_matrix(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_caches
            .get(&id)?
            .world_matrices
            .get(transform_index)
            .copied()
    }

    /// Hierarchy world matrix left-multiplied by [`RenderSpaceState::root_transform`].
    ///
    /// Use only when a host contract explicitly requires this composite. Default rendering uses
    /// [`Self::world_matrix`].
    #[cfg(test)]
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
    pub fn flush_world_caches(&mut self) -> Result<SceneCacheFlushReport, SceneError> {
        profiling::scope!("scene::flush_world_caches");
        use rayon::prelude::*;

        let mut report = SceneCacheFlushReport::default();
        self.world_flush_scratch.dirty_spaces.clear();
        self.world_flush_scratch
            .dirty_spaces
            .extend(self.world_dirty.iter().copied());

        // Drop caches for dirty spaces that no longer exist and drain caches for surviving
        // spaces into a work vec. This runs on the main thread because it mutates `self`.
        let mut work: Vec<(RenderSpaceId, WorldTransformCache)> =
            Vec::with_capacity(self.world_flush_scratch.dirty_spaces.len());
        for id in self.world_flush_scratch.dirty_spaces.iter().copied() {
            if !self.spaces.contains_key(&id) {
                self.world_caches.remove(&id);
                self.world_dirty.remove(&id);
                continue;
            }
            let cache = self.world_caches.remove(&id).unwrap_or_default();
            work.push((id, cache));
        }

        if work.is_empty() {
            return Ok(report);
        }
        let work_units = {
            profiling::scope!("scene::flush_world_caches::estimate_parallel_work");
            work.iter()
                .filter_map(|(id, _cache)| self.spaces.get(id).map(|space| space.nodes.len()))
                .sum::<usize>()
        };
        let admission =
            world_cache_flush_admission(work.len(), work_units, current_reference_worker_count());
        record_parallel_admission("scene_world_cache_flush", work_units, work.len(), admission);

        // `&self.spaces` is a shared borrow across rayon workers; `BTreeMap::get` is `Sync` for
        // `Sync` keys and values. Each task owns its own cache.
        let spaces = &self.spaces;
        let compute_one = |(id, mut cache): (RenderSpaceId, WorldTransformCache)| {
            // Space removed between drain and dispatch -- preserve cache as-is so the reinsert
            // step below drops it via the `Ok` path (caller treats this as a no-op).
            let Some(space) = spaces.get(&id) else {
                return (id, Ok(cache));
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
        };
        let results: Vec<(RenderSpaceId, Result<WorldTransformCache, SceneError>)> =
            if !admission.is_parallel() {
                work.into_iter().map(compute_one).collect()
            } else {
                work.into_par_iter()
                    .with_min_len(WORLD_CACHE_FLUSH_PARALLEL_CHUNK_SPACES)
                    .map(compute_one)
                    .collect()
            };

        let mut first_err: Option<SceneError> = None;
        for (id, result) in results {
            match result {
                Ok(cache) => {
                    self.world_caches.insert(id, cache);
                    self.world_dirty.remove(&id);
                    report.flushed_spaces.push(id);
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
        Ok(report)
    }

    /// Applies [`FrameSubmitData`]: transforms, meshes, skinned meshes, and lights in host order.
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
    ) -> Result<SceneApplyReport, SceneError> {
        profiling::scope!("scene::apply_frame_submit");
        warn_if_multiple_active_non_overlay_spaces(data);
        let mut report = SceneApplyReport::new(data.frame_index);

        // Clear last frame's per-space removal events; Phase B refills them, Phase C consumes.
        // Retain the per-space `Vec` allocations to keep the steady-state path allocation-free.
        for v in self.apply_scratch.transform_removals_by_space.values_mut() {
            v.clear();
        }

        // Reuse the cross-frame scratch HashSet and Vec; both are cleared on entry and put back
        // before this method returns so steady-state apply does not allocate either container.
        let mut seen = std::mem::take(&mut self.apply_scratch.seen);
        seen.clear();
        let mut extracted_per_space = std::mem::take(&mut self.apply_scratch.extracted);
        extracted_per_space.clear();
        extracted_per_space.reserve(data.render_spaces.len());

        // Phase A: serial pre-extract + ensure entries + apply header fields.
        {
            profiling::scope!("scene::apply_frame_submit::extract");
            for update in &data.render_spaces {
                let id = RenderSpaceId(update.id);
                seen.insert(id);
                report.note_submitted_space(id);
                let header_dirty = render_world_header_changed(self.spaces.get(&id), update);
                let space = self.spaces.entry(id).or_insert_with(|| RenderSpaceState {
                    id,
                    ..Default::default()
                });
                space.id = id;
                space.apply_update_header(update);
                let current_node_count = space.nodes.len();
                self.world_caches.entry(id).or_default();

                let extracted = extract_render_space_update(shm, update, data.frame_index)?;
                if header_dirty || extracted_update_affects_render_world(&extracted) {
                    report.note_changed_space(id);
                }
                if header_dirty || extracted_update_affects_reflection_probes(&extracted) {
                    report.note_reflection_probe_dirty_space(id);
                }
                note_render_world_dirty_for_extracted_update(
                    &mut report,
                    id,
                    header_dirty,
                    current_node_count,
                    Some(&*space),
                    &extracted,
                );
                extracted_per_space.push(extracted);
            }
        }

        // Phase B: per-space apply (parallel for >1 space, serial otherwise). Drains
        // `extracted_per_space`; the outer Vec keeps its capacity for next frame.
        self.apply_extracted_per_space(&mut extracted_per_space)?;

        // Phase C: light updates (still serial: shared LightCache). Before applying each space's
        // update we roll pre-existing cached `transform_id`s forward through any transform
        // swap-removes that ran in Phase B -- mirrors the host's `RenderableIndex` reindexing so a
        // light whose transform was swap-moved into a freed slot keeps pointing at it.
        {
            profiling::scope!("scene::apply_frame_submit::lights");
            for update in &data.render_spaces {
                let view = light_updates_view(update);
                if let Some(removals) = self
                    .apply_scratch
                    .transform_removals_by_space
                    .get(&RenderSpaceId(view.space_id))
                {
                    self.light_cache
                        .fixup_for_transform_removals(view.space_id, removals);
                }
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

        self.remove_render_spaces_not_in_submit(&seen, &mut report.removed_spaces);

        // Restore the scratch containers (capacities retained for next frame).
        seen.clear();
        self.apply_scratch.seen = seen;
        debug_assert!(extracted_per_space.is_empty());
        self.apply_scratch.extracted = extracted_per_space;
        Ok(report)
    }

    /// Drops render spaces that were absent from this submit's id set.
    fn remove_render_spaces_not_in_submit(
        &mut self,
        seen: &HashSet<RenderSpaceId>,
        removed: &mut Vec<RenderSpaceId>,
    ) {
        self.apply_scratch.remove_spaces.clear();
        self.apply_scratch
            .remove_spaces
            .extend(self.spaces.keys().copied().filter(|id| !seen.contains(id)));
        for id in self.apply_scratch.remove_spaces.iter().copied() {
            removed.push(id);
            self.light_cache.remove_space(id.0);
            self.spaces.remove(&id);
            self.world_caches.remove(&id);
            self.world_dirty.remove(&id);
            self.apply_scratch.transform_removals_by_space.remove(&id);
        }
    }
}

impl SceneMeshRendererRead for SceneCoordinator {
    fn static_mesh_renderers(&self, id: RenderSpaceId) -> Option<&[super::StaticMeshRenderer]> {
        self.spaces
            .get(&id)
            .map(|space| space.static_mesh_renderers.as_slice())
    }

    fn skinned_mesh_renderers(&self, id: RenderSpaceId) -> Option<&[super::SkinnedMeshRenderer]> {
        self.spaces
            .get(&id)
            .map(|space| space.skinned_mesh_renderers.as_slice())
    }
}

impl WorldMeshSceneRead for SceneCoordinator {
    fn lod_groups(&self, id: RenderSpaceId) -> Option<&[super::LodGroupEntry]> {
        self.spaces
            .get(&id)
            .map(|space| space.lod_groups.as_slice())
    }

    fn billboard_render_buffers(
        &self,
        id: RenderSpaceId,
    ) -> Option<&[super::BillboardRenderBufferEntry]> {
        self.spaces
            .get(&id)
            .map(|space| space.billboard_render_buffers.as_slice())
    }

    fn mesh_render_buffers(&self, id: RenderSpaceId) -> Option<&[super::MeshRenderBufferEntry]> {
        self.spaces
            .get(&id)
            .map(|space| space.mesh_render_buffers.as_slice())
    }

    fn trail_render_buffers(&self, id: RenderSpaceId) -> Option<&[super::TrailRenderBufferEntry]> {
        self.spaces
            .get(&id)
            .map(|space| space.trail_render_buffers.as_slice())
    }
}

impl SceneReflectionProbeRead for SceneCoordinator {
    fn reflection_probes(&self, id: RenderSpaceId) -> Option<&[super::ReflectionProbeEntry]> {
        self.spaces
            .get(&id)
            .map(|space| space.reflection_probes.as_slice())
    }
}

impl SceneSpaceRead for SceneCoordinator {
    type Space<'a>
        = RenderSpaceView<'a>
    where
        Self: 'a;
    type RenderSpaceIds<'a>
        = std::vec::IntoIter<RenderSpaceId>
    where
        Self: 'a;

    fn render_space_ids(&self) -> Self::RenderSpaceIds<'_> {
        let mut ids: Vec<RenderSpaceId> = self.spaces.keys().copied().collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids.into_iter()
    }

    fn space(&self, id: RenderSpaceId) -> Option<Self::Space<'_>> {
        self.spaces.get(&id).map(RenderSpaceView::new)
    }

    fn active_main_space(&self) -> Option<Self::Space<'_>> {
        self.spaces
            .values()
            .filter(|s| s.is_active && !s.is_overlay)
            .min_by_key(|s| s.id.0)
            .map(RenderSpaceView::new)
    }

    fn active_main_ambient_light(&self) -> RenderSH2 {
        self.active_main_space()
            .map(|s| s.ambient_light())
            .unwrap_or_default()
    }

    fn active_main_render_context(&self) -> RenderingContext {
        self.active_main_space()
            .map_or(RenderingContext::UserView, |space| {
                space.main_render_context()
            })
    }

    fn render_context_affects_draw_prep(&self, context: RenderingContext) -> bool {
        self.spaces
            .values()
            .any(|space| space.has_draw_prep_overrides_in_context(context))
    }
}

impl SceneTransformRead for SceneCoordinator {
    fn world_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4> {
        SceneCoordinator::world_matrix_for_context(self, id, transform_index, context)
    }

    fn world_matrix_for_render_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
        head_output_transform: Mat4,
    ) -> Option<Mat4> {
        SceneCoordinator::world_matrix_for_render_context(
            self,
            id,
            transform_index,
            context,
            head_output_transform,
        )
    }

    fn overlay_layer_model_matrix_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> Option<Mat4> {
        SceneCoordinator::overlay_layer_model_matrix_for_context(self, id, transform_index, context)
    }

    fn transform_special_layer(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<LayerType> {
        SceneCoordinator::transform_special_layer(self, id, transform_index)
    }

    fn transform_is_in_overlay_layer(&self, id: RenderSpaceId, transform_index: usize) -> bool {
        SceneCoordinator::transform_is_in_overlay_layer(self, id, transform_index)
    }

    fn transform_has_degenerate_scale_for_context(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
        context: RenderingContext,
    ) -> bool {
        SceneCoordinator::transform_has_degenerate_scale_for_context(
            self,
            id,
            transform_index,
            context,
        )
    }

    fn overridden_material_asset_id(
        &self,
        space_id: RenderSpaceId,
        context: RenderingContext,
        skinned: bool,
        renderable_index: usize,
        slot_index: usize,
    ) -> Option<i32> {
        SceneCoordinator::overridden_material_asset_id(
            self,
            space_id,
            context,
            skinned,
            renderable_index,
            slot_index,
        )
    }
}

impl SceneLightRead for SceneCoordinator {
    fn resolve_lights_for_render_context_into(
        &self,
        id: RenderSpaceId,
        context: RenderingContext,
        head_output_transform: Mat4,
        out: &mut Vec<ResolvedLight>,
    ) {
        SceneCoordinator::resolve_lights_for_render_context_into(
            self,
            id,
            context,
            head_output_transform,
            out,
        );
    }

    fn candidate_light_count_for_render_space_filter(
        &self,
        render_space_filter: Option<RenderSpaceId>,
    ) -> usize {
        SceneCoordinator::candidate_light_count_for_render_space_filter(self, render_space_filter)
    }
}
