//! Two-phase per-tick scene apply: serial shared-memory pre-extract + parallel per-space mutation.
//!
//! `apply_frame_submit` historically iterated [`crate::shared::RenderSpaceUpdate`] chunks serially
//! because every per-space helper takes `&mut SharedMemoryAccessor`. This module splits that work
//! into:
//!
//! 1. **Phase A (serial):** [`extract_render_space_update`] reads every shared-memory descriptor
//!    referenced by the host update for one render space into owned [`Vec`]s held in
//!    [`ExtractedRenderSpaceUpdate`].
//! 2. **Phase B (parallel):** [`mutate::apply_extracted_render_space_update`] mutates the per-space
//!    [`crate::scene::RenderSpaceState`] and [`crate::scene::WorldTransformCache`] using only the
//!    owned payloads. Distinct render spaces own disjoint state, so the apply step can fan out
//!    across rayon workers.
//!
//! Light updates ([`crate::scene::LightCache`]) target shared state and stay serial in a Phase C
//! pass after the parallel apply completes, matching the plan's "simpler first cut".
//!
//! Submodules carve the pipeline by phase: [`extract`] (Phase A), [`mutate`] (Phase B),
//! [`lights`] (Phase C views), and [`work_units`] (Phase B parallelism policy). This facade
//! re-exports the coordinator-facing API and hosts the [`SceneCoordinator`] fan-out driver.

mod extract;
mod lights;
mod mutate;
mod work_units;

pub(in crate::scene::coordinator) use extract::{
    ExtractedRenderSpaceUpdate, extract_render_space_update,
};
pub(in crate::scene::coordinator) use lights::light_updates_view;
/// Re-exported for the coordinator apply tests; production callers reach Phase B through
/// [`SceneCoordinator::apply_extracted_per_space`].
#[cfg(test)]
pub(in crate::scene::coordinator) use mutate::{
    PerSpaceApplyInputs, apply_extracted_render_space_update,
};

use mutate::apply_work_slot_mutation;
use work_units::{
    APPLY_PARALLEL_CHUNK_SPACES, MIN_APPLY_PARALLEL_WORK_UNITS,
    apply_parallel_admission_with_workers, apply_work_units, dominant_slot_work_units,
    extracted_apply_work_units, is_extracted_empty, space_split_apply_preferred,
};

use crate::cpu_parallelism::{current_reference_worker_count, record_parallel_admission};
use crate::scene::error::SceneError;
use crate::scene::ids::RenderSpaceId;
use crate::scene::transforms::TransformRemovalEvent;

use super::{ApplyWorkSlot, SceneCoordinator};

impl SceneCoordinator {
    /// Drains per-space state, runs Phase B (parallel where it pays), and re-inserts the results.
    ///
    /// Drives the rayon fan-out used by [`SceneCoordinator::apply_frame_submit`]. For one or zero
    /// entries we stay serial to skip rayon dispatch overhead. Per-space dirty cache marks are
    /// merged into [`SceneCoordinator::world_dirty`] on the main thread before reinsert. The
    /// caller's `extracted_per_space` is drained in place so the backing [`Vec`] capacity persists
    /// across frames.
    pub(super) fn apply_extracted_per_space(
        &mut self,
        extracted_per_space: &mut Vec<ExtractedRenderSpaceUpdate>,
    ) -> Result<(), SceneError> {
        if extracted_per_space.is_empty() {
            return Ok(());
        }
        profiling::scope!("scene::apply_frame_submit::apply");

        // Hoist the per-frame work buffer out of `self` so we can fill it while still mutating
        // `self.spaces` / `self.world_caches`. `mem::take` preserves the underlying allocation
        // so steady-state apply does not allocate.
        let mut work = std::mem::take(&mut self.apply_work_scratch);
        debug_assert!(work.is_empty());
        {
            profiling::scope!("scene::apply::lift");
            for extracted in extracted_per_space.drain(..) {
                let id = extracted.space_id;
                // Header fields were already applied in Phase A. If the host sent no body
                // payloads for this space this tick, skip the lift/reinsert pair entirely so the
                // steady-state path never moves a [`RenderSpaceState`].
                if is_extracted_empty(&extracted) {
                    continue;
                }
                let work_units = extracted_apply_work_units(&extracted);
                let Some(space) = self.spaces.remove(&id) else {
                    continue;
                };
                let cache = self.world_caches.remove(&id).unwrap_or_default();
                let removal_events = self
                    .transform_removals_by_space
                    .remove(&id)
                    .unwrap_or_default();
                work.push(ApplyWorkSlot {
                    id,
                    space,
                    cache,
                    extracted,
                    work_units,
                    removal_events,
                    world_dirty: false,
                });
            }
        }

        let work_units = {
            profiling::scope!("scene::apply::estimate_parallel_work");
            apply_work_units(&work)
        };
        let worker_count = current_reference_worker_count();
        let admission = apply_parallel_admission_with_workers(work.len(), work_units, worker_count);
        record_parallel_admission("scene_apply", work_units, work.len(), admission);
        if space_split_apply_preferred(
            work.len(),
            work_units,
            dominant_slot_work_units(&work),
            worker_count,
        ) {
            profiling::scope!("scene::apply::mutate::space_split");
            for slot in &mut work {
                if slot.work_units >= MIN_APPLY_PARALLEL_WORK_UNITS {
                    profiling::scope!("scene::apply::mutate::space_split_slot");
                    apply_work_slot_mutation(slot);
                } else {
                    apply_work_slot_mutation(slot);
                }
            }
            self.reinsert_applied_work_slots(&mut work);
            self.apply_work_scratch = work;
            return Ok(());
        }
        if !admission.is_parallel() {
            profiling::scope!("scene::apply::mutate::serial_small_batch");
            for slot in &mut work {
                profiling::scope!("scene::apply::mutate::serial_slot");
                apply_work_slot_mutation(slot);
            }
            self.reinsert_applied_work_slots(&mut work);
            self.apply_work_scratch = work;
            return Ok(());
        }

        use rayon::prelude::*;
        {
            profiling::scope!("scene::apply::mutate");
            work.par_iter_mut()
                .with_min_len(APPLY_PARALLEL_CHUNK_SPACES)
                .for_each(|slot| {
                    profiling::scope!("scene::apply::mutate::worker_slot");
                    apply_work_slot_mutation(slot);
                });
        };
        self.reinsert_applied_work_slots(&mut work);
        self.apply_work_scratch = work;
        Ok(())
    }

    fn reinsert_applied_work_slots(&mut self, work: &mut Vec<ApplyWorkSlot>) {
        profiling::scope!("scene::apply::reinsert");
        for slot in work.drain(..) {
            if slot.world_dirty {
                self.world_dirty.insert(slot.id);
            }
            self.spaces.insert(slot.id, slot.space);
            self.world_caches.insert(slot.id, slot.cache);
            self.stash_transform_removals(slot.id, slot.removal_events);
        }
    }

    /// Moves a per-space transform-removal buffer back into
    /// [`SceneCoordinator::transform_removals_by_space`] so Phase C can read it.
    fn stash_transform_removals(
        &mut self,
        id: RenderSpaceId,
        removals: Vec<TransformRemovalEvent>,
    ) {
        self.transform_removals_by_space.insert(id, removals);
    }
}
