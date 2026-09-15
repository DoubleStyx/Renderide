//! Coalesced spatial-index and LOD-bound refits for prepared-renderable patch batches.

use hashbrown::HashSet;

use crate::scene::{RenderSpaceId, WorldMeshSceneRead};

use super::FramePreparedRenderables;

impl FramePreparedRenderables {
    /// Starts a top-level stable-patch batch that coalesces spatial and LOD refits by render space.
    pub(in crate::world_mesh::draw_prep) fn begin_spatial_lod_refit_batch(&mut self) {
        debug_assert!(
            !self.spatial_lod_refit_batch_active,
            "prepared spatial/LOD refit batches must not nest"
        );
        self.pending_spatial_lod_refit_spaces.clear();
        self.spatial_lod_refit_batch_active = true;
    }

    /// Flushes one top-level stable-patch batch over the union of touched render spaces.
    ///
    /// Returns the number of spatial spaces actually refit. Structural metadata rebuilds clear
    /// queued spaces because their full spatial/LOD reconstruction already supersedes the refit.
    pub(in crate::world_mesh::draw_prep) fn flush_spatial_lod_refit_batch<S>(
        &mut self,
        scene: &S,
    ) -> usize
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        debug_assert!(
            self.spatial_lod_refit_batch_active,
            "prepared spatial/LOD refit flush requires an active batch"
        );
        self.spatial_lod_refit_batch_active = false;
        if self.pending_spatial_lod_refit_spaces.is_empty() {
            return 0;
        }

        // Move the set out while mutably refitting `self`, then restore its allocation for the
        // next frame instead of allocating a fresh union set every flush.
        let mut touched = std::mem::take(&mut self.pending_spatial_lod_refit_spaces);
        let spatial_refit_count = self.refit_cached_spatial_and_lods_for_touched(scene, &touched);
        touched.clear();
        self.pending_spatial_lod_refit_spaces = touched;
        spatial_refit_count
    }

    /// Refits cached spatial data and refreshes LOD metadata after dynamic bounds changed.
    ///
    /// Prepared LOD groups cache the union of their renderer AABBs, so updating draw-row cull
    /// geometry without rebuilding them leaves LOD selection on stale bounds even when the
    /// spatial index itself was refit.
    pub(in crate::world_mesh::draw_prep) fn refit_cached_spatial_and_lods_for_spaces<S, I>(
        &mut self,
        scene: &S,
        space_ids: I,
    ) -> usize
    where
        S: WorldMeshSceneRead + ?Sized,
        I: IntoIterator<Item = RenderSpaceId>,
    {
        if self.spatial_lod_refit_batch_active {
            self.pending_spatial_lod_refit_spaces.extend(space_ids);
            return 0;
        }
        self.refit_cached_spatial_and_lods_for_spaces_with_runs(scene, space_ids, None)
    }

    /// Same as [`Self::refit_cached_spatial_and_lods_for_spaces`], with the runs the caller changed.
    ///
    /// `Some` lets the spatial index skip its O(scene) sweep and touch only the changed entries and
    /// their BVH ancestors. Batched calls fall back to the conservative sweep on flush.
    pub(in crate::world_mesh::draw_prep) fn refit_cached_spatial_and_lods_for_spaces_with_runs<S, I>(
        &mut self,
        scene: &S,
        space_ids: I,
        changed_runs: Option<&HashSet<usize>>,
    ) -> usize
    where
        S: WorldMeshSceneRead + ?Sized,
        I: IntoIterator<Item = RenderSpaceId>,
    {
        if self.spatial_lod_refit_batch_active {
            self.pending_spatial_lod_refit_spaces.extend(space_ids);
            return 0;
        }
        let touched = space_ids.into_iter().collect::<HashSet<_>>();
        // Bounds patches rewrite rows in place, so their runs join whatever the caller named. If
        // nothing is named and no bounds patch is pending, fall back to the conservative sweep.
        let pending = std::mem::take(&mut self.pending_bounds_patch_runs);
        let merged = match changed_runs {
            Some(named) if !pending.is_empty() => {
                Some(named.union(&pending).copied().collect::<HashSet<_>>())
            }
            Some(named) => Some(named.clone()),
            None if !pending.is_empty() => Some(pending),
            None => None,
        };
        let spatial_refit_count = self.spatial.refit_spaces_for_runs(
            &self.draws,
            &self.runs,
            touched.iter().copied(),
            merged.as_ref(),
        );
        let changed_ordinals = merged.as_ref().map(|runs| self.ordinals_for_runs(runs));
        self.refresh_lod_groups_for_bounds_refit(scene, &touched, changed_ordinals.as_ref());
        spatial_refit_count
    }

    /// Executes an eager spatial/LOD refit for a deduplicated set of render spaces.
    pub(in crate::world_mesh::draw_prep) fn refit_cached_spatial_and_lods_for_touched<S>(
        &mut self,
        scene: &S,
        touched: &HashSet<RenderSpaceId>,
    ) -> usize
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        let pending = std::mem::take(&mut self.pending_bounds_patch_runs);
        let spatial_refit_count = self.spatial.refit_spaces_for_runs(
            &self.draws,
            &self.runs,
            touched.iter().copied(),
            (!pending.is_empty()).then_some(&pending),
        );
        let changed_ordinals = (!pending.is_empty()).then(|| self.ordinals_for_runs(&pending));
        self.refresh_lod_groups_for_bounds_refit(scene, touched, changed_ordinals.as_ref());
        spatial_refit_count
    }
}
