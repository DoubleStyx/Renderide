//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores renderer-facing draw templates that are expensive to
//! rediscover every frame.

mod refresh;
mod state;

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{
    MeshRendererOverrideTarget, RenderSpaceId, RenderWorldMaterialOverrideDirty,
    RenderWorldRendererDirty, RenderWorldRendererKind, RenderWorldTransformDirty, SceneApplyReport,
    SceneCacheFlushReport, SceneCoordinator,
};
use crate::shared::RenderingContext;

use super::prepared_renderables::FramePreparedRenderables;
use refresh::{DirtyRendererSet, RefreshOutcome, refresh_render_world_space, refresh_renderer_set};
use state::RenderWorldSpace;

/// Transform-root dirty records assigned to one expansion worker.
const DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS: usize = 1;
/// Render spaces assigned to one mesh-asset dirty expansion worker.
const MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES: usize = 1;
/// Dirty render spaces assigned to one retained-cache refresh worker.
const DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES: usize = 1;
/// Active render spaces assigned to one prepared-snapshot rebuild worker.
const SNAPSHOT_REBUILD_PARALLEL_CHUNK_SPACES: usize = 1;
/// Transform-root dirty input count at which expansion uses Rayon.
const DIRTY_ROOT_EXPANSION_PARALLEL_MIN_ITEMS: usize =
    DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS * 2;
/// Render-space count at which mesh-asset dirty expansion uses Rayon.
const MESH_ASSET_DIRTY_EXPANSION_PARALLEL_MIN_SPACES: usize =
    MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES * 2;
/// Dirty render-space count at which retained cache refresh uses Rayon.
const DIRTY_SPACE_REFRESH_PARALLEL_MIN_SPACES: usize =
    DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES * 2;
/// Active render-space count required before snapshot rebuild fan-out is considered.
const SNAPSHOT_REBUILD_PARALLEL_MIN_SPACES: usize = SNAPSHOT_REBUILD_PARALLEL_CHUNK_SPACES * 2;
/// Retained draw-template count required before snapshot rebuild fan-out is considered.
const SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS: usize = SNAPSHOT_REBUILD_PARALLEL_MIN_SPACES;

/// Maintenance counters for backend-owned retained render-world caches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderWorldMaintenanceStats {
    /// Renderer records whose retained templates were requested dirty this frame.
    pub dirty_renderer_count: usize,
    /// Renderer records actually refreshed this frame.
    pub refreshed_renderer_count: usize,
    /// Draw templates regenerated while refreshing dirty renderer records.
    pub refreshed_template_count: usize,
    /// Mesh asset ids consumed from the mesh-pool mutation log this frame.
    pub mesh_asset_invalidation_count: usize,
    /// Render spaces rebuilt through the full-space fallback this frame.
    pub full_space_rebuild_count: usize,
    /// Full render-world rebuild requests processed this frame.
    pub full_world_rebuild_count: usize,
    /// Retained draw templates currently cached after maintenance.
    pub retained_template_count: usize,
    /// Frames where this render world proved its retained snapshot did not need rebuilding.
    pub steady_state_skip_count: usize,
}

impl RenderWorldMaintenanceStats {
    /// Adds another render world's counters into this aggregate.
    pub fn accumulate(&mut self, other: Self) {
        self.dirty_renderer_count += other.dirty_renderer_count;
        self.refreshed_renderer_count += other.refreshed_renderer_count;
        self.refreshed_template_count += other.refreshed_template_count;
        self.mesh_asset_invalidation_count += other.mesh_asset_invalidation_count;
        self.full_space_rebuild_count += other.full_space_rebuild_count;
        self.full_world_rebuild_count += other.full_world_rebuild_count;
        self.retained_template_count += other.retained_template_count;
        self.steady_state_skip_count += other.steady_state_skip_count;
    }
}

/// Persistent renderer-facing cache of expanded world-mesh renderables.
pub struct RenderWorld {
    /// Per-space retained renderer template records.
    spaces: HashMap<RenderSpaceId, RenderWorldSpace>,
    /// Spaces requiring full retained-template rebuild.
    dirty_spaces: HashSet<RenderSpaceId>,
    /// Individual renderer records requiring retained-template refresh.
    dirty_renderers: HashSet<RenderWorldRendererDirty>,
    /// Transform-root dirties deferred until world-cache flush has completed.
    dirty_transform_roots: Vec<RenderWorldTransformDirty>,
    /// Mesh assets whose referencing renderer records need refresh.
    dirty_mesh_assets: HashSet<i32>,
    /// Whether the next prepare must rebuild every scene space.
    full_rebuild_requested: bool,
    /// Mesh-pool mutation generation consumed by this cache.
    mesh_pool_generation: u64,
    /// Dense prepared snapshot consumed by per-view draw collection.
    prepared: FramePreparedRenderables,
    /// Most recent maintenance counters.
    maintenance_stats: RenderWorldMaintenanceStats,
}

/// Returns whether `node_id` is equal to or below `root_id` in the supplied parent table.
fn node_is_descendant_or_self(parents: &[i32], node_id: i32, root_id: i32) -> bool {
    if node_id < 0 || root_id < 0 {
        return false;
    }
    let mut current = node_id;
    for _ in 0..=parents.len() {
        if current == root_id {
            return true;
        }
        let Some(&parent) = parents.get(current as usize) else {
            return false;
        };
        if parent < 0 {
            return false;
        }
        current = parent;
    }
    false
}

/// Returns whether `node_id` is below any root in `roots`.
fn node_is_under_any_root(parents: &[i32], node_id: i32, roots: &[i32]) -> bool {
    roots
        .iter()
        .any(|&root| node_is_descendant_or_self(parents, node_id, root))
}

/// Result of expanding one transform-root dirty input.
enum TransformDirtyExpansion {
    /// The render space was removed from the scene.
    Removed(RenderSpaceId),
    /// The render space has no retained cache and needs a full rebuild.
    FullSpace(RenderSpaceId),
    /// The dirty roots expanded to renderer records.
    Renderers(Vec<RenderWorldRendererDirty>),
    /// No retained renderer records were affected.
    Empty,
}

/// Worker-owned full-space refresh payload.
struct DirtySpaceRefreshWork {
    /// Render space being refreshed.
    id: RenderSpaceId,
    /// Retained cache removed from [`RenderWorld::spaces`] for worker-owned mutation.
    cached: RenderWorldSpace,
    /// Refresh counters produced by the worker.
    outcome: RefreshOutcome,
}

/// Worker-owned partial-renderer refresh payload.
struct DirtyRendererRefreshWork {
    /// Render space containing the dirty renderer records.
    id: RenderSpaceId,
    /// Dirty renderer records grouped for this space.
    dirty_set: DirtyRendererSet,
    /// Retained cache removed from [`RenderWorld::spaces`] for worker-owned mutation.
    cached: RenderWorldSpace,
    /// Refresh counters produced by the worker.
    outcome: RefreshOutcome,
}

impl RenderWorld {
    /// Creates an empty render-world cache.
    pub fn new(render_context: RenderingContext) -> Self {
        Self {
            spaces: HashMap::new(),
            dirty_spaces: HashSet::new(),
            dirty_renderers: HashSet::new(),
            dirty_transform_roots: Vec::new(),
            dirty_mesh_assets: HashSet::new(),
            full_rebuild_requested: true,
            mesh_pool_generation: 0,
            prepared: FramePreparedRenderables::empty(render_context),
            maintenance_stats: RenderWorldMaintenanceStats::default(),
        }
    }

    /// Marks spaces or renderer records touched by scene apply as needing maintenance.
    pub fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        let has_fine_dirty = !report.render_world_dirty.is_empty();
        if has_fine_dirty {
            for &id in &report.render_world_dirty.full_spaces {
                self.dirty_spaces.insert(id);
            }
            for &dirty in &report.render_world_dirty.renderers {
                self.note_renderer_dirty(dirty);
            }
            self.dirty_transform_roots
                .extend(report.render_world_dirty.transform_roots.iter().cloned());
            for &dirty in &report.render_world_dirty.material_overrides {
                self.note_material_override_dirty(dirty);
            }
        } else {
            for &id in &report.changed_spaces {
                self.dirty_spaces.insert(id);
            }
        }
        for &id in &report.removed_spaces {
            self.remove_space(id);
        }
        if !report.removed_spaces.is_empty() {
            self.full_rebuild_requested = true;
        }
    }

    /// Observes world-cache flushes after scene apply.
    pub fn note_cache_flush_report(&self, _report: &SceneCacheFlushReport) {}

    /// Returns the prepared draw snapshot for this frame, refreshing dirty cached records first.
    pub fn prepare_for_frame(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> &FramePreparedRenderables {
        profiling::scope!("mesh::render_world::prepare_for_frame");
        let mut stats = RenderWorldMaintenanceStats::default();
        let context_changed = self.prepared.render_context() != render_context;
        if context_changed {
            self.full_rebuild_requested = true;
        }
        self.note_mesh_pool_delta(mesh_pool, &mut stats);

        let full_rebuild = self.full_rebuild_requested;
        if full_rebuild {
            stats.full_world_rebuild_count = 1;
            self.mark_all_scene_spaces_dirty(scene);
        }

        self.expand_deferred_dirty_inputs(scene);
        stats.dirty_renderer_count = self.dirty_renderers.len();

        let mut snapshot_dirty = if self.dirty_spaces.is_empty() {
            full_rebuild || context_changed
        } else {
            let outcome = self.refresh_dirty_spaces(scene, mesh_pool, render_context);
            stats.full_space_rebuild_count += outcome.full_space_count;
            stats.refreshed_renderer_count += outcome.renderer_count;
            stats.refreshed_template_count += outcome.template_count;
            true
        };
        if !self.dirty_renderers.is_empty() {
            let outcome = self.refresh_dirty_renderers(scene, mesh_pool, render_context);
            stats.refreshed_renderer_count += outcome.renderer_count;
            stats.refreshed_template_count += outcome.template_count;
            snapshot_dirty |= outcome.renderer_count > 0;
        }

        if snapshot_dirty {
            profiling::scope!("mesh::render_world::rebuild_snapshot");
            self.rebuild_prepared_snapshot(scene, render_context);
        } else {
            stats.steady_state_skip_count = 1;
        }
        self.full_rebuild_requested = false;
        stats.retained_template_count = self.retained_template_count();
        self.maintenance_stats = stats;
        &self.prepared
    }

    /// Prepared draw snapshot from the most recent [`Self::prepare_for_frame`] call.
    pub(crate) fn prepared(&self) -> &FramePreparedRenderables {
        &self.prepared
    }

    /// Maintenance counters from the most recent [`Self::prepare_for_frame`] call.
    pub fn maintenance_stats(&self) -> RenderWorldMaintenanceStats {
        self.maintenance_stats
    }

    /// Removes all retained state for a render space.
    fn remove_space(&mut self, id: RenderSpaceId) {
        self.spaces.remove(&id);
        self.dirty_spaces.remove(&id);
        self.dirty_renderers.retain(|dirty| dirty.space_id != id);
        self.dirty_transform_roots
            .retain(|dirty| dirty.space_id != id);
    }

    /// Records one renderer row dirty unless its whole space is already dirty.
    fn note_renderer_dirty(&mut self, dirty: RenderWorldRendererDirty) {
        if self.dirty_spaces.contains(&dirty.space_id) {
            return;
        }
        self.dirty_renderers.insert(dirty);
    }

    /// Records a material override dirty event for this render context.
    fn note_material_override_dirty(&mut self, dirty: RenderWorldMaterialOverrideDirty) {
        if dirty.context != self.prepared.render_context() {
            return;
        }
        match dirty.target {
            MeshRendererOverrideTarget::Static(index) if index >= 0 => {
                self.note_renderer_dirty(RenderWorldRendererDirty {
                    space_id: dirty.space_id,
                    kind: RenderWorldRendererKind::Static,
                    renderable_index: index as usize,
                });
            }
            MeshRendererOverrideTarget::Skinned(index) if index >= 0 => {
                self.note_renderer_dirty(RenderWorldRendererDirty {
                    space_id: dirty.space_id,
                    kind: RenderWorldRendererKind::Skinned,
                    renderable_index: index as usize,
                });
            }
            MeshRendererOverrideTarget::Static(_)
            | MeshRendererOverrideTarget::Skinned(_)
            | MeshRendererOverrideTarget::Unknown => {
                self.dirty_spaces.insert(dirty.space_id);
            }
        }
    }

    /// Consumes mesh-pool mutations into mesh-asset dirty records or a full rebuild fallback.
    fn note_mesh_pool_delta(
        &mut self,
        mesh_pool: &MeshPool,
        stats: &mut RenderWorldMaintenanceStats,
    ) {
        let delta = mesh_pool.mutation_delta_since(self.mesh_pool_generation);
        if delta.current_generation == self.mesh_pool_generation {
            return;
        }
        self.mesh_pool_generation = delta.current_generation;
        if delta.requires_full_rebuild {
            self.full_rebuild_requested = true;
            return;
        }
        stats.mesh_asset_invalidation_count += delta.changed_asset_ids.len();
        for &asset_id in delta.changed_asset_ids {
            self.dirty_mesh_assets.insert(asset_id);
        }
    }

    /// Marks every live scene space dirty for a full rebuild.
    fn mark_all_scene_spaces_dirty(&mut self, scene: &SceneCoordinator) {
        profiling::scope!("mesh::render_world::mark_all_scene_spaces_dirty");
        self.spaces.retain(|id, _| scene.space(*id).is_some());
        for id in scene.render_space_ids() {
            self.dirty_spaces.insert(id);
        }
        self.dirty_renderers.clear();
        self.dirty_transform_roots.clear();
        self.dirty_mesh_assets.clear();
    }

    /// Expands deferred transform-root and mesh-asset dirties into renderer-record dirties.
    fn expand_deferred_dirty_inputs(&mut self, scene: &SceneCoordinator) {
        self.expand_dirty_transform_roots(scene);
        self.expand_dirty_mesh_assets();
    }

    /// Expands transform-root dirties to descendant renderer records.
    fn expand_dirty_transform_roots(&mut self, scene: &SceneCoordinator) {
        if self.dirty_transform_roots.is_empty() {
            return;
        }
        profiling::scope!("mesh::render_world::expand_transform_roots");
        let roots = std::mem::take(&mut self.dirty_transform_roots);
        let expansions = if roots.len() >= DIRTY_ROOT_EXPANSION_PARALLEL_MIN_ITEMS {
            roots
                .par_iter()
                .with_min_len(DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS)
                .map(|dirty| self.expand_transform_dirty(scene, dirty))
                .collect::<Vec<_>>()
        } else {
            roots
                .iter()
                .map(|dirty| self.expand_transform_dirty(scene, dirty))
                .collect()
        };
        self.apply_transform_dirty_expansions(expansions);
    }

    /// Expands one transform-root dirty input using retained node reverse indexes.
    fn expand_transform_dirty(
        &self,
        scene: &SceneCoordinator,
        dirty: &RenderWorldTransformDirty,
    ) -> TransformDirtyExpansion {
        if self.dirty_spaces.contains(&dirty.space_id) {
            return TransformDirtyExpansion::Empty;
        }
        let Some(space_view) = scene.space(dirty.space_id) else {
            return TransformDirtyExpansion::Removed(dirty.space_id);
        };
        let Some(cached) = self.spaces.get(&dirty.space_id) else {
            return TransformDirtyExpansion::FullSpace(dirty.space_id);
        };
        let parents = space_view.node_parents();
        let mut renderers = Vec::new();
        for (&node_id, refs) in &cached.node_index {
            if !node_is_under_any_root(parents, node_id, &dirty.root_node_ids) {
                continue;
            }
            renderers.extend(refs.iter().map(|renderer| RenderWorldRendererDirty {
                space_id: dirty.space_id,
                kind: renderer.kind,
                renderable_index: renderer.index,
            }));
        }
        if renderers.is_empty() {
            TransformDirtyExpansion::Empty
        } else {
            TransformDirtyExpansion::Renderers(renderers)
        }
    }

    /// Applies transform dirty expansion results to the retained cache's dirty sets.
    fn apply_transform_dirty_expansions(&mut self, expansions: Vec<TransformDirtyExpansion>) {
        for expansion in expansions {
            match expansion {
                TransformDirtyExpansion::Removed(space_id) => self.remove_space(space_id),
                TransformDirtyExpansion::FullSpace(space_id) => {
                    self.dirty_spaces.insert(space_id);
                }
                TransformDirtyExpansion::Renderers(renderers) => {
                    for dirty in renderers {
                        self.note_renderer_dirty(dirty);
                    }
                }
                TransformDirtyExpansion::Empty => {}
            }
        }
    }

    /// Expands dirty mesh asset ids to renderer records through retained reverse indexes.
    fn expand_dirty_mesh_assets(&mut self) {
        if self.dirty_mesh_assets.is_empty() {
            return;
        }
        profiling::scope!("mesh::render_world::expand_mesh_asset_dirties");
        let dirty_mesh_assets = std::mem::take(&mut self.dirty_mesh_assets);
        let spaces = self.spaces.iter().collect::<Vec<_>>();
        let collect_for_space = |(space_id, space): &(&RenderSpaceId, &RenderWorldSpace)| {
            let mut renderer_dirties = Vec::new();
            if self.dirty_spaces.contains(*space_id) {
                return renderer_dirties;
            }
            for asset_id in &dirty_mesh_assets {
                if let Some(renderers) = space.mesh_asset_index.get(asset_id) {
                    renderer_dirties.extend(renderers.iter().map(|renderer| {
                        RenderWorldRendererDirty {
                            space_id: **space_id,
                            kind: renderer.kind,
                            renderable_index: renderer.index,
                        }
                    }));
                }
            }
            renderer_dirties
        };
        let renderer_dirties = if spaces.len() >= MESH_ASSET_DIRTY_EXPANSION_PARALLEL_MIN_SPACES {
            spaces
                .par_iter()
                .with_min_len(MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES)
                .flat_map(collect_for_space)
                .collect::<Vec<_>>()
        } else {
            spaces.iter().flat_map(collect_for_space).collect()
        };
        for dirty in renderer_dirties {
            self.note_renderer_dirty(dirty);
        }
    }

    /// Refreshes all spaces marked for full retained-template rebuild.
    fn refresh_dirty_spaces(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> RefreshOutcome {
        profiling::scope!("mesh::render_world::refresh_dirty_spaces");
        let dirty_spaces = std::mem::take(&mut self.dirty_spaces);
        let mut work = Vec::with_capacity(dirty_spaces.len());
        for id in dirty_spaces {
            self.dirty_renderers.retain(|dirty| dirty.space_id != id);
            work.push(DirtySpaceRefreshWork {
                id,
                cached: self.spaces.remove(&id).unwrap_or_default(),
                outcome: RefreshOutcome::default(),
            });
        }
        if work.len() >= DIRTY_SPACE_REFRESH_PARALLEL_MIN_SPACES {
            work.par_iter_mut()
                .with_min_len(DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES)
                .for_each(|work| {
                    profiling::scope!("mesh::render_world::refresh_dirty_spaces::worker");
                    work.outcome = refresh_render_world_space(
                        &mut work.cached,
                        scene,
                        mesh_pool,
                        render_context,
                        work.id,
                    );
                });
        } else {
            for work in &mut work {
                work.outcome = refresh_render_world_space(
                    &mut work.cached,
                    scene,
                    mesh_pool,
                    render_context,
                    work.id,
                );
            }
        }
        let mut outcome = RefreshOutcome::default();
        for work in work {
            outcome.renderer_count += work.outcome.renderer_count;
            outcome.template_count += work.outcome.template_count;
            outcome.full_space_count += work.outcome.full_space_count;
            if scene.space(work.id).is_some() {
                self.spaces.insert(work.id, work.cached);
            }
        }
        outcome
    }

    /// Refreshes individual renderer records marked dirty by scene or mesh-pool events.
    fn refresh_dirty_renderers(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> RefreshOutcome {
        profiling::scope!("mesh::render_world::refresh_dirty_renderers");
        let dirty_renderers = std::mem::take(&mut self.dirty_renderers);
        let mut by_space: HashMap<RenderSpaceId, DirtyRendererSet> = HashMap::new();
        for dirty in dirty_renderers {
            by_space
                .entry(dirty.space_id)
                .or_default()
                .insert(dirty.kind, dirty.renderable_index);
        }

        let mut outcome = RefreshOutcome::default();
        let mut work = Vec::with_capacity(by_space.len());
        for (space_id, dirty_set) in by_space {
            if dirty_set.is_empty() {
                continue;
            }
            if scene.space(space_id).is_none() {
                self.remove_space(space_id);
                continue;
            }
            work.push(DirtyRendererRefreshWork {
                id: space_id,
                dirty_set,
                cached: self.spaces.remove(&space_id).unwrap_or_default(),
                outcome: RefreshOutcome::default(),
            });
        }
        if work.len() >= DIRTY_SPACE_REFRESH_PARALLEL_MIN_SPACES {
            work.par_iter_mut()
                .with_min_len(DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES)
                .for_each(|work| {
                    profiling::scope!("mesh::render_world::refresh_dirty_renderers::worker");
                    work.outcome =
                        refresh_dirty_renderer_work(work, scene, mesh_pool, render_context);
                });
        } else {
            for work in &mut work {
                work.outcome = refresh_dirty_renderer_work(work, scene, mesh_pool, render_context);
            }
        }
        for work in work {
            outcome.renderer_count += work.outcome.renderer_count;
            outcome.template_count += work.outcome.template_count;
            if scene.space(work.id).is_some() {
                self.spaces.insert(work.id, work.cached);
            }
        }
        outcome
    }

    /// Rebuilds the per-view-consumable prepared snapshot from retained renderer templates.
    fn rebuild_prepared_snapshot(
        &mut self,
        scene: &SceneCoordinator,
        render_context: RenderingContext,
    ) {
        profiling::scope!("mesh::render_world::rebuild_prepared_snapshot");
        self.prepared.begin_cached_rebuild(render_context);
        let active_spaces = scene
            .render_space_ids()
            .filter_map(|id| {
                self.spaces
                    .get(&id)
                    .filter(|space| space.active)
                    .map(|s| (id, s))
            })
            .collect::<Vec<_>>();
        let retained_draw_count = active_spaces
            .iter()
            .map(|(_, space)| space.retained_template_count())
            .sum::<usize>();
        if active_spaces.len() >= SNAPSHOT_REBUILD_PARALLEL_MIN_SPACES
            && retained_draw_count >= SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS
        {
            profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::parallel");
            let outputs = active_spaces
                .par_iter()
                .with_min_len(SNAPSHOT_REBUILD_PARALLEL_CHUNK_SPACES)
                .map(|(id, space)| {
                    profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::worker");
                    let mut draws = Vec::with_capacity(space.retained_template_count());
                    space.append_draws_to(&mut draws);
                    (*id, draws)
                })
                .collect::<Vec<_>>();
            for (id, draws) in outputs {
                self.prepared.push_cached_space(id);
                self.prepared.extend_cached_draws(&draws);
            }
        } else {
            profiling::scope!("mesh::render_world::rebuild_prepared_snapshot::serial");
            for (id, space) in active_spaces {
                self.prepared.push_cached_space(id);
                space.append_to_prepared(&mut self.prepared);
            }
        }
        self.prepared.finish_cached_rebuild();
    }

    /// Number of retained draw templates currently cached.
    fn retained_template_count(&self) -> usize {
        self.spaces
            .values()
            .map(RenderWorldSpace::retained_template_count)
            .sum()
    }
}

/// Refreshes one worker-owned retained render-space for a partial renderer dirty set.
fn refresh_dirty_renderer_work(
    work: &mut DirtyRendererRefreshWork,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
) -> RefreshOutcome {
    let Some(space_view) = scene.space(work.id) else {
        work.cached.active = false;
        return RefreshOutcome::default();
    };
    work.cached.active = space_view.is_active();
    if !work.cached.active {
        return RefreshOutcome::default();
    }
    work.cached
        .static_renderers
        .resize_with(space_view.static_mesh_renderers().len(), Default::default);
    work.cached
        .skinned_renderers
        .resize_with(space_view.skinned_mesh_renderers().len(), Default::default);
    refresh_renderer_set(
        &mut work.cached,
        &work.dirty_set,
        space_view,
        scene,
        mesh_pool,
        render_context,
        work.id,
    )
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new(RenderingContext::default())
    }
}

#[cfg(test)]
mod tests;
