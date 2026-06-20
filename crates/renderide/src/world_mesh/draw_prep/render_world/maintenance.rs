//! Dirty expansion and retained render-world maintenance passes.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{
    RenderSpaceId, RenderSpaceRead, RenderWorldBoundsDirty, RenderWorldRendererDirty,
    RenderWorldRendererKind, RenderWorldTransformDirty, WorldMeshSceneRead,
};
use crate::shared::RenderingContext;

use super::refresh::{
    DirtyRendererSet, RefreshOutcome, refresh_render_world_space, refresh_renderer_bounds_set,
    refresh_renderer_set,
};
use super::state::{RenderWorldRendererRef, RenderWorldSpace};
use super::{
    RenderWorld, RenderWorldDirtyReason, RenderWorldMaintenanceStats, dirty_refresh_admission,
    mesh_asset_expansion_admission, node_is_under_any_root, transform_root_expansion_admission,
    transform_root_node_scan_admission, transform_roots_cover_space,
};
use crate::cpu_parallelism::FrameParallelPolicy;

/// Counters gathered while expanding deferred transform-root dirties.
#[derive(Default)]
pub(super) struct TransformDirtyExpansionStats {
    /// Unique transform-root node ids consumed.
    root_count: usize,
    /// Retained node-index entries scanned.
    scanned_node_count: usize,
    /// Renderer records found by retained reverse-index expansion.
    expanded_renderer_count: usize,
    /// Transform roots that covered all cached renderer records in a space.
    full_space_root_count: usize,
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

/// Result of expanding one transform-root dirty input with accounting.
struct TransformDirtyExpansionResult {
    /// Dirty-expansion action to apply to render-world dirty sets.
    expansion: TransformDirtyExpansion,
    /// Number of unique transform-root node ids represented by this coalesced result.
    root_count: usize,
    /// Retained node-index entries scanned while expanding this dirty record.
    scanned_node_count: usize,
    /// Whether the dirty roots covered the entire retained render space.
    full_space_root: bool,
}

/// Worker-owned full-space refresh payload.
struct DirtySpaceRefreshWork {
    /// Render space being refreshed.
    id: RenderSpaceId,
    /// Estimated renderer/template work used for parallel admission.
    estimated_work_units: usize,
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
    /// Estimated renderer/template work used for parallel admission.
    estimated_work_units: usize,
    /// Retained cache removed from [`RenderWorld::spaces`] for worker-owned mutation.
    cached: RenderWorldSpace,
    /// Refresh counters produced by the worker.
    outcome: RefreshOutcome,
}

/// Worker-owned dynamic-bounds refresh payload.
struct DirtyBoundsRefreshWork {
    /// Render space containing the dirty renderer records.
    id: RenderSpaceId,
    /// Dirty bounds records grouped for this space.
    dirty_set: DirtyRendererSet,
    /// Estimated renderer work used for parallel admission.
    estimated_work_units: usize,
    /// Retained cache removed from [`RenderWorld::spaces`] for worker-owned mutation.
    cached: RenderWorldSpace,
    /// Refresh counters produced by the worker.
    outcome: RefreshOutcome,
}

/// Coalesces transform-root dirty inputs by render space while preserving first-seen space order.
fn coalesce_transform_dirty_roots(
    roots: Vec<RenderWorldTransformDirty>,
) -> Vec<RenderWorldTransformDirty> {
    let mut order = Vec::new();
    let mut by_space: HashMap<RenderSpaceId, (Vec<i32>, HashSet<i32>)> = HashMap::new();
    for dirty in roots {
        if !by_space.contains_key(&dirty.space_id) {
            order.push(dirty.space_id);
        }
        let (root_node_ids, seen_roots) = by_space.entry(dirty.space_id).or_default();
        for root_node_id in dirty.root_node_ids {
            if seen_roots.insert(root_node_id) {
                root_node_ids.push(root_node_id);
            }
        }
    }
    order
        .into_iter()
        .filter_map(|space_id| {
            by_space
                .remove(&space_id)
                .map(|(root_node_ids, _)| RenderWorldTransformDirty {
                    space_id,
                    root_node_ids,
                })
        })
        .collect()
}

/// Appends renderer dirty records referenced by one retained node-index entry.
fn push_transform_node_renderer_dirties(
    renderers: &mut Vec<RenderWorldRendererDirty>,
    space_id: RenderSpaceId,
    node_id: i32,
    refs: &[RenderWorldRendererRef],
    parents: &[i32],
    roots: &[i32],
) {
    if !node_is_under_any_root(parents, node_id, roots) {
        return;
    }
    renderers.extend(refs.iter().map(|renderer| RenderWorldRendererDirty {
        space_id,
        kind: renderer.kind,
        renderable_index: renderer.index,
    }));
}

/// Returns retained renderer dirty records for every valid cached renderer in a space.
fn cached_renderer_dirties_for_space(
    space_id: RenderSpaceId,
    cached: &RenderWorldSpace,
) -> Vec<RenderWorldRendererDirty> {
    cached
        .static_renderers
        .iter()
        .enumerate()
        .filter(|(_, renderer)| renderer.node_id >= 0)
        .map(|(renderable_index, _)| RenderWorldRendererDirty {
            space_id,
            kind: RenderWorldRendererKind::Static,
            renderable_index,
        })
        .chain(
            cached
                .skinned_renderers
                .iter()
                .enumerate()
                .filter(|(_, renderer)| renderer.node_id >= 0)
                .map(|(renderable_index, _)| RenderWorldRendererDirty {
                    space_id,
                    kind: RenderWorldRendererKind::Skinned,
                    renderable_index,
                }),
        )
        .collect()
}

impl RenderWorld {
    /// Expands deferred transform-root and mesh-asset dirties into renderer-record dirties.
    pub(super) fn expand_deferred_dirty_inputs<S>(
        &mut self,
        scene: &S,
        stats: &mut RenderWorldMaintenanceStats,
    ) where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        let transform_stats = self.expand_dirty_transform_roots(scene);
        stats.transform_root_dirty_count += transform_stats.root_count;
        stats.transform_root_scanned_node_count += transform_stats.scanned_node_count;
        stats.transform_root_expanded_renderer_count += transform_stats.expanded_renderer_count;
        stats.transform_root_full_space_count += transform_stats.full_space_root_count;
        self.expand_dirty_mesh_assets();
    }

    /// Expands transform-root dirties to descendant renderer records.
    pub(super) fn expand_dirty_transform_roots<S>(
        &mut self,
        scene: &S,
    ) -> TransformDirtyExpansionStats
    where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        if self.dirty_transform_roots.is_empty() {
            return TransformDirtyExpansionStats::default();
        }
        profiling::scope!("mesh::render_world::expand_transform_roots");
        let roots = coalesce_transform_dirty_roots(std::mem::take(&mut self.dirty_transform_roots));
        let policy = FrameParallelPolicy::for_current_thread_pool();
        let expansions = match transform_root_expansion_admission(policy, roots.len()).chunk_size()
        {
            Some(chunk_size) => roots
                .par_iter()
                .with_min_len(chunk_size)
                .map(|dirty| self.expand_transform_dirty(scene, dirty))
                .collect::<Vec<_>>(),
            None => roots
                .iter()
                .map(|dirty| self.expand_transform_dirty(scene, dirty))
                .collect(),
        };
        self.apply_transform_dirty_expansions(expansions)
    }

    /// Expands one transform-root dirty input using retained node reverse indexes.
    fn expand_transform_dirty<S>(
        &self,
        scene: &S,
        dirty: &RenderWorldTransformDirty,
    ) -> TransformDirtyExpansionResult
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        let root_count = dirty.root_node_ids.len();
        if self.dirty_spaces.contains(&dirty.space_id) {
            return TransformDirtyExpansionResult {
                expansion: TransformDirtyExpansion::Empty,
                root_count,
                scanned_node_count: 0,
                full_space_root: false,
            };
        }
        let Some(space_view) = scene.space(dirty.space_id) else {
            return TransformDirtyExpansionResult {
                expansion: TransformDirtyExpansion::Removed(dirty.space_id),
                root_count,
                scanned_node_count: 0,
                full_space_root: false,
            };
        };
        let Some(cached) = self.spaces.get(&dirty.space_id) else {
            return TransformDirtyExpansionResult {
                expansion: TransformDirtyExpansion::FullSpace(dirty.space_id),
                root_count,
                scanned_node_count: 0,
                full_space_root: false,
            };
        };
        let parents = space_view.node_parents();
        let full_space_root = transform_roots_cover_space(parents, &dirty.root_node_ids);
        let renderers = if full_space_root {
            cached_renderer_dirties_for_space(dirty.space_id, cached)
        } else {
            self.expand_transform_dirty_by_node_index(cached, dirty, parents)
        };
        let expansion = if renderers.is_empty() {
            TransformDirtyExpansion::Empty
        } else {
            TransformDirtyExpansion::Renderers(renderers)
        };
        TransformDirtyExpansionResult {
            expansion,
            root_count,
            scanned_node_count: if full_space_root {
                0
            } else {
                cached.node_index.len()
            },
            full_space_root,
        }
    }

    /// Expands one transform dirty by scanning retained node reverse-index entries.
    fn expand_transform_dirty_by_node_index(
        &self,
        cached: &RenderWorldSpace,
        dirty: &RenderWorldTransformDirty,
        parents: &[i32],
    ) -> Vec<RenderWorldRendererDirty> {
        let node_count = cached.node_index.len();
        let policy = FrameParallelPolicy::for_current_thread_pool();
        match transform_root_node_scan_admission(policy, node_count).chunk_size() {
            Some(chunk_size) => {
                profiling::scope!("mesh::render_world::expand_transform_roots::node_scan_parallel");
                let entries = cached.node_index.iter().collect::<Vec<_>>();
                entries
                    .par_chunks(chunk_size)
                    .with_min_len(1)
                    .map(|chunk| {
                        profiling::scope!(
                            "mesh::render_world::expand_transform_roots::node_scan_worker"
                        );
                        let mut renderers = Vec::new();
                        for &(node_id, refs) in chunk {
                            push_transform_node_renderer_dirties(
                                &mut renderers,
                                dirty.space_id,
                                *node_id,
                                refs,
                                parents,
                                &dirty.root_node_ids,
                            );
                        }
                        renderers
                    })
                    .reduce(Vec::new, |mut left, mut right| {
                        left.append(&mut right);
                        left
                    })
            }
            None => {
                let mut renderers = Vec::new();
                for (&node_id, refs) in &cached.node_index {
                    push_transform_node_renderer_dirties(
                        &mut renderers,
                        dirty.space_id,
                        node_id,
                        refs,
                        parents,
                        &dirty.root_node_ids,
                    );
                }
                renderers
            }
        }
    }

    /// Applies transform dirty expansion results to the retained cache's dirty sets.
    fn apply_transform_dirty_expansions(
        &mut self,
        expansions: Vec<TransformDirtyExpansionResult>,
    ) -> TransformDirtyExpansionStats {
        let mut stats = TransformDirtyExpansionStats::default();
        for expansion in expansions {
            stats.root_count += expansion.root_count;
            stats.scanned_node_count += expansion.scanned_node_count;
            stats.full_space_root_count += usize::from(expansion.full_space_root);
            match expansion.expansion {
                TransformDirtyExpansion::Removed(space_id) => self.remove_space(space_id),
                TransformDirtyExpansion::FullSpace(space_id) => {
                    self.note_space_dirty(space_id);
                }
                TransformDirtyExpansion::Renderers(renderers) => {
                    stats.expanded_renderer_count += renderers.len();
                    for dirty in renderers {
                        self.note_bounds_dirty(
                            RenderWorldBoundsDirty {
                                space_id: dirty.space_id,
                                kind: dirty.kind,
                                renderable_index: dirty.renderable_index,
                            },
                            RenderWorldDirtyReason::TransformOnly,
                        );
                    }
                }
                TransformDirtyExpansion::Empty => {}
            }
        }
        stats
    }

    /// Expands dirty mesh asset ids to renderer records through retained reverse indexes.
    pub(super) fn expand_dirty_mesh_assets(&mut self) {
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
        let policy = FrameParallelPolicy::for_current_thread_pool();
        let renderer_dirties =
            match mesh_asset_expansion_admission(policy, spaces.len()).chunk_size() {
                Some(chunk_size) => spaces
                    .par_iter()
                    .with_min_len(chunk_size)
                    .flat_map(collect_for_space)
                    .collect::<Vec<_>>(),
                None => spaces.iter().flat_map(collect_for_space).collect(),
            };
        for dirty in renderer_dirties {
            self.note_renderer_dirty(dirty, RenderWorldDirtyReason::MeshAsset);
        }
    }

    /// Refreshes all spaces marked for full retained-template rebuild.
    pub(super) fn refresh_dirty_spaces<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> RefreshOutcome
    where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        profiling::scope!("mesh::render_world::refresh_dirty_spaces");
        let dirty_spaces = std::mem::take(&mut self.dirty_spaces);
        let mut work = Vec::with_capacity(dirty_spaces.len());
        for id in dirty_spaces {
            self.dirty_renderers.retain(|dirty, _| dirty.space_id != id);
            self.dirty_bounds_renderers
                .retain(|dirty, _| dirty.space_id != id);
            let cached = self.spaces.remove(&id).unwrap_or_default();
            let estimated_work_units = estimate_full_space_refresh_work(&cached, scene, id);
            work.push(DirtySpaceRefreshWork {
                id,
                estimated_work_units,
                cached,
                outcome: RefreshOutcome::default(),
            });
        }
        let estimated_work_units = work.iter().map(|work| work.estimated_work_units).sum();
        let policy = FrameParallelPolicy::for_current_thread_pool();
        match dirty_refresh_admission(policy, work.len(), estimated_work_units).chunk_size() {
            Some(chunk_size) => {
                work.par_iter_mut()
                    .with_min_len(chunk_size)
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
            }
            None => {
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
    pub(super) fn refresh_dirty_renderers<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> RefreshOutcome
    where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        profiling::scope!("mesh::render_world::refresh_dirty_renderers");
        let dirty_renderers = std::mem::take(&mut self.dirty_renderers);
        self.dirty_bounds_renderers.retain(|dirty, _| {
            !dirty_renderers.contains_key(&RenderWorldRendererDirty {
                space_id: dirty.space_id,
                kind: dirty.kind,
                renderable_index: dirty.renderable_index,
            })
        });
        let mut by_space: HashMap<RenderSpaceId, DirtyRendererSet> = HashMap::new();
        for dirty in dirty_renderers.into_keys() {
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
            let estimated_work_units = dirty_set.len();
            work.push(DirtyRendererRefreshWork {
                id: space_id,
                dirty_set,
                estimated_work_units,
                cached: self.spaces.remove(&space_id).unwrap_or_default(),
                outcome: RefreshOutcome::default(),
            });
        }
        let estimated_work_units = work.iter().map(|work| work.estimated_work_units).sum();
        let policy = FrameParallelPolicy::for_current_thread_pool();
        match dirty_refresh_admission(policy, work.len(), estimated_work_units).chunk_size() {
            Some(chunk_size) => {
                work.par_iter_mut()
                    .with_min_len(chunk_size)
                    .for_each(|work| {
                        profiling::scope!("mesh::render_world::refresh_dirty_renderers::worker");
                        work.outcome =
                            refresh_dirty_renderer_work(work, scene, mesh_pool, render_context);
                    });
            }
            None => {
                for work in &mut work {
                    work.outcome =
                        refresh_dirty_renderer_work(work, scene, mesh_pool, render_context);
                }
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

    /// Refreshes dynamic bounds for renderer records marked by transform-only scene changes.
    pub(super) fn refresh_dirty_bounds<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> RefreshOutcome
    where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        profiling::scope!("mesh::render_world::refresh_dirty_bounds");
        let dirty_bounds = std::mem::take(&mut self.dirty_bounds_renderers);
        let mut by_space: HashMap<RenderSpaceId, DirtyRendererSet> = HashMap::new();
        for dirty in dirty_bounds.into_keys() {
            by_space
                .entry(dirty.space_id)
                .or_default()
                .insert(dirty.kind, dirty.renderable_index);
        }

        let mut work = Vec::with_capacity(by_space.len());
        for (space_id, dirty_set) in by_space {
            if dirty_set.is_empty() {
                continue;
            }
            if scene.space(space_id).is_none() {
                self.remove_space(space_id);
                continue;
            }
            let Some(cached) = self.spaces.remove(&space_id) else {
                self.note_space_dirty(space_id);
                continue;
            };
            let estimated_work_units = dirty_set.len();
            work.push(DirtyBoundsRefreshWork {
                id: space_id,
                dirty_set,
                estimated_work_units,
                cached,
                outcome: RefreshOutcome::default(),
            });
        }
        let estimated_work_units = work.iter().map(|work| work.estimated_work_units).sum();
        let policy = FrameParallelPolicy::for_current_thread_pool();
        match dirty_refresh_admission(policy, work.len(), estimated_work_units).chunk_size() {
            Some(chunk_size) => {
                work.par_iter_mut()
                    .with_min_len(chunk_size)
                    .for_each(|work| {
                        profiling::scope!("mesh::render_world::refresh_dirty_bounds::worker");
                        work.outcome =
                            refresh_dirty_bounds_work(work, scene, mesh_pool, render_context);
                    });
            }
            None => {
                for work in &mut work {
                    work.outcome =
                        refresh_dirty_bounds_work(work, scene, mesh_pool, render_context);
                }
            }
        }
        let mut outcome = RefreshOutcome::default();
        let mut refit_spaces = Vec::new();
        for work in work {
            outcome.renderer_count += work.outcome.renderer_count;
            if scene.space(work.id).is_some() {
                if work.cached.active && work.outcome.renderer_count > 0 {
                    self.update_prepared_bounds_for_set(work.id, &work.cached, &work.dirty_set);
                    refit_spaces.push(work.id);
                }
                self.spaces.insert(work.id, work.cached);
            }
        }
        if !refit_spaces.is_empty() {
            outcome.spatial_refit_count = self
                .prepared
                .refit_cached_spatial_for_spaces(refit_spaces.iter().copied());
        }
        outcome
    }

    /// Copies refreshed per-renderer cull geometry into the existing prepared snapshot.
    fn update_prepared_bounds_for_set(
        &mut self,
        space_id: RenderSpaceId,
        cached: &RenderWorldSpace,
        dirty_set: &DirtyRendererSet,
    ) {
        for &index in &dirty_set.static_indices {
            if let Some(record) = cached.static_renderers.get(index) {
                self.prepared.update_cached_renderer_cull_geometry(
                    space_id,
                    false,
                    index,
                    record.instance_id,
                    record.cull_geometry,
                );
            }
        }
        for &index in &dirty_set.skinned_indices {
            if let Some(record) = cached.skinned_renderers.get(index) {
                self.prepared.update_cached_renderer_cull_geometry(
                    space_id,
                    true,
                    index,
                    record.instance_id,
                    record.cull_geometry,
                );
            }
        }
    }
}

/// Estimates full-space retained-refresh work from cached and current scene renderer counts.
fn estimate_full_space_refresh_work(
    cached: &RenderWorldSpace,
    scene: &(impl WorldMeshSceneRead + Sync + ?Sized),
    id: RenderSpaceId,
) -> usize {
    let cached_renderer_count = cached
        .static_renderers
        .len()
        .saturating_add(cached.skinned_renderers.len());
    let scene_renderer_count = scene
        .static_mesh_renderers(id)
        .map_or(0, |renderers| renderers.len())
        .saturating_add(
            scene
                .skinned_mesh_renderers(id)
                .map_or(0, |renderers| renderers.len()),
        );
    cached_renderer_count.max(scene_renderer_count)
}

/// Refreshes one worker-owned retained render-space for a partial renderer dirty set.
fn refresh_dirty_renderer_work(
    work: &mut DirtyRendererRefreshWork,
    scene: &(impl WorldMeshSceneRead + Sync + ?Sized),
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
    work.cached.static_renderers.resize_with(
        scene
            .static_mesh_renderers(work.id)
            .map_or(0, |renderers| renderers.len()),
        Default::default,
    );
    work.cached.skinned_renderers.resize_with(
        scene
            .skinned_mesh_renderers(work.id)
            .map_or(0, |renderers| renderers.len()),
        Default::default,
    );
    refresh_renderer_set(
        &mut work.cached,
        &work.dirty_set,
        scene,
        mesh_pool,
        render_context,
        work.id,
    )
}

/// Refreshes one worker-owned retained render-space for a bounds-only dirty set.
fn refresh_dirty_bounds_work(
    work: &mut DirtyBoundsRefreshWork,
    scene: &(impl WorldMeshSceneRead + Sync + ?Sized),
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
    refresh_renderer_bounds_set(
        &mut work.cached,
        &work.dirty_set,
        scene,
        mesh_pool,
        render_context,
        work.id,
    )
}
