//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores renderer-facing draw templates that are expensive to
//! rediscover every frame.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{
    MeshRendererInstanceId, MeshRendererOverrideTarget, RenderSpaceId,
    RenderWorldMaterialOverrideDirty, RenderWorldRendererDirty, RenderWorldRendererKind,
    RenderWorldTransformDirty, SceneApplyReport, SceneCacheFlushReport, SceneCoordinator,
    SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::shared::RenderingContext;

use super::prepared_renderables::{
    FramePreparedDraw, FramePreparedRenderables, expand_skinned_renderer_into,
    expand_static_renderer_into,
};

/// Renderer count above which retained-template refresh uses Rayon.
const RENDER_WORLD_PARALLEL_MIN_RENDERERS: usize = 256;

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

#[derive(Default)]
struct RenderWorldSpace {
    /// Whether the host render space is active.
    active: bool,
    /// Retained draw templates for static renderers, indexed by scene dense renderer id.
    static_renderers: Vec<RenderWorldRendererTemplate>,
    /// Retained draw templates for skinned renderers, indexed by scene dense renderer id.
    skinned_renderers: Vec<RenderWorldRendererTemplate>,
    /// Reverse map from mesh asset id to renderer records.
    mesh_asset_index: HashMap<i32, Vec<RenderWorldRendererRef>>,
    /// Reverse map from scene node id to renderer records.
    node_index: HashMap<i32, Vec<RenderWorldRendererRef>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RenderWorldRendererRef {
    /// Renderer table containing the record.
    kind: RenderWorldRendererKind,
    /// Dense renderer index in the selected table.
    index: usize,
}

#[derive(Default)]
struct RenderWorldRendererTemplate {
    /// Renderer-local identity that survives dense table reindexing.
    instance_id: MeshRendererInstanceId,
    /// Scene node id used by transform dirty expansion.
    node_id: i32,
    /// Mesh asset id used by mesh-pool dirty expansion.
    mesh_asset_id: i32,
    /// Retained draw templates emitted by this renderer.
    draws: Vec<FramePreparedDraw>,
}

#[derive(Default)]
struct DirtyRendererSet {
    /// Static renderer indices to refresh for one space.
    static_indices: HashSet<usize>,
    /// Skinned renderer indices to refresh for one space.
    skinned_indices: HashSet<usize>,
}

impl DirtyRendererSet {
    /// Number of renderer records in this dirty set.
    fn len(&self) -> usize {
        self.static_indices.len() + self.skinned_indices.len()
    }

    /// Returns whether this set contains no renderer records.
    fn is_empty(&self) -> bool {
        self.static_indices.is_empty() && self.skinned_indices.is_empty()
    }

    /// Inserts one renderer reference.
    fn insert(&mut self, kind: RenderWorldRendererKind, index: usize) {
        match kind {
            RenderWorldRendererKind::Static => {
                self.static_indices.insert(index);
            }
            RenderWorldRendererKind::Skinned => {
                self.skinned_indices.insert(index);
            }
        }
    }
}

#[derive(Default)]
struct RefreshOutcome {
    /// Renderer records refreshed.
    renderer_count: usize,
    /// Draw templates retained by those refreshed records.
    template_count: usize,
    /// Spaces rebuilt through full-space refresh.
    full_space_count: usize,
}

impl RenderWorldRendererTemplate {
    /// Resets scene identity for a missing renderer row while retaining draw allocation.
    fn clear_missing(&mut self) {
        self.instance_id = MeshRendererInstanceId::default();
        self.node_id = -1;
        self.mesh_asset_id = -1;
        self.draws.clear();
    }

    /// Copies identity fields from a static renderer row.
    fn copy_static_identity(&mut self, renderer: &StaticMeshRenderer) {
        self.instance_id = renderer.instance_id;
        self.node_id = renderer.node_id;
        self.mesh_asset_id = renderer.mesh_asset_id;
    }

    /// Copies identity fields from a skinned renderer row.
    fn copy_skinned_identity(&mut self, renderer: &SkinnedMeshRenderer) {
        self.copy_static_identity(&renderer.base);
    }
}

impl RenderWorldSpace {
    /// Number of retained draw templates in this space.
    fn retained_template_count(&self) -> usize {
        self.static_renderers
            .iter()
            .chain(self.skinned_renderers.iter())
            .map(|renderer| renderer.draws.len())
            .sum()
    }

    /// Rebuilds reverse indexes after one or more renderer records changed identity.
    fn rebuild_reverse_indexes(&mut self) {
        profiling::scope!("mesh::render_world::rebuild_reverse_indexes");
        let mesh_asset_index = &mut self.mesh_asset_index;
        let node_index = &mut self.node_index;
        mesh_asset_index.clear();
        node_index.clear();
        for (index, renderer) in self.static_renderers.iter().enumerate() {
            push_reverse_indexes(
                mesh_asset_index,
                node_index,
                RenderWorldRendererRef {
                    kind: RenderWorldRendererKind::Static,
                    index,
                },
                renderer,
            );
        }
        for (index, renderer) in self.skinned_renderers.iter().enumerate() {
            push_reverse_indexes(
                mesh_asset_index,
                node_index,
                RenderWorldRendererRef {
                    kind: RenderWorldRendererKind::Skinned,
                    index,
                },
                renderer,
            );
        }
    }

    /// Extends a prepared snapshot with this space's retained draw templates.
    fn append_to_prepared(&self, prepared: &mut FramePreparedRenderables) {
        for renderer in &self.static_renderers {
            prepared.extend_cached_draws(&renderer.draws);
        }
        for renderer in &self.skinned_renderers {
            prepared.extend_cached_draws(&renderer.draws);
        }
    }
}

/// Adds one renderer record to reverse indexes when it has valid ids.
fn push_reverse_indexes(
    mesh_asset_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    node_index: &mut HashMap<i32, Vec<RenderWorldRendererRef>>,
    renderer_ref: RenderWorldRendererRef,
    renderer: &RenderWorldRendererTemplate,
) {
    if renderer.mesh_asset_id >= 0 {
        mesh_asset_index
            .entry(renderer.mesh_asset_id)
            .or_default()
            .push(renderer_ref);
    }
    if renderer.node_id >= 0 {
        node_index
            .entry(renderer.node_id)
            .or_default()
            .push(renderer_ref);
    }
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

/// Refreshes one static renderer record.
fn refresh_static_renderer_record(
    record: &mut RenderWorldRendererTemplate,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) {
    let Some(space) = scene.space(space_id) else {
        record.clear_missing();
        return;
    };
    let Some(renderer) = space.static_mesh_renderers().get(index) else {
        record.clear_missing();
        return;
    };
    record.copy_static_identity(renderer);
    record.draws.clear();
    let slot_estimate = renderer.material_slots.len().max(1);
    if slot_estimate > record.draws.capacity() {
        record
            .draws
            .reserve(slot_estimate - record.draws.capacity());
    }
    expand_static_renderer_into(
        &mut record.draws,
        scene,
        mesh_pool,
        render_context,
        space_id,
        index,
    );
}

/// Refreshes one skinned renderer record.
fn refresh_skinned_renderer_record(
    record: &mut RenderWorldRendererTemplate,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) {
    let Some(space) = scene.space(space_id) else {
        record.clear_missing();
        return;
    };
    let Some(renderer) = space.skinned_mesh_renderers().get(index) else {
        record.clear_missing();
        return;
    };
    record.copy_skinned_identity(renderer);
    record.draws.clear();
    let slot_estimate = renderer.base.material_slots.len().max(1);
    if slot_estimate > record.draws.capacity() {
        record
            .draws
            .reserve(slot_estimate - record.draws.capacity());
    }
    expand_skinned_renderer_into(
        &mut record.draws,
        scene,
        mesh_pool,
        render_context,
        space_id,
        index,
    );
}

/// Refreshes every retained renderer record for one render space.
fn refresh_render_world_space(
    cached: &mut RenderWorldSpace,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) -> RefreshOutcome {
    profiling::scope!("mesh::render_world::refresh_space");
    let Some(space) = scene.space(id) else {
        cached.active = false;
        cached.static_renderers.clear();
        cached.skinned_renderers.clear();
        cached.rebuild_reverse_indexes();
        return RefreshOutcome::default();
    };
    cached.active = space.is_active();
    cached
        .static_renderers
        .resize_with(space.static_mesh_renderers().len(), Default::default);
    cached
        .skinned_renderers
        .resize_with(space.skinned_mesh_renderers().len(), Default::default);
    if cached.active {
        refresh_all_records(cached, scene, mesh_pool, render_context, id);
    } else {
        for record in cached
            .static_renderers
            .iter_mut()
            .chain(cached.skinned_renderers.iter_mut())
        {
            record.draws.clear();
        }
    }
    cached.rebuild_reverse_indexes();
    RefreshOutcome {
        renderer_count: cached
            .static_renderers
            .len()
            .saturating_add(cached.skinned_renderers.len()),
        template_count: cached.retained_template_count(),
        full_space_count: 1,
    }
}

/// Refreshes all static and skinned records for an active space.
fn refresh_all_records(
    cached: &mut RenderWorldSpace,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) {
    let renderer_count = cached
        .static_renderers
        .len()
        .saturating_add(cached.skinned_renderers.len());
    if renderer_count >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        profiling::scope!("mesh::render_world::refresh_space_parallel");
        cached
            .static_renderers
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, record)| {
                profiling::scope!("mesh::render_world::refresh_static_record_worker");
                refresh_static_renderer_record(record, scene, mesh_pool, render_context, id, index);
            });
        cached
            .skinned_renderers
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, record)| {
                profiling::scope!("mesh::render_world::refresh_skinned_record_worker");
                refresh_skinned_renderer_record(
                    record,
                    scene,
                    mesh_pool,
                    render_context,
                    id,
                    index,
                );
            });
    } else {
        profiling::scope!("mesh::render_world::refresh_space_serial");
        for (index, record) in cached.static_renderers.iter_mut().enumerate() {
            refresh_static_renderer_record(record, scene, mesh_pool, render_context, id, index);
        }
        for (index, record) in cached.skinned_renderers.iter_mut().enumerate() {
            refresh_skinned_renderer_record(record, scene, mesh_pool, render_context, id, index);
        }
    }
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
        let mut renderer_dirties = Vec::new();
        for dirty in roots {
            if self.dirty_spaces.contains(&dirty.space_id) {
                continue;
            }
            let Some(space_view) = scene.space(dirty.space_id) else {
                self.remove_space(dirty.space_id);
                continue;
            };
            let Some(cached) = self.spaces.get(&dirty.space_id) else {
                self.dirty_spaces.insert(dirty.space_id);
                continue;
            };
            let parents = space_view.node_parents();
            for (index, renderer) in cached.static_renderers.iter().enumerate() {
                if node_is_under_any_root(parents, renderer.node_id, &dirty.root_node_ids) {
                    renderer_dirties.push(RenderWorldRendererDirty {
                        space_id: dirty.space_id,
                        kind: RenderWorldRendererKind::Static,
                        renderable_index: index,
                    });
                }
            }
            for (index, renderer) in cached.skinned_renderers.iter().enumerate() {
                if node_is_under_any_root(parents, renderer.node_id, &dirty.root_node_ids) {
                    renderer_dirties.push(RenderWorldRendererDirty {
                        space_id: dirty.space_id,
                        kind: RenderWorldRendererKind::Skinned,
                        renderable_index: index,
                    });
                }
            }
        }
        for dirty in renderer_dirties {
            self.note_renderer_dirty(dirty);
        }
    }

    /// Expands dirty mesh asset ids to renderer records through retained reverse indexes.
    fn expand_dirty_mesh_assets(&mut self) {
        if self.dirty_mesh_assets.is_empty() {
            return;
        }
        profiling::scope!("mesh::render_world::expand_mesh_asset_dirties");
        let dirty_mesh_assets = std::mem::take(&mut self.dirty_mesh_assets);
        let mut renderer_dirties = Vec::new();
        for (space_id, space) in &self.spaces {
            if self.dirty_spaces.contains(space_id) {
                continue;
            }
            for asset_id in &dirty_mesh_assets {
                let Some(renderers) = space.mesh_asset_index.get(asset_id) else {
                    continue;
                };
                for renderer in renderers {
                    renderer_dirties.push(RenderWorldRendererDirty {
                        space_id: *space_id,
                        kind: renderer.kind,
                        renderable_index: renderer.index,
                    });
                }
            }
        }
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
        let mut outcome = RefreshOutcome::default();
        for id in dirty_spaces {
            self.dirty_renderers.retain(|dirty| dirty.space_id != id);
            let mut cached = self.spaces.remove(&id).unwrap_or_default();
            let refreshed =
                refresh_render_world_space(&mut cached, scene, mesh_pool, render_context, id);
            outcome.renderer_count += refreshed.renderer_count;
            outcome.template_count += refreshed.template_count;
            outcome.full_space_count += refreshed.full_space_count;
            if scene.space(id).is_some() {
                self.spaces.insert(id, cached);
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
        for (space_id, dirty_set) in by_space {
            if dirty_set.is_empty() {
                continue;
            }
            let Some(space_view) = scene.space(space_id) else {
                self.remove_space(space_id);
                continue;
            };
            let cached = self.spaces.entry(space_id).or_default();
            cached.active = space_view.is_active();
            if !cached.active {
                continue;
            }
            cached
                .static_renderers
                .resize_with(space_view.static_mesh_renderers().len(), Default::default);
            cached
                .skinned_renderers
                .resize_with(space_view.skinned_mesh_renderers().len(), Default::default);
            let refreshed = refresh_renderer_set(
                cached,
                &dirty_set,
                scene,
                mesh_pool,
                render_context,
                space_id,
            );
            cached.rebuild_reverse_indexes();
            outcome.renderer_count += refreshed.renderer_count;
            outcome.template_count += refreshed.template_count;
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
        for id in scene.render_space_ids() {
            let Some(space) = self.spaces.get(&id).filter(|space| space.active) else {
                continue;
            };
            self.prepared.push_cached_space(id);
            space.append_to_prepared(&mut self.prepared);
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

/// Refreshes all renderer records in a dirty set for one active render space.
fn refresh_renderer_set(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) -> RefreshOutcome {
    let mut outcome = RefreshOutcome {
        renderer_count: dirty_set.len(),
        ..Default::default()
    };
    let dirty_count = dirty_set.len();
    if dirty_count >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        profiling::scope!("mesh::render_world::refresh_renderer_set_parallel");
        cached
            .static_renderers
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, record)| {
                if dirty_set.static_indices.contains(&index) {
                    refresh_static_renderer_record(
                        record,
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                }
            });
        cached
            .skinned_renderers
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, record)| {
                if dirty_set.skinned_indices.contains(&index) {
                    refresh_skinned_renderer_record(
                        record,
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                }
            });
    } else {
        profiling::scope!("mesh::render_world::refresh_renderer_set_serial");
        for &index in &dirty_set.static_indices {
            if let Some(record) = cached.static_renderers.get_mut(index) {
                refresh_static_renderer_record(
                    record,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                    index,
                );
            }
        }
        for &index in &dirty_set.skinned_indices {
            if let Some(record) = cached.skinned_renderers.get_mut(index) {
                refresh_skinned_renderer_record(
                    record,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                    index,
                );
            }
        }
    }
    outcome.template_count = dirty_set
        .static_indices
        .iter()
        .filter_map(|&index| cached.static_renderers.get(index))
        .map(|record| record.draws.len())
        .sum::<usize>()
        + dirty_set
            .skinned_indices
            .iter()
            .filter_map(|&index| cached.skinned_renderers.get(index))
            .map(|record| record.draws.len())
            .sum::<usize>();
    outcome
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new(RenderingContext::default())
    }
}

#[cfg(test)]
mod tests;
