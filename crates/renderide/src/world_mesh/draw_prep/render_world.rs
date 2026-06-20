//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores renderer-facing draw templates that are expensive to
//! rediscover every frame.

mod maintenance;
mod refresh;
mod snapshot;
mod state;

use hashbrown::{HashMap, HashSet};

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission};
use crate::gpu_pools::MeshPool;
use crate::scene::{
    MeshRendererOverrideTarget, RenderSpaceId, RenderWorldBoundsDirty,
    RenderWorldMaterialOverrideDirty, RenderWorldRendererDirty, RenderWorldRendererKind,
    RenderWorldTransformDirty, SceneApplyReport, SceneCacheFlushReport, WorldMeshSceneRead,
};
use crate::shared::RenderingContext;

use super::prepared_renderables::FramePreparedRenderables;
use snapshot::SnapshotRebuildStats;
use state::RenderWorldSpace;

/// Transform-root dirty records assigned to one expansion worker.
const DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS: usize = 1;
/// Retained node-index entries required before one root expansion scans in parallel.
const DIRTY_ROOT_NODE_SCAN_PARALLEL_MIN_NODES: usize = 128;
/// Retained node-index entries assigned to one root-expansion scan task.
const DIRTY_ROOT_NODE_SCAN_PARALLEL_CHUNK_NODES: usize = 64;
/// Render spaces assigned to one mesh-asset dirty expansion worker.
const MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES: usize = 1;
/// Dirty render spaces assigned to one retained-cache refresh worker.
const DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES: usize = 1;
/// Prepared-snapshot copy tasks assigned to one rebuild worker.
const SNAPSHOT_REBUILD_PARALLEL_CHUNK_TASKS: usize = 1;
/// Estimated dirty renderer/template work required before retained cache refresh uses Rayon.
const DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS: usize = 64;
/// Retained draw templates targeted for one prepared-snapshot rebuild task.
const SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES: usize = 256;
/// Retained draw-template count required before snapshot rebuild fan-out is considered.
const SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS: usize =
    SNAPSHOT_REBUILD_PARALLEL_TARGET_CHUNK_TEMPLATES * 2;

/// Returns the admission decision for transform-root dirty expansion.
fn transform_root_expansion_admission(
    policy: FrameParallelPolicy,
    root_count: usize,
) -> ParallelAdmission {
    policy.admit_independent_items(
        FrameCpuWorkload::independent_items(root_count),
        DIRTY_ROOT_EXPANSION_PARALLEL_CHUNK_ITEMS,
    )
}

/// Returns the admission decision for a large retained node-index scan.
fn transform_root_node_scan_admission(
    policy: FrameParallelPolicy,
    node_count: usize,
) -> ParallelAdmission {
    if node_count < DIRTY_ROOT_NODE_SCAN_PARALLEL_MIN_NODES {
        return ParallelAdmission::Serial;
    }
    policy.admit_independent_items(
        FrameCpuWorkload::independent_items(node_count),
        DIRTY_ROOT_NODE_SCAN_PARALLEL_CHUNK_NODES,
    )
}

/// Returns the admission decision for mesh-asset dirty expansion.
fn mesh_asset_expansion_admission(
    policy: FrameParallelPolicy,
    space_count: usize,
) -> ParallelAdmission {
    policy.admit_independent_items(
        FrameCpuWorkload::independent_items(space_count),
        MESH_ASSET_DIRTY_EXPANSION_PARALLEL_CHUNK_SPACES,
    )
}

/// Returns the admission decision for dirty retained-cache refresh.
fn dirty_refresh_admission(
    policy: FrameParallelPolicy,
    space_count: usize,
    estimated_work_units: usize,
) -> ParallelAdmission {
    if estimated_work_units < DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS {
        return ParallelAdmission::Serial;
    }
    policy.admit_independent_items(
        FrameCpuWorkload::new(0, estimated_work_units, space_count),
        DIRTY_SPACE_REFRESH_PARALLEL_CHUNK_SPACES,
    )
}

/// Returns the admission decision for retained prepared-snapshot rebuild.
fn snapshot_rebuild_admission(
    policy: FrameParallelPolicy,
    task_count: usize,
    retained_draw_count: usize,
) -> ParallelAdmission {
    if retained_draw_count < SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS {
        return ParallelAdmission::Serial;
    }
    policy.admit_independent_items(
        FrameCpuWorkload::new(0, retained_draw_count, task_count),
        SNAPSHOT_REBUILD_PARALLEL_CHUNK_TASKS,
    )
}

/// Maintenance counters for backend-owned retained render-world caches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderWorldMaintenanceStats {
    /// Renderer records dirtied by topology or renderer-state changes this frame.
    pub topology_dirty_count: usize,
    /// Renderer records dirtied by material override changes this frame.
    pub material_dirty_count: usize,
    /// Renderer records dirtied only by transform or bounds changes this frame.
    pub transform_only_dirty_count: usize,
    /// Unique transform-root node ids consumed while expanding deferred scene changes.
    pub transform_root_dirty_count: usize,
    /// Retained node-index entries scanned while expanding transform-root dirties.
    pub transform_root_scanned_node_count: usize,
    /// Renderer records found by transform-root dirty expansion.
    pub transform_root_expanded_renderer_count: usize,
    /// Transform-root dirties that covered an entire retained render space.
    pub transform_root_full_space_count: usize,
    /// Renderer records dirtied by mesh-asset mutations this frame.
    pub mesh_asset_dirty_renderer_count: usize,
    /// Renderer records whose retained templates were requested dirty this frame.
    pub dirty_renderer_count: usize,
    /// Renderer records whose retained bounds were requested dirty this frame.
    pub bounds_dirty_renderer_count: usize,
    /// Renderer records whose retained bounds were refreshed this frame.
    pub bounds_refreshed_renderer_count: usize,
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
    /// Prepared snapshots rebuilt only because generated particle meshes changed.
    pub particle_snapshot_rebuild_count: usize,
    /// Prepared-snapshot copy tasks built while rebuilding retained templates.
    pub snapshot_rebuild_task_count: usize,
    /// Retained draw templates considered while rebuilding prepared snapshots.
    pub snapshot_retained_draw_count: usize,
    /// Render spaces reused from the previous prepared snapshot during a partial rebuild.
    pub snapshot_reused_space_count: usize,
    /// Prepared spatial indexes rebuilt because run membership changed.
    pub spatial_rebuild_count: usize,
    /// Prepared spatial indexes refit because dynamic bounds changed.
    pub spatial_refit_count: usize,
    /// Retained draw templates currently cached after maintenance.
    pub retained_template_count: usize,
    /// Render-world caches serving contexts with no draw-prep overrides.
    pub context_invariant_count: usize,
    /// Frames where this render world proved its retained snapshot did not need rebuilding.
    pub steady_state_skip_count: usize,
}

impl RenderWorldMaintenanceStats {
    /// Builds the profiling sample emitted for retained render-world maintenance.
    pub fn profile_sample(self) -> crate::profiling::RenderWorldMaintenanceProfileSample {
        crate::profiling::RenderWorldMaintenanceProfileSample {
            topology_dirty_count: self.topology_dirty_count,
            material_dirty_count: self.material_dirty_count,
            transform_only_dirty_count: self.transform_only_dirty_count,
            transform_root_dirty_count: self.transform_root_dirty_count,
            transform_root_scanned_node_count: self.transform_root_scanned_node_count,
            transform_root_expanded_renderer_count: self.transform_root_expanded_renderer_count,
            transform_root_full_space_count: self.transform_root_full_space_count,
            mesh_asset_dirty_renderer_count: self.mesh_asset_dirty_renderer_count,
            dirty_renderer_count: self.dirty_renderer_count,
            bounds_dirty_renderer_count: self.bounds_dirty_renderer_count,
            bounds_refreshed_renderer_count: self.bounds_refreshed_renderer_count,
            refreshed_renderer_count: self.refreshed_renderer_count,
            refreshed_template_count: self.refreshed_template_count,
            mesh_asset_invalidation_count: self.mesh_asset_invalidation_count,
            full_world_rebuild_count: self.full_world_rebuild_count,
            particle_snapshot_rebuild_count: self.particle_snapshot_rebuild_count,
            snapshot_rebuild_task_count: self.snapshot_rebuild_task_count,
            snapshot_retained_draw_count: self.snapshot_retained_draw_count,
            snapshot_reused_space_count: self.snapshot_reused_space_count,
            spatial_rebuild_count: self.spatial_rebuild_count,
            spatial_refit_count: self.spatial_refit_count,
            retained_template_count: self.retained_template_count,
            context_invariant_count: self.context_invariant_count,
            steady_state_skip_count: self.steady_state_skip_count,
        }
    }

    /// Adds another render world's counters into this aggregate.
    pub fn accumulate(&mut self, other: Self) {
        self.topology_dirty_count += other.topology_dirty_count;
        self.material_dirty_count += other.material_dirty_count;
        self.transform_only_dirty_count += other.transform_only_dirty_count;
        self.transform_root_dirty_count += other.transform_root_dirty_count;
        self.transform_root_scanned_node_count += other.transform_root_scanned_node_count;
        self.transform_root_expanded_renderer_count += other.transform_root_expanded_renderer_count;
        self.transform_root_full_space_count += other.transform_root_full_space_count;
        self.mesh_asset_dirty_renderer_count += other.mesh_asset_dirty_renderer_count;
        self.dirty_renderer_count += other.dirty_renderer_count;
        self.bounds_dirty_renderer_count += other.bounds_dirty_renderer_count;
        self.bounds_refreshed_renderer_count += other.bounds_refreshed_renderer_count;
        self.refreshed_renderer_count += other.refreshed_renderer_count;
        self.refreshed_template_count += other.refreshed_template_count;
        self.mesh_asset_invalidation_count += other.mesh_asset_invalidation_count;
        self.full_space_rebuild_count += other.full_space_rebuild_count;
        self.full_world_rebuild_count += other.full_world_rebuild_count;
        self.particle_snapshot_rebuild_count += other.particle_snapshot_rebuild_count;
        self.snapshot_rebuild_task_count += other.snapshot_rebuild_task_count;
        self.snapshot_retained_draw_count += other.snapshot_retained_draw_count;
        self.snapshot_reused_space_count += other.snapshot_reused_space_count;
        self.spatial_rebuild_count += other.spatial_rebuild_count;
        self.spatial_refit_count += other.spatial_refit_count;
        self.retained_template_count += other.retained_template_count;
        self.context_invariant_count += other.context_invariant_count;
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
    /// Individual renderer records requiring only dynamic bounds refresh.
    dirty_bounds_renderers: HashSet<RenderWorldBoundsDirty>,
    /// Transform-root dirties deferred until world-cache flush has completed.
    dirty_transform_roots: Vec<RenderWorldTransformDirty>,
    /// Mesh assets whose referencing renderer records need refresh.
    dirty_mesh_assets: HashSet<i32>,
    /// Whether generated particle mesh churn requires rebuilding the prepared snapshot.
    particle_snapshot_dirty: bool,
    /// Whether the next prepare must rebuild every scene space.
    full_rebuild_requested: bool,
    /// Mesh-pool mutation generation consumed by this cache.
    mesh_pool_generation: u64,
    /// Whether this cache represents render contexts that have no draw-prep overrides.
    context_invariant: bool,
    /// Pending topology dirty events accumulated since the last maintenance pass.
    pending_topology_dirty_count: usize,
    /// Pending material dirty events accumulated since the last maintenance pass.
    pending_material_dirty_count: usize,
    /// Pending transform-only dirty events accumulated since the last maintenance pass.
    pending_transform_only_dirty_count: usize,
    /// Pending mesh-asset dirty renderer events accumulated since the last maintenance pass.
    pending_mesh_asset_dirty_renderer_count: usize,
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

/// Returns whether a transform root covers every tree in the supplied parent table.
fn transform_roots_cover_space(parents: &[i32], roots: &[i32]) -> bool {
    let mut root_node = None;
    for (node_id, &parent) in parents.iter().enumerate() {
        if parent >= 0 {
            continue;
        }
        if root_node.replace(node_id as i32).is_some() {
            return false;
        }
    }
    let Some(root_node) = root_node else {
        return false;
    };
    roots.contains(&root_node)
}

impl RenderWorld {
    /// Creates an empty render-world cache.
    pub fn new(render_context: RenderingContext) -> Self {
        Self::new_with_context_mode(render_context, false)
    }

    /// Creates an empty render-world cache for contexts with no draw-prep overrides.
    pub fn new_context_invariant(render_context: RenderingContext) -> Self {
        Self::new_with_context_mode(render_context, true)
    }

    /// Creates an empty render-world cache with explicit context compatibility.
    fn new_with_context_mode(render_context: RenderingContext, context_invariant: bool) -> Self {
        Self {
            spaces: HashMap::new(),
            dirty_spaces: HashSet::new(),
            dirty_renderers: HashSet::new(),
            dirty_bounds_renderers: HashSet::new(),
            dirty_transform_roots: Vec::new(),
            dirty_mesh_assets: HashSet::new(),
            particle_snapshot_dirty: false,
            full_rebuild_requested: true,
            mesh_pool_generation: 0,
            context_invariant,
            pending_topology_dirty_count: 0,
            pending_material_dirty_count: 0,
            pending_transform_only_dirty_count: 0,
            pending_mesh_asset_dirty_renderer_count: 0,
            prepared: if context_invariant {
                FramePreparedRenderables::empty_context_invariant(render_context)
            } else {
                FramePreparedRenderables::empty(render_context)
            },
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
                self.pending_topology_dirty_count =
                    self.pending_topology_dirty_count.saturating_add(1);
            }
            for &dirty in &report.render_world_dirty.bounds {
                self.note_bounds_dirty(dirty);
                self.pending_transform_only_dirty_count =
                    self.pending_transform_only_dirty_count.saturating_add(1);
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
    pub fn prepare_for_frame<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
        render_context: RenderingContext,
    ) -> &FramePreparedRenderables
    where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        profiling::scope!("mesh::render_world::prepare_for_frame");
        let mut stats = RenderWorldMaintenanceStats {
            context_invariant_count: usize::from(self.context_invariant),
            ..Default::default()
        };
        let context_changed = !self
            .prepared
            .is_compatible_with_render_context(render_context);
        if context_changed {
            self.full_rebuild_requested = true;
        }
        self.note_mesh_pool_delta(mesh_pool, &mut stats);

        let full_rebuild = self.full_rebuild_requested;
        if full_rebuild {
            stats.full_world_rebuild_count = 1;
            self.mark_all_scene_spaces_dirty(scene);
        }

        self.expand_deferred_dirty_inputs(scene, &mut stats);
        stats.topology_dirty_count = self.pending_topology_dirty_count;
        stats.material_dirty_count = self.pending_material_dirty_count;
        stats.transform_only_dirty_count = self.pending_transform_only_dirty_count;
        stats.mesh_asset_dirty_renderer_count = self.pending_mesh_asset_dirty_renderer_count;
        stats.dirty_renderer_count = self.dirty_renderers.len();
        stats.bounds_dirty_renderer_count = self.dirty_bounds_renderers.len();
        let mut snapshot_dirty_spaces = HashSet::new();
        snapshot_dirty_spaces.extend(self.dirty_spaces.iter().copied());
        snapshot_dirty_spaces.extend(self.dirty_renderers.iter().map(|dirty| dirty.space_id));
        let force_full_snapshot = full_rebuild || context_changed || self.particle_snapshot_dirty;

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
        if !self.dirty_bounds_renderers.is_empty() {
            let outcome = self.refresh_dirty_bounds(scene, mesh_pool, render_context);
            stats.bounds_refreshed_renderer_count += outcome.renderer_count;
            stats.spatial_refit_count += outcome.spatial_refit_count;
        }
        if self.particle_snapshot_dirty {
            stats.particle_snapshot_rebuild_count = 1;
            snapshot_dirty = true;
        }

        if snapshot_dirty {
            profiling::scope!("mesh::render_world::rebuild_snapshot");
            let dirty_spaces = (!force_full_snapshot).then_some(&snapshot_dirty_spaces);
            let snapshot_stats = self.rebuild_prepared_snapshot(
                scene,
                mesh_pool,
                point_render_buffers,
                render_context,
                dirty_spaces,
            );
            stats.snapshot_rebuild_task_count = snapshot_stats.task_count;
            stats.snapshot_retained_draw_count = snapshot_stats.retained_draw_count;
            stats.snapshot_reused_space_count = snapshot_stats.reused_space_count;
            self.particle_snapshot_dirty = false;
            stats.spatial_rebuild_count = 1;
        } else {
            stats.steady_state_skip_count = 1;
        }
        self.full_rebuild_requested = false;
        stats.retained_template_count = self.retained_template_count();
        self.maintenance_stats = stats;
        self.pending_topology_dirty_count = 0;
        self.pending_material_dirty_count = 0;
        self.pending_transform_only_dirty_count = 0;
        self.pending_mesh_asset_dirty_renderer_count = 0;
        crate::profiling::plot_render_world_maintenance(stats.profile_sample());
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
        self.dirty_bounds_renderers
            .retain(|dirty| dirty.space_id != id);
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

    /// Records one renderer row for bounds refresh unless its whole space is already dirty.
    fn note_bounds_dirty(&mut self, dirty: RenderWorldBoundsDirty) {
        if self.dirty_spaces.contains(&dirty.space_id) {
            return;
        }
        if !self.spaces.contains_key(&dirty.space_id) {
            self.dirty_spaces.insert(dirty.space_id);
            return;
        }
        self.dirty_bounds_renderers.insert(dirty);
    }

    /// Records a material override dirty event for this render context.
    fn note_material_override_dirty(&mut self, dirty: RenderWorldMaterialOverrideDirty) {
        if self.context_invariant {
            return;
        }
        if dirty.context != self.prepared.render_context() {
            return;
        }
        match dirty.target {
            MeshRendererOverrideTarget::Static(index) if index >= 0 => {
                self.pending_material_dirty_count =
                    self.pending_material_dirty_count.saturating_add(1);
                self.note_renderer_dirty(RenderWorldRendererDirty {
                    space_id: dirty.space_id,
                    kind: RenderWorldRendererKind::Static,
                    renderable_index: index as usize,
                });
            }
            MeshRendererOverrideTarget::Skinned(index) if index >= 0 => {
                self.pending_material_dirty_count =
                    self.pending_material_dirty_count.saturating_add(1);
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
                self.pending_material_dirty_count =
                    self.pending_material_dirty_count.saturating_add(1);
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
            if crate::particles::is_generated_particle_mesh_asset_id(asset_id) {
                self.particle_snapshot_dirty = true;
                continue;
            }
            self.dirty_mesh_assets.insert(asset_id);
        }
    }

    /// Marks every live scene space dirty for a full rebuild.
    fn mark_all_scene_spaces_dirty(&mut self, scene: &(impl WorldMeshSceneRead + ?Sized)) {
        profiling::scope!("mesh::render_world::mark_all_scene_spaces_dirty");
        self.spaces.retain(|id, _| scene.space(*id).is_some());
        for id in scene.render_space_ids() {
            self.dirty_spaces.insert(id);
        }
        self.dirty_renderers.clear();
        self.dirty_bounds_renderers.clear();
        self.dirty_transform_roots.clear();
        self.dirty_mesh_assets.clear();
    }

    /// Rebuilds the per-view-consumable prepared snapshot from retained renderer templates.
    fn rebuild_prepared_snapshot<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
        render_context: RenderingContext,
        dirty_spaces: Option<&HashSet<RenderSpaceId>>,
    ) -> SnapshotRebuildStats
    where
        S: WorldMeshSceneRead + Sync + ?Sized,
    {
        snapshot::rebuild_prepared_snapshot(
            self,
            scene,
            mesh_pool,
            point_render_buffers,
            render_context,
            dirty_spaces,
        )
    }

    /// Number of retained draw templates currently cached.
    fn retained_template_count(&self) -> usize {
        self.spaces
            .values()
            .map(RenderWorldSpace::retained_template_count)
            .sum()
    }
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new(RenderingContext::default())
    }
}

#[cfg(test)]
mod tests;
