//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores renderer-facing draw templates that are expensive to
//! rediscover every frame.

mod maintenance;
mod mesh_state;
mod refresh;
mod snapshot;
mod state;

use hashbrown::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission};
use crate::gpu_pools::MeshPool;
use crate::scene::{
    MeshRendererOverrideTarget, RenderSpaceId, RenderWorldBoundsDirty,
    RenderWorldMaterialOverrideDirty, RenderWorldRendererDirty, RenderWorldRendererKind,
    RenderWorldTransformDirty, SceneApplyReport, SceneCacheFlushReport, WorldMeshSceneRead,
};
use crate::shared::RenderingContext;

use super::prepared_renderables::FramePreparedRenderables;
use mesh_state::MeshDrawPrepState;
use snapshot::SnapshotRebuildStats;
use state::RenderWorldSpace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderWorldDirtyReason {
    Topology,
    MaterialOverride,
    MeshAsset,
    TransformOnly,
}

impl RenderWorldDirtyReason {
    fn merge(self, other: Self) -> Self {
        if other.priority() < self.priority() {
            other
        } else {
            self
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::Topology => 0,
            Self::MaterialOverride => 1,
            Self::MeshAsset => 2,
            Self::TransformOnly => 3,
        }
    }
}

#[derive(Default)]
struct RenderWorldDirtyReasonCounts {
    topology: usize,
    material: usize,
    transform_only: usize,
    mesh_asset: usize,
}

impl RenderWorldDirtyReasonCounts {
    fn from_dirty_sets(
        renderers: &HashMap<RenderWorldRendererDirty, RenderWorldDirtyReason>,
        bounds: &HashMap<RenderWorldBoundsDirty, RenderWorldDirtyReason>,
    ) -> Self {
        let mut counts = Self::default();
        for &reason in renderers.values().chain(bounds.values()) {
            match reason {
                RenderWorldDirtyReason::Topology => counts.topology += 1,
                RenderWorldDirtyReason::MaterialOverride => counts.material += 1,
                RenderWorldDirtyReason::TransformOnly => counts.transform_only += 1,
                RenderWorldDirtyReason::MeshAsset => counts.mesh_asset += 1,
            }
        }
        counts
    }
}

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

/// Process-local identity source for retained render-world instances.
///
/// [`RenderWorld::prepared_generation`] is monotonic only within one `RenderWorld`. A cross-frame
/// draw-plan cache outlives map-entry replacement, so generation alone cannot distinguish a newly
/// created world from the previous instance at the same generation.
static NEXT_RENDER_WORLD_CACHE_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_render_world_cache_identity() -> u64 {
    NEXT_RENDER_WORLD_CACHE_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

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
    /// Mesh mutations whose draw-preparation-relevant state changed or lacked a retained baseline.
    pub mesh_asset_draw_prep_change_count: usize,
    /// Mesh mutations proven equivalent for draw preparation and suppressed before renderer dirties.
    pub mesh_asset_draw_prep_noop_count: usize,
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
            mesh_asset_draw_prep_change_count: self.mesh_asset_draw_prep_change_count,
            mesh_asset_draw_prep_noop_count: self.mesh_asset_draw_prep_noop_count,
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
        self.mesh_asset_draw_prep_change_count += other.mesh_asset_draw_prep_change_count;
        self.mesh_asset_draw_prep_noop_count += other.mesh_asset_draw_prep_noop_count;
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
    /// Process-unique identity for cross-frame caches retaining rows derived from this world.
    ///
    /// Generations restart when a `RenderWorld` is recreated. Pairing this identity with
    /// [`Self::prepared_generation`] prevents a replacement from aliasing an old retained plan.
    cache_identity: u64,
    /// Per-space retained renderer template records.
    spaces: HashMap<RenderSpaceId, RenderWorldSpace>,
    /// Spaces requiring full retained-template rebuild.
    dirty_spaces: HashSet<RenderSpaceId>,
    /// Individual renderer records requiring retained-template refresh, with the strongest dirty reason.
    dirty_renderers: HashMap<RenderWorldRendererDirty, RenderWorldDirtyReason>,
    /// Individual renderer records requiring only dynamic bounds refresh, with the strongest dirty reason.
    dirty_bounds_renderers: HashMap<RenderWorldBoundsDirty, RenderWorldDirtyReason>,
    /// Transform-root dirties deferred until world-cache flush has completed.
    dirty_transform_roots: Vec<RenderWorldTransformDirty>,
    /// Mesh assets whose referencing renderer records need refresh.
    dirty_mesh_assets: HashSet<i32>,
    /// Exact draw-preparation metadata retained per observed mesh asset.
    mesh_draw_prep_states: HashMap<i32, MeshDrawPrepState>,
    /// Whether generated particle mesh churn requires rebuilding the prepared snapshot.
    particle_snapshot_dirty: bool,
    /// Whether the next prepare must rebuild every scene space.
    full_rebuild_requested: bool,
    /// Mesh-pool mutation generation consumed by this cache.
    mesh_pool_generation: u64,
    /// Whether this cache represents render contexts that have no draw-prep overrides.
    context_invariant: bool,
    /// Dense prepared snapshot consumed by per-view draw collection.
    prepared: FramePreparedRenderables,
    /// Monotonic generation bumped whenever [`Self::prepared`] is rebuilt (any scene change). Lets
    /// per-view collection reuse a cached draw list when the snapshot is unchanged.
    prepared_generation: u64,
    /// Monotonic generation bumped only on non-particle-only snapshot change (static topology,
    /// material override, mesh asset, transform/bounds). Particle-generated mesh churn advances
    /// [`Self::prepared_generation`] every frame but leaves this stable, so camera-independent
    /// GPU-static plan reuse survives particle churn (particle draws are never GPU-static eligible).
    static_generation: u64,
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

fn bounds_dirty_for_renderer(dirty: RenderWorldRendererDirty) -> RenderWorldBoundsDirty {
    RenderWorldBoundsDirty {
        space_id: dirty.space_id,
        kind: dirty.kind,
        renderable_index: dirty.renderable_index,
    }
}

fn renderer_dirty_for_bounds(dirty: RenderWorldBoundsDirty) -> RenderWorldRendererDirty {
    RenderWorldRendererDirty {
        space_id: dirty.space_id,
        kind: dirty.kind,
        renderable_index: dirty.renderable_index,
    }
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
            cache_identity: next_render_world_cache_identity(),
            spaces: HashMap::new(),
            dirty_spaces: HashSet::new(),
            dirty_renderers: HashMap::new(),
            dirty_bounds_renderers: HashMap::new(),
            dirty_transform_roots: Vec::new(),
            dirty_mesh_assets: HashSet::new(),
            mesh_draw_prep_states: HashMap::new(),
            particle_snapshot_dirty: false,
            full_rebuild_requested: true,
            mesh_pool_generation: 0,
            context_invariant,
            prepared: if context_invariant {
                FramePreparedRenderables::empty_context_invariant(render_context)
            } else {
                FramePreparedRenderables::empty(render_context)
            },
            prepared_generation: 0,
            static_generation: 0,
            maintenance_stats: RenderWorldMaintenanceStats::default(),
        }
    }

    /// Generation of the current prepared snapshot; bumped on every rebuild. Per-view collection can
    /// reuse a cached draw list while this is unchanged (the snapshot did not change).
    #[inline]
    pub(crate) fn prepared_generation(&self) -> u64 {
        self.prepared_generation
    }

    /// Generation of the static (non-particle) prepared content. Unchanged across particle-only
    /// frames, so camera-independent GPU-static plan reuse keys on this instead of
    /// [`Self::prepared_generation`].
    #[inline]
    pub(crate) fn static_generation(&self) -> u64 {
        self.static_generation
    }

    /// Stable identity of this retained world instance for cross-frame dependency keys.
    #[inline]
    pub(crate) fn cache_identity(&self) -> u64 {
        self.cache_identity
    }

    /// Marks spaces or renderer records touched by scene apply as needing maintenance.
    pub fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        let has_fine_dirty = !report.render_world_dirty.is_empty();
        if has_fine_dirty {
            for &id in &report.render_world_dirty.full_spaces {
                self.note_space_dirty(id);
            }
            for &dirty in &report.render_world_dirty.renderers {
                self.note_renderer_dirty(dirty, RenderWorldDirtyReason::Topology);
            }
            for &dirty in &report.render_world_dirty.bounds {
                self.note_bounds_dirty(dirty, RenderWorldDirtyReason::TransformOnly);
            }
            self.dirty_transform_roots
                .extend(report.render_world_dirty.transform_roots.iter().cloned());
            for &dirty in &report.render_world_dirty.material_overrides {
                self.note_material_override_dirty(dirty);
            }
        } else {
            for &id in &report.changed_spaces {
                self.note_space_dirty(id);
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
        let dirty_reason_counts = RenderWorldDirtyReasonCounts::from_dirty_sets(
            &self.dirty_renderers,
            &self.dirty_bounds_renderers,
        );
        stats.topology_dirty_count = dirty_reason_counts.topology;
        stats.material_dirty_count = dirty_reason_counts.material;
        stats.transform_only_dirty_count = dirty_reason_counts.transform_only;
        stats.mesh_asset_dirty_renderer_count = dirty_reason_counts.mesh_asset;
        stats.dirty_renderer_count = self.dirty_renderers.len();
        stats.bounds_dirty_renderer_count = self.dirty_bounds_renderers.len();
        let mut snapshot_dirty_spaces = HashSet::new();
        snapshot_dirty_spaces.extend(self.dirty_spaces.iter().copied());
        snapshot_dirty_spaces.extend(self.dirty_renderers.keys().map(|dirty| dirty.space_id));
        let mut particle_only_snapshot_spaces = HashSet::new();
        let force_full_snapshot = full_rebuild || context_changed;

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
        // Any snapshot change up to here is static (topology, material override, mesh asset, full
        // rebuild); the particle branch below is the only particle-driven cause. Records whether the
        // static half of the snapshot changed so GPU-static plan reuse can ignore particle churn.
        let static_snapshot_change = snapshot_dirty;
        let mut prepared_bounds_patched = false;
        if !self.dirty_bounds_renderers.is_empty() {
            let outcome = self.refresh_dirty_bounds(scene, mesh_pool, render_context);
            stats.bounds_refreshed_renderer_count += outcome.renderer_count;
            stats.spatial_refit_count += outcome.spatial_refit_count;
            prepared_bounds_patched = outcome.renderer_count > 0 || outcome.spatial_refit_count > 0;
        }
        if self.particle_snapshot_dirty {
            stats.particle_snapshot_rebuild_count = 1;
            snapshot_dirty = true;
            for id in scene.render_space_ids() {
                if snapshot::space_has_render_buffer_renderers(scene, id) {
                    if !force_full_snapshot && !snapshot_dirty_spaces.contains(&id) {
                        particle_only_snapshot_spaces.insert(id);
                    }
                    snapshot_dirty_spaces.insert(id);
                }
            }
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
                &particle_only_snapshot_spaces,
            );
            stats.snapshot_rebuild_task_count = snapshot_stats.task_count;
            stats.snapshot_retained_draw_count = snapshot_stats.retained_draw_count;
            stats.snapshot_reused_space_count = snapshot_stats.reused_space_count;
            self.particle_snapshot_dirty = false;
            stats.spatial_rebuild_count = 1;
            // The prepared snapshot changed; invalidate any per-view collection reusing it.
            self.prepared_generation = self.prepared_generation.wrapping_add(1);
            if static_snapshot_change {
                self.static_generation = self.static_generation.wrapping_add(1);
            }
        } else if prepared_bounds_patched {
            // Bounds-only maintenance patches matrices/AABBs in the retained prepared rows without
            // rebuilding the snapshot. Cached view draw items embed those values and must not
            // survive this mutation. Transforms/bounds are static content, so both generations bump.
            self.prepared_generation = self.prepared_generation.wrapping_add(1);
            self.static_generation = self.static_generation.wrapping_add(1);
        } else {
            stats.steady_state_skip_count = 1;
        }
        self.full_rebuild_requested = false;
        if full_rebuild {
            self.rebuild_mesh_draw_prep_states(mesh_pool);
        }
        stats.retained_template_count = self.retained_template_count();
        self.maintenance_stats = stats;
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
        self.dirty_renderers.retain(|dirty, _| dirty.space_id != id);
        self.dirty_bounds_renderers
            .retain(|dirty, _| dirty.space_id != id);
        self.dirty_transform_roots
            .retain(|dirty| dirty.space_id != id);
    }

    /// Records a full-space retained-template rebuild and discards redundant finer-grained dirties.
    fn note_space_dirty(&mut self, id: RenderSpaceId) {
        self.dirty_spaces.insert(id);
        self.dirty_renderers.retain(|dirty, _| dirty.space_id != id);
        self.dirty_bounds_renderers
            .retain(|dirty, _| dirty.space_id != id);
        self.dirty_transform_roots
            .retain(|dirty| dirty.space_id != id);
    }

    /// Records one renderer row dirty unless its whole space is already dirty.
    fn note_renderer_dirty(
        &mut self,
        dirty: RenderWorldRendererDirty,
        reason: RenderWorldDirtyReason,
    ) {
        if self.dirty_spaces.contains(&dirty.space_id) {
            return;
        }
        self.dirty_bounds_renderers
            .remove(&bounds_dirty_for_renderer(dirty));
        self.dirty_renderers
            .entry(dirty)
            .and_modify(|existing| *existing = existing.merge(reason))
            .or_insert(reason);
    }

    /// Records one renderer row for bounds refresh unless its whole space is already dirty.
    fn note_bounds_dirty(&mut self, dirty: RenderWorldBoundsDirty, reason: RenderWorldDirtyReason) {
        if self.dirty_spaces.contains(&dirty.space_id) {
            return;
        }
        if self
            .dirty_renderers
            .contains_key(&renderer_dirty_for_bounds(dirty))
        {
            return;
        }
        if !self.spaces.contains_key(&dirty.space_id) {
            self.note_space_dirty(dirty.space_id);
            return;
        }
        self.dirty_bounds_renderers
            .entry(dirty)
            .and_modify(|existing| *existing = existing.merge(reason))
            .or_insert(reason);
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
                self.note_renderer_dirty(
                    RenderWorldRendererDirty {
                        space_id: dirty.space_id,
                        kind: RenderWorldRendererKind::Static,
                        renderable_index: index as usize,
                    },
                    RenderWorldDirtyReason::MaterialOverride,
                );
            }
            MeshRendererOverrideTarget::Skinned(index) if index >= 0 => {
                self.note_renderer_dirty(
                    RenderWorldRendererDirty {
                        space_id: dirty.space_id,
                        kind: RenderWorldRendererKind::Skinned,
                        renderable_index: index as usize,
                    },
                    RenderWorldDirtyReason::MaterialOverride,
                );
            }
            MeshRendererOverrideTarget::Static(_)
            | MeshRendererOverrideTarget::Skinned(_)
            | MeshRendererOverrideTarget::Unknown => {
                self.note_space_dirty(dirty.space_id);
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
            self.mesh_draw_prep_states.clear();
            self.full_rebuild_requested = true;
            return;
        }
        stats.mesh_asset_invalidation_count += delta.changed_asset_ids.len();
        profiling::scope!("mesh::render_world::classify_mesh_asset_deltas");
        for &asset_id in delta.changed_asset_ids {
            if crate::particles::is_generated_particle_mesh_asset_id(asset_id) {
                self.particle_snapshot_dirty = true;
                continue;
            }
            let mesh = mesh_pool.get(asset_id);
            if self
                .mesh_draw_prep_states
                .get(&asset_id)
                .is_some_and(|retained| retained.matches(mesh))
            {
                stats.mesh_asset_draw_prep_noop_count += 1;
                continue;
            }
            self.mesh_draw_prep_states
                .insert(asset_id, MeshDrawPrepState::capture(mesh));
            stats.mesh_asset_draw_prep_change_count += 1;
            self.dirty_mesh_assets.insert(asset_id);
        }
    }

    /// Rebuilds mesh mutation baselines from assets referenced by the refreshed retained world.
    fn rebuild_mesh_draw_prep_states(&mut self, mesh_pool: &MeshPool) {
        profiling::scope!("mesh::render_world::rebuild_mesh_draw_prep_states");
        let asset_ids = self
            .spaces
            .values()
            .flat_map(|space| space.mesh_asset_index.keys().copied())
            .collect::<HashSet<_>>();
        self.mesh_draw_prep_states.clear();
        self.mesh_draw_prep_states.reserve(asset_ids.len());
        for asset_id in asset_ids {
            self.mesh_draw_prep_states.insert(
                asset_id,
                MeshDrawPrepState::capture(mesh_pool.get(asset_id)),
            );
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
        particle_only_spaces: &HashSet<RenderSpaceId>,
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
            particle_only_spaces,
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
