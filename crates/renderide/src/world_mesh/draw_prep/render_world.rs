//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores only renderer-facing expansion products that are
//! expensive to rediscover every frame.

use hashbrown::{HashMap, HashSet};

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;

use super::prepared_renderables::{
    FramePreparedDraw, FramePreparedRenderables, estimated_draw_count, expand_space_into,
    populate_runs_and_material_keys,
};

/// Persistent renderer-facing cache of expanded world-mesh renderables.
pub struct RenderWorld {
    spaces: HashMap<RenderSpaceId, RenderWorldSpace>,
    dirty_spaces: HashSet<RenderSpaceId>,
    full_rebuild_requested: bool,
    mesh_pool_generation: u64,
    prepared: FramePreparedRenderables,
}

#[derive(Default)]
struct RenderWorldSpace {
    active: bool,
    draws: Vec<FramePreparedDraw>,
}

impl RenderWorld {
    /// Creates an empty render-world cache.
    pub fn new(render_context: RenderingContext) -> Self {
        Self {
            spaces: HashMap::new(),
            dirty_spaces: HashSet::new(),
            full_rebuild_requested: true,
            mesh_pool_generation: 0,
            prepared: FramePreparedRenderables::empty(render_context),
        }
    }

    /// Marks spaces touched or removed by scene apply as needing render-world maintenance.
    pub fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        for &id in &report.changed_spaces {
            self.dirty_spaces.insert(id);
        }
        for &id in &report.removed_spaces {
            self.spaces.remove(&id);
            self.dirty_spaces.remove(&id);
        }
        if !report.removed_spaces.is_empty() {
            self.full_rebuild_requested = true;
        }
    }

    /// Marks spaces whose world matrices changed as needing cached bounds refresh.
    pub fn note_cache_flush_report(&mut self, report: &SceneCacheFlushReport) {
        for &id in &report.flushed_spaces {
            self.dirty_spaces.insert(id);
        }
    }

    /// Forces every scene space to be refreshed on the next frame extraction.
    pub fn request_full_rebuild(&mut self) {
        self.full_rebuild_requested = true;
    }

    /// Returns the prepared draw snapshot for this frame, refreshing dirty cached spaces first.
    pub fn prepare_for_frame(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> &FramePreparedRenderables {
        let mesh_pool_generation = mesh_pool.mutation_generation();
        let context_changed = self.prepared.render_context() != render_context;
        let mesh_pool_changed = self.mesh_pool_generation != mesh_pool_generation;
        if context_changed || mesh_pool_changed {
            self.full_rebuild_requested = true;
            self.mesh_pool_generation = mesh_pool_generation;
        }

        let full_rebuild = self.full_rebuild_requested;
        if full_rebuild {
            self.mark_all_scene_spaces_dirty(scene);
        }

        let had_dirty = !self.dirty_spaces.is_empty();
        if had_dirty {
            self.refresh_dirty_spaces(scene, mesh_pool, render_context);
        }

        if full_rebuild || had_dirty || context_changed || mesh_pool_changed {
            self.rebuild_prepared_snapshot(scene, render_context);
        }
        self.full_rebuild_requested = false;
        &self.prepared
    }

    fn mark_all_scene_spaces_dirty(&mut self, scene: &SceneCoordinator) {
        self.spaces.retain(|id, _| scene.space(*id).is_some());
        for id in scene.render_space_ids() {
            self.dirty_spaces.insert(id);
        }
    }

    fn refresh_dirty_spaces(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) {
        let mut dirty_spaces = std::mem::take(&mut self.dirty_spaces);
        for id in dirty_spaces.drain() {
            let Some(space) = scene.space(id) else {
                self.spaces.remove(&id);
                continue;
            };
            let cached = self.spaces.entry(id).or_default();
            cached.active = space.is_active;
            cached.draws.clear();
            if !space.is_active {
                continue;
            }
            let estimate = estimated_draw_count(scene, id);
            if estimate > cached.draws.capacity() {
                cached.draws.reserve(estimate - cached.draws.capacity());
            }
            expand_space_into(&mut cached.draws, scene, mesh_pool, render_context, id);
        }
        self.dirty_spaces = dirty_spaces;
    }

    fn rebuild_prepared_snapshot(
        &mut self,
        scene: &SceneCoordinator,
        render_context: RenderingContext,
    ) {
        self.prepared.render_context = render_context;
        self.prepared.active_space_ids.clear();
        self.prepared.draws.clear();
        self.prepared.runs.clear();
        self.prepared.material_property_keys.clear();

        for id in scene.render_space_ids() {
            let Some(space) = self.spaces.get(&id) else {
                continue;
            };
            if !space.active {
                continue;
            }
            self.prepared.active_space_ids.push(id);
            self.prepared.draws.extend(space.draws.iter().cloned());
        }

        populate_runs_and_material_keys(
            &self.prepared.draws,
            &mut self.prepared.runs,
            &mut self.prepared.material_property_keys,
            &mut self.prepared.material_property_seen_scratch,
        );
    }
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new(RenderingContext::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneCacheFlushReport;

    #[test]
    fn apply_report_marks_changed_spaces_dirty() {
        let mut world = RenderWorld::default();
        world.note_scene_apply_report(&SceneApplyReport {
            frame_index: 7,
            submitted_spaces: vec![RenderSpaceId(1)],
            changed_spaces: vec![RenderSpaceId(1), RenderSpaceId(2)],
            removed_spaces: Vec::new(),
        });

        assert!(world.dirty_spaces.contains(&RenderSpaceId(1)));
        assert!(world.dirty_spaces.contains(&RenderSpaceId(2)));
    }

    #[test]
    fn removed_space_evicts_cached_rows_and_requests_snapshot_rebuild() {
        let mut world = RenderWorld::default();
        world.spaces.insert(
            RenderSpaceId(3),
            RenderWorldSpace {
                active: true,
                draws: Vec::new(),
            },
        );
        world.dirty_spaces.insert(RenderSpaceId(3));

        world.note_scene_apply_report(&SceneApplyReport {
            frame_index: 8,
            submitted_spaces: Vec::new(),
            changed_spaces: Vec::new(),
            removed_spaces: vec![RenderSpaceId(3)],
        });

        assert!(!world.spaces.contains_key(&RenderSpaceId(3)));
        assert!(!world.dirty_spaces.contains(&RenderSpaceId(3)));
        assert!(world.full_rebuild_requested);
    }

    #[test]
    fn cache_flush_marks_bounds_dirty() {
        let mut world = RenderWorld::default();
        world.note_cache_flush_report(&SceneCacheFlushReport {
            flushed_spaces: vec![RenderSpaceId(9)],
        });

        assert!(world.dirty_spaces.contains(&RenderSpaceId(9)));
    }
}
