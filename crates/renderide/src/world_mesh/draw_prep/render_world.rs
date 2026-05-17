//! Persistent CPU render-world cache for world-mesh draw preparation.
//!
//! The scene layer remains the authoritative host-world mirror. This cache lives in the backend
//! side of world-mesh draw prep and stores only renderer-facing expansion products that are
//! expensive to rediscover every frame.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;

use super::prepared_renderables::{
    FramePreparedDraw, FramePreparedRenderables, estimated_draw_count, expand_space_into_aggressive,
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
    chunk_scratch: Vec<Vec<FramePreparedDraw>>,
}

struct DirtyRenderWorldSpace {
    id: RenderSpaceId,
    cached: RenderWorldSpace,
}

fn refresh_render_world_space(
    cached: &mut RenderWorldSpace,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) {
    profiling::scope!("mesh::render_world::refresh_space");
    cached.draws.clear();
    if !cached.active {
        return;
    }
    {
        profiling::scope!("mesh::render_world::reserve_space_draws");
        let estimate = estimated_draw_count(scene, id);
        if estimate > cached.draws.capacity() {
            cached.draws.reserve(estimate - cached.draws.capacity());
        }
    }
    expand_space_into_aggressive(
        &mut cached.draws,
        &mut cached.chunk_scratch,
        scene,
        mesh_pool,
        render_context,
        id,
    );
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

    /// Returns the prepared draw snapshot for this frame, refreshing dirty cached spaces first.
    pub fn prepare_for_frame(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> &FramePreparedRenderables {
        profiling::scope!("mesh::render_world::prepare_for_frame");
        let mesh_pool_generation = mesh_pool.mutation_generation();
        let context_changed = self.prepared.render_context() != render_context;
        let mesh_pool_changed = self.mesh_pool_generation != mesh_pool_generation;
        if context_changed || mesh_pool_changed {
            self.full_rebuild_requested = true;
            self.mesh_pool_generation = mesh_pool_generation;
        }

        let full_rebuild = self.full_rebuild_requested;
        if full_rebuild {
            profiling::scope!("mesh::render_world::mark_all_dirty");
            self.mark_all_scene_spaces_dirty(scene);
        }

        let had_dirty = !self.dirty_spaces.is_empty();
        if had_dirty {
            profiling::scope!("mesh::render_world::refresh_dirty_spaces");
            self.refresh_dirty_spaces(scene, mesh_pool, render_context);
        }

        if full_rebuild || had_dirty || context_changed || mesh_pool_changed {
            profiling::scope!("mesh::render_world::rebuild_snapshot");
            self.rebuild_prepared_snapshot(scene, render_context);
        }
        self.full_rebuild_requested = false;
        &self.prepared
    }

    /// Prepared draw snapshot from the most recent [`Self::prepare_for_frame`] call.
    pub(crate) fn prepared(&self) -> &FramePreparedRenderables {
        &self.prepared
    }

    fn mark_all_scene_spaces_dirty(&mut self, scene: &SceneCoordinator) {
        profiling::scope!("mesh::render_world::mark_all_scene_spaces_dirty");
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
        profiling::scope!("mesh::render_world::refresh_dirty_spaces_inner");
        let mut dirty_spaces = std::mem::take(&mut self.dirty_spaces);
        let mut work = Vec::with_capacity(dirty_spaces.len());
        {
            profiling::scope!("mesh::render_world::collect_dirty_spaces");
            for id in dirty_spaces.drain() {
                let Some(space) = scene.space(id) else {
                    self.spaces.remove(&id);
                    continue;
                };
                let mut cached = self.spaces.remove(&id).unwrap_or_default();
                cached.active = space.is_active();
                work.push(DirtyRenderWorldSpace { id, cached });
            }
        }

        if work.len() < 2 {
            profiling::scope!("mesh::render_world::refresh_dirty_serial");
            for mut slot in work.drain(..) {
                refresh_render_world_space(
                    &mut slot.cached,
                    scene,
                    mesh_pool,
                    render_context,
                    slot.id,
                );
                self.spaces.insert(slot.id, slot.cached);
            }
        } else {
            {
                profiling::scope!("mesh::render_world::refresh_dirty_parallel");
                work.par_iter_mut().for_each(|slot| {
                    profiling::scope!("mesh::render_world::dirty_space_worker");
                    refresh_render_world_space(
                        &mut slot.cached,
                        scene,
                        mesh_pool,
                        render_context,
                        slot.id,
                    );
                });
            }
            {
                profiling::scope!("mesh::render_world::reinsert_dirty_spaces");
                for slot in work.drain(..) {
                    self.spaces.insert(slot.id, slot.cached);
                }
            }
        }
        self.dirty_spaces = dirty_spaces;
    }

    fn rebuild_prepared_snapshot(
        &mut self,
        scene: &SceneCoordinator,
        render_context: RenderingContext,
    ) {
        profiling::scope!("mesh::render_world::rebuild_prepared_snapshot");
        self.prepared.rebuild_from_cached_spaces(
            render_context,
            scene.render_space_ids().filter_map(|id| {
                self.spaces
                    .get(&id)
                    .filter(|s| s.active)
                    .map(|s| (id, s.draws.as_slice()))
            }),
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
    use crate::shared::RenderTransform;

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
                ..Default::default()
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

    #[test]
    fn removed_space_wins_over_changed_space_in_apply_report() {
        let mut world = RenderWorld::default();
        world
            .spaces
            .insert(RenderSpaceId(5), RenderWorldSpace::default());

        world.note_scene_apply_report(&SceneApplyReport {
            frame_index: 9,
            submitted_spaces: vec![RenderSpaceId(5)],
            changed_spaces: vec![RenderSpaceId(5)],
            removed_spaces: vec![RenderSpaceId(5)],
        });

        assert!(!world.spaces.contains_key(&RenderSpaceId(5)));
        assert!(!world.dirty_spaces.contains(&RenderSpaceId(5)));
        assert!(world.full_rebuild_requested);
    }

    #[test]
    fn mark_all_scene_spaces_dirty_retain_only_existing_scene_spaces() {
        let mut scene = SceneCoordinator::new();
        let keep = RenderSpaceId(10);
        scene.test_seed_space_identity_worlds(keep, vec![RenderTransform::default()], vec![-1]);
        let mut world = RenderWorld::default();
        world.spaces.insert(keep, RenderWorldSpace::default());
        world
            .spaces
            .insert(RenderSpaceId(11), RenderWorldSpace::default());

        world.mark_all_scene_spaces_dirty(&scene);

        assert!(world.spaces.contains_key(&keep));
        assert!(!world.spaces.contains_key(&RenderSpaceId(11)));
        assert!(world.dirty_spaces.contains(&keep));
    }

    #[test]
    fn rebuild_prepared_snapshot_skips_inactive_cached_spaces() {
        let mut scene = SceneCoordinator::new();
        let active = RenderSpaceId(20);
        let inactive = RenderSpaceId(21);
        scene.test_seed_space_identity_worlds(active, vec![RenderTransform::default()], vec![-1]);
        scene.test_seed_space_identity_worlds(inactive, vec![RenderTransform::default()], vec![-1]);
        let mut world = RenderWorld::default();
        world.spaces.insert(
            active,
            RenderWorldSpace {
                active: true,
                draws: Vec::new(),
                ..Default::default()
            },
        );
        world.spaces.insert(
            inactive,
            RenderWorldSpace {
                active: false,
                draws: Vec::new(),
                ..Default::default()
            },
        );

        world.rebuild_prepared_snapshot(&scene, RenderingContext::RenderToAsset);

        assert_eq!(
            world.prepared.render_context(),
            RenderingContext::RenderToAsset
        );
        assert_eq!(world.prepared.active_space_ids(), &[active]);
        assert!(world.prepared.draws().is_empty());
    }
}
