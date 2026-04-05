//! Owns all [`RenderSpaceState`](super::render_space::RenderSpaceState) instances and applies per-frame host data.

use std::collections::{HashMap, HashSet};

use glam::Mat4;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::FrameSubmitData;

use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::lights::{
    apply_light_renderables_update, apply_lights_buffer_renderers_update, LightCache, ResolvedLight,
};
use super::math::multiply_root;
use super::render_space::RenderSpaceState;
use super::world::{compute_world_matrices_for_space, ensure_cache_shapes, WorldTransformCache};

/// Scene registry: one entry per host render space, Unity `RenderingManager` dictionary semantics.
pub struct SceneCoordinator {
    spaces: HashMap<RenderSpaceId, RenderSpaceState>,
    world_caches: HashMap<RenderSpaceId, WorldTransformCache>,
    world_dirty: HashSet<RenderSpaceId>,
    light_cache: LightCache,
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
        }
    }

    /// Host light cache (submissions + incremental updates); CPU-side only.
    pub fn light_cache(&self) -> &LightCache {
        &self.light_cache
    }

    /// Mutable light cache ([`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) store, tests).
    pub fn light_cache_mut(&mut self) -> &mut LightCache {
        &mut self.light_cache
    }

    /// Render space ids currently present (stable order not guaranteed).
    pub fn render_space_ids(&self) -> impl Iterator<Item = RenderSpaceId> + '_ {
        self.spaces.keys().copied()
    }

    /// Resolves lights in world space for `id`, including submission-only buffer fallback.
    pub fn resolve_lights_world(&self, id: RenderSpaceId) -> Vec<ResolvedLight> {
        let sid = id.0;
        self.light_cache
            .resolve_lights_with_fallback(sid, |transform_idx| {
                self.world_matrix_with_root(id, transform_idx)
            })
    }

    /// Read-only access for debugging / future systems.
    pub fn space(&self, id: RenderSpaceId) -> Option<&RenderSpaceState> {
        self.spaces.get(&id)
    }

    /// Cached space-local world matrix (`world * root` via [`Self::world_matrix_with_root`]).
    pub fn world_matrix_local(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_caches
            .get(&id)?
            .world_matrices
            .get(transform_index)
            .copied()
    }

    /// Absolute world matrix including render-space root TRS.
    pub fn world_matrix_with_root(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        let local = self.world_matrix_local(id, transform_index)?;
        Some(multiply_root(local, &space.root_transform))
    }

    /// Recomputes cached world matrices for every dirty space (no-op if caches clean).
    pub fn flush_world_caches(&mut self) -> Result<(), SceneError> {
        let dirty: Vec<RenderSpaceId> = self.world_dirty.iter().copied().collect();
        for id in dirty {
            let Some(space) = self.spaces.get(&id) else {
                self.world_caches.remove(&id);
                self.world_dirty.remove(&id);
                continue;
            };
            let n = space.nodes.len();
            let cache = self.world_caches.entry(id).or_default();
            ensure_cache_shapes(cache, n, false);
            compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, cache)?;
            self.world_dirty.remove(&id);
        }
        Ok(())
    }

    /// Applies [`FrameSubmitData`]: transforms, meshes, skinned meshes, lights (Unity order).
    pub fn apply_frame_submit(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &FrameSubmitData,
    ) -> Result<(), SceneError> {
        let mut seen = HashSet::new();

        for update in &data.render_spaces {
            seen.insert(RenderSpaceId(update.id));
            let space = self
                .spaces
                .entry(RenderSpaceId(update.id))
                .or_insert_with(|| RenderSpaceState {
                    id: RenderSpaceId(update.id),
                    ..Default::default()
                });
            space.id = RenderSpaceId(update.id);
            space.apply_update_header(update);

            let cache = self
                .world_caches
                .entry(RenderSpaceId(update.id))
                .or_default();

            let mut transform_removals = Vec::new();
            if let Some(ref tu) = update.transforms_update {
                transform_removals.extend(super::transforms_apply::apply_transforms_update(
                    space,
                    cache,
                    &mut self.world_dirty,
                    RenderSpaceId(update.id),
                    shm,
                    tu,
                    data.frame_index,
                )?);
            }
            if let Some(ref mu) = update.mesh_renderers_update {
                super::mesh_apply::apply_mesh_renderables_update(
                    space,
                    shm,
                    mu,
                    data.frame_index,
                    update.id,
                )?;
            }
            if let Some(ref su) = update.skinned_mesh_renderers_update {
                super::mesh_apply::apply_skinned_mesh_renderables_update(
                    space,
                    shm,
                    su,
                    data.frame_index,
                    update.id,
                    &transform_removals,
                )?;
            }
            if let Some(ref lu) = update.lights_update {
                apply_light_renderables_update(&mut self.light_cache, shm, lu, update.id)?;
            }
            if let Some(ref lbu) = update.lights_buffer_renderers_update {
                apply_lights_buffer_renderers_update(&mut self.light_cache, shm, lbu, update.id)?;
            }
        }

        let to_remove: Vec<RenderSpaceId> = self
            .spaces
            .keys()
            .copied()
            .filter(|id| !seen.contains(id))
            .collect();
        for id in to_remove {
            self.light_cache.remove_space(id.0);
            self.spaces.remove(&id);
            self.world_caches.remove(&id);
            self.world_dirty.remove(&id);
        }

        Ok(())
    }
}
