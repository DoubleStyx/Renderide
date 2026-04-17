//! Owns all [`RenderSpaceState`](super::render_space::RenderSpaceState) instances and applies per-frame host data.

mod world_queries;

use std::collections::{HashMap, HashSet};

use glam::Mat4;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::RenderingContext;
use crate::shared::{FrameSubmitData, RenderSpaceUpdate};

use super::camera_apply;
use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::lights::{
    apply_light_renderables_update, apply_lights_buffer_renderers_update, LightCache, ResolvedLight,
};
use super::math::multiply_root;
use super::render_overrides::{
    apply_render_material_overrides_update, apply_render_transform_overrides_update,
    MeshRendererOverrideTarget,
};
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::{compute_world_matrices_for_space, ensure_cache_shapes, WorldTransformCache};

/// Warns when more than one non-overlay render space is marked active (breaks main-camera assumptions).
fn warn_if_multiple_active_non_overlay_spaces(data: &FrameSubmitData) {
    let active_non_overlay = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .count();
    if active_non_overlay > 1 {
        logger::warn!(
            "FrameSubmitData: {active_non_overlay} active non-overlay render spaces (expected at most one for main camera parity)"
        );
    }
}

/// Best-effort reflection probe SH2 host task markers (Unity `RenderSpace.HandleUpdate` preamble).
fn mark_reflection_probe_sh2_task_failures(shm: &mut SharedMemoryAccessor, data: &FrameSubmitData) {
    for update in &data.render_spaces {
        if let Some(ref sh2) = update.reflection_probe_sh2_taks {
            super::reflection_probe_sh2::mark_reflection_probe_sh2_tasks_failed(shm, sh2);
        }
    }
}

#[cfg(test)]
mod tests;

/// Scene registry: one entry per host render space, Unity `RenderingManager` dictionary semantics.
pub struct SceneCoordinator {
    spaces: HashMap<RenderSpaceId, RenderSpaceState>,
    world_caches: HashMap<RenderSpaceId, WorldTransformCache>,
    world_dirty: HashSet<RenderSpaceId>,
    light_cache: LightCache,
    /// Reused when flushing [`Self::world_dirty`] (avoids per-flush allocation).
    world_dirty_flush_scratch: Vec<RenderSpaceId>,
    /// Reused per [`Self::apply_render_space_update_chunk`] for transform removal events.
    transform_removals_scratch: Vec<TransformRemovalEvent>,
    /// Reused in [`Self::remove_render_spaces_not_in_submit`] for ids to drop after a submit.
    remove_spaces_scratch: Vec<RenderSpaceId>,
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
            world_dirty_flush_scratch: Vec::new(),
            transform_removals_scratch: Vec::new(),
            remove_spaces_scratch: Vec::new(),
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

    /// Number of host render spaces currently tracked.
    pub fn render_space_count(&self) -> usize {
        self.spaces.len()
    }

    /// Total static and skinned mesh renderables across all spaces.
    pub fn total_mesh_renderable_count(&self) -> usize {
        self.spaces
            .values()
            .map(|s| s.static_mesh_renderers.len() + s.skinned_mesh_renderers.len())
            .sum()
    }

    /// Resolves lights in world space for `id`, including submission-only buffer fallback.
    pub fn resolve_lights_world(&self, id: RenderSpaceId) -> Vec<ResolvedLight> {
        let sid = id.0;
        self.light_cache
            .resolve_lights_with_fallback(sid, |transform_idx| self.world_matrix(id, transform_idx))
    }

    /// Appends world-space lights for `id` into `out` (caller typically [`Vec::clear`]s once per frame).
    /// Same semantics as [`Self::resolve_lights_world`] without allocating a new [`Vec`].
    pub fn resolve_lights_world_into(&self, id: RenderSpaceId, out: &mut Vec<ResolvedLight>) {
        let sid = id.0;
        self.light_cache.resolve_lights_with_fallback_into(
            sid,
            |transform_idx| self.world_matrix(id, transform_idx),
            out,
        );
    }

    /// Read-only access for debugging / future systems.
    pub fn space(&self, id: RenderSpaceId) -> Option<&RenderSpaceState> {
        self.spaces.get(&id)
    }

    /// Main non-overlay render space, matching Unity's single active main-space expectation.
    pub fn active_main_space(&self) -> Option<&RenderSpaceState> {
        self.spaces
            .values()
            .filter(|s| s.is_active && !s.is_overlay)
            .min_by_key(|s| s.id.0)
    }

    /// Current head-output render context for the main view.
    pub fn active_main_render_context(&self) -> RenderingContext {
        self.active_main_space()
            .map(RenderSpaceState::main_render_context)
            .unwrap_or(RenderingContext::UserView)
    }

    /// Cached world matrix from the host transform hierarchy (parent chain only).
    ///
    /// This matches object/light/bone placement: [`RenderSpaceState::root_transform`] is **not**
    /// applied here—it drives the view basis via [`RenderSpaceState::view_transform`], not mesh
    /// model matrices.
    pub fn world_matrix(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_caches
            .get(&id)?
            .world_matrices
            .get(transform_index)
            .copied()
    }

    /// Alias for [`Self::world_matrix`].
    pub fn world_matrix_local(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_matrix(id, transform_index)
    }

    /// Hierarchy world matrix left-multiplied by [`RenderSpaceState::root_transform`].
    ///
    /// Use only when a host contract explicitly requires this composite. Default rendering uses
    /// [`Self::world_matrix`].
    pub fn world_matrix_including_space_root(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        let local = self.world_matrix(id, transform_index)?;
        Some(multiply_root(local, &space.root_transform))
    }

    /// Material override for the given renderer + slot in the given render context.
    pub fn overridden_material_asset_id(
        &self,
        space_id: RenderSpaceId,
        context: RenderingContext,
        skinned: bool,
        renderable_index: usize,
        slot_index: usize,
    ) -> Option<i32> {
        let space = self.spaces.get(&space_id)?;
        let target = if skinned {
            MeshRendererOverrideTarget::Skinned(renderable_index as i32)
        } else {
            MeshRendererOverrideTarget::Static(renderable_index as i32)
        };
        space.overridden_material_asset_id(context, target, slot_index)
    }

    /// Recomputes cached world matrices for every dirty space (no-op if caches clean).
    pub fn flush_world_caches(&mut self) -> Result<(), SceneError> {
        self.world_dirty_flush_scratch.clear();
        self.world_dirty_flush_scratch
            .extend(self.world_dirty.iter().copied());
        for id in self.world_dirty_flush_scratch.iter().copied() {
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
        warn_if_multiple_active_non_overlay_spaces(data);
        mark_reflection_probe_sh2_task_failures(shm, data);

        let mut seen = HashSet::new();
        for update in &data.render_spaces {
            seen.insert(RenderSpaceId(update.id));
            self.apply_render_space_update_chunk(shm, data.frame_index, update)?;
        }

        self.remove_render_spaces_not_in_submit(&seen);
        Ok(())
    }

    /// Per-space slice of [`FrameSubmitData`]: cameras, transforms, meshes, overrides, lights.
    fn apply_render_space_update_chunk(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        frame_index: i32,
        update: &RenderSpaceUpdate,
    ) -> Result<(), SceneError> {
        let space = self
            .spaces
            .entry(RenderSpaceId(update.id))
            .or_insert_with(|| RenderSpaceState {
                id: RenderSpaceId(update.id),
                ..Default::default()
            });
        space.id = RenderSpaceId(update.id);
        space.apply_update_header(update);

        if let Some(ref cu) = update.cameras_update {
            camera_apply::apply_camera_renderables_update(space, shm, cu, update.id)?;
        }

        let cache = self
            .world_caches
            .entry(RenderSpaceId(update.id))
            .or_default();

        self.transform_removals_scratch.clear();
        if let Some(ref tu) = update.transforms_update {
            super::transforms_apply::apply_transforms_update(
                space,
                cache,
                &mut self.world_dirty,
                RenderSpaceId(update.id),
                super::transforms_apply::TransformsUpdateBuffers {
                    shm,
                    update: tu,
                    frame_index,
                },
                &mut self.transform_removals_scratch,
            )?;
        }
        let transform_removals = self.transform_removals_scratch.as_slice();
        if let Some(ref mu) = update.mesh_renderers_update {
            super::mesh_apply::apply_mesh_renderables_update(
                space,
                shm,
                mu,
                frame_index,
                update.id,
            )?;
        }
        if let Some(ref su) = update.skinned_mesh_renderers_update {
            super::mesh_apply::apply_skinned_mesh_renderables_update(
                space,
                shm,
                su,
                frame_index,
                update.id,
                transform_removals,
            )?;
        }
        if let Some(ref rtu) = update.render_transform_overrides_update {
            apply_render_transform_overrides_update(
                space,
                shm,
                rtu,
                update.id,
                transform_removals,
            )?;
        }
        if let Some(ref rmu) = update.render_material_overrides_update {
            apply_render_material_overrides_update(space, shm, rmu, update.id, transform_removals)?;
        }
        if let Some(ref lu) = update.lights_update {
            apply_light_renderables_update(&mut self.light_cache, shm, lu, update.id)?;
        }
        if let Some(ref lbu) = update.lights_buffer_renderers_update {
            apply_lights_buffer_renderers_update(&mut self.light_cache, shm, lbu, update.id)?;
        }
        Ok(())
    }

    /// Drops render spaces that were absent from this submit’s id set.
    fn remove_render_spaces_not_in_submit(&mut self, seen: &HashSet<RenderSpaceId>) {
        self.remove_spaces_scratch.clear();
        self.remove_spaces_scratch
            .extend(self.spaces.keys().copied().filter(|id| !seen.contains(id)));
        for id in self.remove_spaces_scratch.iter().copied() {
            self.light_cache.remove_space(id.0);
            self.spaces.remove(&id);
            self.world_caches.remove(&id);
            self.world_dirty.remove(&id);
        }
    }
}
