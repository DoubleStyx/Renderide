//! Retained renderer-template refresh routines.

use hashbrown::HashSet;
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, RenderWorldRendererKind, SceneCoordinator};
use crate::shared::RenderingContext;

use super::super::prepared_renderables::{
    expand_skinned_renderer_into, expand_static_renderer_into,
};
use super::state::{RenderWorldRendererTemplate, RenderWorldSpace};

/// Renderer count above which retained-template refresh uses Rayon.
const RENDER_WORLD_PARALLEL_MIN_RENDERERS: usize = 256;

/// Dirty renderer records grouped by render space.
#[derive(Default)]
pub(super) struct DirtyRendererSet {
    /// Static renderer indices to refresh for one space.
    pub(super) static_indices: HashSet<usize>,
    /// Skinned renderer indices to refresh for one space.
    pub(super) skinned_indices: HashSet<usize>,
}

impl DirtyRendererSet {
    /// Number of renderer records in this dirty set.
    pub(super) fn len(&self) -> usize {
        self.static_indices.len() + self.skinned_indices.len()
    }

    /// Returns whether this set contains no renderer records.
    pub(super) fn is_empty(&self) -> bool {
        self.static_indices.is_empty() && self.skinned_indices.is_empty()
    }

    /// Inserts one renderer reference.
    pub(super) fn insert(&mut self, kind: RenderWorldRendererKind, index: usize) {
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

/// Counts returned by retained renderer-template refreshes.
#[derive(Default)]
pub(super) struct RefreshOutcome {
    /// Renderer records refreshed.
    pub(super) renderer_count: usize,
    /// Draw templates retained by those refreshed records.
    pub(super) template_count: usize,
    /// Spaces rebuilt through full-space refresh.
    pub(super) full_space_count: usize,
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
pub(super) fn refresh_render_world_space(
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

/// Refreshes all renderer records in a dirty set for one active render space.
pub(super) fn refresh_renderer_set(
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
