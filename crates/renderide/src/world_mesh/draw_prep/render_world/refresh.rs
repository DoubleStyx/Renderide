//! Retained renderer-template refresh routines.

use hashbrown::HashSet;
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, RenderSpaceView, RenderWorldRendererKind, SceneCoordinator};
use crate::shared::RenderingContext;

use super::super::prepared_renderables::{
    expand_skinned_renderer_into, expand_static_renderer_into,
};
use super::state::{RenderWorldRendererRef, RenderWorldRendererTemplate, RenderWorldSpace};

/// Records per worker chunk when refreshing dense retained renderer tables.
const RENDER_WORLD_REFRESH_CHUNK_SIZE: usize = 64;
/// Retained-renderer refresh chunks assigned to one Rayon worker leaf.
const RENDER_WORLD_REFRESH_CHUNKS_PER_TASK: usize = 1;
/// Renderer count above which retained-template refresh uses Rayon.
const RENDER_WORLD_PARALLEL_MIN_RENDERERS: usize = RENDER_WORLD_REFRESH_CHUNK_SIZE * 2;

/// Minimum dirty density before a dirty refresh scans whole renderer vectors in parallel.
const RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR: usize = 4;

/// Returns whether a dirty renderer set is dense enough to parallel-scan retained tables.
#[inline]
fn should_parallel_scan_dirty_records(dirty_count: usize, total_count: usize) -> bool {
    dirty_count >= RENDER_WORLD_PARALLEL_MIN_RENDERERS
        && dirty_count.saturating_mul(RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR) >= total_count
}

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

    /// Removes this dirty set's stale identities from a render space's reverse indexes.
    pub(super) fn remove_reverse_indexes_from(&self, cached: &mut RenderWorldSpace) {
        profiling::scope!("mesh::render_world::reverse_index_delta::remove");
        for &index in &self.static_indices {
            cached.remove_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Static,
                index,
            });
        }
        for &index in &self.skinned_indices {
            cached.remove_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Skinned,
                index,
            });
        }
    }

    /// Inserts this dirty set's refreshed identities into a render space's reverse indexes.
    pub(super) fn push_reverse_indexes_into(&self, cached: &mut RenderWorldSpace) {
        profiling::scope!("mesh::render_world::reverse_index_delta::push");
        for &index in &self.static_indices {
            cached.push_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Static,
                index,
            });
        }
        for &index in &self.skinned_indices {
            cached.push_reverse_indexes_for_ref(RenderWorldRendererRef {
                kind: RenderWorldRendererKind::Skinned,
                index,
            });
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
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) {
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
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    index: usize,
) {
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
        refresh_all_records(cached, space, scene, mesh_pool, render_context, id);
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
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    id: RenderSpaceId,
) {
    refresh_all_static_records(
        &mut cached.static_renderers,
        space,
        scene,
        mesh_pool,
        render_context,
        id,
    );
    refresh_all_skinned_records(
        &mut cached.skinned_renderers,
        space,
        scene,
        mesh_pool,
        render_context,
        id,
    );
}

/// Refreshes all static retained records, using Rayon only for at least two chunks.
fn refresh_all_static_records(
    records: &mut [RenderWorldRendererTemplate],
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        profiling::scope!("mesh::render_world::refresh_space_parallel::static");
        records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                profiling::scope!("mesh::render_world::refresh_space_parallel::static_chunk");
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    refresh_static_renderer_record(
                        record,
                        space,
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        start_index + offset,
                    );
                }
            });
    } else {
        profiling::scope!("mesh::render_world::refresh_space_serial::static");
        for (index, record) in records.iter_mut().enumerate() {
            refresh_static_renderer_record(
                record,
                space,
                scene,
                mesh_pool,
                render_context,
                space_id,
                index,
            );
        }
    }
}

/// Refreshes all skinned retained records, using Rayon only for at least two chunks.
fn refresh_all_skinned_records(
    records: &mut [RenderWorldRendererTemplate],
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        profiling::scope!("mesh::render_world::refresh_space_parallel::skinned");
        records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                profiling::scope!("mesh::render_world::refresh_space_parallel::skinned_chunk");
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    refresh_skinned_renderer_record(
                        record,
                        space,
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        start_index + offset,
                    );
                }
            });
    } else {
        profiling::scope!("mesh::render_world::refresh_space_serial::skinned");
        for (index, record) in records.iter_mut().enumerate() {
            refresh_skinned_renderer_record(
                record,
                space,
                scene,
                mesh_pool,
                render_context,
                space_id,
                index,
            );
        }
    }
}

/// Refreshes all renderer records in a dirty set for one active render space.
pub(super) fn refresh_renderer_set(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    space: RenderSpaceView<'_>,
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
    dirty_set.remove_reverse_indexes_from(cached);
    let total_count = cached
        .static_renderers
        .len()
        .saturating_add(cached.skinned_renderers.len());
    if should_parallel_scan_dirty_records(dirty_count, total_count) {
        refresh_renderer_set_parallel(
            cached,
            dirty_set,
            space,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    } else {
        refresh_renderer_set_serial(
            cached,
            dirty_set,
            space,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    }
    dirty_set.push_reverse_indexes_into(cached);
    outcome.template_count = refreshed_dirty_template_count(cached, dirty_set);
    outcome
}

/// Refreshes dirty records by scanning dense renderer tables in parallel chunks.
fn refresh_renderer_set_parallel(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::render_world::refresh_renderer_set_parallel");
    refresh_static_dirty_records_dense_scan(
        &mut cached.static_renderers,
        dirty_set,
        space,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
    refresh_skinned_dirty_records_dense_scan(
        &mut cached.skinned_renderers,
        dirty_set,
        space,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

/// Refreshes dirty static records through a dense scan, parallelizing only at two chunks.
fn refresh_static_dirty_records_dense_scan(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                profiling::scope!(
                    "mesh::render_world::refresh_renderer_set_parallel::static_chunk"
                );
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    let index = start_index + offset;
                    if dirty_set.static_indices.contains(&index) {
                        refresh_static_renderer_record(
                            record,
                            space,
                            scene,
                            mesh_pool,
                            render_context,
                            space_id,
                            index,
                        );
                    }
                }
            });
    } else {
        for (index, record) in records.iter_mut().enumerate() {
            if dirty_set.static_indices.contains(&index) {
                refresh_static_renderer_record(
                    record,
                    space,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                    index,
                );
            }
        }
    }
}

/// Refreshes dirty skinned records through a dense scan, parallelizing only at two chunks.
fn refresh_skinned_dirty_records_dense_scan(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    if records.len() >= RENDER_WORLD_PARALLEL_MIN_RENDERERS {
        records
            .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
            .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                profiling::scope!(
                    "mesh::render_world::refresh_renderer_set_parallel::skinned_chunk"
                );
                let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
                for (offset, record) in chunk.iter_mut().enumerate() {
                    let index = start_index + offset;
                    if dirty_set.skinned_indices.contains(&index) {
                        refresh_skinned_renderer_record(
                            record,
                            space,
                            scene,
                            mesh_pool,
                            render_context,
                            space_id,
                            index,
                        );
                    }
                }
            });
    } else {
        for (index, record) in records.iter_mut().enumerate() {
            if dirty_set.skinned_indices.contains(&index) {
                refresh_skinned_renderer_record(
                    record,
                    space,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                    index,
                );
            }
        }
    }
}

/// Refreshes only the explicit dirty indices without scanning dense renderer tables.
fn refresh_renderer_set_serial(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::render_world::refresh_renderer_set_serial");
    for &index in &dirty_set.static_indices {
        if let Some(record) = cached.static_renderers.get_mut(index) {
            refresh_static_renderer_record(
                record,
                space,
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
                space,
                scene,
                mesh_pool,
                render_context,
                space_id,
                index,
            );
        }
    }
}

/// Counts draw templates retained by the records touched in one dirty set.
fn refreshed_dirty_template_count(
    cached: &RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
) -> usize {
    dirty_set
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
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that sparse dirty sets avoid full-vector parallel scans.
    #[test]
    fn dirty_scan_parallelism_requires_threshold_and_density() {
        assert!(!should_parallel_scan_dirty_records(
            RENDER_WORLD_PARALLEL_MIN_RENDERERS - 1,
            RENDER_WORLD_PARALLEL_MIN_RENDERERS,
        ));
        assert!(!should_parallel_scan_dirty_records(
            RENDER_WORLD_PARALLEL_MIN_RENDERERS,
            RENDER_WORLD_PARALLEL_MIN_RENDERERS * (RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR + 1),
        ));
        assert!(should_parallel_scan_dirty_records(
            RENDER_WORLD_PARALLEL_MIN_RENDERERS,
            RENDER_WORLD_PARALLEL_MIN_RENDERERS * RENDER_WORLD_DIRTY_SCAN_MIN_DENSITY_DIVISOR,
        ));
    }
}
