//! Dirty retained renderer refresh.

use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, WorldMeshSceneRead};
use crate::shared::RenderingContext;

use super::super::state::{RenderWorldRendererTemplate, RenderWorldSpace};
use super::{
    DirtyRendererSet, RENDER_WORLD_PARALLEL_MIN_RENDERERS, RENDER_WORLD_REFRESH_CHUNK_SIZE,
    RENDER_WORLD_REFRESH_CHUNKS_PER_TASK, RefreshOutcome, refresh_skinned_renderer_record,
    refresh_static_renderer_record, should_parallel_scan_dirty_records,
};

/// Refreshes all renderer records in a dirty set for one active render space.
pub(in crate::world_mesh::draw_prep::render_world) fn refresh_renderer_set<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) -> RefreshOutcome
where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
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
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    } else {
        refresh_renderer_set_serial(
            cached,
            dirty_set,
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
fn refresh_renderer_set_parallel<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_renderer_set_parallel");
    refresh_static_dirty_records_dense_scan(
        &mut cached.static_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
    refresh_skinned_dirty_records_dense_scan(
        &mut cached.skinned_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

/// Refreshes dirty static records through a dense scan, parallelizing only at two chunks.
fn refresh_static_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
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
fn refresh_skinned_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
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
    scene: &(impl WorldMeshSceneRead + ?Sized),
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
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
