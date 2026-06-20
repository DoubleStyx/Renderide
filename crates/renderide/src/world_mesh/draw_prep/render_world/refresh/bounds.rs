//! Dynamic bounds-only retained renderer refresh.

use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, WorldMeshSceneRead};
use crate::shared::RenderingContext;

use super::super::state::{RenderWorldRendererTemplate, RenderWorldSpace};
use super::{
    DirtyRendererSet, RENDER_WORLD_REFRESH_CHUNK_SIZE, RENDER_WORLD_REFRESH_CHUNKS_PER_TASK,
    RefreshOutcome, should_parallel_scan_dirty_records, skinned_renderer_cull_geometry,
    static_renderer_cull_geometry,
};

/// Refreshes only dynamic world/cull bounds for all renderer records in a dirty set.
pub(in crate::world_mesh::draw_prep::render_world) fn refresh_renderer_bounds_set<S>(
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
    profiling::scope!("mesh::render_world::refresh_renderer_bounds_set");
    let dirty_count = dirty_set.len();
    let total_count = cached
        .static_renderers
        .len()
        .saturating_add(cached.skinned_renderers.len());
    if should_parallel_scan_dirty_records(dirty_count, total_count) {
        refresh_renderer_bounds_set_parallel(
            cached,
            dirty_set,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    } else {
        refresh_renderer_bounds_set_serial(
            cached,
            dirty_set,
            scene,
            mesh_pool,
            render_context,
            space_id,
        );
    }
    RefreshOutcome {
        renderer_count: dirty_set.len(),
        ..Default::default()
    }
}

/// Refreshes dirty bounds by scanning dense retained renderer tables in parallel chunks.
fn refresh_renderer_bounds_set_parallel<S>(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("mesh::render_world::refresh_renderer_bounds_set_parallel");
    refresh_static_bounds_dirty_records_dense_scan(
        &mut cached.static_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
    refresh_skinned_bounds_dirty_records_dense_scan(
        &mut cached.skinned_renderers,
        dirty_set,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

/// Refreshes dirty static bounds through a dense scan, parallelizing only at two chunks.
fn refresh_static_bounds_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    records
        .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
        .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            profiling::scope!(
                "mesh::render_world::refresh_renderer_bounds_set_parallel::static_chunk"
            );
            let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
            for (offset, record) in chunk.iter_mut().enumerate() {
                let index = start_index + offset;
                if dirty_set.static_indices.contains(&index) {
                    record.cull_geometry = static_renderer_cull_geometry(
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                }
            }
        });
}

/// Refreshes dirty skinned bounds through a dense scan, parallelizing only at two chunks.
fn refresh_skinned_bounds_dirty_records_dense_scan<S>(
    records: &mut [RenderWorldRendererTemplate],
    dirty_set: &DirtyRendererSet,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    records
        .par_chunks_mut(RENDER_WORLD_REFRESH_CHUNK_SIZE)
        .with_min_len(RENDER_WORLD_REFRESH_CHUNKS_PER_TASK)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            profiling::scope!(
                "mesh::render_world::refresh_renderer_bounds_set_parallel::skinned_chunk"
            );
            let start_index = chunk_index * RENDER_WORLD_REFRESH_CHUNK_SIZE;
            for (offset, record) in chunk.iter_mut().enumerate() {
                let index = start_index + offset;
                if dirty_set.skinned_indices.contains(&index) {
                    record.cull_geometry = skinned_renderer_cull_geometry(
                        scene,
                        mesh_pool,
                        render_context,
                        space_id,
                        index,
                    );
                }
            }
        });
}

/// Refreshes only the explicit dirty bounds indices without scanning dense renderer tables.
fn refresh_renderer_bounds_set_serial(
    cached: &mut RenderWorldSpace,
    dirty_set: &DirtyRendererSet,
    scene: &(impl WorldMeshSceneRead + ?Sized),
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    for &index in &dirty_set.static_indices {
        let cull_geometry =
            static_renderer_cull_geometry(scene, mesh_pool, render_context, space_id, index);
        if let Some(record) = cached.static_renderers.get_mut(index) {
            record.cull_geometry = cull_geometry;
        }
    }
    for &index in &dirty_set.skinned_indices {
        let cull_geometry =
            skinned_renderer_cull_geometry(scene, mesh_pool, render_context, space_id, index);
        if let Some(record) = cached.skinned_renderers.get_mut(index) {
            record.cull_geometry = cull_geometry;
        }
    }
}
